"""
LLM-assisted value mapping for messy categorical and numeric data.
Uses Groq API to intelligently map values, with human review as a safeguard.
"""

from dotenv import load_dotenv
load_dotenv()  # Load GROQ_API_KEY from .env

import json
import logging
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class LLMAssistant:
    """
    Leverages Groq API to intelligently map messy values to standardized forms.
    Falls back to basic text cleaning if API unavailable.
    """

    def __init__(self, groq_api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Args:
            groq_api_key: Groq API key. If None, try to load from env var GROQ_API_KEY.
            model: Groq model ID (default: llama-3.3-70b-specdec - latest available model)
        """
        self.groq_api_key = groq_api_key
        self.model = model
        self.client = None
        self.available = False
        self._initialize_client()
        print(f"🚀 Using model: {self.model}")

    def _initialize_client(self):
        """Try to initialize Groq client; set available=False if fails."""
        try:
            from groq import Groq
            import os

            api_key = os.getenv("GROQ_API_KEY")
            
            if not api_key:
                logger.warning("❌ GROQ_API_KEY not found in environment variables")
                print("❌ GROQ_API_KEY not found in environment variables")
                self.available = False
                return
            
            if self.groq_api_key:
                self.client = Groq(api_key=self.groq_api_key)
            else:
                # Will auto-load from GROQ_API_KEY env var
                self.client = Groq()
            
            self.available = True
            logger.info("✅ Groq client initialized successfully")
            print("✅ Groq client initialized successfully")
        except ImportError:
            logger.warning("❌ Groq library not installed. Install with: pip install groq")
            print("❌ Groq library not installed. Install with: pip install groq")
            self.available = False
        except Exception as e:
            logger.warning(f"❌ Failed to initialize Groq: {e}")
            print(f"❌ Failed to initialize Groq: {e}")
            self.available = False

    def map_messy_categories(
        self, unique_values: list, column_name: str, context: str = ""
    ) -> Dict[str, str]:
        """
        [DEPRECATED - use standardize_column_values instead]
        Maps messy category values to standardized forms.
        Now delegates to the generic standardize_column_values method.
        """
        return self.standardize_column_values(unique_values, column_name, context)
    
    def map_messy_numbers(
        self, unique_values: list, column_name: str, context: str = ""
    ) -> Dict[str, Any]:
        """
        [DEPRECATED - use standardize_column_values instead]
        Maps messy numeric values to standardized numbers.
        Now delegates to the generic standardize_column_values method.
        """
        return self.standardize_column_values(unique_values, column_name, context)
    
    def standardize_column_values(
        self, unique_values: list, column_name: str, context: str = ""
    ) -> Dict[str, Any]:
        """
        Generic intelligent standardization that detects inconsistency type automatically.
        Handles: numeric/currency, categorical, formatted numbers, mixed formats, etc.
        
        Args:
            unique_values: List of unique values from the column
            column_name: Name of the column
            context: Optional domain context
            
        Returns:
            Dictionary mapping original values to standardized values.
            Automatically detects what kind of standardization is needed.
        """
        if not self.available:
            return self._fallback_smart_standardize(unique_values)
        
        # Analyze the column to detect inconsistency patterns
        analysis = self._analyze_column_patterns(unique_values)
        
        prompt = f"""You are an expert data standardization specialist. Analyze the column values and standardize them intelligently.

Column Name: {column_name}
{f'Domain Context: {context}' if context else ''}
Column Sample Size: {len(unique_values)} unique values

VALUES TO STANDARDIZE:
{json.dumps(unique_values, indent=2)}

DETECTED PATTERNS:
{json.dumps(analysis, indent=2)}

YOUR TASK:
Standardize these values by:

1. DETECT INCONSISTENCY TYPE:
   - Are these formatting variations of the SAME concept? (e.g., "USA", "us", "U.S.A" → all USA)
   - Are these numeric values with different units/multipliers? (e.g., "5 lakh", "500000", "5.0L" → all 500000)
   - Are these semantic duplicates with spelling/case variations?
   - Are these values with currency symbols, percentages, or other decorators?
   - Are these structured data with different delimiters or formats?

2. STANDARDIZATION RULES (apply MOST APPROPRIATE):
   
   a) IF SEMANTIC DUPLICATES (same concept, different format):
      - "USA", "us", "U.S.A" → consolidate to "usa" (pick canonical form)
      - "New York", "new york", "NY" → consider if they're same (consolidate if yes)
   
   b) IF NUMERIC WITH MULTIPLIERS/UNITS:
      - Extract the numeric base value
      - Apply multipliers: k=1000, m=1000000, b=1000000000
      - IMPORTANT: lakh=100000, crore=10000000 (Indian numbering)
      - Remove currency symbols ($, £, €, ₹, etc.)
      - Handle decimals (1.5, 2.3, etc.)
      - Examples: "5 lakh" → 500000, "2.5 crore" → 25000000, "$100" → 100
   
   c) IF STRUCTURED DATA (dates, times, addresses, phone numbers):
      - "2026-04-20", "04/20/2026", "20-04-2026" → standardize to ISO format: "2026-04-20"
      - "+1 234-567-8900", "1(234)567-8900" → "12345678900"
      - "123 Main St", "123 main street" → "123 main street"
   
   d) IF MIXED TYPES (some numeric, some text):
      - Try to convert all to most appropriate type (usually numeric if >80% are numeric)
      - Keep as text if mixed semantic meaning
   
   e) IF TRULY DIFFERENT ENTITIES:
      - "apple" (fruit) vs "apple" (company) - if genuinely different, keep separate
      - "red" vs "blue" - genuinely different colors, keep separate
      - Only consolidate when you're 100% confident they mean the EXACT same thing

3. CONSOLIDATION SAFETY:
   - Conservative: when uncertain → KEEP SEPARATE (don't over-consolidate)
   - Preserve distinctness: "active" ≠ "inactive", "yes" ≠ "no"
   - Many-to-one mapping only for TRUE DUPLICATES or DIRECT SYNONYMS

4. OUTPUT FORMAT:
   Return ONLY valid JSON with no markdown.
   Format: {{"original_value": standardized_value, ...}}
   
   Examples:
   - Categorical: {{"USA": "usa", "us": "usa", "U.S.A": "usa"}}
   - Numeric: {{"5 lakh": 500000, "50 lac": 5000000, "5.5 crore": 55000000}}
   - Dates: {{"04/20/2026": "2026-04-20", "2026-04-20": "2026-04-20"}}
   - Mixed: {{"Product A": "product_a", "product a": "product_a", "prod_a": "product_a"}}

IMPORTANT:
- Return ONLY the JSON mapping, no explanations or markdown
- Each original value must have exactly one standardized output
- If you cannot standardize a value with confidence → return it unchanged
- Null/None values → map to null
"""

        try:
            print(f"\n📡 Calling Groq API for intelligent standardization: {column_name}")
            print(f"   Unique values to process: {unique_values[:10]}")
            
            message = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.choices[0].message.content.strip()
            print(f"✅ Groq API response received for {column_name}")
            print(f"   Raw response (first 500 chars): {response_text[:500]}")
            
            mapping = self._extract_json(response_text)
            logger.info(f"✓ LLM generated standardization for {column_name}: {len(mapping)} entries")
            print(f"✓ Initial mapping from LLM: {mapping}")
            
            # Post-process: Convert string numbers to actual numbers
            mapping_before_coerce = mapping.copy()
            mapping = self._coerce_numeric_values(mapping)
            
            if mapping != mapping_before_coerce:
                print(f"   After numeric coercion:")
                for k, v_before in mapping_before_coerce.items():
                    v_after = mapping.get(k)
                    if v_after != v_before:
                        print(f"     '{k}': {repr(v_before)} ({type(v_before).__name__}) → {repr(v_after)} ({type(v_after).__name__})")
            
            print(f"   Final mapping before validation: {mapping}")
            
            # Validate consolidation
            mapping = self._validate_consolidation(mapping, unique_values, column_name)
            print(f"   Final mapping after validation: {mapping}")
            return mapping
            
        except Exception as e:
            logger.error(f"LLM API error for {column_name}: {e}")
            print(f"🔴 API CALL FAILED for {column_name}: {e}")
            return self._fallback_smart_standardize(unique_values)
    
    def _analyze_column_patterns(self, unique_values: list) -> Dict[str, Any]:
        """Analyze patterns in column values to guide LLM."""
        import re
        
        analysis = {
            "total_unique_values": len(unique_values),
            "detected_patterns": [],
            "sample_values": unique_values[:5],
        }
        
        # Check for numeric/currency patterns
        numeric_count = 0
        currency_symbols = set()
        multipliers = set()
        
        for val in unique_values:
            val_str = str(val).lower()
            
            # Numeric detection
            if re.search(r'\d+\.?\d*', val_str):
                numeric_count += 1
                
                # Currency symbols
                if any(sym in val_str for sym in ['$', '€', '£', '₹', '¥']):
                    currency_symbols.add([s for s in ['$', '€', '£', '₹', '¥'] if s in val_str][0])
                
                # Multiplier words
                multiplier_words = ['lakh', 'lac', 'crore', 'cr', 'thousand', 'k', 'million', 'm', 'billion', 'b', 'h', 'hundred']
                for word in multiplier_words:
                    if word in val_str:
                        multipliers.add(word)
        
        if numeric_count > 0:
            analysis["detected_patterns"].append(f"numeric_values ({numeric_count}/{len(unique_values)})")
        
        if currency_symbols:
            analysis["detected_patterns"].append(f"currency_symbols: {list(currency_symbols)}")
        
        if multipliers:
            analysis["detected_patterns"].append(f"multipliers: {list(multipliers)}")
        
        # Check for date patterns
        date_pattern = r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}-\w+-\d{4}'
        if any(re.search(date_pattern, str(val)) for val in unique_values):
            analysis["detected_patterns"].append("date_values")
        
        # Check for case variation
        has_mixed_case = len([v for v in unique_values if str(v).lower() != str(v).upper()]) > 0
        if has_mixed_case:
            analysis["detected_patterns"].append("case_variations")
        
        # Check for whitespace variation
        has_whitespace_var = any(' ' in str(v) for v in unique_values)
        if has_whitespace_var:
            analysis["detected_patterns"].append("whitespace_variations")
        
        return analysis
    
    def _fallback_smart_standardize(self, unique_values: list) -> Dict[str, Any]:
        """Smart fallback that handles common standardizations without LLM."""
        import re
        
        mapping = {}
        
        for val in unique_values:
            if pd.isna(val):
                mapping[str(val)] = None
                continue
            
            val_str = str(val).strip()
            standardized = val_str
            
            # Try numeric extraction with multipliers (IMPORTANT: lakh/crore must come first)
            numeric_match = re.search(r'([\d.]+)\s*([a-z]*)', val_str.lower())
            if numeric_match:
                try:
                    num = float(numeric_match.group(1))
                    multiplier = numeric_match.group(2).strip()
                    
                    multipliers = {
                        'thousand': 1000, 'k': 1000,
                        'million': 1000000, 'm': 1000000,
                        'billion': 1000000000, 'b': 1000000000,
                        'lakh': 100000, 'lac': 100000, 'l': 100000,
                        'crore': 10000000, 'cr': 10000000,
                        'hundred': 100, 'h': 100
                    }
                    
                    if multiplier in multipliers:
                        standardized = num * multipliers[multiplier]
                        print(f"  ✅ {val_str} → {standardized} (multiplier: {multiplier})")
                    elif multiplier == '':
                        standardized = num
                    else:
                        # Keep as text if multiplier unknown
                        standardized = val_str.lower().strip()
                except Exception as e:
                    print(f"  ⚠️  Could not parse numeric value in {val_str}: {e}")
                    standardized = val_str.lower().strip()
            else:
                # No numeric match - text standardization: lowercase, trim whitespace
                standardized = val_str.lower().strip()
            
            mapping[val_str] = standardized
        
        return mapping
        """
        Use LLM to map messy number-like strings to standardized numeric values.
        Examples: "$100" → 100, "5 mil" → 5000000, "1.5k" → 1500

        Args:
            unique_values: List of unique values from the column
            column_name: Name of the column
            context: Optional context (e.g., "annual salary", "population")

        Returns:
            Dictionary mapping original values to numeric values.
            If API fails, returns best-effort numeric extraction.
        """
        if not self.available:
            logger.warning("Groq unavailable. Using basic numeric extraction fallback.")
            return self._fallback_numeric_extract(unique_values)

        prompt = f"""You are a numeric standardization expert. Convert messy numeric strings to standardized numbers.

Column: {column_name}
{f'Context: {context}' if context else ''}

Values to standardize:
{json.dumps(unique_values, indent=2)}

STANDARDIZATION RULES:

1. COMMON PATTERNS:
   a) Currency Symbols: Remove $ £ € ¥ etc., keep numeric value
   b) Punctuation: Remove commas (1,000 → 1000), keep decimals (1.5)
   c) Whitespace: Strip leading/trailing spaces
   d) Hidden multipliers: Identify words meaning "thousands", "millions", "lakhs", "crores", etc.

2. WORD MULTIPLIERS (apply these):
   - "thousand", "k", "K", "k." = × 1,000
   - "million", "m", "M", "m." = × 1,000,000
   - "billion", "b", "B", "b." = × 1,000,000,000
   - "hundred", "hundred" = × 100
   - "lakh", "lac", "l", "L" = × 100,000  [IMPORTANT for Indian numbers]
   - "crore", "cr", "Cr" = × 10,000,000  [IMPORTANT for Indian numbers]
   - "lakh crore", "lakh-crore" = × 10,000,000,000
   - For domain-specific multipliers: Use context or keep raw if unsure

3. DECIMAL HANDLING:
   - Preserve decimal precision: 1.5 stays 1.5
   - Handle different decimal separators if context allows (. or ,)

4. EDGE CASES:
   - "none", "N/A", "null", "?" → null (return as null)
   - If value is already numeric → return as-is
   - If partially numeric (e.g., "Version 2.0") → extract number (2.0)
   - If you cannot convert with confidence → return null

5. VERIFICATION:
   - Sanity check: converted values should be reasonable given context
   - If value seems wrong (e.g., -1000000 for positive count), flag with null

OUTPUT: Return ONLY valid JSON (no markdown, no explanation).
Format: {{"original_value": numeric_value_or_null, ...}}
Example: {{"$100": 100, "5M": 5000000, "N/A": null}}
"""

        try:
            print(f"📡 Calling Groq API for numeric mapping: {column_name}")
            message = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.choices[0].message.content.strip()
            print(f"✅ Groq API response received for {column_name}")
            mapping = self._extract_json(response_text)
            logger.info(f"✓ LLM generated numeric mapping for {column_name}: {len(mapping)} entries")
            print(f"✓ Numeric mapping created: {len(mapping)} value pairs")
            return mapping

        except Exception as e:
            logger.error(f"LLM API error for {column_name}: {e}")
            print(f"🔴 API CALL FAILED for {column_name}: {e}")
            return self._fallback_numeric_extract(unique_values)

    def _validate_consolidation(self, mapping: Dict[str, str], original_values: list, column_name: str) -> Dict[str, str]:
        """
        Validate that LLM consolidation isn't too aggressive.
        Flags suspicious mappings where semantically different values got consolidated.
        """
        # Count how many source values map to each target
        target_counts = {}
        for original, target in mapping.items():
            if target not in target_counts:
                target_counts[target] = []
            target_counts[target].append(original)
        
        # Flag suspicious consolidations
        for target, sources in target_counts.items():
            # If many different sources map to one target, warn
            if len(sources) > 5:
                print(f"⚠️  WARNING: {column_name} - {len(sources)} different values → '{target}'")
                print(f"    Sources: {sources}")
                print(f"    👉 REVIEW THIS in data editor - may be over-consolidation!")
        
        return mapping
    
    def _coerce_numeric_values(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string representations of numbers to actual numeric values."""
        coerced = {}
        
        for original, standardized in mapping.items():
            if standardized is None:
                coerced[original] = None
            elif isinstance(standardized, (int, float)):
                # Already numeric
                coerced[original] = standardized
            elif isinstance(standardized, str):
                # Try to convert to numeric
                try:
                    # Try integer first
                    if '.' not in standardized:
                        coerced[original] = int(standardized)
                    else:
                        coerced[original] = float(standardized)
                except (ValueError, TypeError):
                    # Keep as string if not numeric
                    coerced[original] = standardized
            else:
                coerced[original] = standardized
        
        return coerced

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from response, handling markdown code blocks and formatting."""
        text = text.strip()
        print(f"  🔍 Raw response text ({len(text)} chars): {text[:200]}...")

        # Remove markdown code blocks if present
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        # Try to find JSON object if wrapped in other text
        if not text.startswith("{"):
            start = text.find("{")
            if start != -1:
                # Find matching closing brace
                brace_count = 0
                for i in range(start, len(text)):
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            text = text[start:i+1]
                            break

        print(f"  🔍 Extracted text ({len(text)} chars): {text}")
        
        try:
            result = json.loads(text)
            print(f"  ✅ JSON parsed successfully: {len(result)} keys")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nText: {text[:500]}")
            print(f"  ❌ JSON parse error: {e}")
            return {}

    def _fallback_text_clean(self, unique_values: list) -> Dict[str, str]:
        """Basic fallback: lowercase, strip whitespace, remove special chars."""
        mapping = {}
        for val in unique_values:
            if pd.isna(val):
                mapping[str(val)] = val
            else:
                cleaned = str(val).lower().strip()
                # Remove some common special chars but keep alphanumeric and spaces
                cleaned = "".join(c if c.isalnum() or c.isspace() else "" for c in cleaned)
                mapping[str(val)] = cleaned
        return mapping

    def _fallback_numeric_extract(self, unique_values: list) -> Dict[str, Any]:
        """Basic fallback: extract numeric + multiplier from string."""
        import re

        mapping = {}
        multipliers = {
            "thousand": 1000,
            "k": 1000,
            "million": 1000000,
            "m": 1000000,
            "billion": 1000000000,
            "b": 1000000000,
            "lakh": 100000,
            "crore": 10000000,
        }
        
        for val in unique_values:
            if pd.isna(val):
                mapping[str(val)] = None
            else:
                val_str = str(val).lower()
                
                # Try to extract number + multiplier
                match = re.search(r"([\d.]+)\s*([a-z]*)", val_str)
                if match:
                    try:
                        num = float(match.group(1))
                        multiplier_str = match.group(2).strip()
                        
                        # Apply multiplier if found
                        if multiplier_str in multipliers:
                            mapping[str(val)] = num * multipliers[multiplier_str]
                        else:
                            mapping[str(val)] = num
                    except ValueError:
                        mapping[str(val)] = None
                else:
                    mapping[str(val)] = None
        
        return mapping

    def validate_mapping(
        self, mapping: Dict[str, Any], expected_type: str = "str"
    ) -> tuple[bool, str]:
        """
        Validate that mapping values make sense.
        Returns (is_valid, message)
        """
        if not isinstance(mapping, dict):
            return False, "Mapping is not a dictionary"

        if expected_type == "numeric":
            for key, val in mapping.items():
                if val is not None and not isinstance(val, (int, float)):
                    return False, f"Expected numeric value for '{key}', got {type(val).__name__}"
        elif expected_type == "str":
            for key, val in mapping.items():
                if val is not None and not isinstance(val, str):
                    return False, f"Expected string value for '{key}', got {type(val).__name__}"

        return True, "Mapping valid"

    def apply_mapping_to_dataframe(
        self, df: pd.DataFrame, column: str, mapping: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Safely apply a mapping to a DataFrame column.
        Unmapped values remain unchanged.
        """
        df = df.copy()
        df[column] = df[column].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)
        return df

    def standardize_column_headers(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Use LLM to standardize messy column headers to clean snake_case.
        
        Args:
            df: DataFrame with messy column names
            
        Returns:
            Dictionary mapping old column names to standardized names
        """
        messy_headers = list(df.columns)
        
        if not self.available:
            logger.warning("⚠️ Groq unavailable. Using basic header cleaning fallback.")
            return self._fallback_header_clean(messy_headers)
        
        prompt = f"""
You are a data schema expert. Standardize these messy column names into clean, consistent names.

Messy column headers:
{json.dumps(messy_headers, indent=2)}

STANDARDIZATION RULES - FOLLOW STRICTLY:

1. CONVERT TO LOWERCASE SNAKE_CASE:
   "Employee_Nbr" → "employee_nbr"
   "Emp Number" → "emp_number"
   "ID emp" → "id_emp"

2. REMOVE SPECIAL CHARACTERS:
   "Col#1" → "col_1"
   "Value($)" → "value"
   "Price (%)" → "price"

3. CONSOLIDATE VARIATIONS OF SAME FIELD:
   ["Emp_Nbr", "Employee Number", "ID_emp", "employee #"] → all become "employee_id" (pick ONE canonical)
   ["custID", "customer_id", "Customer ID"] → all become "customer_id"

4. BE SEMANTIC:
   "Emp_Nbr" → "employee_id" (not "emp_nbr")
   "Qty" → "quantity" (expand abbreviations)
   "Desc" → "description"
   "Comm" → "commission" (or "comment" depending on context - pick most likely)

5. REMOVE REDUNDANCY:
   If columns are: ["User_ID", "User_Name", "User_Email"]
   → ["user_id", "user_name", "user_email"] keep them clear

OUTPUT: Return ONLY valid JSON mapping.
Each key is an original column name, each value is the standardized name.
Example: {{"Emp_Nbr": "employee_id", "Employee Number": "employee_id", "First Name": "first_name"}}
"""
        
        try:
            print(f"📡 Calling Groq API for header standardization ({len(messy_headers)} columns)")
            message = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.choices[0].message.content.strip()
            mapping = self._extract_json(response_text)
            print(f"✅ Header standardization complete: {len(mapping)} columns mapped")
            print(f"   Sample: {dict(list(mapping.items())[:3])}")
            return mapping
        except Exception as e:
            logger.error(f"Header standardization error: {e}")
            print(f"🔴 Header standardization failed: {e}")
            return self._fallback_header_clean(messy_headers)

    def detect_pii_in_dataframe(self, df: pd.DataFrame) -> Dict[str, list]:
        """
        Detect Personally Identifiable Information (PII) in the DataFrame.
        ONLY scans text columns with LOW-MEDIUM cardinality (skips high-cardinality like names/titles).
        Ignores numeric and datetime columns.
        
        Args:
            df: DataFrame to scan
            
        Returns:
            Dictionary mapping column names to list of detected PII types and sample values
        """
        import re
        
        pii_results = {}
        pii_patterns = {
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",  # XXX-XX-XXXX
            "Credit_Card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # XXXX-XXXX-XXXX-XXXX
            "Phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
            "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "Passport": r"\b[A-Z]{1,2}\d{6,9}\b",  # Passport format
            "IP_Address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        }
        
        for col in df.columns:
            # Only check text columns, skip pure numeric/datetime
            if df[col].dtype in ['float64', 'int64', 'int32', 'datetime64']:
                continue
            
            # Skip high-cardinality text columns (likely names, journalist names, titles, etc)
            unique_count = df[col].nunique()
            cardinality_ratio = unique_count / max(len(df), 1)
            if cardinality_ratio > 0.5:  # More than 50% unique = skip (too high cardinality)
                continue
            
            col_pii = []
            try:
                # Convert to string for pattern matching
                col_data = df[col].astype(str)
                
                for pii_type, pattern in pii_patterns.items():
                    matches = col_data.str.findall(pattern)
                    if any(matches.apply(len) > 0):
                        # Found PII
                        found_values = []
                        for match_list in matches:
                            if match_list:
                                found_values.extend(match_list)
                        
                        if found_values:
                            col_pii.append({
                                "type": pii_type,
                                "count": len(set(found_values)),
                                "sample": found_values[0] if found_values else None
                            })
            except Exception as e:
                logger.debug(f"PII scan error for {col}: {e}")
            
            if col_pii:
                pii_results[col] = col_pii
        
        return pii_results

    def _fallback_header_clean(self, headers: list) -> Dict[str, str]:
        """Basic fallback: convert to lowercase snake_case."""
        import re
        mapping = {}
        for header in headers:
            # Convert to lowercase and replace spaces/special chars with underscores
            cleaned = str(header).lower()
            cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned)  # Replace non-alphanumeric with _
            cleaned = re.sub(r'^_+|_+$', '', cleaned)  # Remove leading/trailing underscores
            mapping[header] = cleaned
        return mapping

    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Automatically detect and categorize missing value patterns.
        No human input needed - just flag issues and patterns.
        
        Returns:
            Dictionary with missing value analysis by column and across dataset
        """
        import numpy as np
        
        results = {
            "overall_missing_pct": (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
            "columns_with_missing": {},
            "problematic_columns": [],  # >80% missing or >98% same value
            "patterns": {}
        }
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_pct > 0:
                # Calculate distinctness (uniqueness)
                unique_count = df[col].nunique()
                distinctness = (unique_count / len(df)) * 100
                
                results["columns_with_missing"][col] = {
                    "missing_count": int(missing_count),
                    "missing_pct": round(missing_pct, 2),
                    "unique_values": unique_count,
                    "distinctness_pct": round(distinctness, 2)
                }
                
                # Flag problematic columns
                if missing_pct > 80:
                    results["problematic_columns"].append({
                        "column": col,
                        "issue": "sparse_data",
                        "severity": "high",
                        "recommendation": "Consider dropping - mostly null"
                    })
                
                if distinctness < 2:  # Only 1 or 2 unique values
                    results["problematic_columns"].append({
                        "column": col,
                        "issue": "low_cardinality",
                        "severity": "medium",
                        "recommendation": "Low information - might be categorical flag"
                    })
        
        # Detect missing value PATTERNS (MCAR vs MNAR hints)
        if results["columns_with_missing"]:
            # Check correlation of nullness across columns
            null_corr = df.isna().corr()
            high_corr_pairs = []
            for i in range(len(null_corr.columns)):
                for j in range(i+1, len(null_corr.columns)):
                    corr_val = null_corr.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation
                        high_corr_pairs.append({
                            "col1": null_corr.columns[i],
                            "col2": null_corr.columns[j],
                            "correlation": round(corr_val, 3),
                            "note": "Nullness is correlated - possibly MNAR"
                        })
            
            if high_corr_pairs:
                results["patterns"]["correlated_nullness"] = high_corr_pairs
        
        return results

    def infer_data_types(self, df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Intelligently infer what data types columns SHOULD be.
        No human input - automatic detection with reasoning.
        
        Returns:
            Dictionary mapping column names to inferred type + confidence
        """
        import re
        import numpy as np
        from datetime import datetime
        
        results = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                results[col] = {
                    "current_type": str(df[col].dtype),
                    "inferred_type": "unknown",
                    "confidence": 0,
                    "reason": "All values missing"
                }
                continue
            
            current_type = str(df[col].dtype)
            inferred_type = current_type
            confidence = 100
            reason = []
            
            # Sample for performance
            sample = col_data.sample(n=min(sample_size, len(col_data)), random_state=42)
            
            # Try numeric inference
            numeric_matches = 0
            for val in sample:
                try:
                    float(str(val).replace(',', '').replace('$', ''))
                    numeric_matches += 1
                except (ValueError, TypeError):
                    pass
            
            numeric_pct = (numeric_matches / len(sample)) * 100
            
            if numeric_pct > 80 and current_type == 'object':
                inferred_type = 'numeric'
                confidence = numeric_pct
                reason.append(f"{numeric_pct:.1f}% of values are numeric-like")
            
            # Try date inference
            if inferred_type == current_type:
                date_matches = 0
                date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
                
                for val in sample:
                    for fmt in date_formats:
                        try:
                            datetime.strptime(str(val), fmt)
                            date_matches += 1
                            break
                        except (ValueError, TypeError):
                            pass
                
                date_pct = (date_matches / len(sample)) * 100
                if date_pct > 70 and current_type == 'object':
                    inferred_type = 'datetime'
                    confidence = date_pct
                    reason.append(f"Matches common date formats ({date_pct:.1f}%)")
            
            # Boolean inference
            if inferred_type == current_type and current_type == 'object':
                bool_values = {'true', 'false', 'yes', 'no', '1', '0', 'y', 'n', 'enabled', 'disabled'}
                bool_matches = sum(1 for v in sample if str(v).lower() in bool_values)
                bool_pct = (bool_matches / len(sample)) * 100
                
                if bool_pct > 90 and len(col_data.unique()) <= 5:
                    inferred_type = 'categorical/boolean'
                    confidence = bool_pct
                    reason.append(f"Only {len(col_data.unique())} unique boolean-like values")
            
            # Categorical inference (high cardinality = stay as is)
            if inferred_type == current_type and current_type == 'object':
                unique_count = col_data.nunique()
                unique_pct = (unique_count / len(col_data)) * 100
                
                if unique_pct < 5:
                    inferred_type = 'categorical'
                    confidence = 95
                    reason.append(f"Low cardinality: {unique_count} unique values")
                else:
                    inferred_type = 'text'
                    confidence = 80
                    reason.append(f"High cardinality: {unique_count} unique values")
            
            results[col] = {
                "current_type": current_type,
                "inferred_type": inferred_type,
                "confidence": round(confidence, 1),
                "unique_values": int(col_data.nunique()),
                "reason": "; ".join(reason) if reason else "Retained current type"
            }
        
        return results

    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Automatically detect outliers in numeric columns using IQR method.
        No human input - just flag for review.
        
        Returns:
            Dictionary with outlier detection results by column
        """
        import numpy as np
        
        results = {
            "numeric_columns": {},
            "outlier_rows": []  # Row indices with outliers
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if len(col_data) < 4:  # Need enough data for meaningful stats
                continue
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            if len(outliers) > 0:
                outlier_pct = (len(outliers) / len(col_data)) * 100
                results["numeric_columns"][col] = {
                    "outlier_count": int(len(outliers)),
                    "outlier_pct": round(outlier_pct, 2),
                    "lower_bound": round(lower_bound, 2),
                    "upper_bound": round(upper_bound, 2),
                    "min_outlier": float(outliers.min()),
                    "max_outlier": float(outliers.max()),
                    "recommendation": "Extreme" if outlier_pct > 5 else "Review if expected"
                }
                
                # Track row indices with outliers
                outlier_indices = col_data[
                    (col_data < lower_bound) | (col_data > upper_bound)
                ].index.tolist()
                
                for idx in outlier_indices:
                    if idx not in [r["row_index"] for r in results["outlier_rows"]]:
                        results["outlier_rows"].append({
                            "row_index": int(idx),
                            "columns": [col]
                        })
                    else:
                        # Add to existing row's column list
                        for row_info in results["outlier_rows"]:
                            if row_info["row_index"] == idx:
                                row_info["columns"].append(col)
        
        # Flag rows with multiple outliers
        multi_outlier_rows = [r for r in results["outlier_rows"] if len(r["columns"]) > 1]
        if multi_outlier_rows:
            results["suspicious_rows"] = multi_outlier_rows
        
        return results

    def detect_duplicates(self, df: pd.DataFrame, fuzzy: bool = False) -> Dict[str, Any]:
        """
        Automatically detect exact and near-duplicate rows.
        No human input - just identify suspects.
        
        Args:
            df: DataFrame to scan
            fuzzy: If True, also check for fuzzy/near-duplicates (slower)
        
        Returns:
            Dictionary with duplicate detection results
        """
        results = {
            "total_rows": len(df),
            "exact_duplicates": {},
            "duplicate_row_indices": [],
            "suggestion": ""
        }
        
        # Check for completely identical rows
        duplicated_mask = df.duplicated(keep=False)
        duplicate_rows = df[duplicated_mask]
        
        if len(duplicate_rows) > 0:
            dup_count = len(duplicate_rows) - len(df[duplicated_mask].drop_duplicates())
            results["exact_duplicates"]["found"] = True
            results["exact_duplicates"]["count"] = int(dup_count)
            results["exact_duplicates"]["pct"] = round((dup_count / len(df)) * 100, 2)
            results["duplicate_row_indices"] = df[duplicated_mask].index.tolist()
        else:
            results["exact_duplicates"]["found"] = False
            results["exact_duplicates"]["count"] = 0
        
        # Check for mostly-identical rows (fuzzy)
        if fuzzy:
            from difflib import SequenceMatcher
            
            fuzzy_duplicates = []
            for i in range(len(df)):
                for j in range(i+1, min(i+10, len(df))):  # Compare with next 10 rows
                    # Convert rows to strings and compare
                    row_i = str(df.iloc[i].values)
                    row_j = str(df.iloc[j].values)
                    
                    similarity = SequenceMatcher(None, row_i, row_j).ratio()
                    if similarity > 0.95:  # >95% similar
                        fuzzy_duplicates.append({
                            "row_i": int(i),
                            "row_j": int(j),
                            "similarity": round(similarity, 3)
                        })
            
            if fuzzy_duplicates:
                results["fuzzy_duplicates"] = fuzzy_duplicates[:20]  # Top 20
        
        # Generate suggestion
        if results["exact_duplicates"]["found"]:
            if results["exact_duplicates"]["pct"] > 10:
                results["suggestion"] = "⚠️ HIGH: Consider removing duplicates before analysis"
            else:
                results["suggestion"] = "✓ LOW: Small number of duplicates, review before removal"
        
        return results

    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive automated data quality report.
        Combines all checks into one actionable summary.
        Zero human input needed - just presents findings.
        """
        print("📊 Generating comprehensive data quality report...")
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
            "missing_values": self.analyze_missing_values(df),
            "data_types": self.infer_data_types(df),
            "outliers": self.detect_outliers(df),
            "duplicates": self.detect_duplicates(df, fuzzy=False),
            "pii_data": self.detect_pii_in_dataframe(df),
            "headers": self.standardize_column_headers(df),
            "statistics": self.calculate_statistical_profile(df),
            "domain_validation": self.validate_domain_rules(df),
            "action_items": []
        }
        
        # Generate prioritized action items
        priorities = [
            # Critical
            ("PII_DETECTED", lambda: len(report["pii_data"]) > 0, 
             "🔴 CRITICAL: PII Detected - Review and mask sensitive data immediately"),
            
            ("HIGH_DUPLICATES", lambda: report["duplicates"]["exact_duplicates"].get("pct", 0) > 10,
             "🔴 CRITICAL: >10% duplicate rows - Remove before modeling"),
            
            ("HIGH_MISSING", lambda: report["missing_values"]["overall_missing_pct"] > 30,
             "🟠 SERIOUS: >30% missing data - Consider imputation strategy"),
            
            ("DOMAIN_VIOLATIONS", lambda: len(report["domain_validation"]["violations"]) > 0,
             "🟠 SERIOUS: Domain validation failures - Review data quality"),
            
            # Important
            ("TYPE_MISMATCH", lambda: any(
                d["current_type"] != d["inferred_type"] 
                for d in report["data_types"].values()
            ), "🟡 IMPORTANT: Data type mismatches detected - Consider conversion"),
            
            ("SPARSE_COLS", lambda: len(report["missing_values"]["problematic_columns"]) > 0,
             "🟡 IMPORTANT: Sparse/low-info columns found - Consider dropping"),
            
            # Advisory
            ("OUTLIERS", lambda: len(report["outliers"]["numeric_columns"]) > 0,
             "🔵 ADVISORY: Outliers detected - Review if domain-expected"),
            
            ("SKEWED_DATA", lambda: any(
                abs(d.get("skewness", 0)) > 2 
                for d in report["statistics"]["distributions"].values()
            ), "🔵 ADVISORY: Highly skewed distributions - Consider transformation"),
            
            ("HIGH_CORRELATION", lambda: len(report["statistics"]["high_correlations"]) > 0,
             "🔵 ADVISORY: Multicollinearity detected - Review feature selection"),
        ]
        
        for issue_code, condition, message in priorities:
            if condition():
                report["action_items"].append({
                    "code": issue_code,
                    "message": message,
                    "addressed": False
                })
        
        print(f"✅ Report generated: {len(report['action_items'])} action items")
        return report

    def calculate_statistical_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate distribution statistics for numeric columns.
        No input needed - automatic analysis.
        
        Returns:
            Dictionary with skewness, kurtosis, correlation, percentiles
        """
        import numpy as np
        from scipy import stats
        
        results = {
            "numeric_summary": {},
            "distributions": {},
            "correlation_matrix": None,
            "high_correlations": []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return results
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if len(col_data) < 2:
                continue
            
            # Basic stats
            results["numeric_summary"][col] = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q25": float(col_data.quantile(0.25)),
                "q75": float(col_data.quantile(0.75)),
            }
            
            # Distribution shape
            try:
                skewness = float(stats.skew(col_data))
                kurtosis = float(stats.kurtosis(col_data))
                
                # Interpret skewness
                if abs(skewness) < 0.5:
                    skew_interpretation = "symmetric"
                elif skewness > 0:
                    skew_interpretation = "right-skewed (long tail right)"
                else:
                    skew_interpretation = "left-skewed (long tail left)"
                
                results["distributions"][col] = {
                    "skewness": round(skewness, 3),
                    "skewness_interpretation": skew_interpretation,
                    "kurtosis": round(kurtosis, 3),
                    "kurtosis_interpretation": "heavy-tailed" if kurtosis > 3 else "light-tailed",
                    "normality_hint": "approximately normal" if abs(skewness) < 0.5 and kurtosis < 3 else "non-normal"
                }
            except Exception as e:
                results["distributions"][col] = {"error": str(e)}
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr()
                
                # Find high correlations (>0.7 or <-0.7)
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append({
                                "col1": corr_matrix.columns[i],
                                "col2": corr_matrix.columns[j],
                                "correlation": round(corr_val, 3),
                                "strength": "strong positive" if corr_val > 0.7 else "strong negative"
                            })
                
                results["correlation_matrix"] = corr_matrix.to_dict()
                results["high_correlations"] = sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)
            except Exception as e:
                results["correlation_error"] = str(e)
        
        return results

    def validate_domain_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply common domain-specific validation rules.
        No input needed - automatic validation against common patterns.
        
        Returns:
            Dictionary with validation results and violations
        """
        import re
        from datetime import datetime
        
        results = {
            "total_validations": 0,
            "validations": {},
            "violations": []
        }
        
        # Define common domain rules
        rules = {
            "postal_code": {
                "pattern": r"^\d{5}(-\d{4})?$",
                "description": "US ZIP code format (12345 or 12345-6789)",
                "columns": ["zip", "postal_code", "zipcode", "zip_code"]
            },
            "email": {
                "pattern": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$",
                "description": "Valid email format",
                "columns": ["email", "mail", "e_mail", "email_address"]
            },
            "phone": {
                "pattern": r"^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$",
                "description": "Valid phone number format",
                "columns": ["phone", "phone_number", "contact", "telephone"]
            },
            "url": {
                "pattern": r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&/=]*)$",
                "description": "Valid URL format",
                "columns": ["url", "website", "link", "web_url"]
            },
            "date_in_future": {
                "check": "date_future",
                "description": "Date must be in the past",
                "columns": ["date", "created_date", "date_of_birth", "birthdate", "dob"]
            },
            "positive_number": {
                "check": "positive",
                "description": "Value must be positive",
                "columns": ["price", "amount", "quantity", "count", "age", "salary"]
            }
        }
        
        for col in df.columns:
            col_lower = col.lower()
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            violations_in_col = []
            
            # Check each rule
            for rule_name, rule_config in rules.items():
                # Check if column matches rule's column patterns
                column_matches = any(pattern in col_lower for pattern in rule_config.get("columns", []))
                
                if not column_matches:
                    continue
                
                results["total_validations"] += 1
                
                # Pattern-based validation
                if "pattern" in rule_config:
                    pattern = rule_config["pattern"]
                    invalid_count = 0
                    invalid_samples = []
                    
                    for val in col_data:
                        val_str = str(val).strip()
                        if not re.match(pattern, val_str):
                            invalid_count += 1
                            if len(invalid_samples) < 3:  # Keep first 3 samples
                                invalid_samples.append(val_str)
                    
                    if invalid_count > 0:
                        invalid_pct = (invalid_count / len(col_data)) * 100
                        violations_in_col.append({
                            "rule": rule_name,
                            "invalid_count": invalid_count,
                            "invalid_pct": round(invalid_pct, 2),
                            "samples": invalid_samples,
                            "description": rule_config["description"],
                            "severity": "high" if invalid_pct > 20 else "medium" if invalid_pct > 5 else "low"
                        })
                
                # Date future check
                elif rule_config.get("check") == "date_future":
                    future_count = 0
                    today = datetime.now().date()
                    
                    for val in col_data:
                        try:
                            # Try multiple date formats
                            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
                                try:
                                    val_date = datetime.strptime(str(val), fmt).date()
                                    if val_date > today:
                                        future_count += 1
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            pass
                    
                    if future_count > 0:
                        future_pct = (future_count / len(col_data)) * 100
                        violations_in_col.append({
                            "rule": rule_name,
                            "future_count": future_count,
                            "future_pct": round(future_pct, 2),
                            "description": rule_config["description"],
                            "severity": "high"
                        })
                
                # Positive number check
                elif rule_config.get("check") == "positive":
                    negative_count = 0
                    try:
                        for val in col_data:
                            if pd.notna(val) and float(val) < 0:
                                negative_count += 1
                    except (ValueError, TypeError):
                        pass  # Not numeric, skip
                    
                    if negative_count > 0:
                        negative_pct = (negative_count / len(col_data)) * 100
                        violations_in_col.append({
                            "rule": rule_name,
                            "negative_count": negative_count,
                            "negative_pct": round(negative_pct, 2),
                            "description": rule_config["description"],
                            "severity": "medium" if negative_pct > 5 else "low"
                        })
            
            if violations_in_col:
                results["validations"][col] = violations_in_col
                results["violations"].extend(violations_in_col)
        
        return results
