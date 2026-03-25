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
        Use LLM to intelligently map messy category values to standardized forms.

        Args:
            unique_values: List of unique values from the column
            column_name: Name of the column (for context)
            context: Optional domain context (e.g., "country codes", "product categories")

        Returns:
            Dictionary mapping original values to standardized values.
            If API fails, returns basic cleaned version (lowercase, stripped).
        """
        if not self.available:
            logger.warning(f"⚠️ Groq unavailable for {column_name}. Using basic text cleaning fallback.")
            print(f"⚠️ Groq unavailable for {column_name}. Using fallback...")
            return self._fallback_text_clean(unique_values)

        prompt = f"""
You are a data consolidation expert. Your ONLY job is to identify which values are VARIATIONS OF THE SAME THING and map them to ONE canonical value.

Column: {column_name}
{f'Context: {context}' if context else ''}

Unique values to consolidate:
{json.dumps(unique_values, indent=2)}

CONSOLIDATION RULES - FOLLOW STRICTLY:

1. CASE INSENSITIVITY:
   "Channel", "channel", "CHANNEL", "cHaNnEl" → ALL map to ONE (e.g., "channel")

2. FORMAT VARIATIONS - TREAT AS IDENTICAL:
   "channel_a", "channel-a", "channela", "channel a", "Channel A" → ALL are the SAME channel
   "New York", "NY", "NEW YORK", "newyork" → ALL map to ONE (e.g., "new_york")

3. WHITESPACE & PUNCTUATION NORMALIZATION:
   "Product Name", "product-name", "product_name", "productname" → SAME VALUE
   Remove/normalize: spaces, dashes, underscores, periods, commas they don't change meaning

4. COMMON ABBREVIATIONS = FULL FORM = ACRONYM (GROUP ALL TOGETHER):
   "USA", "US", "United States", "united states", "u.s.a.", "us" → ONE canonical (e.g., "united_states")
   "NYC", "New York City", "new york city", "newyorkc" → ONE canonical (e.g., "new_york_city")
   "LOL", "lol", "laugh out loud" → ONE canonical

5. SIMILAR SOUNDING / SEMANTICALLY IDENTICAL:
   "yes", "yeah", "yep", "y", "true" → ONE (e.g., "yes")
   "no", "nope", "n", "false", "nah" → ONE (e.g., "no")
   "active", "enabled", "on", "working" → ONE (e.g., "active")
   "inactive", "disabled", "off", "down" → ONE (e.g., "inactive")

6. MINIMIZE CARDINALITY AGGRESSIVELY:
   If you have 30 channels, group by semantic similarity
   Example: ["Fox News", "FOXNEWS", "fox-news", "fox"] → map ALL to "foxnews"
   Example: ["Channel_A", "ChannelA", "channel-a"] → map ALL to "channel_a"

STRATEGY FOR MULTI-CATEGORY COLUMNS (if {column_name} has many distinct themes):
- Group by PREFIX: channels starting with "news" vs "sports" vs "entertainment"
- Group by SIMILARITY: typos, capitalization, spacing variations → one canonical
- When in doubt, CONSOLIDATE not SEPARATE

OUTPUT: Return ONLY valid JSON (no markdown blocks, no explanation).
Each key is an original value, each value is its canonical/consolidated form.
Example: {{"Channel_A": "channel_a", "ChannelA": "channel_a", "channel-a": "channel_a"}}
"""

        try:
            print(f"📡 Calling Groq API for category mapping: {column_name}")
            message = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.choices[0].message.content.strip()
            print(f"✅ Groq API response received for {column_name}")

            # Extract JSON from response (handle markdown code blocks)
            mapping = self._extract_json(response_text)
            logger.info(f"✓ LLM generated mapping for {column_name}: {len(mapping)} entries")
            print(f"✓ Mapping created: {len(mapping)} value pairs")
            print(f"  Raw response preview: {response_text[:300]}")
            print(f"  Parsed mapping: {mapping}")
            return mapping

        except Exception as e:
            logger.error(f"LLM API error for {column_name}: {e}")
            print(f"🔴 API CALL FAILED for {column_name}: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Full error: {str(e)}")
            return self._fallback_text_clean(unique_values)

    def map_messy_numbers(
        self, unique_values: list, column_name: str, context: str = ""
    ) -> Dict[str, Any]:
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

        prompt = f"""
You are a data standardization expert. Your task is to convert messy numeric strings 
to standardized numeric values. Be very explicit with multipliers.

Column: {column_name}
{f'Context: {context}' if context else ''}

Unique values to convert:
{json.dumps(unique_values, indent=2)}

Return ONLY a valid JSON object mapping original values to numeric values.
Example output format:
{{"$100": 100, "5 million": 5000000, "10 lakh": 1000000, "2 crore": 20000000, "1.5k": 1500, "N/A": null}}

CRITICAL RULES - follow exactly:
1. For currency: Remove $, commas, convert to number (e.g., "$1,000" → 1000)
2. For word multipliers, apply these EXACTLY:
   - "thousand" or "k" or "K" = multiply by 1000
   - "million" or "m" or "M" (if standalone word, not within a number) = multiply by 1000000
   - "billion" or "b" or "B" = multiply by 1000000000
   - "lakh" = multiply by 100000 (Indian system)
   - "crore" = multiply by 10000000 (Indian system)
3. Examples:
   - "5 million" → 5 * 1000000 = 5000000
   - "10 lakh" → 10 * 100000 = 1000000
   - "2 crore" → 2 * 10000000 = 20000000
   - "1.5k" → 1.5 * 1000 = 1500
   - "100" → 100
4. Handle decimals: "1.5k" = 1500, "0.5 million" = 500000
5. For non-numeric values or if you cannot convert, use null
6. RETURN VALID JSON ONLY - no other text, no markdown blocks
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
