"""
Pattern-based detection and normalization for ambiguous/mixed-type columns.
Detects common patterns (numeric text, categorical variations) and prompts user for mappings.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List


class PatternDetector:
    """
    Detects columns with ambiguous patterns:
    - Numeric text (5 million, 2 lakh, 3.5K, etc.)
    - Categorical variations (m/male/man, f/female/fem, etc.)
    - Mixed types within a column
    """

    def __init__(self):
        self.numeric_patterns = {
            'million': 1_000_000,
            'million': 1_000_000,
            'lakh': 100_000,
            'crore': 10_000_000,
            'thousand': 1_000,
            'k': 1_000,
            'b': 1_000_000_000,
            'trillion': 1_000_000_000_000,
        }

        # Common categorical mappings (for detection only, user confirms)
        self.categorical_mappings = {
            'gender': {
                'male': ['m', 'male', 'Male', 'boy', 'male '],
                'female': ['f', 'Female', 'female', 'woman', 'girl'],
                'other': ['other', 'u', 'unknown']
            },
            'boolean': {
                'yes': ['y', 'yes', 'true', '1', 'on'],
                'no': ['n', 'no', 'false', '0', 'off']
            }
        }

    def detect_ambiguous_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Scan columns and return detected patterns needing user intervention.
        
        Returns:
            {col_name: {type: 'numeric_text'|'categorical_mixed'|'mixed_types', 
                       samples: [...], detected_pattern: {...}}}
        """
        ambiguous = {}

        for col in df.columns:
            # Skip purely numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            col_data = df[col].dropna().astype(str)
            if len(col_data) == 0:
                continue

            # Check for numeric text patterns (5 million, 2 lakh, etc)
            numeric_text_matches = self._detect_numeric_text(col_data)
            if numeric_text_matches:
                ambiguous[col] = {
                    'type': 'numeric_text',
                    'samples': col_data.head(5).tolist(),
                    'matches_found': numeric_text_matches,
                    'match_pct': len(numeric_text_matches) / len(col_data) * 100
                }
                continue

            # Check for mixed categorical variations (gender, yes/no, etc)
            categorical_match = self._detect_categorical_variations(col_data)
            if categorical_match:
                ambiguous[col] = {
                    'type': 'categorical_mixed',
                    'samples': col_data.head(5).tolist(),
                    'detected_category': categorical_match['category'],
                    'variations': categorical_match['variations'],
                    'match_pct': categorical_match['match_pct']
                }

        return ambiguous

    def _detect_numeric_text(self, col_data: pd.Series) -> List[Tuple[str, float]]:
        """Detect values like '5 million', '2.5 lakh', etc."""
        pattern_str = '|'.join(self.numeric_patterns.keys())
        regex = rf'([\d.]+)\s*({pattern_str})'
        
        matches = []
        for val in col_data.unique():
            match = re.search(regex, str(val).lower())
            if match:
                matches.append(val)
        
        return matches

    def _detect_categorical_variations(self, col_data: pd.Series) -> Dict:
        """Detect if column is actually a categorical with variations (m/male/man, etc)."""
        unique_vals = col_data.unique()
        total_unique = len(unique_vals)
        
        # Only detect if low cardinality (likely categorical)
        if total_unique > 10:
            return None

        for category_type, variants_map in self.categorical_mappings.items():
            all_variants = [v for vars in variants_map.values() for v in vars]
            
            matched_cols = [val for val in unique_vals 
                          if str(val).lower().strip() in all_variants]
            
            if len(matched_cols) / total_unique >= 0.7:  # 70%+ match
                return {
                    'category': category_type,
                    'variations': list(matched_cols),
                    'match_pct': len(matched_cols) / total_unique * 100
                }

        return None


class PatternNormalizer:
    """
    Applies user-defined mappings to normalize ambiguous columns.
    """

    def __init__(self, user_mappings: Dict = None):
        """
        user_mappings format:
        {
            'col_name': {
                'type': 'numeric_text' | 'categorical',
                'mappings': {
                    '5 million': 5000000,  # for numeric_text
                    'm': 'male',           # for categorical
                    'fem': 'female'
                }
            }
        }
        """
        self.user_mappings = user_mappings or {}

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply user mappings to columns."""
        df = df.copy()

        for col, mapping_config in self.user_mappings.items():
            if col not in df.columns:
                continue

            mapping_type = mapping_config.get('type')
            mappings = mapping_config.get('mappings', {})

            if mapping_type == 'numeric_text':
                df[col] = df[col].apply(
                    lambda x: self._normalize_numeric_text(x, mappings)
                )
            elif mapping_type == 'categorical':
                df[col] = df[col].apply(
                    lambda x: self._normalize_categorical(x, mappings)
                )

        return df

    def _normalize_numeric_text(self, value, mappings: Dict):
        """Convert '5 million' to 5000000, etc."""
        if pd.isna(value):
            return np.nan

        value_str = str(value).strip()

        # Check if it's in user mappings
        if value_str in mappings:
            return mappings[value_str]

        # Try generic pattern matching
        pattern_str = '|'.join(['million', 'lakh', 'crore', 'thousand', 'k', 'b', 'trillion'])
        regex = rf'([\d.]+)\s*({pattern_str})'
        match = re.search(regex, value_str.lower())

        if match:
            num_str, unit = match.groups()
            multipliers = {
                'million': 1_000_000,
                'lakh': 100_000,
                'crore': 10_000_000,
                'thousand': 1_000,
                'k': 1_000,
                'b': 1_000_000_000,
                'trillion': 1_000_000_000_000,
            }
            try:
                return float(num_str) * multipliers.get(unit.lower(), 1)
            except:
                return np.nan

        # Try direct conversion
        try:
            return float(value_str)
        except:
            return np.nan

    def _normalize_categorical(self, value, mappings: Dict):
        """Map m → male, fem → female, etc."""
        if pd.isna(value):
            return np.nan

        value_str = str(value).lower().strip()

        # Exact match in mappings
        if value_str in mappings:
            return mappings[value_str]

        # If not mapped, return original (user may have custom values)
        return value
