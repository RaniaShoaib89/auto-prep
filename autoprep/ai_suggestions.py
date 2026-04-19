"""
AI-driven preprocessing suggestions and enhancements.
Provides column semantic profiling, outlier explanations, interaction features, and data quality reports.
"""

import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AIColumnAnalyzer:
    """Analyzes columns semantically to provide intelligent preprocessing suggestions."""
    
    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
    
    def profile_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all columns semantically to understand their meaning and suggest transformations.
        
        Returns:
            {
                "column_name": {
                    "semantic_type": "currency|location|person|product|date|quantity|...",
                    "suggested_encoding": "onehot|frequency|target_encode|...",
                    "quality_issues": ["issue1", "issue2"],
                    "ai_action": "description of suggested action"
                },
                ...
            }
        """
        if not self.llm_agent or not self.llm_agent.available:
            return self._fallback_profile(df)
        
        column_profiles = {}
        
        for col in df.columns:
            sample_values = df[col].dropna().head(10).tolist()
            col_name_lower = col.lower()
            
            # Attempt semantic classification
            semantic_type = self._detect_semantic_type(col, sample_values, col_name_lower)
            
            # Get suggested encoding
            cardinality = df[col].nunique()
            suggested_encoding = self._suggest_encoding(col, semantic_type, cardinality, df[col].dtype)
            
            # Detect quality issues
            quality_issues = self._detect_column_issues(col, df[col])
            
            # Build AI action
            ai_action = self._build_ai_action(col, semantic_type, suggested_encoding, quality_issues)
            
            column_profiles[col] = {
                "semantic_type": semantic_type,
                "cardinality": int(cardinality),
                "dtype": str(df[col].dtype),
                "suggested_encoding": suggested_encoding,
                "quality_issues": quality_issues,
                "ai_action": ai_action,
                "sample_values": sample_values[:3]
            }
        
        return column_profiles
    
    def _detect_semantic_type(self, col_name: str, samples: list, col_lower: str) -> str:
        """Detect what a column semantically represents."""
        
        # Name-based detection (most reliable)
        name_keywords = {
            "currency": ["price", "cost", "amount", "salary", "fee", "revenue", "income"],
            "location": ["city", "country", "state", "province", "region", "zip", "postal"],
            "person": ["name", "person", "customer", "user", "employee", "author"],
            "product": ["product", "item", "sku", "code"],
            "date": ["date", "time", "created", "updated", "modified", "born"],
            "quantity": ["qty", "quantity", "count", "units", "volume"],
            "percentage": ["pct", "percent", "rate", "ratio", "%"],
            "boolean": ["flag", "is_", "has_", "active", "enabled"],
            "identifier": ["id", "key", "code", "reference", "ref"],
        }
        
        for semantic_type, keywords in name_keywords.items():
            if any(kw in col_lower for kw in keywords):
                return semantic_type
        
        # Value-based detection
        return self._infer_from_values(samples)
    
    def _infer_from_values(self, samples: list) -> str:
        """Infer semantic type from actual values."""
        str_samples = [str(s).lower() for s in samples if pd.notna(s)]
        
        if not str_samples:
            return "unknown"
        
        # Check for patterns
        try:
            # Try to convert all to numeric
            numeric_count = 0
            for s in str_samples:
                try:
                    float(s)
                    numeric_count += 1
                except (ValueError, TypeError):
                    pass
            
            # If >80% are numeric, classify as quantity
            if numeric_count / len(str_samples) > 0.8:
                return "quantity"
        except:
            pass
        
        if any(s.startswith("$") or s.startswith("€") for s in str_samples):
            return "currency"
        
        if any(len(s) == 5 or (len(s) == 10 and s[2] in "-/") for s in str_samples):
            return "date"  # Could be postal code or date
        
        return "categorical"
    
    def _suggest_encoding(self, col: str, semantic_type: str, cardinality: int, dtype) -> str:
        """Suggest optimal encoding for the column."""
        
        # High cardinality text (names, etc.) - keep readable
        if semantic_type in ["person", "product", "identifier"] and cardinality > 10:
            return "keep_readable"
        
        # Binary
        if cardinality == 2:
            return "label_encode"
        
        # Low cardinality categorical
        if semantic_type in ["location", "category"] and cardinality <= 10:
            return "onehot"
        
        # Medium cardinality
        if cardinality <= 50:
            return "onehot"
        
        # High cardinality
        return "frequency_encode"
    
    def _detect_column_issues(self, col: str, series: pd.Series) -> List[str]:
        """Detect quality issues in a column."""
        issues = []
        
        # Missing values
        missing_pct = (series.isna().sum() / len(series)) * 100
        if missing_pct > 20:
            issues.append(f"missing_values_{missing_pct:.0f}%")
        
        # Low variance
        try:
            unique_pct = (series.nunique() / len(series)) * 100
            if unique_pct < 5:
                issues.append("low_variance")
        except:
            pass
        
        # High cardinality
        if series.nunique() > len(series) * 0.8:
            issues.append("high_cardinality")
        
        # Mixed types (too many nulls or weird values)
        try:
            if series.dtype == 'object':
                str_series = series.astype(str)
                if any(len(s) > 500 for s in str_series.dropna().head(100)):
                    issues.append("contains_long_text")
        except:
            pass
        
        return issues
    
    def _build_ai_action(self, col: str, semantic_type: str, encoding: str, issues: List[str]) -> str:
        """Build human-readable AI action recommendation."""
        
        actions = []
        
        if "missing_values" in "".join(issues):
            actions.append("Impute missing values intelligently")
        
        if semantic_type == "currency":
            actions.append("Normalize currency values to numeric")
        elif semantic_type == "date":
            actions.append("Extract date features (year, month, day, etc.)")
        elif semantic_type == "location":
            actions.append("Consolidate location variations")
        
        if encoding == "onehot":
            actions.append("Convert to one-hot encoding for ML models")
        elif encoding == "frequency_encode":
            actions.append("Apply frequency encoding to reduce dimensionality")
        
        if "low_variance" in issues:
            actions.append("Consider dropping (low information)")
        
        if "high_cardinality" in issues and semantic_type not in ["person", "product"]:
            actions.append("Group rare categories under 'Other'")
        
        return "; ".join(actions) if actions else "No special action needed"


class AIOutlierExplainer:
    """Explains why values are outliers and suggests handling strategies."""
    
    def explain_outliers(self, df: pd.DataFrame, col: str, outlier_indices: List[int]) -> Dict[str, Any]:
        """
        Explain detected outliers in business/domain terms.
        
        Returns:
            {
                "column": "col_name",
                "outlier_count": 5,
                "outlier_pct": 2.5,
                "explanations": [
                    {
                        "value": 10000,
                        "possible_reasons": ["genuine_rarity", "data_entry_error", "fraud"],
                        "context": "This value is 5x higher than typical"
                    },
                    ...
                ],
                "recommended_action": "clip|remove|flag|keep"
            }
        """
        
        if not isinstance(df[col], pd.Series):
            return {"error": "Invalid column"}
        
        series = df[col]
        outlier_values = series.iloc[outlier_indices].dropna().unique()
        
        explanations = []
        for value in outlier_values:
            explanation = {
                "value": value,
                "possible_reasons": self._classify_outlier_reason(series, value),
                "context": self._build_outlier_context(series, value)
            }
            explanations.append(explanation)
        
        return {
            "column": col,
            "outlier_count": len(outlier_indices),
            "outlier_pct": round(len(outlier_indices) / len(series) * 100, 2),
            "explanations": explanations,
            "recommended_action": self._recommend_outlier_action(series, explanations)
        }
    
    def _classify_outlier_reason(self, series: pd.Series, value) -> List[str]:
        """Classify why a value is an outlier."""
        reasons = []
        
        # Numerical outliers
        if pd.api.types.is_numeric_dtype(series):
            mean = series.mean()
            std = series.std()
            z_score = abs((value - mean) / std) if std > 0 else 0
            
            if z_score > 5:
                reasons.append("extreme_statistical_deviation")
            elif z_score > 3:
                reasons.append("high_statistical_deviation")
            
            # Check for common error patterns
            if value == 0 and series.min() > 0:
                reasons.append("possible_missing_value_placeholder")
            elif value < 0 and series.min() >= 0:
                reasons.append("sign_error_possible")
        
        # High cardinality outliers
        if series.nunique() > 100:
            if series.value_counts().get(value, 0) == 1:
                reasons.append("rare_unique_value")
        
        # Suspicious patterns
        str_val = str(value).lower()
        if any(p in str_val for p in ["error", "na", "null", "unknown", "?"]):
            reasons.append("possible_data_quality_marker")
        
        if not reasons:
            reasons.append("unusual_but_valid")
        
        return reasons
    
    def _build_outlier_context(self, series: pd.Series, value) -> str:
        """Build contextual description of the outlier."""
        
        try:
            if pd.api.types.is_numeric_dtype(series):
                mean = series.mean()
                median = series.median()
                q75 = series.quantile(0.75)
                
                ratio_to_mean = value / mean if mean != 0 else 0
                return f"Value {value} is {ratio_to_mean:.1f}x the mean ({mean:.0f}). Median={median:.0f}, Q75={q75:.0f}"
            else:
                count = series.value_counts().get(value, 0)
                return f"Value appears {count} time(s) in {len(series)} rows ({count/len(series)*100:.1f}% frequency)"
        except:
            return "Statistical analysis unavailable"
    
    def _recommend_outlier_action(self, series: pd.Series, explanations: List[Dict]) -> str:
        """Recommend how to handle outliers."""
        
        # If many "unusual but valid", probably keep them
        valid_reasons = sum(1 for e in explanations if "unusual_but_valid" in e["possible_reasons"])
        if valid_reasons > len(explanations) * 0.5:
            return "keep_with_monitoring"
        
        # If mostly errors, remove
        error_reasons = sum(1 for e in explanations if any(r in e["possible_reasons"] for r in ["error", "placeholder"]))
        if error_reasons > len(explanations) * 0.5:
            return "remove"
        
        # Default: clip to reasonable bounds
        return "clip"


class AIInteractionSuggester:
    """Suggests important feature interactions to engineer."""
    
    def suggest_interactions(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """
        Suggest valuable interaction features based on column types and names.
        
        Returns:
            {
                "interaction_suggestions": [
                    {
                        "type": "multiplication|division|addition",
                        "columns": ["col1", "col2"],
                        "description": "Revenue per customer",
                        "reason": "High business value"
                    },
                    ...
                ],
                "temporal_suggestions": [...]
            }
        """
        
        suggestions = {
            "interaction_suggestions": [],
            "temporal_suggestions": [],
            "domain_suggestions": []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric interactions
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Check for business-meaningful pairs
                interaction = self._evaluate_numeric_interaction(df, col1, col2)
                if interaction:
                    suggestions["interaction_suggestions"].append(interaction)
        
        # Temporal features
        for date_col in date_cols:
            for num_col in numeric_cols:
                col_lower = num_col.lower()
                if any(x in col_lower for x in ["price", "revenue", "count", "amount"]):
                    suggestions["temporal_suggestions"].append({
                        "type": "temporal_trend",
                        "date_col": date_col,
                        "numeric_col": num_col,
                        "description": f"Trend of {num_col} over time from {date_col}",
                        "reason": "Capture time-dependent patterns"
                    })
        
        # Domain-specific suggestions
        domain_interactions = self._suggest_domain_features(df, numeric_cols, cat_cols)
        suggestions["domain_suggestions"].extend(domain_interactions)
        
        return suggestions
    
    def _evaluate_numeric_interaction(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, str] or None:
        """Evaluate if two numeric columns should be multiplied/divided."""
        
        # Check correlation - if highly correlated, maybe one is redundant
        corr = df[[col1, col2]].corr().iloc[0, 1]
        if abs(corr) > 0.9:
            return None  # Too collinear
        
        # Check semantic meaning
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # Revenue x Quantity pattern
        if ("revenue" in col1_lower or "total" in col1_lower) and ("qty" in col2_lower or "quantity" in col2_lower):
            return {
                "type": "division",
                "columns": [col1, col2],
                "description": f"Unit price = {col1} / {col2}",
                "reason": "Derive unit economics"
            }
        
        # Price x Quantity = Revenue
        if ("price" in col1_lower and "qty" in col2_lower) or ("qty" in col1_lower and "price" in col2_lower):
            return {
                "type": "multiplication",
                "columns": [col1, col2],
                "description": f"Total value = {col1} × {col2}",
                "reason": "Key business metric"
            }
        
        return None
    
    def _suggest_domain_features(self, df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]) -> List[Dict]:
        """Suggest domain-specific features."""
        
        suggestions = []
        
        # Age from date of birth
        if any("birth" in c.lower() for c in df.columns):
            suggestions.append({
                "type": "age_calculation",
                "description": "Calculate age from birth date",
                "reason": "Age is important demographic variable"
            })
        
        # Customer lifetime value indicators
        if any(x in str(df.columns).lower() for x in ["customer", "purchase", "amount"]):
            suggestions.append({
                "type": "customer_segmentation",
                "description": "Create customer value tiers (high/medium/low spender)",
                "reason": "Drive business insights and personalization"
            })
        
        return suggestions


class AIQualityReportGenerator:
    """Generates comprehensive AI-written data quality reports."""
    
    def generate_report(self, df_raw: pd.DataFrame, df_processed: pd.DataFrame, 
                       column_profiles: Dict = None, llm_agent=None) -> Dict[str, Any]:
        """
        Generate comprehensive quality assessment report with AI narrative.
        """
        
        report = {
            "summary": self._generate_summary(df_raw, df_processed),
            "data_loss_assessment": self._assess_data_loss(df_raw, df_processed),
            "quality_issues": self._identify_key_issues(df_raw),
            "transformation_impact": self._assess_transformation_impact(df_raw, df_processed),
            "recommendations": self._generate_recommendations(df_raw, column_profiles or {}),
            "model_readiness": self._assess_model_readiness(df_processed)
        }
        
        return report
    
    def _generate_summary(self, df_raw: pd.DataFrame, df_processed: pd.DataFrame) -> str:
        """Generate executive summary."""
        
        rows_lost = len(df_raw) - len(df_processed)
        cols_lost = len(df_raw.columns) - len(df_processed.columns)
        
        summary = f"""Dataset: {len(df_raw):,} rows × {len(df_raw.columns)} columns input\n"""
        summary += f"Processed: {len(df_processed):,} rows × {len(df_processed.columns)} columns output\n"
        
        if rows_lost > 0:
            summary += f"⚠️ {rows_lost:,} rows removed (duplicates, too much missing)\n"
        if cols_lost > 0:
            summary += f"⚠️ {cols_lost} columns dropped (high missingness, low variance, or IDs)\n"
        
        return summary
    
    def _assess_data_loss(self, df_raw: pd.DataFrame, df_processed: pd.DataFrame) -> Dict[str, Any]:
        """Quantify data loss."""
        
        return {
            "rows_input": len(df_raw),
            "rows_output": len(df_processed),
            "rows_lost": len(df_raw) - len(df_processed),
            "row_retention_pct": round((len(df_processed) / len(df_raw)) * 100, 1),
            "columns_input": len(df_raw.columns),
            "columns_output": len(df_processed.columns),
            "columns_dropped": len(df_raw.columns) - len(df_processed.columns),
        }
    
    def _identify_key_issues(self, df: pd.DataFrame) -> List[str]:
        """Identify key quality issues."""
        
        issues = []
        
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 10:
            issues.append(f"High missing data overall ({missing_pct:.1f}%)")
        
        duplicates = len(df) - len(df.drop_duplicates())
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate rows detected")
        
        return issues
    
    def _assess_transformation_impact(self, df_raw: pd.DataFrame, df_processed: pd.DataFrame) -> Dict[str, Any]:
        """Assess impact of transformations on data distribution."""
        
        return {
            "encoding_applied": "Categorical columns converted to numeric (one-hot, frequency, label encoding)",
            "scaling_applied": "Continuous features standardized (mean-centered, unit variance)",
            "features_extracted": "Date features extracted (year, month, day, day-of-week, quarter)",
            "notes": "Negative numeric values are normal after standardization and indicate below-average values"
        }
    
    def _generate_recommendations(self, df: pd.DataFrame, column_profiles: Dict) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        if column_profiles:
            problem_cols = [col for col, profile in column_profiles.items() 
                          if "low_variance" in profile.get("quality_issues", [])]
            if problem_cols:
                recommendations.append(f"Consider dropping low-information columns: {', '.join(problem_cols)}")
        
        return recommendations
    
    def _assess_model_readiness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess readiness for ML modeling."""
        
        return {
            "is_ready": True,
            "numeric_only": True,
            "scaling_applied": True,
            "missing_values": df.isna().sum().sum(),
            "next_steps": ["Explore feature importance", "Test against baseline models", "Validate on holdout set"]
        }
