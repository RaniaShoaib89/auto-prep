#!/usr/bin/env python
"""Comprehensive test of AutoPrep system on unclean movie data"""
from autoprep.pipeline import AutoPrepPipeline
from autoprep.loader import DataLoader
from autoprep.profiler import DataProfiler
import pandas as pd

print("\n" + "="*80)
print("AUTOPREP SYSTEM TEST ON UNCLEAN MOVIE DATA")
print("="*80)

# Load raw data
print("\n1. RAW DATA DIAGNOSTICS")
print("-" * 80)
loader = DataLoader()
df_raw = loader.load_data('unclean_data.csv')
print(f"Raw Data Shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
print(f"\nMissing Values:")
missing = df_raw.isnull().sum()
print(missing[missing > 0])

# Profile raw data
profiler = DataProfiler()
raw_profile = profiler.profile(df_raw)
print(f"\nRaw Data - Numeric Columns: {len(raw_profile['numerical'])}")
print(f"Raw Data - Categorical Columns: {len(raw_profile['categorical'])}")

# Run pipeline
print("\n2. RUNNING PIPELINE")
print("-" * 80)
pipeline = AutoPrepPipeline(
    missing_strategy='auto',
    missing_threshold=0.5,
    outlier_method='iqr',
    outlier_action='clip',
    encoding_strategy='auto',
    extract_date_features=False,
    drop_identifiers=True,
    drop_low_variance=True,
    drop_high_correlation=True,
    visualize=False,
    interactive_mode=False,
)

try:
    df_processed, report = pipeline.run('unclean_data.csv')
    print("Pipeline Status: SUCCESS")
except Exception as e:
    print(f"Pipeline Status: FAILED - {e}")
    exit(1)

# Results
print("\n3. PROCESSING SUMMARY")
print("-" * 80)
print(f"Input:  {report['raw_profile']['shape']['rows']:,} rows x {report['raw_profile']['shape']['cols']} cols")
print(f"Output: {report['processed_profile']['shape']['rows']:,} rows x {report['processed_profile']['shape']['cols']} cols")
print(f"\nCleaning Report:")
print(f"  Duplicates removed: {report['cleaning'].get('duplicates_removed', 0)}")
print(f"  Columns dropped (high missing): {len(report['cleaning'].get('columns_dropped_high_missing', []))}")
print(f"  Columns with missing values filled: {len(report['cleaning'].get('missing_values_filled', {}))}")
print(f"  Outliers handled: {len(report['cleaning'].get('outliers_handled', {}))}")

print(f"\nEncoding Report:")
print(f"  Encoding strategy: {report['encoding'].get('strategy_used', 'N/A')}")
print(f"  Columns encoded: {report['encoding'].get('n_columns_encoded', 0)}")

print(f"\nFeature Engineering Report:")
print(f"  Features extracted: {report['feature_engineering'].get('features_extracted', 0)}")
print(f"  Low variance cols dropped: {len(report['feature_engineering'].get('low_variance_dropped', []))}")
print(f"  High correlation cols dropped: {len(report['feature_engineering'].get('high_correlation_dropped', []))}")

# Final data quality
print("\n4. PROCESSED DATA QUALITY")
print("-" * 80)
print(f"Final Shape: {df_processed.shape[0]} rows x {df_processed.shape[1]} cols")
print(f"Data Types: {df_processed.dtypes.value_counts().to_dict()}")
print(f"Missing Values: {df_processed.isnull().sum().sum()} (0 = perfect)")
print(f"\nFirst 3 rows:")
print(df_processed.head(3).to_string())

print("\n" + "="*80)
print("TEST COMPLETE - System is ready for use!")
print("="*80 + "\n")
