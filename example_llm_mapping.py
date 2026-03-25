"""
Example: LLM-Assisted Data Mapping Workflow
============================================

This example demonstrates:
1. Profile detection (Messy Categories & Messy Numbers)
2. AI-powered value mapping (via Groq)
3. Graceful fallback when API unavailable
4. Human review via Streamlit data_editor
5. Safe application to DataFrame
"""

import pandas as pd
from autoprep.profiler import DataProfiler
from autoprep.llm_agent import LLMAssistant


def example_profile_detection():
    """Step 1: Detect columns eligible for AI mapping"""
    # Sample messy data
    df = pd.DataFrame({
        "gender": ["F", "M", "Female", "m", "woman", "MALE"],  # Profile A: 3-50 unique
        "salary": ["100k", "$150,000", "5 mil", "2.5M", "None", "1k"],  # Profile B: 80%+ digits
        "age": [25, 30, 35, 40, 45, 50],  # Numeric - won't match either profile
    })
    
    profiler = DataProfiler(missing_green_zone=(0.0, 0.10))
    candidates = profiler.detect_llm_candidates(df)
    
    print("=" * 60)
    print("PROFILE DETECTION RESULTS")
    print("=" * 60)
    
    print("\n📌 Profile A (Messy Categories - 3-50 unique):")
    for col, info in candidates["profile_a_messy_categories"].items():
        print(f"  ✓ {col}: {info['n_unique']} unique values")
        print(f"    Values: {info['unique_values'][:5]}")
    
    print("\n📌 Profile B (Messy Numbers - 80%+ digits):")
    for col, info in candidates["profile_b_messy_numbers"].items():
        print(f"  ✓ {col}: {info['digit_ratio']:.1%} contain digits")
        print(f"    Values: {info['unique_values'][:5]}")
    
    return df, profiler, candidates


def example_ai_mapping():
    """Step 2: Use Groq to generate intelligent mappings"""
    df, profiler, candidates = example_profile_detection()
    
    print("\n" + "=" * 60)
    print("AI MAPPING GENERATION (via Groq)")
    print("=" * 60)
    
    llm = LLMAssistant()  # Reads GROQ_API_KEY from env by default
    
    if not llm.available:
        print("\n⚠️  AI Engine UNAVAILABLE - demonstrating fallback")
        return df, llm
    
    print("\n🤖 Generating mappings for Profile A (Categories)...")
    gender_candidates = candidates["profile_a_messy_categories"].get("gender", {})
    if gender_candidates:
        gender_mapping = llm.map_messy_categories(
            unique_values=gender_candidates["unique_values"],
            column_name="gender",
            context="biological gender classification"
        )
        print(f"   Generated: {gender_mapping}")
    
    print("\n🤖 Generating mappings for Profile B (Numbers)...")
    salary_candidates = candidates["profile_b_messy_numbers"].get("salary", {})
    if salary_candidates:
        salary_mapping = llm.map_messy_numbers(
            unique_values=salary_candidates["unique_values"],
            column_name="salary",
            context="annual salary in USD"
        )
        print(f"   Generated: {salary_mapping}")
    
    return df, llm


def example_error_handling():
    """Step 3: Demonstrate error handling & fallback"""
    df = pd.DataFrame({
        "city": ["New York", "NY", "new york", "NYC", "N.Y."],
    })
    
    print("\n" + "=" * 60)
    print("ERROR HANDLING & FALLBACK LOGIC")
    print("=" * 60)
    
    llm = LLMAssistant()
    
    if not llm.available:
        print("\n✓ API unavailable - using fallback cleaning")
        mapping = llm.map_messy_categories(
            unique_values=df["city"].unique().tolist(),
            column_name="city"
        )
        print(f"  Fallback mapping: {mapping}")
        
        # Apply fallback
        df["city"] = df["city"].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)
        print(f"  Result:\n{df}")
    else:
        print("\n✓ API available - normal workflow")
    
    return df


def example_validation():
    """Step 4: Validate mappings before application"""
    mapping_good = {"100k": 100000, "1M": 1000000}
    mapping_bad = {"100k": "one hundred thousand"}  # Wrong type for numeric
    
    print("\n" + "=" * 60)
    print("MAPPING VALIDATION")
    print("=" * 60)
    
    llm = LLMAssistant()
    
    is_valid, msg = llm.validate_mapping(mapping_good, expected_type="numeric")
    print(f"\n✓ Valid mapping: {is_valid} - {msg}")
    print(f"  {mapping_good}")
    
    is_valid, msg = llm.validate_mapping(mapping_bad, expected_type="numeric")
    print(f"\n✗ Invalid mapping: {is_valid} - {msg}")
    print(f"  {mapping_bad}")


def example_apply_mapping():
    """Step 5: Safely apply validated mapping to DataFrame"""
    df = pd.DataFrame({
        "salary": ["100k", "100k", "5M", "1k", None],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    })
    
    mapping = {"100k": 100000, "5M": 5000000, "1k": 1000}
    
    print("\n" + "=" * 60)
    print("APPLYING MAPPING TO DATAFRAME")
    print("=" * 60)
    
    llm = LLMAssistant()
    
    print("\nBefore:")
    print(df)
    
    df_mapped = llm.apply_mapping_to_dataframe(df, "salary", mapping)
    
    print("\nAfter:")
    print(df_mapped)


if __name__ == "__main__":
    # Run all examples
    print("\n🚀 LLM-ASSISTED DATA MAPPING WORKFLOW\n")
    
    example_profile_detection()
    example_ai_mapping()
    example_error_handling()
    example_validation()
    example_apply_mapping()
    
    print("\n✅ All examples complete!")
    print("\nNext: Integrate into Streamlit app for interactive review")
