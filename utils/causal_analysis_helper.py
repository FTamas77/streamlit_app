"""
Data analysis helper for understanding why causal effects are zero
"""
import pandas as pd
import numpy as np

def analyze_causal_relationship(data, treatment_col, outcome_col, encoded_data=None):
    """
    Comprehensive analysis of the relationship between treatment and outcome
    
    Args:
        data: Original dataframe
        treatment_col: Name of treatment variable
        outcome_col: Name of outcome variable
        encoded_data: Encoded dataframe if available
    """
    print("="*60)
    print(f"CAUSAL RELATIONSHIP ANALYSIS: {treatment_col} → {outcome_col}")
    print("="*60)
    
    # Determine which data to analyze
    if treatment_col.endswith('_Code') and encoded_data is not None:
        analysis_data = encoded_data
        print(f"📊 Using ENCODED data for analysis")
    else:
        analysis_data = data
        print(f"📊 Using ORIGINAL data for analysis")
    
    print(f"Data shape: {analysis_data.shape}")
    print(f"Available columns: {list(analysis_data.columns)}")
    
    # Check if variables exist
    if treatment_col not in analysis_data.columns:
        print(f"❌ Treatment variable '{treatment_col}' not found!")
        return
    
    if outcome_col not in analysis_data.columns:
        print(f"❌ Outcome variable '{outcome_col}' not found!")
        return
    
    print(f"✅ Both variables found in data")
    print()
    
    # Basic statistics
    print("📈 TREATMENT VARIABLE ANALYSIS:")
    print(f"   Variable: {treatment_col}")
    print(f"   Type: {analysis_data[treatment_col].dtype}")
    print(f"   Unique values: {analysis_data[treatment_col].nunique()}")
    print(f"   Range: [{analysis_data[treatment_col].min():.3f}, {analysis_data[treatment_col].max():.3f}]")
    print(f"   Mean: {analysis_data[treatment_col].mean():.3f}")
    print(f"   Std: {analysis_data[treatment_col].std():.3f}")
    
    if analysis_data[treatment_col].nunique() <= 10:
        print(f"   Value counts:")
        for val, count in analysis_data[treatment_col].value_counts().items():
            print(f"     {val}: {count} occurrences ({count/len(analysis_data)*100:.1f}%)")
    print()
    
    print("📉 OUTCOME VARIABLE ANALYSIS:")
    print(f"   Variable: {outcome_col}")
    print(f"   Type: {analysis_data[outcome_col].dtype}")
    print(f"   Unique values: {analysis_data[outcome_col].nunique()}")
    print(f"   Range: [{analysis_data[outcome_col].min():.3f}, {analysis_data[outcome_col].max():.3f}]")
    print(f"   Mean: {analysis_data[outcome_col].mean():.3f}")
    print(f"   Std: {analysis_data[outcome_col].std():.3f}")
    print()
    
    # Correlation analysis
    correlation = analysis_data[treatment_col].corr(analysis_data[outcome_col])
    print("🔗 RELATIONSHIP ANALYSIS:")
    print(f"   Pearson correlation: {correlation:.4f}")
    
    if abs(correlation) < 0.1:
        print("   ⚠️  VERY WEAK correlation - causal effect likely to be zero")
    elif abs(correlation) < 0.3:
        print("   📊 WEAK correlation - small causal effect possible")
    elif abs(correlation) < 0.7:
        print("   📈 MODERATE correlation - detectable causal effect likely")
    else:
        print("   🔥 STRONG correlation - significant causal effect expected")
    
    # Group analysis for categorical treatment
    if analysis_data[treatment_col].nunique() <= 10:
        print(f"\n📊 OUTCOME BY TREATMENT GROUP:")
        group_stats = analysis_data.groupby(treatment_col)[outcome_col].agg(['count', 'mean', 'std']).round(3)
        print(group_stats)
        
        # Check for meaningful differences between groups
        group_means = analysis_data.groupby(treatment_col)[outcome_col].mean()
        max_diff = group_means.max() - group_means.min()
        print(f"\n   Maximum difference between groups: {max_diff:.3f}")
        
        if max_diff < analysis_data[outcome_col].std() * 0.2:
            print("   ⚠️  Group differences are VERY SMALL relative to overall variation")
            print("   💡 This explains why causal effect is near zero")
        elif max_diff < analysis_data[outcome_col].std() * 0.5:
            print("   📊 Group differences are SMALL but may be detectable")
        else:
            print("   📈 Group differences are SUBSTANTIAL - should be detectable")
    
    print()
    print("🎯 RECOMMENDATIONS:")
    
    if abs(correlation) < 0.1:
        print("   • The variables appear to have no meaningful relationship")
        print("   • Consider:")
        print("     - Is there a theoretical reason to expect a causal relationship?")
        print("     - Are there confounding variables that need to be controlled?")
        print("     - Is the sample size sufficient?")
        print("     - Could there be non-linear relationships?")
    
    if analysis_data[treatment_col].nunique() <= 10:
        group_stats = analysis_data.groupby(treatment_col)[outcome_col].mean()
        if group_stats.std() < 0.1:
            print("   • Treatment groups have very similar outcome means")
            print("   • This suggests no causal effect exists")
    
    print("   • To improve causal detection:")
    print("     - Collect more data if sample size < 200")
    print("     - Include relevant confounding variables")
    print("     - Check for data quality issues")
    print("     - Consider domain constraints in causal discovery")
    
    print("="*60)
    
    return {
        'correlation': correlation,
        'treatment_stats': analysis_data[treatment_col].describe(),
        'outcome_stats': analysis_data[outcome_col].describe(),
        'group_analysis': analysis_data.groupby(treatment_col)[outcome_col].describe() if analysis_data[treatment_col].nunique() <= 10 else None
    }
