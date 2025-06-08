#!/usr/bin/env python3
"""
Error analysis and visualization script for reimbursement models.
Provides detailed error analysis including percentile plots and categorical breakdowns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from final_submission_engine import FinalSubmissionEngine
import json

class ErrorAnalyzer:
    def __init__(self, model_type='advanced'):
        self.model_type = model_type
        self.engine = FinalSubmissionEngine(model_type=model_type)
        self.results = None
        
    def analyze_errors(self):
        """Perform comprehensive error analysis."""
        print(f"Analyzing errors for {self.model_type} model...")
        
        # Get detailed predictions
        self.results = self.engine.get_detailed_predictions()
        df = pd.DataFrame(self.results)
        
        # Basic statistics
        errors = df['error'].values
        print(f"\nError Statistics:")
        print(f"  Total cases: {len(errors)}")
        print(f"  Mean Absolute Error: ${np.mean(errors):.2f}")
        print(f"  Median Error: ${np.median(errors):.2f}")
        print(f"  Std Dev: ${np.std(errors):.2f}")
        print(f"  Min Error: ${np.min(errors):.2f}")
        print(f"  Max Error: ${np.max(errors):.2f}")
        
        # Percentile analysis
        percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
        print(f"\nError Percentiles:")
        for p in percentiles:
            value = np.percentile(errors, p)
            print(f"  {p}th percentile: ${value:.2f}")
        
        # Threshold analysis
        thresholds = [0.01, 0.1, 1, 5, 10, 25, 50, 100]
        print(f"\nAccuracy at Thresholds:")
        for thresh in thresholds:
            accuracy = np.mean(errors <= thresh) * 100
            print(f"  Within ${thresh:>6}: {accuracy:>5.1f}%")
        
        return df
    
    def plot_error_distribution(self, df):
        """Plot error distribution and percentiles."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Error Analysis for {self.model_type.title()} Model', fontsize=16)
        
        errors = df['error'].values
        
        # 1. Error histogram
        axes[0,0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: ${np.mean(errors):.2f}')
        axes[0,0].axvline(np.median(errors), color='green', linestyle='--', label=f'Median: ${np.median(errors):.2f}')
        axes[0,0].set_xlabel('Absolute Error ($)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Error Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Log scale error histogram
        axes[0,1].hist(errors[errors > 0], bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Absolute Error ($)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Error Distribution (Log Scale)')
        axes[0,1].set_yscale('log')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Percentile plot
        percentiles = np.arange(1, 100, 1)
        percentile_values = [np.percentile(errors, p) for p in percentiles]
        axes[1,0].plot(percentiles, percentile_values, linewidth=2)
        axes[1,0].set_xlabel('Percentile')
        axes[1,0].set_ylabel('Error ($)')
        axes[1,0].set_title('Error by Percentile')
        axes[1,0].grid(True, alpha=0.3)
        
        # Add key percentile lines
        key_percentiles = [90, 95, 99]
        for p in key_percentiles:
            value = np.percentile(errors, p)
            axes[1,0].axhline(value, color='red', linestyle='--', alpha=0.7)
            axes[1,0].text(p, value, f' {p}%: ${value:.1f}', verticalalignment='bottom')
        
        # 4. Cumulative accuracy
        thresholds = np.logspace(-2, 3, 100)  # From 0.01 to 1000
        accuracies = [np.mean(errors <= t) * 100 for t in thresholds]
        axes[1,1].semilogx(thresholds, accuracies, linewidth=2)
        axes[1,1].set_xlabel('Error Threshold ($)')
        axes[1,1].set_ylabel('Accuracy (%)')
        axes[1,1].set_title('Cumulative Accuracy')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add key threshold lines
        key_thresholds = [1, 5, 10, 25]
        for t in key_thresholds:
            acc = np.mean(errors <= t) * 100
            axes[1,1].axvline(t, color='red', linestyle='--', alpha=0.7)
            axes[1,1].text(t, acc, f' ${t}: {acc:.1f}%', rotation=90, verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.model_type}_error_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nError distribution plot saved as {self.model_type}_error_analysis.png")
        return fig
    
    def plot_categorical_errors(self, df):
        """Plot errors by different categories."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Categorical Error Analysis for {self.model_type.title()} Model', fontsize=16)
        
        # 1. Error by trip duration
        axes[0,0].boxplot([df[df['days'] == d]['error'].values for d in sorted(df['days'].unique())], 
                         labels=sorted(df['days'].unique()))
        axes[0,0].set_xlabel('Trip Duration (days)')
        axes[0,0].set_ylabel('Error ($)')
        axes[0,0].set_title('Error by Trip Duration')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Error by efficiency bins
        df['efficiency_bin'] = pd.cut(df['miles_per_day'], bins=[0, 50, 100, 200, 300, 500, np.inf], 
                                    labels=['<50', '50-100', '100-200', '200-300', '300-500', '>500'])
        efficiency_groups = [df[df['efficiency_bin'] == cat]['error'].values for cat in df['efficiency_bin'].cat.categories]
        axes[0,1].boxplot([group for group in efficiency_groups if len(group) > 0], 
                         labels=[cat for cat, group in zip(df['efficiency_bin'].cat.categories, efficiency_groups) if len(group) > 0])
        axes[0,1].set_xlabel('Miles per Day')
        axes[0,1].set_ylabel('Error ($)')
        axes[0,1].set_title('Error by Efficiency')
        axes[0,1].grid(True, alpha=0.3)
        plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Error by spending bins
        df['spending_bin'] = pd.cut(df['receipts_per_day'], bins=[0, 50, 100, 200, 400, np.inf], 
                                  labels=['<50', '50-100', '100-200', '200-400', '>400'])
        spending_groups = [df[df['spending_bin'] == cat]['error'].values for cat in df['spending_bin'].cat.categories]
        axes[0,2].boxplot([group for group in spending_groups if len(group) > 0], 
                         labels=[cat for cat, group in zip(df['spending_bin'].cat.categories, spending_groups) if len(group) > 0])
        axes[0,2].set_xlabel('Receipts per Day ($)')
        axes[0,2].set_ylabel('Error ($)')
        axes[0,2].set_title('Error by Spending Level')
        axes[0,2].grid(True, alpha=0.3)
        plt.setp(axes[0,2].xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Error scatter: predicted vs actual
        axes[1,0].scatter(df['expected'], df['predicted'], alpha=0.6, s=20)
        min_val = min(df['expected'].min(), df['predicted'].min())
        max_val = max(df['expected'].max(), df['predicted'].max())
        axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[1,0].set_xlabel('Expected ($)')
        axes[1,0].set_ylabel('Predicted ($)')
        axes[1,0].set_title('Predicted vs Expected')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Error vs total trip cost
        df['total_input'] = df['days'] * 100 + df['miles'] * 0.5 + df['receipts']  # Rough trip cost estimate
        axes[1,1].scatter(df['total_input'], df['error'], alpha=0.6, s=20)
        axes[1,1].set_xlabel('Estimated Trip Cost ($)')
        axes[1,1].set_ylabel('Error ($)')
        axes[1,1].set_title('Error vs Trip Complexity')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Receipt ending analysis
        df['receipt_ending'] = df['receipts'].apply(lambda x: f"{x:.2f}"[-2:])
        ending_counts = df['receipt_ending'].value_counts().head(10)
        ending_errors = [df[df['receipt_ending'] == ending]['error'].mean() for ending in ending_counts.index]
        axes[1,2].bar(range(len(ending_counts)), ending_errors)
        axes[1,2].set_xlabel('Receipt Ending')
        axes[1,2].set_ylabel('Mean Error ($)')
        axes[1,2].set_title('Error by Receipt Ending')
        axes[1,2].set_xticks(range(len(ending_counts)))
        axes[1,2].set_xticklabels(ending_counts.index, rotation=45)
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_type}_categorical_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Categorical analysis plot saved as {self.model_type}_categorical_analysis.png")
        return fig
    
    def analyze_worst_cases(self, df, top_n=20):
        """Analyze the worst performing cases."""
        worst_cases = df.nlargest(top_n, 'error')
        
        print(f"\nTop {top_n} Worst Cases:")
        print("-" * 80)
        for _, case in worst_cases.iterrows():
            print(f"Case {case['case_id']:3d}: {case['days']}d, {case['miles']:6.1f}mi, ${case['receipts']:8.2f} "
                  f"-> Expected: ${case['expected']:7.2f}, Predicted: ${case['predicted']:7.2f}, "
                  f"Error: ${case['error']:6.2f}")
        
        # Analyze patterns in worst cases
        print(f"\nWorst Cases Analysis:")
        print(f"  Average trip length: {worst_cases['days'].mean():.1f} days")
        print(f"  Average efficiency: {worst_cases['miles_per_day'].mean():.1f} miles/day")
        print(f"  Average spending: ${worst_cases['receipts_per_day'].mean():.2f}/day")
        print(f"  Average error: ${worst_cases['error'].mean():.2f}")
        
        return worst_cases
    
    def compare_models(self, other_model_types):
        """Compare multiple models."""
        comparison_data = []
        
        # Add current model
        df_current = pd.DataFrame(self.results)
        errors_current = df_current['error'].values
        comparison_data.append({
            'model': self.model_type,
            'mae': np.mean(errors_current),
            'median': np.median(errors_current),
            'p95': np.percentile(errors_current, 95),
            'within_1': np.mean(errors_current <= 1) * 100,
            'within_5': np.mean(errors_current <= 5) * 100,
            'within_10': np.mean(errors_current <= 10) * 100
        })
        
        # Add other models
        for model_type in other_model_types:
            try:
                engine = FinalSubmissionEngine(model_type=model_type)
                results = engine.get_detailed_predictions()
                errors = np.array([r['error'] for r in results])
                
                comparison_data.append({
                    'model': model_type,
                    'mae': np.mean(errors),
                    'median': np.median(errors),
                    'p95': np.percentile(errors, 95),
                    'within_1': np.mean(errors <= 1) * 100,
                    'within_5': np.mean(errors <= 5) * 100,
                    'within_10': np.mean(errors <= 10) * 100
                })
            except Exception as e:
                print(f"Could not load {model_type} model: {e}")
        
        comparison_df = pd.DataFrame(comparison_data)
        print(f"\nModel Comparison:")
        print(comparison_df.round(2))
        
        return comparison_df
    
    def run_full_analysis(self, compare_models=None):
        """Run complete error analysis."""
        print(f"Running full error analysis for {self.model_type} model...")
        
        # Basic analysis
        df = self.analyze_errors()
        
        # Create plots
        self.plot_error_distribution(df)
        self.plot_categorical_errors(df)
        
        # Worst cases
        self.analyze_worst_cases(df)
        
        # Model comparison if requested
        if compare_models:
            self.compare_models(compare_models)
        
        print(f"\nAnalysis complete! Check the generated PNG files for visualizations.")
        
        return df

def main():
    """Run error analysis on the current best model."""
    analyzer = ErrorAnalyzer(model_type='advanced')
    df = analyzer.run_full_analysis()
    
    # Save detailed results
    df.to_csv('detailed_predictions.csv', index=False)
    print("\nDetailed predictions saved to detailed_predictions.csv")

if __name__ == "__main__":
    main()