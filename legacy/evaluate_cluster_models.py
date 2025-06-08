#!/usr/bin/env python3
"""
Convenience script for evaluating specific cluster model configurations.
Usage:
  python3 evaluate_cluster_models.py --all                    # Evaluate all available models
  python3 evaluate_cluster_models.py --clusters 5            # Evaluate specific cluster count
  python3 evaluate_cluster_models.py --compare 3 5 7         # Compare specific configurations
"""

import sys
import argparse
from final_submission_engine import FinalSubmissionEngine

def evaluate_specific_cluster(n_clusters):
    """Evaluate a specific cluster configuration."""
    print(f"Evaluating {n_clusters}-cluster model:")
    print("=" * 50)
    
    try:
        engine = FinalSubmissionEngine(
            model_type='hybrid_enhanced_clustering',
            n_clusters=n_clusters
        )
        
        # Public cases evaluation
        mae = engine.evaluate_public_cases()
        
        # Edge cases evaluation
        print(f"\nEdge case evaluation:")
        edge_cases = [
            (8, 482, 1411.49, 631.81, "Case 548"),
            (6, 204, 818.99, 628.40, "Case 82"),
            (10, 1192, 23.47, 1157.87, "Case 522"),
            (7, 1006, 1181.33, 2279.82, "Case 149"),
            (2, 384, 495.49, 290.36, "Case 89")
        ]
        
        total_edge_error = 0
        for days, miles, receipts, expected, name in edge_cases:
            predicted = engine.calculate_reimbursement(days, miles, receipts)
            error = abs(predicted - expected)
            total_edge_error += error
            print(f"  {name}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")
        
        avg_edge_error = total_edge_error / len(edge_cases)
        print(f"\nSummary for {n_clusters} clusters:")
        print(f"  Overall MAE: ${mae:.2f}")
        print(f"  Edge case MAE: ${avg_edge_error:.2f}")
        
        return mae, avg_edge_error
        
    except Exception as e:
        print(f"Error evaluating {n_clusters}-cluster model: {e}")
        return None, None

def compare_configurations(cluster_list):
    """Compare multiple cluster configurations."""
    print("Comparing cluster configurations:")
    print("=" * 60)
    
    results = []
    
    for n_clusters in cluster_list:
        print(f"\n{n_clusters} clusters:")
        mae, edge_mae = evaluate_specific_cluster(n_clusters)
        if mae is not None:
            results.append((n_clusters, mae, edge_mae))
    
    if results:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY:")
        print(f"{'Clusters':<10} {'Overall MAE':<15} {'Edge MAE':<15} {'Total Score':<15}")
        print("-" * 60)
        
        for n_clusters, mae, edge_mae in results:
            # Combined score (weighted average: 70% overall, 30% edge cases)
            total_score = 0.7 * mae + 0.3 * edge_mae
            print(f"{n_clusters:<10} ${mae:<14.2f} ${edge_mae:<14.2f} ${total_score:<14.2f}")
        
        # Find best overall and best edge case
        best_overall = min(results, key=lambda x: x[1])
        best_edge = min(results, key=lambda x: x[2])
        best_combined = min(results, key=lambda x: 0.7 * x[1] + 0.3 * x[2])
        
        print(f"\nBest overall MAE: {best_overall[0]} clusters (${best_overall[1]:.2f})")
        print(f"Best edge case MAE: {best_edge[0]} clusters (${best_edge[2]:.2f})")
        print(f"Best combined score: {best_combined[0]} clusters (${0.7 * best_combined[1] + 0.3 * best_combined[2]:.2f})")

def main():
    parser = argparse.ArgumentParser(description='Evaluate cluster model configurations')
    parser.add_argument('--all', action='store_true', help='Evaluate all available models')
    parser.add_argument('--clusters', type=int, help='Evaluate specific cluster count')
    parser.add_argument('--compare', type=int, nargs='+', help='Compare specific configurations')
    
    args = parser.parse_args()
    
    if args.all:
        # Use the existing evaluation method
        dummy_engine = FinalSubmissionEngine(model_type='advanced')
        dummy_engine.evaluate_all_cluster_models()
        
    elif args.clusters:
        evaluate_specific_cluster(args.clusters)
        
    elif args.compare:
        compare_configurations(args.compare)
        
    else:
        # Default: evaluate optimal model
        print("Evaluating optimal hybrid model:")
        try:
            engine = FinalSubmissionEngine(model_type='hybrid_enhanced_clustering')
            engine.evaluate_public_cases()
        except Exception as e:
            print(f"Error: {e}")
            print("\nAvailable options:")
            print("  --all              Evaluate all cluster configurations")
            print("  --clusters N       Evaluate specific cluster count")
            print("  --compare N1 N2    Compare specific configurations")

if __name__ == "__main__":
    main()