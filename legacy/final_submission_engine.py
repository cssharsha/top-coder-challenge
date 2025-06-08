#!/usr/bin/env python3
"""
Final submission engine integrating multiple models with the existing interface.
This maintains compatibility with run.sh while supporting various model types.
"""

import json
import os
import importlib
import sys
from advanced_reimbursement_engine import AdvancedReimbursementEngine

class FinalSubmissionEngine:
    def __init__(self, model_type='advanced', model_path=None, n_clusters=None):
        self.model_type = model_type
        self.n_clusters = n_clusters
        
        # Determine model path based on cluster specification
        if model_path:
            self.model_path = model_path
        elif model_type == 'hybrid_enhanced_clustering' and n_clusters:
            self.model_path = f'hybrid_enhanced_clustering_{n_clusters}clusters_model.pkl'
        else:
            self.model_path = f'{model_type}_model.pkl'
            
        self.engine = None
        self.load_model()
    
    def load_model(self):
        """Load the specified model type."""
        # Suppress output for run.sh compatibility
        suppress_output = len(sys.argv) > 1 and 'run.sh' in str(sys.argv)
        
        if self.model_type == 'advanced':
            self.engine = AdvancedReimbursementEngine()
        elif self.model_type == 'parametric':
            from parametric_reimbursement_engine import ParametricReimbursementEngine
            self.engine = ParametricReimbursementEngine()
        elif self.model_type == 'hybrid_enhanced_clustering':
            from hybrid_enhanced_clustering_engine import HybridEnhancedClusteringEngine
            self.engine = HybridEnhancedClusteringEngine()
        elif self.model_type == 'enhanced_parametric':
            from enhanced_parametric_engine import EnhancedParametricEngine
            self.engine = EnhancedParametricEngine()
        elif self.model_type == 'clustering_regime':
            from clustering_regime_engine import ClusteringRegimeEngine
            self.engine = ClusteringRegimeEngine()
        elif self.model_type == 'learned_regime':
            from learned_regime_engine import LearnedRegimeEngine
            self.engine = LearnedRegimeEngine()
        else:
            # Dynamic loading for future model types
            try:
                module_name = f"{self.model_type}_reimbursement_engine"
                class_name = f"{self.model_type.title()}ReimbursementEngine"
                module = importlib.import_module(module_name)
                engine_class = getattr(module, class_name)
                self.engine = engine_class()
            except (ImportError, AttributeError):
                if not suppress_output:
                    print(f"Model type '{self.model_type}' not found, falling back to advanced model")
                self.engine = AdvancedReimbursementEngine()
                self.model_type = 'advanced'
                self.model_path = 'advanced_model.pkl'
        
        try:
            self.engine.load(self.model_path)
            # if not suppress_output:
            #     print(f"Loaded {self.model_type} model from {self.model_path}")
        except FileNotFoundError:
            if not suppress_output:
                print(f"{self.model_type} model not found, training new model...")
            self.train_model(suppress_output)
    
    def train_model(self, suppress_output=False):
        """Train the specified model if it doesn't exist."""
        try:
            with open('../public_cases.json', 'r') as f:
                cases = json.load(f)
        except FileNotFoundError:
            # Try relative path from different directories
            with open('public_cases.json', 'r') as f:
                cases = json.load(f)
                
        self.engine.train(cases)
        self.engine.save(self.model_path)
        if not suppress_output:
            print(f"{self.model_type} model trained and saved as {self.model_path}")
    
    def get_detailed_predictions(self):
        """Get detailed predictions for error analysis."""
        try:
            with open('../public_cases.json', 'r') as f:
                cases = json.load(f)
        except FileNotFoundError:
            with open('public_cases.json', 'r') as f:
                cases = json.load(f)
        
        results = []
        for i, case in enumerate(cases):
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            predicted = self.calculate_reimbursement(days, miles, receipts)
            error = abs(predicted - expected)
            
            results.append({
                'case_id': i,
                'days': days,
                'miles': miles,
                'receipts': receipts,
                'expected': expected,
                'predicted': predicted,
                'error': error,
                'miles_per_day': miles / max(days, 1),
                'receipts_per_day': receipts / max(days, 1)
            })
        
        return results
    
    def calculate_reimbursement(self, trip_duration_days, miles_traveled, total_receipts_amount):
        """
        Calculate reimbursement amount for given trip parameters.
        
        Args:
            trip_duration_days (int): Number of days for the trip
            miles_traveled (float): Total miles traveled
            total_receipts_amount (float): Total amount of receipts submitted
            
        Returns:
            float: Calculated reimbursement amount
        """
        return self.engine.predict(trip_duration_days, miles_traveled, total_receipts_amount)
    
    def evaluate_public_cases(self):
        """Evaluate performance on public test cases."""
        try:
            with open('../public_cases.json', 'r') as f:
                cases = json.load(f)
        except FileNotFoundError:
            with open('public_cases.json', 'r') as f:
                cases = json.load(f)
        
        errors = []
        exact_matches = 0
        close_matches = 0
        
        for case in cases:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            predicted = self.calculate_reimbursement(days, miles, receipts)
            error = abs(predicted - expected)
            errors.append(error)
            
            if error <= 0.01:
                exact_matches += 1
            if error <= 1.0:
                close_matches += 1
        
        mae = sum(errors) / len(errors)
        max_error = max(errors)
        within_1 = sum(1 for e in errors if e <= 1) / len(errors) * 100
        within_5 = sum(1 for e in errors if e <= 5) / len(errors) * 100
        within_10 = sum(1 for e in errors if e <= 10) / len(errors) * 100
        
        print(f"Final Model Evaluation:")
        print(f"  Cases: {len(cases)}")
        print(f"  Mean Absolute Error: ${mae:.2f}")
        print(f"  Maximum Error: ${max_error:.2f}")
        print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/len(cases)*100:.1f}%)")
        print(f"  Close matches (±$1.00): {close_matches} ({close_matches/len(cases)*100:.1f}%)")
        print(f"  Within $1: {within_1:.1f}%")
        print(f"  Within $5: {within_5:.1f}%")
        print(f"  Within $10: {within_10:.1f}%")
        
        return mae
    
    def evaluate_all_cluster_models(self, cluster_range=None):
        """Evaluate all available cluster model configurations."""
        if cluster_range is None:
            cluster_range = range(3, 12)  # Default range
        
        print("Evaluating all cluster model configurations:")
        print("=" * 60)
        
        results = []
        
        for n_clusters in cluster_range:
            model_path = f'hybrid_enhanced_clustering_{n_clusters}clusters_model.pkl'
            
            # Check if model exists
            if not os.path.exists(model_path):
                print(f"  {n_clusters} clusters: Model not found")
                continue
            
            try:
                # Create engine for this cluster configuration
                engine = FinalSubmissionEngine(
                    model_type='hybrid_enhanced_clustering',
                    n_clusters=n_clusters
                )
                
                # Evaluate
                mae = engine.evaluate_public_cases()
                results.append((n_clusters, mae))
                
                print(f"  {n_clusters} clusters: MAE = ${mae:.2f}")
                
            except Exception as e:
                print(f"  {n_clusters} clusters: Error - {e}")
        
        if results:
            # Find best configuration
            best_clusters, best_mae = min(results, key=lambda x: x[1])
            print(f"\nBest configuration: {best_clusters} clusters (MAE = ${best_mae:.2f})")
            
            # Test edge cases with best model
            print(f"\nTesting edge cases with best model ({best_clusters} clusters):")
            best_engine = FinalSubmissionEngine(
                model_type='hybrid_enhanced_clustering',
                n_clusters=best_clusters
            )
            
            edge_cases = [
                (8, 482, 1411.49, 631.81, "Case 548"),
                (6, 204, 818.99, 628.40, "Case 82"),
                (10, 1192, 23.47, 1157.87, "Case 522"),
                (7, 1006, 1181.33, 2279.82, "Case 149"),
                (2, 384, 495.49, 290.36, "Case 89")
            ]
            
            total_edge_error = 0
            for days, miles, receipts, expected, name in edge_cases:
                predicted = best_engine.calculate_reimbursement(days, miles, receipts)
                error = abs(predicted - expected)
                total_edge_error += error
                print(f"  {name}: Expected ${expected:.2f}, Predicted ${predicted:.2f}, Error ${error:.2f}")
            
            avg_edge_error = total_edge_error / len(edge_cases)
            print(f"  Average edge case error: ${avg_edge_error:.2f}")
        
        return results

def main():
    """Main function for standalone testing."""
    print("Initializing Final Submission Engine...")
    
    # Check if we should evaluate all cluster models
    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate-clusters':
        dummy_engine = FinalSubmissionEngine(model_type='advanced')  # Just for the method
        dummy_engine.evaluate_all_cluster_models()
        return
    
    engine = FinalSubmissionEngine()
    
    # Evaluate on public cases
    engine.evaluate_public_cases()
    
    # Test key examples from interviews
    print(f"\nKey interview examples:")
    examples = [
        (5, 200, 500, "Kevin's sweet spot"),
        (3, 93, 1.42, "Low receipts case"),
        (8, 600, 1200, "Long expensive trip"),
        (1, 1082, 1809.49, "Extreme case"),
        (4, 180, 99.99, "Receipt ending .99")
    ]
    
    for days, miles, receipts, description in examples:
        result = engine.calculate_reimbursement(days, miles, receipts)
        print(f"  {description}: {days}d, {miles}mi, ${receipts:.2f} -> ${result:.2f}")

if __name__ == "__main__":
    main()
