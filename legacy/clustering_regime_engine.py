#!/usr/bin/env python3
"""
Clustering-Based Regime Engine
Discovers natural patterns in all 1000 cases using clustering, then learns 
regime-specific parameters for each cluster through optimization.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class ClusteringRegimeEngine:
    def __init__(self, n_clusters=6):
        self.n_clusters = n_clusters
        self.cluster_model = None
        self.cluster_params = {}  # Parameters for each cluster
        self.regime_classifier = None  # Predicts cluster from new inputs
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.cluster_labels = None
        self.cluster_info = {}
        
        # Edge cases for validation
        self.edge_cases = [
            (8, 482, 1411.49, 631.81, "Case 548"),
            (6, 204, 818.99, 628.40, "Case 82"),
            (10, 1192, 23.47, 1157.87, "Case 522"),
            (7, 1006, 1181.33, 2279.82, "Case 149"),
            (2, 384, 495.49, 290.36, "Case 89")
        ]
    
    def extract_clustering_features(self, cases):
        """Extract comprehensive features for clustering analysis."""
        features = []
        targets = []
        
        for case in cases:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            miles_per_day = miles / max(days, 1)
            receipts_per_day = receipts / max(days, 1)
            
            # Comprehensive feature set
            feature_vector = [
                # Basic inputs
                days, miles, receipts,
                
                # Derived ratios
                miles_per_day, receipts_per_day,
                
                # Efficiency measures
                expected / max(days, 1),           # Expected per day
                expected / max(miles, 1),          # Expected per mile
                expected / max(receipts, 1),       # Expected per receipt dollar
                
                # Pattern indicators
                days * miles_per_day,              # Trip intensity
                receipts_per_day / max(miles_per_day, 1), # Expense efficiency
                miles / max(receipts, 1),          # Miles per expense dollar
                
                # Categorical encodings
                float(days <= 2),                  # Short trip
                float(3 <= days <= 4),             # Medium-short trip
                float(5 <= days <= 6),             # Medium trip
                float(days >= 7),                  # Long trip
                float(miles_per_day < 50),         # Low mobility
                float(50 <= miles_per_day <= 150), # Normal mobility
                float(150 <= miles_per_day <= 300), # High mobility
                float(miles_per_day > 300),        # Extreme mobility
                float(receipts_per_day < 50),      # Low spending
                float(50 <= receipts_per_day <= 150), # Normal spending
                float(receipts_per_day > 150),     # High spending
                
                # Interaction patterns
                float(days <= 3 and receipts_per_day > 200),    # Suspicious short expensive
                float(days >= 7 and receipts_per_day < 30),     # Long minimal expense
                float(miles >= 500 and receipts_per_day <= 100), # High mile efficient
                float(days >= 5 and miles_per_day > 200),       # Business intensive
                
                # Receipt ending patterns (discovered quirks)
                float(f"{receipts:.2f}".endswith('.49')),
                float(f"{receipts:.2f}".endswith('.99')),
            ]
            
            features.append(feature_vector)
            targets.append(expected)
        
        return np.array(features), np.array(targets)
    
    def perform_clustering_analysis(self, features, targets, cases):
        """Perform clustering analysis to discover natural patterns."""
        print("Performing clustering analysis...")
        
        # Normalize features for clustering
        features_normalized = self.feature_scaler.fit_transform(features)
        
        # Try different numbers of clusters and find optimal
        silhouette_scores = []
        cluster_range = range(3, 10)
        
        print("Finding optimal number of clusters...")
        for n in cluster_range:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_normalized)
            score = silhouette_score(features_normalized, labels)
            silhouette_scores.append(score)
            print(f"  {n} clusters: silhouette score = {score:.3f}")
        
        # Use best number of clusters
        best_n = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {best_n}")
        self.n_clusters = best_n
        
        # Perform final clustering
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.cluster_model.fit_predict(features_normalized)
        
        # Analyze clusters
        self.analyze_clusters(cases, targets)
        
        return self.cluster_labels
    
    def analyze_clusters(self, cases, targets):
        """Analyze the discovered clusters to understand patterns."""
        print(f"\nCluster Analysis ({self.n_clusters} clusters):")
        print("=" * 60)
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_cases = [cases[i] for i in range(len(cases)) if cluster_mask[i]]
            cluster_targets = targets[cluster_mask]
            
            if len(cluster_cases) == 0:
                continue
            
            # Calculate cluster statistics
            days_list = [c['input']['trip_duration_days'] for c in cluster_cases]
            miles_list = [c['input']['miles_traveled'] for c in cluster_cases]
            receipts_list = [c['input']['total_receipts_amount'] for c in cluster_cases]
            
            cluster_stats = {
                'size': len(cluster_cases),
                'avg_days': np.mean(days_list),
                'avg_miles': np.mean(miles_list),
                'avg_receipts': np.mean(receipts_list),
                'avg_expected': np.mean(cluster_targets),
                'avg_receipts_per_day': np.mean([r/max(d,1) for r, d in zip(receipts_list, days_list)]),
                'avg_miles_per_day': np.mean([m/max(d,1) for m, d in zip(miles_list, days_list)]),
            }
            
            self.cluster_info[cluster_id] = cluster_stats
            
            print(f"\nCluster {cluster_id}: {cluster_stats['size']} cases")
            print(f"  Avg trip: {cluster_stats['avg_days']:.1f}d, "
                  f"{cluster_stats['avg_miles']:.0f}mi, ${cluster_stats['avg_receipts']:.0f}")
            print(f"  Per day: {cluster_stats['avg_miles_per_day']:.0f}mi/d, "
                  f"${cluster_stats['avg_receipts_per_day']:.0f}/d")
            print(f"  Avg expected: ${cluster_stats['avg_expected']:.0f}")
            
            # Check if any edge cases are in this cluster
            edge_cases_in_cluster = []
            for days, miles, receipts, expected, name in self.edge_cases:
                # Find which cluster this edge case would belong to
                edge_features = self.extract_clustering_features([{
                    'input': {'trip_duration_days': days, 'miles_traveled': miles, 'total_receipts_amount': receipts},
                    'expected_output': expected
                }])[0]
                edge_normalized = self.feature_scaler.transform([edge_features[0]])
                edge_cluster = self.cluster_model.predict(edge_normalized)[0]
                
                if edge_cluster == cluster_id:
                    edge_cases_in_cluster.append(name)
            
            if edge_cases_in_cluster:
                print(f"  Edge cases: {', '.join(edge_cases_in_cluster)}")
    
    def learn_cluster_parameters(self, cluster_id, train_cases, val_cases):
        """Learn optimal parameters for a specific cluster using train/val split."""
        if len(train_cases) < 5:  # Too few cases to optimize
            return self.get_default_parameters()
        
        print(f"\nOptimizing parameters for Cluster {cluster_id} (Train: {len(train_cases)}, Val: {len(val_cases)})...")
        
        # Initialize parameters for this cluster
        params = {
            'base_per_diem': 100.0,
            'mile_rate_1': 0.58,
            'mile_rate_2': 0.50,
            'mile_rate_3': 0.42,
            'mile_tier_1': 100.0,
            'mile_tier_2': 300.0,
            'receipt_rate': 0.80,
            'expense_cap': 80.0,
            'penalty_mild': 0.90,
            'penalty_moderate': 0.75,
            'penalty_severe': 0.50,
            'efficiency_bonus': 1.0,
            'high_mile_bonus': 1.0,
            'minimal_expense_bonus': 0.0,
        }
        
        # Define bounds
        bounds = [
            (80, 120),    # base_per_diem
            (0.45, 0.65), # mile_rate_1
            (0.35, 0.55), # mile_rate_2
            (0.30, 0.50), # mile_rate_3
            (75, 150),    # mile_tier_1
            (200, 400),   # mile_tier_2
            (0.60, 0.95), # receipt_rate
            (50, 150),    # expense_cap
            (0.80, 0.95), # penalty_mild
            (0.60, 0.85), # penalty_moderate
            (0.30, 0.70), # penalty_severe
            (0.90, 1.30), # efficiency_bonus
            (0.90, 1.50), # high_mile_bonus
            (0.0, 0.20),  # minimal_expense_bonus
        ]
        
        def cluster_objective(param_values):
            # Update parameters
            param_names = list(params.keys())
            for i, value in enumerate(param_values):
                params[param_names[i]] = value
            
            # Calculate validation error (what we optimize for)
            val_error = 0.0
            if len(val_cases) > 0:
                for case in val_cases:
                    days = case['input']['trip_duration_days']
                    miles = case['input']['miles_traveled']
                    receipts = case['input']['total_receipts_amount']
                    expected = case['expected_output']
                    
                    try:
                        predicted = self.predict_with_cluster_params(days, miles, receipts, params)
                        error = abs(predicted - expected)
                        val_error += error
                    except:
                        val_error += 1000  # Penalty for invalid parameters
                
                val_error /= len(val_cases)
            else:
                # If no validation cases, use training error
                for case in train_cases:
                    days = case['input']['trip_duration_days']
                    miles = case['input']['miles_traveled']
                    receipts = case['input']['total_receipts_amount']
                    expected = case['expected_output']
                    
                    try:
                        predicted = self.predict_with_cluster_params(days, miles, receipts, params)
                        error = abs(predicted - expected)
                        val_error += error
                    except:
                        val_error += 1000
                
                val_error /= len(train_cases)
            
            return val_error
        
        # Optimize
        initial_values = list(params.values())
        
        try:
            result = optimize.minimize(
                cluster_objective,
                initial_values,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150}
            )
            
            if result.success:
                # Update parameters with optimized values
                param_names = list(params.keys())
                for i, value in enumerate(result.x):
                    params[param_names[i]] = value
                
                # Calculate training error for comparison
                train_error = 0.0
                for case in train_cases:
                    days = case['input']['trip_duration_days']
                    miles = case['input']['miles_traveled']
                    receipts = case['input']['total_receipts_amount']
                    expected = case['expected_output']
                    predicted = self.predict_with_cluster_params(days, miles, receipts, params)
                    train_error += abs(predicted - expected)
                train_error /= len(train_cases)
                
                print(f"  Optimization successful: Train MAE = ${train_error:.2f}, Val MAE = ${result.fun:.2f}")
            else:
                print(f"  Optimization failed, using defaults")
                
        except Exception as e:
            print(f"  Optimization error: {e}, using defaults")
        
        return params
    
    def predict_with_cluster_params(self, days, miles, receipts, params):
        """Predict using cluster-specific parameters."""
        miles_per_day = miles / max(days, 1)
        receipts_per_day = receipts / max(days, 1)
        
        # Per diem
        per_diem = days * params['base_per_diem']
        
        # Mileage with cluster-specific rates
        if miles <= params['mile_tier_1']:
            mileage = miles * params['mile_rate_1']
        elif miles <= params['mile_tier_2']:
            mileage = (params['mile_tier_1'] * params['mile_rate_1'] + 
                      (miles - params['mile_tier_1']) * params['mile_rate_2'])
        else:
            mileage = (params['mile_tier_1'] * params['mile_rate_1'] + 
                      (params['mile_tier_2'] - params['mile_tier_1']) * params['mile_rate_2'] +
                      (miles - params['mile_tier_2']) * params['mile_rate_3'])
        
        # Receipts
        receipt_reimb = receipts * params['receipt_rate']
        
        # Apply cluster-specific adjustments
        base_total = per_diem + mileage + receipt_reimb
        
        # Expense penalties based on cluster-learned cap
        if receipts_per_day > params['expense_cap']:
            violation_ratio = receipts_per_day / params['expense_cap']
            if violation_ratio < 1.3:
                penalty = params['penalty_mild']
            elif violation_ratio < 1.8:
                penalty = params['penalty_moderate']
            else:
                penalty = params['penalty_severe']
            
            base_total *= penalty
        
        # Efficiency bonuses
        if 150 <= miles_per_day <= 250:
            base_total *= params['efficiency_bonus']
        
        # High mileage bonus
        if miles >= 600 and receipts_per_day <= 120:
            base_total *= params['high_mile_bonus']
        
        # Minimal expense bonus
        if receipts_per_day <= 30 and miles >= 200:
            base_total += miles * params['minimal_expense_bonus']
        
        return base_total
    
    def get_default_parameters(self):
        """Get default parameters for clusters with too few cases."""
        return {
            'base_per_diem': 100.0,
            'mile_rate_1': 0.58,
            'mile_rate_2': 0.50,
            'mile_rate_3': 0.42,
            'mile_tier_1': 100.0,
            'mile_tier_2': 300.0,
            'receipt_rate': 0.80,
            'expense_cap': 80.0,
            'penalty_mild': 0.90,
            'penalty_moderate': 0.75,
            'penalty_severe': 0.50,
            'efficiency_bonus': 1.05,
            'high_mile_bonus': 1.10,
            'minimal_expense_bonus': 0.10,
        }
    
    def create_stratified_splits(self, cases, cluster_labels):
        """Create 60/20/20 stratified splits maintaining cluster distribution."""
        print("\nCreating 60/20/20 stratified splits by clusters...")
        
        # Convert to arrays for easier handling
        cases_array = np.array(cases)
        
        # First split: 80% train+val, 20% test (stratified by cluster)
        train_val_idx, test_idx = train_test_split(
            np.arange(len(cases)), 
            test_size=0.2, 
            random_state=42, 
            stratify=cluster_labels
        )
        
        # Second split: 60% train, 20% val from the 80% (stratified by cluster)  
        train_val_labels = cluster_labels[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.25,  # 0.25 * 0.8 = 0.2 (20% of total)
            random_state=42,
            stratify=train_val_labels
        )
        
        # Create splits
        train_cases = [cases[i] for i in train_idx]
        val_cases = [cases[i] for i in val_idx]
        test_cases = [cases[i] for i in test_idx]
        
        train_labels = cluster_labels[train_idx]
        val_labels = cluster_labels[val_idx]
        test_labels = cluster_labels[test_idx]
        
        # Verify cluster distribution
        print(f"Split sizes: Train={len(train_cases)} ({len(train_cases)/len(cases)*100:.1f}%), "
              f"Val={len(val_cases)} ({len(val_cases)/len(cases)*100:.1f}%), "
              f"Test={len(test_cases)} ({len(test_cases)/len(cases)*100:.1f}%)")
        
        print("Cluster distribution verification:")
        for cluster_id in range(self.n_clusters):
            train_count = np.sum(train_labels == cluster_id)
            val_count = np.sum(val_labels == cluster_id)
            test_count = np.sum(test_labels == cluster_id)
            total_count = train_count + val_count + test_count
            
            if total_count > 0:
                print(f"  Cluster {cluster_id}: Train={train_count} ({train_count/total_count*100:.1f}%), "
                      f"Val={val_count} ({val_count/total_count*100:.1f}%), "
                      f"Test={test_count} ({test_count/total_count*100:.1f}%)")
        
        return {
            'train': {'cases': train_cases, 'labels': train_labels, 'indices': train_idx},
            'val': {'cases': val_cases, 'labels': val_labels, 'indices': val_idx},
            'test': {'cases': test_cases, 'labels': test_labels, 'indices': test_idx}
        }
    
    def train_regime_classifier(self, features, splits):
        """Train a classifier to predict cluster membership using stratified splits."""
        print("\nTraining regime classifier on stratified data...")
        
        # Use the stratified train/test split
        train_features = features[splits['train']['indices']]
        test_features = features[splits['test']['indices']]
        train_labels = splits['train']['labels']
        test_labels = splits['test']['labels']
        
        # Train classifier
        self.regime_classifier = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42
        )
        self.regime_classifier.fit(train_features, train_labels)
        
        # Evaluate
        train_accuracy = self.regime_classifier.score(train_features, train_labels)
        test_accuracy = self.regime_classifier.score(test_features, test_labels)
        
        print(f"Regime classifier accuracy: Train={train_accuracy:.3f}, Test={test_accuracy:.3f}")
        
        return self.regime_classifier
    
    def evaluate_on_test_set(self, test_split):
        """Evaluate the clustering model on the held-out test set."""
        print(f"\nEvaluating on held-out test set ({len(test_split['cases'])} cases)...")
        
        total_error = 0.0
        cluster_errors = {i: [] for i in range(self.n_clusters)}
        
        for i, case in enumerate(test_split['cases']):
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            true_cluster = test_split['labels'][i]
            
            predicted = self.predict(days, miles, receipts)
            predicted_cluster = self.predict_cluster(days, miles, receipts)
            error = abs(predicted - expected)
            
            total_error += error
            cluster_errors[true_cluster].append(error)
        
        # Overall test performance
        avg_error = total_error / len(test_split['cases'])
        print(f"Test Set Results:")
        print(f"  Overall MAE: ${avg_error:.2f}")
        
        # Per-cluster test performance
        print("  Per-cluster test performance:")
        for cluster_id in range(self.n_clusters):
            if cluster_errors[cluster_id]:
                cluster_mae = np.mean(cluster_errors[cluster_id])
                cluster_count = len(cluster_errors[cluster_id])
                print(f"    Cluster {cluster_id}: {cluster_count} cases, MAE = ${cluster_mae:.2f}")
        
        return avg_error
    
    def train(self, cases):
        """Train the clustering-based regime engine."""
        print("Training Clustering-Based Regime Engine")
        print("=" * 50)
        print(f"Training on {len(cases)} cases...")
        
        # Extract features
        features, targets = self.extract_clustering_features(cases)
        print(f"Extracted {features.shape[1]} features per case")
        
        # Perform clustering analysis
        cluster_labels = self.perform_clustering_analysis(features, targets, cases)
        
        # Create stratified splits maintaining cluster distribution
        splits = self.create_stratified_splits(cases, cluster_labels)
        
        # Learn parameters for each cluster using train/val splits
        print(f"\nLearning cluster-specific parameters with train/val optimization...")
        for cluster_id in range(self.n_clusters):
            # Get cluster cases from train and validation sets
            train_cluster_cases = [case for i, case in enumerate(splits['train']['cases']) 
                                 if splits['train']['labels'][i] == cluster_id]
            val_cluster_cases = [case for i, case in enumerate(splits['val']['cases']) 
                               if splits['val']['labels'][i] == cluster_id]
            
            if len(train_cluster_cases) > 0:
                self.cluster_params[cluster_id] = self.learn_cluster_parameters(
                    cluster_id, train_cluster_cases, val_cluster_cases
                )
        
        # Train regime classifier using stratified splits
        self.train_regime_classifier(features, splits)
        
        # Evaluate on test set
        self.evaluate_on_test_set(splits['test'])
        
        # Evaluate on edge cases
        self.test_edge_cases()
        
        # Store splits for later analysis
        self.splits = splits
        
        return self
    
    def predict_cluster(self, days, miles, receipts):
        """Predict which cluster a new case belongs to."""
        if self.regime_classifier is None:
            return 0  # Default cluster
        
        # Extract features for this case
        case_features = self.extract_clustering_features([{
            'input': {'trip_duration_days': days, 'miles_traveled': miles, 'total_receipts_amount': receipts},
            'expected_output': 0  # Dummy value
        }])[0]
        
        # Predict cluster
        cluster_id = self.regime_classifier.predict([case_features[0]])[0]
        return cluster_id
    
    def predict(self, days, miles, receipts):
        """Main prediction method using clustering."""
        # Predict cluster
        cluster_id = self.predict_cluster(days, miles, receipts)
        
        # Get cluster parameters
        if cluster_id in self.cluster_params:
            params = self.cluster_params[cluster_id]
        else:
            params = self.get_default_parameters()
        
        # Predict using cluster-specific parameters
        prediction = self.predict_with_cluster_params(days, miles, receipts, params)
        
        return prediction
    
    def test_edge_cases(self):
        """Test the clustering approach on edge cases."""
        print(f"\nTesting Edge Cases with Clustering Approach:")
        print("=" * 60)
        
        total_error = 0.0
        for days, miles, receipts, expected, name in self.edge_cases:
            predicted = self.predict(days, miles, receipts)
            cluster_id = self.predict_cluster(days, miles, receipts)
            error = abs(predicted - expected)
            total_error += error
            
            print(f"\n{name}")
            print(f"  Input: {days}d, {miles}mi, ${receipts:.2f}")
            print(f"  Expected: ${expected:.2f}, Predicted: ${predicted:.2f}")
            print(f"  Error: ${error:.2f}")
            print(f"  Assigned to Cluster: {cluster_id}")
            if cluster_id in self.cluster_info:
                print(f"  Cluster profile: {self.cluster_info[cluster_id]['size']} cases, "
                      f"avg ${self.cluster_info[cluster_id]['avg_expected']:.0f}")
        
        avg_error = total_error / len(self.edge_cases)
        print(f"\nEdge Case Results:")
        print(f"  Average Error: ${avg_error:.2f}")
        print(f"  Total Error: ${total_error:.2f}")
        
        return avg_error
    
    def save(self, filepath):
        """Save the clustering regime model."""
        model_data = {
            'n_clusters': self.n_clusters,
            'cluster_model': self.cluster_model,
            'cluster_params': self.cluster_params,
            'regime_classifier': self.regime_classifier,
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler,
            'cluster_labels': self.cluster_labels,
            'cluster_info': self.cluster_info,
            'edge_cases': self.edge_cases
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Clustering regime model saved to {filepath}")
    
    def load(self, filepath):
        """Load the clustering regime model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_clusters = model_data['n_clusters']
        self.cluster_model = model_data['cluster_model']
        self.cluster_params = model_data['cluster_params']
        self.regime_classifier = model_data['regime_classifier']
        self.scaler = model_data['scaler']
        self.feature_scaler = model_data['feature_scaler']
        self.cluster_labels = model_data['cluster_labels']
        self.cluster_info = model_data['cluster_info']
        self.edge_cases = model_data.get('edge_cases', [])
        
        return self

def main():
    """Train and test the clustering-based regime engine."""
    print("Clustering-Based Regime Engine")
    print("=" * 40)
    
    # Load data
    try:
        with open('public_cases.json', 'r') as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} training cases")
    except FileNotFoundError:
        print("Error: public_cases.json not found!")
        return
    
    # Create and train engine
    engine = ClusteringRegimeEngine()
    engine.train(cases)
    
    # Save the model
    engine.save('clustering_regime_model.pkl')
    
    print("\nClustering-based regime engine training complete!")

if __name__ == "__main__":
    main()