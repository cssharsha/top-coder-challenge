#!/usr/bin/env python3
"""
Hybrid Enhanced-Clustering Engine
Combines clustering-based regime detection with enhanced parametric prediction logic.
Uses full dataset (1000 cases) with proper 60/20/20 stratified splits.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class HybridEnhancedClusteringEngine:
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters
        self.cluster_model = None
        self.cluster_params = {}  # Enhanced parametric parameters per cluster
        self.regime_classifier = None
        self.ml_models = {}  # ML model per cluster
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
            
            # Comprehensive feature set for clustering
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
                
                # Receipt ending patterns
                float(f"{receipts:.2f}".endswith('.49')),
                float(f"{receipts:.2f}".endswith('.99')),
            ]
            
            features.append(feature_vector)
            targets.append(expected)
        
        return np.array(features), np.array(targets)
    
    def perform_clustering_analysis(self, features, targets, cases):
        """Perform clustering analysis with optimal cluster selection."""
        print("Performing clustering analysis...")
        
        # Normalize features for clustering
        features_normalized = self.feature_scaler.fit_transform(features)
        
        # If n_clusters is specified, use it directly
        if self.n_clusters is not None:
            print(f"Using specified number of clusters: {self.n_clusters}")
            self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.cluster_labels = self.cluster_model.fit_predict(features_normalized)
            self.analyze_clusters(cases, targets)
            return self.cluster_labels
        
        # Comprehensive cluster evaluation
        cluster_range = range(3, 12)
        evaluation_results = []
        
        print("Evaluating cluster configurations...")
        for n in cluster_range:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_normalized)
            
            # Silhouette score (clustering quality)
            silhouette = silhouette_score(features_normalized, labels)
            
            # Cluster size analysis (optimization feasibility)
            cluster_sizes = [np.sum(labels == i) for i in range(n)]
            min_cluster_size = min(cluster_sizes)
            avg_cluster_size = np.mean(cluster_sizes)
            cluster_balance = min_cluster_size / avg_cluster_size  # How balanced are clusters
            
            # Inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            
            # Edge case distribution analysis
            edge_case_distribution = self.analyze_edge_case_distribution(features_normalized, labels, kmeans)
            
            evaluation_results.append({
                'n_clusters': n,
                'silhouette': silhouette,
                'min_cluster_size': min_cluster_size,
                'avg_cluster_size': avg_cluster_size,
                'cluster_balance': cluster_balance,
                'inertia': inertia,
                'edge_case_spread': edge_case_distribution,
                'kmeans_model': kmeans,
                'labels': labels
            })
            
            print(f"  {n} clusters: silhouette={silhouette:.3f}, min_size={min_cluster_size}, "
                  f"balance={cluster_balance:.2f}, edge_spread={edge_case_distribution}")
        
        # Store all configurations for multi-model training
        self.all_cluster_configs = evaluation_results
        
        # Select optimal number of clusters using composite score
        best_config = self.select_optimal_clusters(evaluation_results)
        
        print(f"\nOptimal configuration selected: {best_config['n_clusters']} clusters")
        print(f"  Silhouette: {best_config['silhouette']:.3f}")
        print(f"  Min cluster size: {best_config['min_cluster_size']}")
        print(f"  Cluster balance: {best_config['cluster_balance']:.2f}")
        print(f"  Edge case spread: {best_config['edge_case_spread']}")
        
        # Store optimal configuration
        self.optimal_config = best_config
        
        # Use the optimal configuration as default
        self.n_clusters = best_config['n_clusters']
        self.cluster_model = best_config['kmeans_model']
        self.cluster_labels = best_config['labels']
        
        # Analyze final clusters
        self.analyze_clusters(cases, targets)
        
        return self.cluster_labels
    
    def analyze_edge_case_distribution(self, features_normalized, labels, kmeans_model):
        """Analyze how edge cases are distributed across clusters."""
        edge_clusters = []
        
        for days, miles, receipts, expected, name in self.edge_cases:
            # Extract features for this edge case
            edge_features = self.extract_clustering_features([{
                'input': {'trip_duration_days': days, 'miles_traveled': miles, 'total_receipts_amount': receipts},
                'expected_output': expected
            }])[0]
            edge_normalized = self.feature_scaler.transform([edge_features[0]])
            edge_cluster = kmeans_model.predict(edge_normalized)[0]
            edge_clusters.append(edge_cluster)
        
        # Calculate edge case spread (how many different clusters contain edge cases)
        unique_edge_clusters = len(set(edge_clusters))
        return unique_edge_clusters
    
    def select_optimal_clusters(self, evaluation_results):
        """Select optimal number of clusters using composite scoring."""
        
        # Normalize scores for comparison
        silhouettes = [r['silhouette'] for r in evaluation_results]
        min_sizes = [r['min_cluster_size'] for r in evaluation_results]
        balances = [r['cluster_balance'] for r in evaluation_results]
        edge_spreads = [r['edge_case_spread'] for r in evaluation_results]
        
        # Normalize to 0-1 range
        def normalize(values):
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [1.0] * len(values)
            return [(v - min_val) / (max_val - min_val) for v in values]
        
        norm_silhouettes = normalize(silhouettes)
        norm_min_sizes = normalize(min_sizes)
        norm_balances = normalize(balances)
        
        # For edge case spread, we want moderate values (not too concentrated, not too dispersed)
        # Penalize both extremes
        norm_edge_spreads = []
        optimal_spread = 3  # Ideally spread across 3-4 clusters
        for spread in edge_spreads:
            penalty = abs(spread - optimal_spread) / optimal_spread
            norm_edge_spreads.append(max(0, 1 - penalty))
        
        # Composite scoring with weights
        composite_scores = []
        for i, result in enumerate(evaluation_results):
            n = result['n_clusters']
            
            # Weights for different criteria
            silhouette_weight = 0.35      # Clustering quality
            min_size_weight = 0.25        # Optimization feasibility (min cluster size)
            balance_weight = 0.20         # Cluster balance
            edge_spread_weight = 0.20     # Edge case distribution
            
            # Penalty for too many clusters (complexity)
            complexity_penalty = max(0, (n - 8) * 0.05)  # Penalize >8 clusters
            
            # Bonus for minimum viable cluster sizes (≥20 for reliable optimization)
            min_size_bonus = 0.1 if result['min_cluster_size'] >= 20 else 0
            
            score = (norm_silhouettes[i] * silhouette_weight +
                    norm_min_sizes[i] * min_size_weight +
                    norm_balances[i] * balance_weight +
                    norm_edge_spreads[i] * edge_spread_weight +
                    min_size_bonus - complexity_penalty)
            
            composite_scores.append(score)
            
            print(f"    {n} clusters: composite_score={score:.3f} "
                  f"(sil={norm_silhouettes[i]:.2f}, size={norm_min_sizes[i]:.2f}, "
                  f"bal={norm_balances[i]:.2f}, edge={norm_edge_spreads[i]:.2f})")
        
        # Select best configuration
        best_idx = np.argmax(composite_scores)
        return evaluation_results[best_idx]
    
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
    
    def get_cluster_specific_param_template(self, cluster_id):
        """Get cluster-specific parameter template based on cluster characteristics."""
        cluster_stats = self.cluster_info[cluster_id]
        
        # Base template - enhanced parametric parameters
        base_params = {
            # Base per diem rates
            'base_per_diem': 100.0,
            'day_5_bonus': 1.05,
            'sweet_spot_bonus': 1.03,
            'long_trip_penalty': 0.95,
            'short_trip_penalty': 0.98,
            
            # Mileage parameters
            'mile_tier1_limit': 100.0,
            'mile_tier2_limit': 300.0,
            'mile_tier3_limit': 600.0,
            'mile_rate1': 0.58,
            'mile_rate2': 0.50,
            'mile_rate3': 0.42,
            'mile_rate4': 0.35,
            
            # Receipt processing
            'receipt_normal_rate': 0.85,
            'receipt_optimal_rate': 0.90,
            'receipt_high_rate': 0.70,
            'receipt_extreme_rate': 0.50,
            
            # Efficiency bonuses
            'efficiency_bonus': 1.05,
            'high_mile_bonus': 1.10,
            'minimal_expense_bonus': 0.05,
            
            # Pattern penalties
            'extreme_spending_penalty': 0.60,
            'short_expensive_penalty': 0.70,
            'long_minimal_penalty': 0.85,
            
            # Receipt ending bonuses
            'ending_49_bonus': 1.02,
            'ending_99_bonus': 1.01,
            
            # ML weight
            'ml_weight': 0.80,
        }
        
        # Customize based on cluster characteristics
        avg_receipts_per_day = cluster_stats['avg_receipts_per_day']
        avg_miles_per_day = cluster_stats['avg_miles_per_day']
        avg_days = cluster_stats['avg_days']
        
        # Adjust parameters based on cluster profile
        if avg_receipts_per_day > 200:  # High expense cluster
            base_params['receipt_high_rate'] = 0.60
            base_params['extreme_spending_penalty'] = 0.50
        elif avg_receipts_per_day < 50:  # Low expense cluster
            base_params['receipt_optimal_rate'] = 0.95
            base_params['minimal_expense_bonus'] = 0.15
        
        if avg_miles_per_day > 200:  # High mobility cluster
            base_params['efficiency_bonus'] = 1.15
            base_params['high_mile_bonus'] = 1.20
        elif avg_miles_per_day < 50:  # Low mobility cluster
            base_params['mile_rate1'] = 0.55
            
        if avg_days > 8:  # Long trip cluster
            base_params['long_trip_penalty'] = 0.90
        elif avg_days < 3:  # Short trip cluster
            base_params['short_trip_penalty'] = 0.95
            base_params['short_expensive_penalty'] = 0.60
        
        return base_params
    
    def enhanced_parametric_prediction(self, days, miles, receipts, cluster_params):
        """Enhanced parametric prediction using cluster-specific parameters."""
        p = cluster_params
        
        # Base per diem calculation
        base_rate = p['base_per_diem']
        if days == 5:
            per_diem_rate = base_rate * p['day_5_bonus']
        elif 4 <= days <= 6:
            per_diem_rate = base_rate * p['sweet_spot_bonus']
        elif days >= 8:
            per_diem_rate = base_rate * p['long_trip_penalty']
        elif days <= 2:
            per_diem_rate = base_rate * p['short_trip_penalty']
        else:
            per_diem_rate = base_rate
            
        per_diem = days * per_diem_rate
        
        # Mileage calculation with tiers
        miles_per_day = miles / max(days, 1)
        
        if miles <= p['mile_tier1_limit']:
            mileage_reimb = miles * p['mile_rate1']
        elif miles <= p['mile_tier2_limit']:
            mileage_reimb = (p['mile_tier1_limit'] * p['mile_rate1'] + 
                           (miles - p['mile_tier1_limit']) * p['mile_rate2'])
        elif miles <= p['mile_tier3_limit']:
            mileage_reimb = (p['mile_tier1_limit'] * p['mile_rate1'] + 
                           (p['mile_tier2_limit'] - p['mile_tier1_limit']) * p['mile_rate2'] +
                           (miles - p['mile_tier2_limit']) * p['mile_rate3'])
        else:
            mileage_reimb = (p['mile_tier1_limit'] * p['mile_rate1'] + 
                           (p['mile_tier2_limit'] - p['mile_tier1_limit']) * p['mile_rate2'] +
                           (p['mile_tier3_limit'] - p['mile_tier2_limit']) * p['mile_rate3'] +
                           (miles - p['mile_tier3_limit']) * p['mile_rate4'])
        
        # Receipt processing
        receipts_per_day = receipts / max(days, 1)
        
        if receipts_per_day <= 75:
            receipt_reimb = receipts * p['receipt_optimal_rate']
        elif receipts_per_day <= 150:
            receipt_reimb = receipts * p['receipt_normal_rate']
        elif receipts_per_day <= 250:
            receipt_reimb = receipts * p['receipt_high_rate']
        else:
            receipt_reimb = receipts * p['receipt_extreme_rate']
        
        # Receipt ending bonuses
        receipt_str = f"{receipts:.2f}"
        if receipt_str.endswith('.49'):
            receipt_reimb *= p['ending_49_bonus']
        elif receipt_str.endswith('.99'):
            receipt_reimb *= p['ending_99_bonus']
        
        # Calculate base total
        total = per_diem + mileage_reimb + receipt_reimb
        
        # Apply efficiency bonuses
        if 150 <= miles_per_day <= 250:
            total *= p['efficiency_bonus']
        
        # High mileage bonus
        if miles >= 600 and receipts_per_day <= 120:
            total *= p['high_mile_bonus']
        
        # Minimal expense bonus
        if receipts_per_day <= 30 and miles >= 200:
            total += miles * p['minimal_expense_bonus']
        
        # Pattern penalties
        if receipts_per_day > 300:  # Extreme spending
            total *= p['extreme_spending_penalty']
        elif days <= 2 and receipts_per_day > 200:  # Short expensive
            total *= p['short_expensive_penalty']
        elif days >= 8 and receipts_per_day < 20:  # Long minimal
            total *= p['long_minimal_penalty']
        
        return total
    
    def extract_enhanced_features(self, days, miles, receipts):
        """Extract enhanced features for ML model."""
        miles_per_day = miles / max(days, 1)
        receipts_per_day = receipts / max(days, 1)
        
        features = [
            # Base features
            days, miles, receipts, miles_per_day, receipts_per_day,
            
            # Polynomial features
            days * miles, days * receipts, miles * receipts,
            days ** 2, miles ** 2, receipts ** 2,
            np.sqrt(miles), np.sqrt(receipts),
            np.log1p(days), np.log1p(miles), np.log1p(receipts),
            
            # Pattern features
            float(f"{receipts:.2f}".endswith('.49')),
            float(f"{receipts:.2f}".endswith('.99')),
            float(days == 5),
            float(days >= 8),
            float(days <= 2),
            float(150 <= miles_per_day <= 250),
            float(miles_per_day > 300),
            float(receipts_per_day > 200),
            float(receipts_per_day < 50),
            
            # Interaction features
            float(days >= 7 and miles_per_day > 100 and receipts_per_day > 100),
            float(days <= 3 and miles_per_day < 100 and receipts_per_day > 150),
        ]
        
        return np.array(features)
    
    def learn_cluster_parameters(self, cluster_id, train_cases, val_cases):
        """Learn optimal parameters for a specific cluster."""
        if len(train_cases) < 10:  # Too few cases to optimize
            return self.get_cluster_specific_param_template(cluster_id)
        
        print(f"\nOptimizing parameters for Cluster {cluster_id} (Train: {len(train_cases)}, Val: {len(val_cases)})...")
        
        # Get cluster-specific template
        params = self.get_cluster_specific_param_template(cluster_id)
        
        # Train ML model for this cluster
        X_train, y_train = [], []
        for case in train_cases:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            features = self.extract_enhanced_features(days, miles, receipts)
            X_train.append(features)
            y_train.append(case['expected_output'])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        # Train cluster-specific ML model
        ml_model = GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42)
        ml_model.fit(X_train, y_train)
        self.ml_models[cluster_id] = ml_model
        
        # Define parameter bounds (complete set)
        param_bounds = {
            'base_per_diem': (80, 120),
            'day_5_bonus': (1.0, 1.15),
            'sweet_spot_bonus': (1.0, 1.10),
            'long_trip_penalty': (0.85, 1.0),
            'short_trip_penalty': (0.90, 1.05),
            'mile_tier1_limit': (75, 150),
            'mile_tier2_limit': (250, 400),
            'mile_tier3_limit': (500, 800),
            'mile_rate1': (0.50, 0.65),
            'mile_rate2': (0.40, 0.60),
            'mile_rate3': (0.35, 0.50),
            'mile_rate4': (0.25, 0.45),
            'receipt_normal_rate': (0.75, 0.95),
            'receipt_optimal_rate': (0.85, 0.98),
            'receipt_high_rate': (0.60, 0.80),
            'receipt_extreme_rate': (0.40, 0.70),
            'efficiency_bonus': (1.0, 1.20),
            'high_mile_bonus': (1.0, 1.30),
            'minimal_expense_bonus': (0.0, 0.20),
            'extreme_spending_penalty': (0.40, 0.80),
            'short_expensive_penalty': (0.50, 0.85),
            'long_minimal_penalty': (0.70, 0.95),
            'ending_49_bonus': (1.0, 1.05),
            'ending_99_bonus': (1.0, 1.03),
            'ml_weight': (0.60, 0.90),
        }
        
        # Enhanced optimization with progress tracking
        self.cluster_iteration_count = 0
        self.cluster_best_loss = float('inf')
        
        def cluster_objective(param_values):
            # Update parameters
            param_names = list(params.keys())
            for i, value in enumerate(param_values):
                params[param_names[i]] = value
            
            # Calculate validation error with enhanced loss function
            val_errors = []
            train_errors = []
            
            # Validation set evaluation
            if len(val_cases) > 0:
                for case in val_cases:
                    days = case['input']['trip_duration_days']
                    miles = case['input']['miles_traveled']
                    receipts = case['input']['total_receipts_amount']
                    expected = case['expected_output']
                    
                    try:
                        # Rule-based prediction
                        rule_pred = self.enhanced_parametric_prediction(days, miles, receipts, params)
                        
                        # ML prediction
                        features = self.extract_enhanced_features(days, miles, receipts).reshape(1, -1)
                        ml_pred = ml_model.predict(features)[0]
                        
                        # Blend predictions
                        predicted = params['ml_weight'] * ml_pred + (1 - params['ml_weight']) * rule_pred
                        error = abs(predicted - expected)
                        val_errors.append(error)
                    except:
                        val_errors.append(1000)  # Penalty for invalid parameters
                
                val_mae = np.mean(val_errors)
            else:
                val_mae = 0
            
            # Training set evaluation for combined loss
            for case in train_cases[:min(len(train_cases), 50)]:  # Sample for efficiency
                days = case['input']['trip_duration_days']
                miles = case['input']['miles_traveled']
                receipts = case['input']['total_receipts_amount']
                expected = case['expected_output']
                
                try:
                    rule_pred = self.enhanced_parametric_prediction(days, miles, receipts, params)
                    features = self.extract_enhanced_features(days, miles, receipts).reshape(1, -1)
                    ml_pred = ml_model.predict(features)[0]
                    predicted = params['ml_weight'] * ml_pred + (1 - params['ml_weight']) * rule_pred
                    error = abs(predicted - expected)
                    train_errors.append(error)
                except:
                    train_errors.append(1000)
            
            # Enhanced loss function (similar to enhanced_parametric_engine)
            all_errors = val_errors + train_errors
            mae = np.mean(all_errors)
            
            # Penalty for large errors (focus on reducing worst cases)
            large_error_penalty = np.mean([e for e in all_errors if e > 100]) if any(e > 100 for e in all_errors) else 0
            
            # Combined loss with emphasis on validation if available
            if len(val_cases) > 0:
                loss = val_mae + 0.2 * large_error_penalty
            else:
                loss = mae + 0.3 * large_error_penalty
            
            # Track progress
            self.cluster_iteration_count += 1
            if loss < self.cluster_best_loss:
                self.cluster_best_loss = loss
            
            if self.cluster_iteration_count % 25 == 0:
                print(f"    Iter {self.cluster_iteration_count}, Best: ${self.cluster_best_loss:.2f}, Current: ${loss:.2f}")
            
            return loss
        
        # Enhanced optimization with multiple attempts
        param_names = list(params.keys())
        initial_values = [params[name] for name in param_names]
        bounds = [param_bounds[name] for name in param_names]
        
        # Test initial parameters
        initial_loss = cluster_objective(initial_values)
        print(f"    Initial loss: ${initial_loss:.2f}")
        
        # Multiple optimization attempts with different starting points
        best_result = None
        best_loss = float('inf')
        
        for attempt in range(3):  # 3 attempts like enhanced_parametric_engine
            print(f"    Optimization attempt {attempt + 1}/3...")
            
            # Reset progress tracking for each attempt
            self.cluster_iteration_count = 0
            self.cluster_best_loss = float('inf')
            
            # Add noise for different starting points (except first attempt)
            if attempt > 0:
                noisy_initial = []
                for i, (value, (low, high)) in enumerate(zip(initial_values, bounds)):
                    noise = np.random.normal(0, 0.05) * value  # 5% noise
                    noisy_value = np.clip(value + noise, low, high)
                    noisy_initial.append(noisy_value)
                start_values = noisy_initial
            else:
                start_values = initial_values
            
            try:
                result = optimize.minimize(
                    cluster_objective,
                    start_values,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-6}  # Increased iterations and tighter tolerance
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
                    print(f"    New best loss: ${best_loss:.2f}")
                    
            except Exception as e:
                print(f"    Attempt {attempt + 1} failed: {e}")
        
        if best_result is not None and best_result.fun < initial_loss:
            # Update parameters with best result
            for i, value in enumerate(best_result.x):
                params[param_names[i]] = value
            
            improvement = initial_loss - best_result.fun
            print(f"  Optimization successful: Val MAE = ${best_result.fun:.2f} (improved by ${improvement:.2f})")
        else:
            print(f"  Optimization failed to improve, using template (initial: ${initial_loss:.2f})")
        
        return params
    
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
        """Train a classifier to predict cluster membership."""
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
    
    def train(self, cases, save_all_models=True):
        """Train the hybrid enhanced-clustering engine for all cluster configurations."""
        print("Training Hybrid Enhanced-Clustering Engine")
        print("=" * 50)
        print(f"Training on {len(cases)} cases...")
        
        # Extract features for clustering
        features, targets = self.extract_clustering_features(cases)
        print(f"Extracted {features.shape[1]} features per case")
        
        # Perform clustering analysis (gets all configurations)
        cluster_labels = self.perform_clustering_analysis(features, targets, cases)
        
        if save_all_models and hasattr(self, 'all_cluster_configs'):
            print(f"\nTraining models for all {len(self.all_cluster_configs)} cluster configurations...")
            
            # Train and save model for each cluster configuration
            for i, config in enumerate(self.all_cluster_configs):
                print(f"\n{'='*60}")
                print(f"Training model for {config['n_clusters']} clusters ({i+1}/{len(self.all_cluster_configs)})")
                print(f"{'='*60}")
                
                # Temporarily set this configuration
                temp_n_clusters = self.n_clusters
                temp_cluster_model = self.cluster_model
                temp_cluster_labels = self.cluster_labels
                temp_cluster_params = self.cluster_params.copy()
                temp_ml_models = self.ml_models.copy()
                
                # Use this configuration
                self.n_clusters = config['n_clusters']
                self.cluster_model = config['kmeans_model']
                self.cluster_labels = config['labels']
                self.cluster_params = {}
                self.ml_models = {}
                
                try:
                    # Train this configuration
                    test_mae = self.train_single_configuration(cases, features)
                    
                    # Save this configuration
                    model_filename = f'hybrid_enhanced_clustering_{config["n_clusters"]}clusters_model.pkl'
                    self.save(model_filename)
                    
                    print(f"Model saved: {model_filename} (Test MAE: ${test_mae:.2f})")
                    
                except Exception as e:
                    print(f"Failed to train {config['n_clusters']} cluster model: {e}")
                
                # Restore temporary state for next iteration
                self.cluster_params = temp_cluster_params.copy()
                self.ml_models = temp_ml_models.copy()
            
            # Restore optimal configuration
            self.n_clusters = temp_n_clusters
            self.cluster_model = temp_cluster_model
            self.cluster_labels = temp_cluster_labels
            
            print(f"\n{'='*60}")
            print(f"Training optimal model ({self.n_clusters} clusters)")
            print(f"{'='*60}")
        
        # Train the optimal configuration
        final_test_mae = self.train_single_configuration(cases, features)
        
        # Save optimal model
        self.save('hybrid_enhanced_clustering_model.pkl')
        print(f"\nOptimal model saved: hybrid_enhanced_clustering_model.pkl (Test MAE: ${final_test_mae:.2f})")
        
        return self
    
    def train_single_configuration(self, cases, features):
        """Train a single cluster configuration."""
        
        # Create stratified splits maintaining cluster distribution
        splits = self.create_stratified_splits(cases, self.cluster_labels)
        
        # Train regime classifier
        self.train_regime_classifier(features, splits)
        
        # Learn parameters for each cluster using train/val splits
        print(f"Learning cluster-specific enhanced parameters...")
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
        
        # Evaluate on test set
        test_mae = self.evaluate_on_test_set(splits['test'])
        
        # Test edge cases
        edge_mae = self.test_edge_cases()
        
        # Store splits for later analysis
        self.splits = splits
        
        return test_mae
    
    def update_todo_status(self, todo_id, status):
        """Helper to update todo status (placeholder for actual implementation)."""
        pass  # Will be implemented by TodoWrite calls
    
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
        """Main prediction method using hybrid approach."""
        # Predict cluster
        cluster_id = self.predict_cluster(days, miles, receipts)
        
        # Get cluster parameters and ML model
        if cluster_id in self.cluster_params:
            params = self.cluster_params[cluster_id]
            ml_model = self.ml_models.get(cluster_id)
        else:
            # Use default parameters if cluster not found
            params = self.get_cluster_specific_param_template(0)
            ml_model = self.ml_models.get(0)
        
        # Rule-based prediction
        rule_pred = self.enhanced_parametric_prediction(days, miles, receipts, params)
        
        # ML prediction
        if ml_model is not None:
            features = self.extract_enhanced_features(days, miles, receipts).reshape(1, -1)
            ml_pred = ml_model.predict(features)[0]
            
            # Blend predictions using cluster-specific weight
            final_pred = params['ml_weight'] * ml_pred + (1 - params['ml_weight']) * rule_pred
        else:
            final_pred = rule_pred
        
        return final_pred
    
    def evaluate_on_test_set(self, test_split):
        """Evaluate the hybrid model on the held-out test set."""
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
    
    def test_edge_cases(self):
        """Test the hybrid approach on edge cases."""
        print(f"\nTesting Edge Cases with Hybrid Enhanced-Clustering Approach:")
        print("=" * 70)
        
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
        """Save the hybrid model."""
        model_data = {
            'n_clusters': self.n_clusters,
            'cluster_model': self.cluster_model,
            'cluster_params': self.cluster_params,
            'regime_classifier': self.regime_classifier,
            'ml_models': self.ml_models,
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler,
            'cluster_labels': self.cluster_labels,
            'cluster_info': self.cluster_info,
            'edge_cases': self.edge_cases,
            'optimal_config': getattr(self, 'optimal_config', None),
            'all_cluster_configs': getattr(self, 'all_cluster_configs', None)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Hybrid enhanced-clustering model ({self.n_clusters} clusters) saved to {filepath}")
    
    def load(self, filepath):
        """Load the hybrid model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_clusters = model_data['n_clusters']
        self.cluster_model = model_data['cluster_model']
        self.cluster_params = model_data['cluster_params']
        self.regime_classifier = model_data['regime_classifier']
        self.ml_models = model_data['ml_models']
        self.scaler = model_data['scaler']
        self.feature_scaler = model_data['feature_scaler']
        self.cluster_labels = model_data['cluster_labels']
        self.cluster_info = model_data['cluster_info']
        self.edge_cases = model_data.get('edge_cases', [])
        self.optimal_config = model_data.get('optimal_config', None)
        self.all_cluster_configs = model_data.get('all_cluster_configs', None)
        
        return self

def main():
    """Train and test the hybrid enhanced-clustering engine."""
    print("Hybrid Enhanced-Clustering Engine")
    print("=" * 40)
    
    # Load full dataset (1000 cases)
    try:
        with open('../public_cases.json', 'r') as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} training cases from public dataset")
    except FileNotFoundError:
        print("Error: public_cases.json not found!")
        return
    
    # Create and train hybrid engine
    engine = HybridEnhancedClusteringEngine()
    engine.train(cases)
    
    # Save the model
    engine.save('hybrid_enhanced_clustering_model.pkl')
    
    print("\nHybrid enhanced-clustering engine training complete!")

if __name__ == "__main__":
    main()