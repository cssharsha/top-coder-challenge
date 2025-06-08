#!/usr/bin/env python3
"""
Advanced reimbursement engine incorporating all interview insights:
- Kevin's efficiency sweet spots and statistical patterns
- Lisa's 5-day bonus and receipt ending quirks
- Marcus's monthly variations and efficiency rewards
- Jennifer's trip length preferences
- Complex interaction effects and data splits
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class AdvancedReimbursementEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.bias_correction = 0
        self.weights = {}
        
    def extract_comprehensive_features(self, days, miles, receipts):
        """Extract comprehensive features based on all interview insights."""
        
        # Basic features
        miles_per_day = miles / max(days, 1)
        receipts_per_day = receipts / max(days, 1)
        
        # Kevin's efficiency categories (180-220 sweet spot)
        efficiency_optimal = 1.0 if 180 <= miles_per_day <= 220 else 0.0
        efficiency_good = 1.0 if 150 <= miles_per_day <= 250 else 0.0
        efficiency_poor = 1.0 if miles_per_day < 50 or miles_per_day > 400 else 0.0
        
        # Lisa's trip length patterns (5-day bonus, 4-6 day sweet spot)
        is_5_day = 1.0 if days == 5 else 0.0
        is_sweet_spot = 1.0 if 4 <= days <= 6 else 0.0
        is_long_trip = 1.0 if days >= 8 else 0.0
        is_short_trip = 1.0 if days <= 2 else 0.0
        
        # Receipt ending patterns (.49/.99 bonuses)
        receipt_str = f"{receipts:.2f}"
        ends_49 = 1.0 if receipt_str.endswith('.49') else 0.0
        ends_99 = 1.0 if receipt_str.endswith('.99') else 0.0
        ends_lucky = ends_49 + ends_99
        
        # Kevin's spending thresholds
        spending_low = 1.0 if receipts_per_day < 75 else 0.0
        spending_medium = 1.0 if 75 <= receipts_per_day <= 120 else 0.0
        spending_high = 1.0 if 120 < receipts_per_day <= 200 else 0.0
        spending_extreme = 1.0 if receipts_per_day > 300 else 0.0
        
        # Mileage tiers (Lisa's observations)
        miles_tier_1 = min(miles, 100)  # First 100 miles at full rate
        miles_tier_2 = max(0, min(miles - 100, 200))  # Next 200 miles
        miles_tier_3 = max(0, min(miles - 300, 300))  # Next 300 miles
        miles_tier_4 = max(0, miles - 600)  # Beyond 600 miles
        
        # Marcus's efficiency rewards
        high_effort = 1.0 if (days >= 5 and miles_per_day > 200) else 0.0
        balanced_trip = 1.0 if (days >= 3 and 100 <= miles_per_day <= 300 and receipts_per_day <= 150) else 0.0
        
        # Problem combinations (penalties)
        vacation_penalty = 1.0 if (days >= 8 and receipts_per_day > 200) else 0.0
        lazy_trip = 1.0 if (days >= 5 and miles_per_day < 60) else 0.0
        expensive_short = 1.0 if (days <= 2 and receipts_per_day > 300) else 0.0
        
        # Advanced interaction features
        efficiency_spending_ratio = miles_per_day / max(receipts_per_day, 1)
        total_value = days * miles * receipts / 10000  # Normalized total value
        
        # Polynomial features for key variables
        days_sq = days ** 2
        miles_sq = miles ** 2
        receipts_sq = receipts ** 2
        
        # Logarithmic features for diminishing returns
        log_days = np.log1p(days)
        log_miles = np.log1p(miles)
        log_receipts = np.log1p(receipts)
        
        # Complex interactions
        days_miles_interaction = days * miles / 1000
        days_receipts_interaction = days * receipts / 1000
        miles_receipts_interaction = miles * receipts / 10000
        
        # Trip efficiency score (Kevin's k-means inspiration)
        efficiency_score = (miles_per_day * 0.5 + 
                          (200 - abs(miles_per_day - 200)) * 0.3 +
                          (100 - abs(receipts_per_day - 100)) * 0.2)
        
        features = [
            # Basic features (0-4)
            days, miles, receipts, miles_per_day, receipts_per_day,
            
            # Trip categorization (5-9)
            is_5_day, is_sweet_spot, is_long_trip, is_short_trip, efficiency_optimal,
            
            # Receipt patterns (10-12)
            ends_49, ends_99, ends_lucky,
            
            # Spending categories (13-16)
            spending_low, spending_medium, spending_high, spending_extreme,
            
            # Mileage tiers (17-20)
            miles_tier_1, miles_tier_2, miles_tier_3, miles_tier_4,
            
            # Behavioral patterns (21-24)
            high_effort, balanced_trip, vacation_penalty, lazy_trip,
            
            # Problem combinations (25-27)
            expensive_short, efficiency_good, efficiency_poor,
            
            # Advanced features (28-35)
            efficiency_spending_ratio, total_value, days_sq, miles_sq, 
            receipts_sq, log_days, log_miles, log_receipts,
            
            # Interactions (36-39)
            days_miles_interaction, days_receipts_interaction, 
            miles_receipts_interaction, efficiency_score,
            
            # Additional Kevin-inspired features (40-44)
            np.sqrt(miles), np.sqrt(receipts), 
            miles * efficiency_optimal, receipts * is_sweet_spot,
            days * efficiency_optimal
        ]
        
        return np.array(features)
    
    def advanced_rule_based(self, days, miles, receipts):
        """Enhanced rule-based calculation incorporating all insights."""
        
        # Base per diem with Lisa's 5-day bonus
        base_per_diem = 100
        if days == 5:
            per_diem_rate = 115  # Stronger 5-day bonus
        elif 4 <= days <= 6:
            per_diem_rate = 108  # Sweet spot bonus
        elif days >= 10:
            per_diem_rate = 85   # Long trip penalty
        elif days <= 2:
            per_diem_rate = 95   # Short trip penalty
        else:
            per_diem_rate = base_per_diem
            
        per_diem = days * per_diem_rate
        
        # Kevin's mileage tiers with efficiency bonuses
        miles_per_day = miles / max(days, 1)
        
        # Tiered mileage calculation
        if miles <= 100:
            mileage_reimb = miles * 0.58
        elif miles <= 300:
            mileage_reimb = 100 * 0.58 + (miles - 100) * 0.52
        elif miles <= 600:
            mileage_reimb = 100 * 0.58 + 200 * 0.52 + (miles - 300) * 0.45
        else:
            mileage_reimb = 100 * 0.58 + 200 * 0.52 + 300 * 0.45 + (miles - 600) * 0.38
        
        # Kevin's efficiency bonuses
        if 180 <= miles_per_day <= 220:
            mileage_reimb *= 1.15  # Optimal efficiency bonus
        elif 150 <= miles_per_day <= 250:
            mileage_reimb *= 1.08  # Good efficiency bonus
        elif miles_per_day > 400:
            mileage_reimb *= 0.75  # Unrealistic efficiency penalty
        elif miles_per_day < 30:
            mileage_reimb *= 0.85  # Low efficiency penalty
            
        # Enhanced receipt processing with Kevin's thresholds
        receipts_per_day = receipts / max(days, 1)
        
        if receipts_per_day > 300:
            receipt_reimb = receipts * 0.35  # Extreme spending penalty
        elif receipts_per_day > 150:
            receipt_reimb = receipts * 0.65  # High spending penalty
        elif receipts_per_day < 20:
            receipt_reimb = max(0, receipts * 0.5)  # Very low penalty
        elif receipts_per_day < 50:
            receipt_reimb = receipts * 0.80  # Low spending penalty
        elif 75 <= receipts_per_day <= 120:
            receipt_reimb = receipts * 0.95  # Kevin's optimal range
        else:
            receipt_reimb = receipts * 0.88  # Normal range
            
        # Lisa's receipt ending bonuses
        receipt_str = f"{receipts:.2f}"
        if receipt_str.endswith('.49'):
            receipt_reimb *= 1.025  # Slightly stronger .49 bonus
        elif receipt_str.endswith('.99'):
            receipt_reimb *= 1.015  # .99 bonus
            
        total = per_diem + mileage_reimb + receipt_reimb
        
        # Kevin's combination bonuses/penalties
        if days == 5 and 180 <= miles_per_day <= 220 and receipts_per_day < 100:
            total *= 1.12  # "Sweet spot combo"
            
        if days >= 8 and receipts_per_day > 200:
            total *= 0.80  # "Vacation penalty"
            
        if days >= 8 and miles_per_day < 60:
            total *= 0.82  # Long trip with low efficiency
            
        if days <= 2 and receipts_per_day > 300:
            total *= 0.65  # Expensive short trip
            
        # Marcus's high effort bonus
        if days >= 5 and miles_per_day > 200:
            total *= 1.05  # High effort bonus
            
        return total
    
    def create_train_val_test_split(self, cases, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """Create 60/20/20 split as requested."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # First split: separate test set
        train_val, test = train_test_split(cases, test_size=test_ratio, random_state=42)
        
        # Second split: separate train and validation
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(train_val, test_size=val_size_adjusted, random_state=42)
        
        print(f"Data split: Train={len(train)} ({len(train)/len(cases)*100:.1f}%), "
              f"Val={len(val)} ({len(val)/len(cases)*100:.1f}%), "
              f"Test={len(test)} ({len(test)/len(cases)*100:.1f}%)")
        
        return train, val, test
    
    def train(self, cases):
        """Train ensemble with proper data splits."""
        print("Training advanced reimbursement model with interview insights...")
        
        # Create data splits
        train_cases, val_cases, test_cases = self.create_train_val_test_split(cases)
        
        # Extract features and targets for all splits
        def process_cases(case_list):
            features, targets = [], []
            for case in case_list:
                days = case['input']['trip_duration_days']
                miles = case['input']['miles_traveled']
                receipts = case['input']['total_receipts_amount']
                output = case['expected_output']
                
                feature_vector = self.extract_comprehensive_features(days, miles, receipts)
                features.append(feature_vector)
                targets.append(output)
            return np.array(features), np.array(targets)
        
        X_train, y_train = process_cases(train_cases)
        X_val, y_val = process_cases(val_cases)
        X_test, y_test = process_cases(test_cases)
        
        # Feature scaling
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.scalers['standard'].transform(X_val)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train multiple models with different approaches
        models_config = {
            'rf_deep': RandomForestRegressor(n_estimators=500, max_depth=25, min_samples_split=2, random_state=42),
            'rf_wide': RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, random_state=43),
            'gb_strong': GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42),
            'gb_gentle': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=43),
            'extra_trees': ExtraTreesRegressor(n_estimators=400, max_depth=20, random_state=42),
            'ridge': Ridge(alpha=10.0),
            'elastic': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'mlp': MLPRegressor(hidden_layer_sizes=(200, 100, 50), max_iter=1000, random_state=42)
        }
        
        # Train and validate each model
        val_scores = {}
        for name, model in models_config.items():
            print(f"Training {name}...")
            if name in ['ridge', 'elastic', 'mlp']:
                model.fit(X_train_scaled, y_train)
                val_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
            
            val_mae = mean_absolute_error(y_val, val_pred)
            val_scores[name] = val_mae
            self.models[name] = model
            print(f"  {name} validation MAE: ${val_mae:.2f}")
        
        # Calculate ensemble weights based on inverse validation error
        total_inv_error = sum(1/score for score in val_scores.values())
        self.weights = {name: (1/score)/total_inv_error for name, score in val_scores.items()}
        
        print(f"\nModel weights: {dict(sorted(self.weights.items(), key=lambda x: x[1], reverse=True))}")
        
        # Calculate bias correction on validation set
        ensemble_val_pred = self._ensemble_predict(X_val, X_val_scaled)
        self.bias_correction = np.mean(y_val - ensemble_val_pred)
        
        # Final test set evaluation
        ensemble_test_pred = self._ensemble_predict(X_test, X_test_scaled) + self.bias_correction
        test_mae = mean_absolute_error(y_test, ensemble_test_pred)
        
        print(f"\nFinal Results:")
        print(f"  Validation MAE: ${mean_absolute_error(y_val, ensemble_val_pred + self.bias_correction):.2f}")
        print(f"  Test MAE: ${test_mae:.2f}")
        print(f"  Bias correction: ${self.bias_correction:.2f}")
        
        # Store test cases for later analysis
        self.test_cases = test_cases
        self.test_features = X_test
        self.test_features_scaled = X_test_scaled
        self.test_targets = y_test
        
        return self
    
    def _ensemble_predict(self, X_raw, X_scaled):
        """Generate ensemble predictions."""
        predictions = []
        
        for name, model in self.models.items():
            if name in ['ridge', 'elastic', 'mlp']:
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X_raw)
            predictions.append(pred * self.weights[name])
        
        return np.sum(predictions, axis=0)
    
    def predict(self, days, miles, receipts):
        """Predict reimbursement with advanced ensemble."""
        if not self.models:
            return self.advanced_rule_based(days, miles, receipts)
        
        # Extract features
        features = self.extract_comprehensive_features(days, miles, receipts).reshape(1, -1)
        features_scaled = self.scalers['standard'].transform(features)
        
        # Get ensemble prediction
        ml_prediction = self._ensemble_predict(features, features_scaled)[0] + self.bias_correction
        
        # Get rule-based prediction for blending
        rule_prediction = self.advanced_rule_based(days, miles, receipts)
        
        # Adaptive blending based on confidence
        # Use more ML for cases in training distribution, more rules for edge cases
        efficiency = miles / max(days, 1)
        receipts_per_day = receipts / max(days, 1)
        
        # Higher ML weight for typical cases
        if (2 <= days <= 8 and 50 <= efficiency <= 300 and receipts_per_day <= 250):
            ml_weight = 0.92
        else:
            ml_weight = 0.85  # More conservative for edge cases
            
        final_prediction = ml_weight * ml_prediction + (1 - ml_weight) * rule_prediction
        
        return final_prediction
    
    def analyze_test_performance(self):
        """Analyze performance on test set by different categories."""
        if not hasattr(self, 'test_cases'):
            print("No test data available for analysis")
            return
            
        predictions = []
        for case in self.test_cases:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            pred = self.predict(days, miles, receipts)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        errors = np.abs(predictions - self.test_targets)
        
        print(f"\nTest Set Analysis:")
        print(f"  MAE: ${np.mean(errors):.2f}")
        print(f"  Within $1: {np.mean(errors <= 1)*100:.1f}%")
        print(f"  Within $5: {np.mean(errors <= 5)*100:.1f}%")
        print(f"  Within $10: {np.mean(errors <= 10)*100:.1f}%")
        print(f"  Within $25: {np.mean(errors <= 25)*100:.1f}%")
        
        # Category analysis
        categories = {
            '5-day trips': [i for i, case in enumerate(self.test_cases) if case['input']['trip_duration_days'] == 5],
            'High efficiency': [i for i, case in enumerate(self.test_cases) 
                              if case['input']['miles_traveled']/case['input']['trip_duration_days'] >= 180],
            'Receipt .49/.99': [i for i, case in enumerate(self.test_cases) 
                              if f"{case['input']['total_receipts_amount']:.2f}".endswith(('.49', '.99'))]
        }
        
        for cat_name, indices in categories.items():
            if indices:
                cat_errors = errors[indices]
                print(f"  {cat_name}: {len(indices)} cases, MAE=${np.mean(cat_errors):.2f}")
    
    def save(self, filepath):
        """Save the trained ensemble."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'weights': self.weights,
            'bias_correction': self.bias_correction
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load a trained ensemble."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.weights = model_data['weights']
        self.bias_correction = model_data['bias_correction']
        return self

def main():
    """Train and evaluate the advanced model."""
    print("Loading test cases...")
    with open('../public_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Train advanced model
    engine = AdvancedReimbursementEngine()
    engine.train(cases)
    
    # Analyze performance
    engine.analyze_test_performance()
    
    # Save model
    engine.save('advanced_model.pkl')
    print("\nAdvanced model saved as advanced_model.pkl!")
    
    # Test some examples
    print("\nExample predictions:")
    examples = [
        (5, 200, 500),      # Kevin's sweet spot
        (3, 93, 1.42),      # Low receipts
        (1, 1082, 1809.49), # Extreme case
        (8, 400, 1200),     # Long expensive trip
        (4, 180, 99.99)     # .99 ending
    ]
    
    for days, miles, receipts in examples:
        result = engine.predict(days, miles, receipts)
        print(f"  {days}d, {miles}mi, ${receipts:.2f} -> ${result:.2f}")

if __name__ == "__main__":
    main()