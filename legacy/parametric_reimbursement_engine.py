#!/usr/bin/env python3
"""
Parametric reimbursement engine where all rules are learnable parameters.
Uses optimization to find optimal thresholds, rates, and coefficients.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
import json

class ParametricReimbursementEngine:
    def __init__(self):
        self.params = None
        self.ml_model = None
        self.param_bounds = None
        self.best_loss = float('inf')
        self.iteration_count = 0
        
    def init_parameters(self):
        """Initialize parameters with reasonable defaults based on interview insights."""
        # Per diem parameters
        self.params = {
            # Base per diem rates
            'base_per_diem': 100.0,
            'day_5_bonus': 1.15,      # 5-day trip bonus multiplier
            'sweet_spot_bonus': 1.08,  # 4-6 day bonus
            'long_trip_penalty': 0.85, # 8+ day penalty
            'short_trip_penalty': 0.95, # 1-2 day penalty
            
            # Trip length thresholds
            'short_trip_threshold': 2.0,
            'sweet_spot_min': 4.0,
            'sweet_spot_max': 6.0,
            'long_trip_threshold': 8.0,
            
            # Mileage tier thresholds and rates
            'mile_tier1_limit': 100.0,
            'mile_tier2_limit': 300.0,
            'mile_tier3_limit': 600.0,
            'mile_rate1': 0.58,
            'mile_rate2': 0.52,
            'mile_rate3': 0.45,
            'mile_rate4': 0.38,
            
            # Efficiency parameters
            'efficiency_optimal_min': 180.0,
            'efficiency_optimal_max': 220.0,
            'efficiency_good_min': 150.0,
            'efficiency_good_max': 250.0,
            'efficiency_optimal_bonus': 1.15,
            'efficiency_good_bonus': 1.08,
            'efficiency_poor_threshold': 50.0,
            'efficiency_extreme_threshold': 400.0,
            'efficiency_poor_penalty': 0.85,
            'efficiency_extreme_penalty': 0.75,
            
            # Receipt spending thresholds and rates
            'receipt_very_low_threshold': 20.0,
            'receipt_low_threshold': 50.0,
            'receipt_medium_threshold': 120.0,
            'receipt_high_threshold': 200.0,
            'receipt_extreme_threshold': 300.0,
            'receipt_very_low_rate': 0.5,
            'receipt_low_rate': 0.80,
            'receipt_normal_rate': 0.88,
            'receipt_optimal_rate': 0.95,
            'receipt_high_rate': 0.65,
            'receipt_extreme_rate': 0.35,
            
            # Receipt ending bonuses
            'ending_49_bonus': 1.025,
            'ending_99_bonus': 1.015,
            
            # Combination effects
            'sweet_spot_combo_bonus': 1.12,  # 5 days + optimal efficiency + low spending
            'vacation_penalty': 0.80,        # Long trip + high spending
            'lazy_trip_penalty': 0.82,       # Long trip + low efficiency
            'expensive_short_penalty': 0.65, # Short trip + extreme spending
            'high_effort_bonus': 1.05,       # Long trip + high efficiency
            
            # ML blending weight
            'ml_weight': 0.85,
        }
        
        # Define bounds for parameter optimization
        self.param_bounds = {
            'base_per_diem': (80.0, 150.0),
            'day_5_bonus': (1.0, 1.3),
            'sweet_spot_bonus': (1.0, 1.2),
            'long_trip_penalty': (0.7, 1.0),
            'short_trip_penalty': (0.8, 1.1),
            
            'short_trip_threshold': (1.5, 3.0),
            'sweet_spot_min': (3.0, 5.0),
            'sweet_spot_max': (5.0, 7.0),
            'long_trip_threshold': (7.0, 10.0),
            
            'mile_tier1_limit': (50.0, 150.0),
            'mile_tier2_limit': (200.0, 400.0),
            'mile_tier3_limit': (500.0, 800.0),
            'mile_rate1': (0.4, 0.7),
            'mile_rate2': (0.35, 0.6),
            'mile_rate3': (0.3, 0.55),
            'mile_rate4': (0.25, 0.5),
            
            'efficiency_optimal_min': (150.0, 200.0),
            'efficiency_optimal_max': (200.0, 250.0),
            'efficiency_good_min': (120.0, 180.0),
            'efficiency_good_max': (220.0, 300.0),
            'efficiency_optimal_bonus': (1.05, 1.25),
            'efficiency_good_bonus': (1.0, 1.15),
            'efficiency_poor_threshold': (30.0, 80.0),
            'efficiency_extreme_threshold': (300.0, 500.0),
            'efficiency_poor_penalty': (0.7, 0.95),
            'efficiency_extreme_penalty': (0.6, 0.9),
            
            'receipt_very_low_threshold': (10.0, 40.0),
            'receipt_low_threshold': (30.0, 80.0),
            'receipt_medium_threshold': (80.0, 150.0),
            'receipt_high_threshold': (150.0, 250.0),
            'receipt_extreme_threshold': (250.0, 400.0),
            'receipt_very_low_rate': (0.3, 0.8),
            'receipt_low_rate': (0.6, 0.9),
            'receipt_normal_rate': (0.75, 0.95),
            'receipt_optimal_rate': (0.85, 1.0),
            'receipt_high_rate': (0.5, 0.8),
            'receipt_extreme_rate': (0.2, 0.5),
            
            'ending_49_bonus': (1.0, 1.05),
            'ending_99_bonus': (1.0, 1.03),
            
            'sweet_spot_combo_bonus': (1.0, 1.2),
            'vacation_penalty': (0.6, 0.9),
            'lazy_trip_penalty': (0.7, 0.95),
            'expensive_short_penalty': (0.5, 0.8),
            'high_effort_bonus': (1.0, 1.15),
            
            'ml_weight': (0.6, 0.95),
        }
    
    def parametric_rule_based(self, days, miles, receipts):
        """Calculate reimbursement using learned parameters."""
        p = self.params
        
        # Per diem calculation with learned thresholds
        base_rate = p['base_per_diem']
        
        if days == 5:
            per_diem_rate = base_rate * p['day_5_bonus']
        elif p['sweet_spot_min'] <= days <= p['sweet_spot_max']:
            per_diem_rate = base_rate * p['sweet_spot_bonus']
        elif days >= p['long_trip_threshold']:
            per_diem_rate = base_rate * p['long_trip_penalty']
        elif days <= p['short_trip_threshold']:
            per_diem_rate = base_rate * p['short_trip_penalty']
        else:
            per_diem_rate = base_rate
            
        per_diem = days * per_diem_rate
        
        # Mileage calculation with learned tiers
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
        
        # Efficiency adjustments with learned parameters
        if p['efficiency_optimal_min'] <= miles_per_day <= p['efficiency_optimal_max']:
            mileage_reimb *= p['efficiency_optimal_bonus']
        elif p['efficiency_good_min'] <= miles_per_day <= p['efficiency_good_max']:
            mileage_reimb *= p['efficiency_good_bonus']
        elif miles_per_day < p['efficiency_poor_threshold']:
            mileage_reimb *= p['efficiency_poor_penalty']
        elif miles_per_day > p['efficiency_extreme_threshold']:
            mileage_reimb *= p['efficiency_extreme_penalty']
        
        # Receipt processing with learned thresholds
        receipts_per_day = receipts / max(days, 1)
        
        if receipts_per_day > p['receipt_extreme_threshold']:
            receipt_reimb = receipts * p['receipt_extreme_rate']
        elif receipts_per_day > p['receipt_high_threshold']:
            receipt_reimb = receipts * p['receipt_high_rate']
        elif receipts_per_day < p['receipt_very_low_threshold']:
            receipt_reimb = receipts * p['receipt_very_low_rate']
        elif receipts_per_day < p['receipt_low_threshold']:
            receipt_reimb = receipts * p['receipt_low_rate']
        elif p['receipt_low_threshold'] <= receipts_per_day <= p['receipt_medium_threshold']:
            receipt_reimb = receipts * p['receipt_optimal_rate']
        else:
            receipt_reimb = receipts * p['receipt_normal_rate']
        
        # Receipt ending bonuses
        receipt_str = f"{receipts:.2f}"
        if receipt_str.endswith('.49'):
            receipt_reimb *= p['ending_49_bonus']
        elif receipt_str.endswith('.99'):
            receipt_reimb *= p['ending_99_bonus']
        
        total = per_diem + mileage_reimb + receipt_reimb
        
        # Combination effects with learned parameters
        is_optimal_efficiency = p['efficiency_optimal_min'] <= miles_per_day <= p['efficiency_optimal_max']
        is_low_spending = receipts_per_day < p['receipt_medium_threshold']
        
        if days == 5 and is_optimal_efficiency and is_low_spending:
            total *= p['sweet_spot_combo_bonus']
        elif days >= p['long_trip_threshold'] and receipts_per_day > p['receipt_high_threshold']:
            total *= p['vacation_penalty']
        elif days >= p['long_trip_threshold'] and miles_per_day < p['efficiency_poor_threshold']:
            total *= p['lazy_trip_penalty']
        elif days <= p['short_trip_threshold'] and receipts_per_day > p['receipt_extreme_threshold']:
            total *= p['expensive_short_penalty']
        elif days >= 5 and miles_per_day > 200:
            total *= p['high_effort_bonus']
        
        return total
    
    def extract_ml_features(self, days, miles, receipts):
        """Extract features for ML component."""
        miles_per_day = miles / max(days, 1)
        receipts_per_day = receipts / max(days, 1)
        
        features = [
            days, miles, receipts,
            miles_per_day, receipts_per_day,
            days * miles, days * receipts, miles * receipts,
            days ** 2, miles ** 2, receipts ** 2,
            np.sqrt(miles), np.sqrt(receipts),
            np.log1p(days), np.log1p(miles), np.log1p(receipts),
            float(f"{receipts:.2f}".endswith('.49')),
            float(f"{receipts:.2f}".endswith('.99')),
            float(days == 5),
            float(days >= 8),
            float(days <= 2),
            float(150 <= miles_per_day <= 250),
            float(miles_per_day > 300),
            float(receipts_per_day > 200),
            float(receipts_per_day < 50)
        ]
        
        return np.array(features)
    
    def objective_function(self, param_values, X_train, y_train, X_val, y_val):
        """Objective function for parameter optimization."""
        # Update parameters
        param_names = list(self.params.keys())
        for i, value in enumerate(param_values):
            self.params[param_names[i]] = value
        
        # Calculate predictions
        predictions = []
        for i in range(len(X_train)):
            days = X_train[i][0]
            miles = X_train[i][1] 
            receipts = X_train[i][2]
            
            rule_pred = self.parametric_rule_based(days, miles, receipts)
            if self.ml_model is not None:
                ml_pred = self.ml_model.predict([X_train[i]])[0]
                pred = self.params['ml_weight'] * ml_pred + (1 - self.params['ml_weight']) * rule_pred
            else:
                pred = rule_pred
            predictions.append(pred)
        
        # Calculate validation loss
        val_predictions = []
        for i in range(len(X_val)):
            days = X_val[i][0]
            miles = X_val[i][1]
            receipts = X_val[i][2]
            
            rule_pred = self.parametric_rule_based(days, miles, receipts)
            if self.ml_model is not None:
                ml_pred = self.ml_model.predict([X_val[i]])[0]
                pred = self.params['ml_weight'] * ml_pred + (1 - self.params['ml_weight']) * rule_pred
            else:
                pred = rule_pred
            val_predictions.append(pred)
        
        loss = mean_absolute_error(y_val, val_predictions)
        
        # Track progress
        self.iteration_count += 1
        if loss < self.best_loss:
            self.best_loss = loss
        
        if self.iteration_count % 50 == 0:
            print(f"  Iteration {self.iteration_count}, Best Loss: ${self.best_loss:.2f}, Current: ${loss:.2f}")
        
        return loss
    
    def train(self, cases):
        """Train the parametric model with iterative optimization."""
        print("Training parametric reimbursement model...")
        
        # Prepare data
        X, y = [], []
        for case in cases:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            features = self.extract_ml_features(days, miles, receipts)
            X.append(features)
            y.append(case['expected_output'])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Initialize parameters
        self.init_parameters()
        
        # Phase 1: Train ML component
        print("Phase 1: Training ML component...")
        self.ml_model = GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42)
        self.ml_model.fit(X_train, y_train)
        ml_val_pred = self.ml_model.predict(X_val)
        ml_mae = mean_absolute_error(y_val, ml_val_pred)
        print(f"  ML component MAE: ${ml_mae:.2f}")
        
        # Phase 2: Optimize parameters iteratively
        print("Phase 2: Optimizing rule parameters...")
        
        # Convert parameters to optimization format
        param_names = list(self.params.keys())
        initial_values = [self.params[name] for name in param_names]
        bounds = [self.param_bounds[name] for name in param_names]
        
        self.iteration_count = 0
        self.best_loss = float('inf')
        
        # Use multiple optimization attempts
        best_result = None
        best_params = None
        
        for attempt in range(3):
            print(f"  Optimization attempt {attempt + 1}/3...")
            
            # Add some noise to initial values for different starting points
            if attempt > 0:
                initial_values = [self.params[name] + np.random.normal(0, 0.1) * self.params[name] 
                                for name in param_names]
                # Clip to bounds
                for i, (low, high) in enumerate(bounds):
                    initial_values[i] = np.clip(initial_values[i], low, high)
            
            result = optimize.minimize(
                self.objective_function,
                initial_values,
                args=(X_train, y_train, X_val, y_val),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_params = result.x.copy()
                
        # Set best parameters
        for i, name in enumerate(param_names):
            self.params[name] = best_params[i]
        
        print(f"  Final parameter optimization MAE: ${best_result.fun:.2f}")
        
        # Phase 3: Final evaluation
        print("Phase 3: Final evaluation...")
        test_predictions = []
        for i in range(len(X_test)):
            days = X_test[i][0]
            miles = X_test[i][1]
            receipts = X_test[i][2]
            pred = self.predict(days, miles, receipts)
            test_predictions.append(pred)
        
        test_mae = mean_absolute_error(y_test, test_predictions)
        print(f"  Final test MAE: ${test_mae:.2f}")
        
        # Save test data for later analysis
        self.test_X = X_test
        self.test_y = y_test
        
        return self
    
    def predict(self, days, miles, receipts):
        """Predict using optimized parameters and ML ensemble."""
        if self.params is None:
            raise ValueError("Model not trained yet")
        
        # Rule-based prediction with learned parameters
        rule_pred = self.parametric_rule_based(days, miles, receipts)
        
        # ML prediction
        if self.ml_model is not None:
            features = self.extract_ml_features(days, miles, receipts).reshape(1, -1)
            ml_pred = self.ml_model.predict(features)[0]
            
            # Blend with learned weight
            final_pred = (self.params['ml_weight'] * ml_pred + 
                         (1 - self.params['ml_weight']) * rule_pred)
        else:
            final_pred = rule_pred
        
        return final_pred
    
    def print_learned_parameters(self):
        """Print the learned parameters."""
        if self.params is None:
            print("No parameters learned yet")
            return
            
        print("\nLearned Parameters:")
        print("=" * 50)
        
        categories = {
            'Per Diem': ['base_per_diem', 'day_5_bonus', 'sweet_spot_bonus', 'long_trip_penalty', 'short_trip_penalty'],
            'Trip Thresholds': ['short_trip_threshold', 'sweet_spot_min', 'sweet_spot_max', 'long_trip_threshold'],
            'Mileage Tiers': ['mile_tier1_limit', 'mile_tier2_limit', 'mile_tier3_limit', 'mile_rate1', 'mile_rate2', 'mile_rate3', 'mile_rate4'],
            'Efficiency': ['efficiency_optimal_min', 'efficiency_optimal_max', 'efficiency_optimal_bonus', 'efficiency_good_min', 'efficiency_good_max', 'efficiency_good_bonus'],
            'Receipts': ['receipt_low_threshold', 'receipt_medium_threshold', 'receipt_high_threshold', 'receipt_normal_rate', 'receipt_optimal_rate'],
            'Combinations': ['sweet_spot_combo_bonus', 'vacation_penalty', 'ml_weight']
        }
        
        for category, param_list in categories.items():
            print(f"\n{category}:")
            for param in param_list:
                if param in self.params:
                    print(f"  {param}: {self.params[param]:.3f}")
    
    def save(self, filepath):
        """Save the trained model."""
        model_data = {
            'params': self.params,
            'ml_model': self.ml_model,
            'param_bounds': self.param_bounds
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.params = model_data['params']
        self.ml_model = model_data['ml_model']
        self.param_bounds = model_data['param_bounds']
        return self

def main():
    """Train and evaluate the parametric model."""
    print("Loading test cases...")
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Train parametric model
    engine = ParametricReimbursementEngine()
    engine.train(cases)
    
    # Print learned parameters
    engine.print_learned_parameters()
    
    # Save model
    engine.save('parametric_model.pkl')
    print("\nParametric model saved!")
    
    # Test examples
    print("\nExample predictions:")
    examples = [
        (5, 200, 500),      # Sweet spot combo
        (3, 93, 1.42),      # Low receipts
        (8, 600, 1200),     # Long expensive trip
        (1, 1082, 1809.49), # Extreme case
    ]
    
    for days, miles, receipts in examples:
        result = engine.predict(days, miles, receipts)
        print(f"  {days}d, {miles}mi, ${receipts:.2f} -> ${result:.2f}")

if __name__ == "__main__":
    main()
