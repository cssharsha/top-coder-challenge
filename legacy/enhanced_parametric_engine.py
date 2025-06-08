#!/usr/bin/env python3
"""
Enhanced parametric reimbursement engine that specifically handles high error cases.
Adds trip-receipt correlation features and extreme case penalties.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
import json

class EnhancedParametricEngine:
    def __init__(self):
        self.params = None
        self.ml_model = None
        self.param_bounds = None
        self.best_loss = float('inf')
        self.iteration_count = 0
        
    def init_parameters(self):
        """Initialize parameters with focus on correlation patterns."""
        self.params = {
            # Base per diem rates
            'base_per_diem': 100.0,
            'day_5_bonus': 1.15,
            'sweet_spot_bonus': 1.08,
            'long_trip_penalty': 0.85,
            'short_trip_penalty': 0.95,
            
            # Trip length thresholds
            'short_trip_threshold': 2.0,
            'sweet_spot_min': 4.0,
            'sweet_spot_max': 6.0,
            'long_trip_threshold': 8.0,
            
            # Mileage parameters
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
            'efficiency_optimal_bonus': 1.15,
            'efficiency_poor_threshold': 50.0,
            'efficiency_extreme_threshold': 400.0,
            'efficiency_poor_penalty': 0.85,
            'efficiency_extreme_penalty': 0.75,
            
            # Receipt base rates
            'receipt_normal_rate': 0.88,
            'receipt_optimal_rate': 0.95,
            'receipt_high_rate': 0.65,
            'receipt_extreme_rate': 0.35,
            
            # NEW: Trip-Receipt Correlation Parameters
            'expected_receipt_per_day': 75.0,  # Expected baseline
            'receipt_tolerance_factor': 0.5,    # How much deviation is OK
            'correlation_penalty_strength': 0.3, # How harsh the penalty
            
            # NEW: Extreme Case Detection
            'extreme_spending_multiplier': 3.0,  # 3x expected = extreme
            'extreme_penalty_cap': 0.4,         # Maximum penalty factor
            'minimal_spending_threshold': 10.0,  # Very low spending
            'minimal_spending_penalty': 0.6,    # Penalty for minimal spending
            
            # NEW: Trip Pattern Penalties
            'short_expensive_threshold': 200.0,  # $/day for short trips
            'short_expensive_penalty': 0.5,     # Heavy penalty
            'long_minimal_threshold': 20.0,     # $/day for long trips  
            'long_minimal_penalty': 0.7,        # Penalty for suspiciously low
            
            # NEW: Mile-Receipt Interaction
            'high_mile_high_receipt_bonus': 1.1, # Legitimate business travel
            'high_mile_low_receipt_penalty': 0.8, # Suspicious pattern
            'low_mile_high_receipt_penalty': 0.6, # Possible abuse
            
            # Receipt ending bonuses
            'ending_49_bonus': 1.025,
            'ending_99_bonus': 1.015,
            
            # Combination effects
            'vacation_penalty': 0.80,
            'ml_weight': 0.85,
        }
        
        # Enhanced bounds
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
            'efficiency_optimal_bonus': (1.05, 1.25),
            'efficiency_poor_threshold': (30.0, 80.0),
            'efficiency_extreme_threshold': (300.0, 500.0),
            'efficiency_poor_penalty': (0.7, 0.95),
            'efficiency_extreme_penalty': (0.6, 0.9),
            
            'receipt_normal_rate': (0.75, 0.95),
            'receipt_optimal_rate': (0.85, 1.0),
            'receipt_high_rate': (0.5, 0.8),
            'receipt_extreme_rate': (0.2, 0.5),
            
            # NEW bounds
            'expected_receipt_per_day': (50.0, 120.0),
            'receipt_tolerance_factor': (0.3, 0.8),
            'correlation_penalty_strength': (0.1, 0.5),
            'extreme_spending_multiplier': (2.0, 5.0),
            'extreme_penalty_cap': (0.3, 0.6),
            'minimal_spending_threshold': (5.0, 30.0),
            'minimal_spending_penalty': (0.4, 0.8),
            'short_expensive_threshold': (150.0, 300.0),
            'short_expensive_penalty': (0.3, 0.7),
            'long_minimal_threshold': (10.0, 40.0),
            'long_minimal_penalty': (0.5, 0.9),
            'high_mile_high_receipt_bonus': (1.0, 1.2),
            'high_mile_low_receipt_penalty': (0.6, 0.9),
            'low_mile_high_receipt_penalty': (0.4, 0.8),
            
            'ending_49_bonus': (1.0, 1.05),
            'ending_99_bonus': (1.0, 1.03),
            'vacation_penalty': (0.6, 0.9),
            'ml_weight': (0.6, 0.95),
        }
    
    def calculate_trip_receipt_correlation_penalty(self, days, receipts):
        """Calculate penalty based on trip-receipt correlation mismatch."""
        p = self.params
        
        receipts_per_day = receipts / max(days, 1)
        expected_per_day = p['expected_receipt_per_day']
        
        # Calculate deviation from expected
        deviation_ratio = abs(receipts_per_day - expected_per_day) / expected_per_day
        
        # Apply penalty if deviation exceeds tolerance
        if deviation_ratio > p['receipt_tolerance_factor']:
            excess_deviation = deviation_ratio - p['receipt_tolerance_factor']
            penalty_factor = 1.0 - (excess_deviation * p['correlation_penalty_strength'])
            return max(penalty_factor, p['extreme_penalty_cap'])
        
        return 1.0
    
    def detect_extreme_patterns(self, days, miles, receipts):
        """Detect and penalize extreme spending patterns."""
        p = self.params
        
        receipts_per_day = receipts / max(days, 1)
        miles_per_day = miles / max(days, 1)
        penalty_factor = 1.0
        
        # Extreme spending detection
        if receipts_per_day > p['expected_receipt_per_day'] * p['extreme_spending_multiplier']:
            penalty_factor *= p['extreme_penalty_cap']
        
        # Minimal spending on long trips (suspicious)
        if days >= p['long_trip_threshold'] and receipts_per_day < p['long_minimal_threshold']:
            penalty_factor *= p['long_minimal_penalty']
        
        # Very expensive short trips (possible abuse)
        if days <= p['short_trip_threshold'] and receipts_per_day > p['short_expensive_threshold']:
            penalty_factor *= p['short_expensive_penalty']
        
        # Mile-receipt interaction patterns
        high_miles = miles_per_day > 150
        high_receipts = receipts_per_day > p['expected_receipt_per_day'] * 1.5
        low_receipts = receipts_per_day < p['expected_receipt_per_day'] * 0.5
        low_miles = miles_per_day < 50
        
        if high_miles and high_receipts:
            penalty_factor *= p['high_mile_high_receipt_bonus']  # Legitimate business travel
        elif high_miles and low_receipts:
            penalty_factor *= p['high_mile_low_receipt_penalty']  # Suspicious
        elif low_miles and high_receipts:
            penalty_factor *= p['low_mile_high_receipt_penalty']  # Possible abuse
        
        return penalty_factor
    
    def enhanced_parametric_calculation(self, days, miles, receipts):
        """Enhanced calculation with correlation and extreme case handling."""
        p = self.params
        
        # Base per diem calculation
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
        
        # Efficiency adjustments
        if p['efficiency_optimal_min'] <= miles_per_day <= p['efficiency_optimal_max']:
            mileage_reimb *= p['efficiency_optimal_bonus']
        elif miles_per_day < p['efficiency_poor_threshold']:
            mileage_reimb *= p['efficiency_poor_penalty']
        elif miles_per_day > p['efficiency_extreme_threshold']:
            mileage_reimb *= p['efficiency_extreme_penalty']
        
        # Receipt processing - start with base rate
        receipts_per_day = receipts / max(days, 1)
        
        # Determine base receipt rate
        if receipts_per_day < p['minimal_spending_threshold']:
            receipt_reimb = receipts * p['minimal_spending_penalty']
        elif receipts_per_day <= p['expected_receipt_per_day']:
            receipt_reimb = receipts * p['receipt_optimal_rate']
        elif receipts_per_day <= p['expected_receipt_per_day'] * 2:
            receipt_reimb = receipts * p['receipt_normal_rate']
        elif receipts_per_day <= p['expected_receipt_per_day'] * 3:
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
        
        # Apply NEW correlation-based penalties
        correlation_penalty = self.calculate_trip_receipt_correlation_penalty(days, receipts)
        extreme_penalty = self.detect_extreme_patterns(days, miles, receipts)
        
        # Combine penalties (multiplicative)
        combined_penalty = correlation_penalty * extreme_penalty
        
        # Apply final penalty
        total *= combined_penalty
        
        return total
    
    def extract_enhanced_features(self, days, miles, receipts):
        """Extract enhanced features including correlation measures."""
        miles_per_day = miles / max(days, 1)
        receipts_per_day = receipts / max(days, 1)
        
        # NEW: Correlation features
        expected_receipts = days * 75  # Baseline expectation
        receipt_deviation = abs(receipts - expected_receipts) / max(expected_receipts, 1)
        receipt_day_ratio = receipts_per_day / 75.0  # Normalized spending rate
        
        # NEW: Pattern detection features
        is_extreme_spending = float(receipts_per_day > 225)  # 3x normal
        is_minimal_spending = float(receipts_per_day < 15)   # Very low
        is_short_expensive = float(days <= 2 and receipts_per_day > 200)
        is_long_minimal = float(days >= 8 and receipts_per_day < 20)
        
        # NEW: Mile-receipt interaction
        mile_receipt_balance = miles_per_day / max(receipts_per_day, 1)
        high_mile_low_receipt = float(miles_per_day > 150 and receipts_per_day < 50)
        low_mile_high_receipt = float(miles_per_day < 50 and receipts_per_day > 150)
        
        features = [
            # Base features
            days, miles, receipts, miles_per_day, receipts_per_day,
            
            # Original polynomial features
            days * miles, days * receipts, miles * receipts,
            days ** 2, miles ** 2, receipts ** 2,
            np.sqrt(miles), np.sqrt(receipts),
            np.log1p(days), np.log1p(miles), np.log1p(receipts),
            
            # Original pattern features
            float(f"{receipts:.2f}".endswith('.49')),
            float(f"{receipts:.2f}".endswith('.99')),
            float(days == 5),
            float(days >= 8),
            float(days <= 2),
            float(150 <= miles_per_day <= 250),
            float(miles_per_day > 300),
            float(receipts_per_day > 200),
            float(receipts_per_day < 50),
            
            # NEW: Enhanced correlation features
            receipt_deviation,
            receipt_day_ratio,
            is_extreme_spending,
            is_minimal_spending,
            is_short_expensive,
            is_long_minimal,
            mile_receipt_balance,
            high_mile_low_receipt,
            low_mile_high_receipt,
            
            # NEW: Trip category combinations
            float(days >= 7 and miles_per_day > 100 and receipts_per_day > 100),  # Legitimate business
            float(days <= 3 and miles_per_day < 100 and receipts_per_day > 150), # Suspicious pattern
        ]
        
        return np.array(features)
    
    def objective_function(self, param_values, X_train, y_train, X_val, y_val):
        """Enhanced objective function with focus on high error cases."""
        # Update parameters
        param_names = list(self.params.keys())
        for i, value in enumerate(param_values):
            self.params[param_names[i]] = value
        
        # Calculate validation predictions
        val_predictions = []
        for i in range(len(X_val)):
            days = X_val[i][0]
            miles = X_val[i][1]
            receipts = X_val[i][2]
            
            rule_pred = self.enhanced_parametric_calculation(days, miles, receipts)
            if self.ml_model is not None:
                ml_pred = self.ml_model.predict([X_val[i]])[0]
                pred = self.params['ml_weight'] * ml_pred + (1 - self.params['ml_weight']) * rule_pred
            else:
                pred = rule_pred
            val_predictions.append(pred)
        
        # Calculate loss with emphasis on high error cases
        errors = np.abs(np.array(val_predictions) - y_val)
        
        # Standard MAE
        mae = np.mean(errors)
        
        # Penalty for large errors (focus on reducing worst cases)
        large_error_penalty = np.mean(errors[errors > 100]) if np.any(errors > 100) else 0
        
        # Combined loss
        loss = mae + 0.3 * large_error_penalty
        
        # Track progress
        self.iteration_count += 1
        if loss < self.best_loss:
            self.best_loss = loss
        
        if self.iteration_count % 50 == 0:
            print(f"  Iteration {self.iteration_count}, Best Loss: ${self.best_loss:.2f}, Current: ${loss:.2f}")
        
        return loss
    
    def train(self, cases):
        """Train the enhanced parametric model."""
        print("Training enhanced parametric model with correlation features...")
        
        # Prepare data
        X, y = [], []
        for case in cases:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            features = self.extract_enhanced_features(days, miles, receipts)
            X.append(features)
            y.append(case['expected_output'])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Initialize parameters
        self.init_parameters()
        
        # Train ML component
        print("Training ML component...")
        self.ml_model = GradientBoostingRegressor(n_estimators=300, max_depth=10, random_state=42)
        self.ml_model.fit(X_train, y_train)
        
        # Optimize parameters with focus on high error cases
        print("Optimizing parameters for extreme cases...")
        param_names = list(self.params.keys())
        initial_values = [self.params[name] for name in param_names]
        bounds = [self.param_bounds[name] for name in param_names]
        
        self.iteration_count = 0
        self.best_loss = float('inf')
        
        # Multiple optimization attempts
        best_result = None
        best_params = None
        
        for attempt in range(2):
            print(f"  Optimization attempt {attempt + 1}/2...")
            
            result = optimize.minimize(
                self.objective_function,
                initial_values,
                args=(X_train, y_train, X_val, y_val),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-6}
            )
            
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_params = result.x.copy()
        
        # Set best parameters
        for i, name in enumerate(param_names):
            self.params[name] = best_params[i]
        
        print(f"Final optimization MAE: ${best_result.fun:.2f}")
        
        # Test performance
        test_predictions = []
        for i in range(len(X_test)):
            days = X_test[i][0]
            miles = X_test[i][1]
            receipts = X_test[i][2]
            pred = self.predict(days, miles, receipts)
            test_predictions.append(pred)
        
        test_mae = mean_absolute_error(y_test, test_predictions)
        print(f"Final test MAE: ${test_mae:.2f}")
        
        return self
    
    def predict(self, days, miles, receipts):
        """Enhanced prediction with correlation handling."""
        if self.params is None:
            raise ValueError("Model not trained yet")
        
        # Rule-based prediction with enhanced correlation handling
        rule_pred = self.enhanced_parametric_calculation(days, miles, receipts)
        
        # ML prediction
        if self.ml_model is not None:
            features = self.extract_enhanced_features(days, miles, receipts).reshape(1, -1)
            ml_pred = self.ml_model.predict(features)[0]
            
            # Blend predictions
            final_pred = (self.params['ml_weight'] * ml_pred + 
                         (1 - self.params['ml_weight']) * rule_pred)
        else:
            final_pred = rule_pred
        
        return final_pred
    
    def test_problematic_cases(self):
        """Test the specific high error cases."""
        problem_cases = [
            (8, 482, 1411.49, 631.81, "Case 548"),
            (6, 204, 818.99, 628.40, "Case 82"), 
            (10, 1192, 23.47, 1157.87, "Case 522"),
            (7, 1006, 1181.33, 2279.82, "Case 149"),
            (2, 384, 495.49, 290.36, "Case 89")
        ]
        
        print("\nTesting problematic cases:")
        print("-" * 60)
        
        for days, miles, receipts, expected, case_name in problem_cases:
            predicted = self.predict(days, miles, receipts)
            error = abs(predicted - expected)
            print(f"{case_name}: {days}d, {miles}mi, ${receipts:.2f}")
            print(f"  Expected: ${expected:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")
    
    def save(self, filepath):
        """Save the enhanced model."""
        model_data = {
            'params': self.params,
            'ml_model': self.ml_model,
            'param_bounds': self.param_bounds
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load the enhanced model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.params = model_data['params']
        self.ml_model = model_data['ml_model']
        self.param_bounds = model_data['param_bounds']
        return self

def main():
    """Train and test the enhanced parametric model."""
    print("Loading test cases...")
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Train enhanced model
    engine = EnhancedParametricEngine()
    engine.train(cases)
    
    # Test problematic cases
    engine.test_problematic_cases()
    
    # Save model
    engine.save('enhanced_parametric_model.pkl')
    print("\nEnhanced parametric model saved!")

if __name__ == "__main__":
    main()