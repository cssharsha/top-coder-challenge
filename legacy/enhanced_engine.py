#!/usr/bin/env python3
"""
Enhanced reimbursement engine that handles discovered edge cases:
- Receipt ending bonuses (.49, .99)
- Long trip penalties
- Extreme spending adjustments
- Over-prediction bias correction
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pickle

class EnhancedReimbursementEngine:
    def __init__(self):
        self.ml_model = None
        self.feature_means = None
        self.feature_stds = None
        self.bias_correction = 0
        
    def extract_features(self, days, miles, receipts):
        """Extract enhanced features including edge case indicators."""
        miles_per_day = miles / days
        receipts_per_day = receipts / days
        
        # Receipt ending features (discovered pattern)
        receipt_str = f"{receipts:.2f}"
        ends_49 = 1.0 if receipt_str.endswith('.49') else 0.0
        ends_99 = 1.0 if receipt_str.endswith('.99') else 0.0
        
        # Trip category features
        is_long_trip = 1.0 if days >= 8 else 0.0
        is_short_trip = 1.0 if days <= 2 else 0.0
        is_medium_trip = 1.0 if 4 <= days <= 6 else 0.0
        
        # Efficiency categories
        is_low_efficiency = 1.0 if miles_per_day < 50 else 0.0
        is_high_efficiency = 1.0 if miles_per_day > 300 else 0.0
        is_optimal_efficiency = 1.0 if 150 <= miles_per_day <= 250 else 0.0
        
        # Spending categories
        is_low_spending = 1.0 if receipts_per_day < 50 else 0.0
        is_high_spending = 1.0 if receipts_per_day > 200 else 0.0
        is_extreme_spending = 1.0 if receipts_per_day > 400 else 0.0
        
        # Interaction features for edge cases
        long_trip_low_efficiency = is_long_trip * is_low_efficiency
        short_trip_high_spending = is_short_trip * is_high_spending
        
        features = [
            days,                           # 0: trip duration
            miles,                          # 1: total miles
            receipts,                       # 2: total receipts
            miles_per_day,                  # 3: efficiency
            receipts_per_day,               # 4: spending rate
            days * miles,                   # 5: interaction
            days * receipts,                # 6: interaction
            miles * receipts,               # 7: interaction
            days ** 2,                      # 8: quadratic
            miles ** 2,                     # 9: quadratic
            receipts ** 2,                  # 10: quadratic
            np.sqrt(miles),                 # 11: sqrt miles
            np.sqrt(receipts),              # 12: sqrt receipts
            np.log(days + 1),               # 13: log days
            np.log(miles + 1),              # 14: log miles
            np.log(receipts + 1),           # 15: log receipts
            ends_49,                        # 16: receipt ends in .49
            ends_99,                        # 17: receipt ends in .99
            is_long_trip,                   # 18: 8+ days
            is_short_trip,                  # 19: 1-2 days
            is_medium_trip,                 # 20: 4-6 days
            is_low_efficiency,              # 21: <50 mi/day
            is_high_efficiency,             # 22: >300 mi/day
            is_optimal_efficiency,          # 23: 150-250 mi/day
            is_low_spending,                # 24: <$50/day
            is_high_spending,               # 25: >$200/day
            is_extreme_spending,            # 26: >$400/day
            long_trip_low_efficiency,       # 27: problem combination
            short_trip_high_spending,       # 28: problem combination
        ]
        
        return np.array(features)
    
    def rule_based_with_edge_cases(self, days, miles, receipts):
        """Enhanced rule-based calculation with edge case handling."""
        # Base calculation
        per_diem_base = 100
        
        # Adjust per diem for trip length
        if days >= 10:
            per_diem_rate = 85  # Long trip penalty
        elif days == 5:
            per_diem_rate = 110  # Sweet spot bonus
        else:
            per_diem_rate = per_diem_base
        
        per_diem = days * per_diem_rate
        
        # Mileage with complex tiers
        miles_per_day = miles / days
        if miles <= 50:
            mileage_reimb = miles * 0.45  # Low mileage penalty
        elif miles <= 100:
            mileage_reimb = miles * 0.58
        elif miles <= 300:
            mileage_reimb = 100 * 0.58 + (miles - 100) * 0.50
        elif miles <= 600:
            mileage_reimb = 100 * 0.58 + 200 * 0.50 + (miles - 300) * 0.42
        else:
            # Very high mileage - diminishing returns
            mileage_reimb = 100 * 0.58 + 200 * 0.50 + 300 * 0.42 + (miles - 600) * 0.35
        
        # Efficiency adjustments
        if miles_per_day > 400:
            mileage_reimb *= 0.8  # Penalty for unrealistic efficiency
        elif 150 <= miles_per_day <= 250:
            mileage_reimb *= 1.1  # Efficiency bonus
        
        # Receipt handling with complex rules
        receipts_per_day = receipts / days
        
        if receipts_per_day > 300:
            # Extreme spending penalty
            receipt_reimb = receipts * 0.4
        elif receipts_per_day > 150:
            # High spending penalty
            receipt_reimb = receipts * 0.65
        elif receipts_per_day < 20:
            # Very low spending penalty
            receipt_reimb = receipts * 0.6
        elif receipts_per_day < 50:
            # Low spending penalty
            receipt_reimb = receipts * 0.85
        else:
            # Normal range
            receipt_reimb = receipts * 0.9
        
        # Receipt ending bonuses (discovered bug/feature)
        receipt_str = f"{receipts:.2f}"
        if receipt_str.endswith('.49'):
            receipt_reimb *= 1.02  # Small bonus for .49
        elif receipt_str.endswith('.99'):
            receipt_reimb *= 1.01  # Small bonus for .99
        
        # Special case adjustments
        total = per_diem + mileage_reimb + receipt_reimb
        
        # Long trip with low efficiency penalty
        if days >= 8 and miles_per_day < 60:
            total *= 0.85
        
        # Short trip with extreme spending penalty
        if days <= 2 and receipts_per_day > 300:
            total *= 0.7
        
        return total
    
    def train(self, cases):
        """Train multiple models and ensemble them."""
        print("Training enhanced reimbursement model...")
        
        # Extract features and targets
        features = []
        targets = []
        
        for case in cases:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            output = case['expected_output']
            
            feature_vector = self.extract_features(days, miles, receipts)
            features.append(feature_vector)
            targets.append(output)
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Normalize features
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0)
        features_normalized = (features - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Train ensemble of models
        models = [
            RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42),
            GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42),
            RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=3, random_state=43)
        ]
        
        trained_models = []
        for model in models:
            model.fit(features_normalized, targets)
            trained_models.append(model)
        
        self.ml_model = trained_models
        
        # Calculate bias correction (over-prediction tendency)
        ensemble_predictions = self._ensemble_predict(features_normalized)
        self.bias_correction = np.mean(targets - ensemble_predictions)
        
        # Training accuracy
        corrected_predictions = ensemble_predictions + self.bias_correction
        mae = np.mean(np.abs(corrected_predictions - targets))
        print(f"Training MAE: ${mae:.2f}")
        print(f"Bias correction: ${self.bias_correction:.2f}")
        
        return self
    
    def _ensemble_predict(self, features_normalized):
        """Get ensemble prediction from multiple models."""
        if not self.ml_model:
            return np.array([0])
        
        predictions = []
        for model in self.ml_model:
            pred = model.predict(features_normalized)
            predictions.append(pred)
        
        # Weighted average (equal weights for now)
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def predict(self, days, miles, receipts):
        """Predict reimbursement with enhanced model."""
        if self.ml_model is None:
            return self.rule_based_with_edge_cases(days, miles, receipts)
        
        # Extract and normalize features
        features = self.extract_features(days, miles, receipts)
        features_normalized = (features - self.feature_means) / (self.feature_stds + 1e-8)
        features_normalized = features_normalized.reshape(1, -1)
        
        # Get ensemble ML prediction
        ml_prediction = self._ensemble_predict(features_normalized)[0]
        
        # Apply bias correction
        ml_prediction += self.bias_correction
        
        # Get enhanced rule-based prediction
        rule_prediction = self.rule_based_with_edge_cases(days, miles, receipts)
        
        # Blend predictions with higher ML weight
        final_prediction = 0.9 * ml_prediction + 0.1 * rule_prediction
        
        return final_prediction
    
    def save(self, filepath):
        """Save the trained model."""
        model_data = {
            'ml_model': self.ml_model,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'bias_correction': self.bias_correction
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.ml_model = model_data['ml_model']
        self.feature_means = model_data['feature_means']
        self.feature_stds = model_data['feature_stds']
        self.bias_correction = model_data['bias_correction']
        return self

def load_test_cases():
    """Load the public test cases from JSON file."""
    with open('../public_cases.json', 'r') as f:
        return json.load(f)

def evaluate_model(engine, cases):
    """Evaluate the enhanced model."""
    errors = []
    receipt_49_errors = []
    receipt_99_errors = []
    other_errors = []
    
    for case in cases:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = engine.predict(days, miles, receipts)
        error = abs(predicted - expected)
        errors.append(error)
        
        # Track receipt ending errors
        receipt_str = f"{receipts:.2f}"
        if receipt_str.endswith('.49'):
            receipt_49_errors.append(error)
        elif receipt_str.endswith('.99'):
            receipt_99_errors.append(error)
        else:
            other_errors.append(error)
    
    mae = np.mean(errors)
    max_error = np.max(errors)
    
    within_10 = sum(1 for e in errors if e <= 10) / len(errors) * 100
    within_25 = sum(1 for e in errors if e <= 25) / len(errors) * 100
    within_50 = sum(1 for e in errors if e <= 50) / len(errors) * 100
    
    print(f"Enhanced Model Results:")
    print(f"  Mean Absolute Error: ${mae:.2f}")
    print(f"  Maximum Error: ${max_error:.2f}")
    print(f"  Cases within $10: {within_10:.1f}%")
    print(f"  Cases within $25: {within_25:.1f}%")
    print(f"  Cases within $50: {within_50:.1f}%")
    
    if receipt_49_errors:
        print(f"  .49 ending errors: ${np.mean(receipt_49_errors):.2f} avg")
    if receipt_99_errors:
        print(f"  .99 ending errors: ${np.mean(receipt_99_errors):.2f} avg")
    if other_errors:
        print(f"  Other ending errors: ${np.mean(other_errors):.2f} avg")
    
    return mae

def main():
    """Train and evaluate the enhanced model."""
    print("Loading test cases...")
    cases = load_test_cases()
    
    # Train enhanced model
    engine = EnhancedReimbursementEngine()
    engine.train(cases)
    
    # Evaluate
    evaluate_model(engine, cases)
    
    # Save
    engine.save('enhanced_model.pkl')
    print("Enhanced model saved!")

if __name__ == "__main__":
    main()