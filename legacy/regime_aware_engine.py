#!/usr/bin/env python3
"""
Regime-Aware Reimbursement Engine - Phase 1: Manual Hard Cap Discovery
Uses the 5 problematic edge cases to discover optimal caps and penalties.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import json

class RegimeAwareEngine:
    def __init__(self):
        self.caps = None
        self.penalties = None
        self.ml_model = None
        self.manual_mode = True  # Start with manual tuning
        
    def init_manual_caps(self):
        """Initialize manual caps based on edge case analysis."""
        # These will be manually tuned based on the 5 edge cases
        self.caps = {
            # Per-day expense caps by trip length
            'short_trip_cap': 75.0,    # 1-2 days: very strict
            'medium_trip_cap': 85.0,   # 3-6 days: moderate  
            'long_trip_cap': 70.0,     # 7+ days: strict again
            
            # Trip length thresholds
            'short_threshold': 2,
            'medium_threshold': 6,
            
            # Base rates
            'base_per_diem': 100.0,
            'mile_rate_tier1': 0.58,
            'mile_rate_tier2': 0.50,
            'mile_rate_tier3': 0.42,
            'mile_tier1': 100,
            'mile_tier2': 300,
            'mile_tier3': 600,
        }
        
        # Penalty factors (how much to reduce reimbursement)
        self.penalties = {
            'mild_violation': 0.85,      # 15% reduction
            'moderate_violation': 0.70,  # 30% reduction  
            'severe_violation': 0.50,    # 50% reduction
            'extreme_violation': 0.35,   # 65% reduction
            
            # Violation thresholds (multiplier of cap)
            'mild_threshold': 1.2,       # 20% over cap
            'moderate_threshold': 1.5,   # 50% over cap
            'severe_threshold': 2.0,     # 100% over cap
            'extreme_threshold': 3.0,    # 200% over cap
        }
    
    def classify_regime(self, days, miles, receipts):
        """Classify the case into different regimes for targeted handling."""
        receipts_per_day = receipts / max(days, 1)
        miles_per_day = miles / max(days, 1)
        
        # Determine trip length category
        if days <= self.caps['short_threshold']:
            trip_type = 'short'
            expense_cap = self.caps['short_trip_cap']
        elif days <= self.caps['medium_threshold']:
            trip_type = 'medium'  
            expense_cap = self.caps['medium_trip_cap']
        else:
            trip_type = 'long'
            expense_cap = self.caps['long_trip_cap']
        
        # Calculate expense violation level
        if receipts_per_day <= expense_cap:
            expense_regime = 'normal'
            violation_factor = 0
        else:
            violation_ratio = receipts_per_day / expense_cap
            
            if violation_ratio < self.penalties['mild_threshold']:
                expense_regime = 'mild_violation'
                violation_factor = violation_ratio
            elif violation_ratio < self.penalties['moderate_threshold']:
                expense_regime = 'moderate_violation'
                violation_factor = violation_ratio
            elif violation_ratio < self.penalties['severe_threshold']:
                expense_regime = 'severe_violation'
                violation_factor = violation_ratio
            else:
                expense_regime = 'extreme_violation'
                violation_factor = violation_ratio
        
        # Detect anomaly patterns
        anomaly_flags = []
        if days >= 8 and receipts_per_day < 10:  # Long trip, minimal expenses
            anomaly_flags.append('long_minimal')
        if days <= 2 and receipts_per_day > 200:  # Short trip, huge expenses
            anomaly_flags.append('short_excessive')
        if miles_per_day > 300 and receipts_per_day < 30:  # High miles, low expenses
            anomaly_flags.append('high_mile_low_expense')
        
        return {
            'trip_type': trip_type,
            'expense_regime': expense_regime,
            'violation_factor': violation_factor,
            'anomaly_flags': anomaly_flags,
            'expense_cap': expense_cap,
            'receipts_per_day': receipts_per_day
        }
    
    def calculate_base_reimbursement(self, days, miles, receipts):
        """Calculate base reimbursement using standard rules."""
        # Per diem calculation
        per_diem = days * self.caps['base_per_diem']
        
        # Mileage calculation with tiers
        if miles <= self.caps['mile_tier1']:
            mileage = miles * self.caps['mile_rate_tier1']
        elif miles <= self.caps['mile_tier2']:
            mileage = (self.caps['mile_tier1'] * self.caps['mile_rate_tier1'] + 
                      (miles - self.caps['mile_tier1']) * self.caps['mile_rate_tier2'])
        elif miles <= self.caps['mile_tier3']:
            mileage = (self.caps['mile_tier1'] * self.caps['mile_rate_tier1'] + 
                      (self.caps['mile_tier2'] - self.caps['mile_tier1']) * self.caps['mile_rate_tier2'] +
                      (miles - self.caps['mile_tier2']) * self.caps['mile_rate_tier3'])
        else:
            mileage = (self.caps['mile_tier1'] * self.caps['mile_rate_tier1'] + 
                      (self.caps['mile_tier2'] - self.caps['mile_tier1']) * self.caps['mile_rate_tier2'] +
                      (self.caps['mile_tier3'] - self.caps['mile_tier2']) * self.caps['mile_rate_tier3'] +
                      (miles - self.caps['mile_tier3']) * 0.35)
        
        # Receipt reimbursement (base rate before penalties)
        receipt_reimb = receipts * 0.80  # Start with 80% base rate
        
        return per_diem + mileage + receipt_reimb
    
    def apply_regime_adjustments(self, base_amount, regime_info):
        """Apply regime-specific adjustments based on classification."""
        adjusted_amount = base_amount
        
        # Apply expense violation penalties
        if regime_info['expense_regime'] != 'normal':
            penalty_factor = self.penalties[regime_info['expense_regime']]
            
            # Apply penalty only to receipt portion (more targeted)
            # For now, apply to whole amount (can be refined)
            adjusted_amount *= penalty_factor
            
        # Apply anomaly-specific adjustments
        for anomaly in regime_info['anomaly_flags']:
            if anomaly == 'long_minimal':
                # Long trip with minimal expenses - might be legitimate, small penalty
                adjusted_amount *= 0.95
            elif anomaly == 'short_excessive':
                # Short trip with huge expenses - suspicious, big penalty
                adjusted_amount *= 0.60
            elif anomaly == 'high_mile_low_expense':
                # High miles, low expenses - could be efficient travel, small bonus
                adjusted_amount *= 1.05
        
        return adjusted_amount
    
    def predict_manual(self, days, miles, receipts):
        """Manual prediction using hard-coded caps and rules."""
        if self.caps is None:
            self.init_manual_caps()
        
        # Classify the case
        regime_info = self.classify_regime(days, miles, receipts)
        
        # Calculate base reimbursement
        base_amount = self.calculate_base_reimbursement(days, miles, receipts)
        
        # Apply regime adjustments
        final_amount = self.apply_regime_adjustments(base_amount, regime_info)
        
        return final_amount, regime_info
    
    def test_edge_cases(self):
        """Test the 5 problematic edge cases with current caps."""
        edge_cases = [
            (8, 482, 1411.49, 631.81, "Case 548: Long trip, high expenses"),
            (6, 204, 818.99, 628.40, "Case 82: Medium trip, high expenses"),
            (10, 1192, 23.47, 1157.87, "Case 522: Very long trip, minimal expenses"),
            (7, 1006, 1181.33, 2279.82, "Case 149: Long trip, high miles+expenses"),
            (2, 384, 495.49, 290.36, "Case 89: Short trip, high miles+expenses")
        ]
        
        print("Testing Edge Cases with Current Caps:")
        print("=" * 70)
        
        total_error = 0
        for days, miles, receipts, expected, description in edge_cases:
            predicted, regime_info = self.predict_manual(days, miles, receipts)
            error = abs(predicted - expected)
            total_error += error
            
            print(f"\n{description}")
            print(f"  Input: {days}d, {miles}mi, ${receipts:.2f}")
            print(f"  Expected: ${expected:.2f}, Predicted: ${predicted:.2f}")
            print(f"  Error: ${error:.2f}")
            print(f"  Regime: {regime_info['trip_type']} trip, {regime_info['expense_regime']}")
            print(f"  Per-day: ${regime_info['receipts_per_day']:.2f} (cap: ${regime_info['expense_cap']:.2f})")
            if regime_info['anomaly_flags']:
                print(f"  Anomalies: {', '.join(regime_info['anomaly_flags'])}")
        
        avg_error = total_error / len(edge_cases)
        print(f"\nAverage Error: ${avg_error:.2f}")
        print(f"Total Error: ${total_error:.2f}")
        
        return avg_error
    
    def tune_caps_manually(self):
        """Interactive manual tuning of caps based on edge case performance."""
        print("Starting Manual Cap Tuning...")
        print("Current caps:", self.caps)
        
        # Test with initial caps
        initial_error = self.test_edge_cases()
        
        print(f"\nInitial average error: ${initial_error:.2f}")
        print("\nRecommended Adjustments Based on Edge Case Analysis:")
        
        # Analyze each edge case and suggest cap adjustments
        print("\nCase 548 (8d, $1411.49 -> $631.81): Long trip with extreme expenses")
        print("  Current long_trip_cap: $70/day, Actual: $176/day = 2.5x violation")
        print("  Needs severe penalty (current 50% may not be enough)")
        
        print("\nCase 82 (6d, $818.99 -> $628.40): Medium trip with high expenses") 
        print("  Current medium_trip_cap: $85/day, Actual: $136/day = 1.6x violation")
        print("  Needs moderate penalty")
        
        print("\nCase 522 (10d, $23.47 -> $1157.87): Long trip, minimal expenses")
        print("  Very low expenses but high expected payout - special case")
        print("  May need different base calculation for high-mile low-expense")
        
        print("\nCase 149 (7d, $1181.33 -> $2279.82): Should pay MORE, not less")
        print("  High miles (1006) + reasonable expenses = legitimate business travel")
        print("  Need bonus for high-efficiency + reasonable expenses")
        
        print("\nCase 89 (2d, $495.49 -> $290.36): Short trip, extreme expenses")
        print("  Current short_trip_cap: $75/day, Actual: $247/day = 3.3x violation")
        print("  Needs extreme penalty")
        
        # Suggest new caps
        print("\nSuggested Cap Adjustments:")
        suggested_caps = {
            'short_trip_cap': 60.0,    # Tighter for short trips
            'medium_trip_cap': 80.0,   # Slightly tighter
            'long_trip_cap': 65.0,     # Tighter for long trips
        }
        
        suggested_penalties = {
            'mild_violation': 0.90,      # Less harsh for mild
            'moderate_violation': 0.65,  # Harsher for moderate
            'severe_violation': 0.45,    # Much harsher for severe
            'extreme_violation': 0.30,   # Very harsh for extreme
        }
        
        print("New caps:", suggested_caps)
        print("New penalties:", suggested_penalties)
        
        return suggested_caps, suggested_penalties
    
    def apply_manual_tuning(self, new_caps=None, new_penalties=None):
        """Apply manually tuned caps and test performance."""
        if new_caps:
            self.caps.update(new_caps)
        if new_penalties:
            self.penalties.update(new_penalties)
        
        print("\nTesting with updated caps...")
        new_error = self.test_edge_cases()
        
        return new_error
    
    def save(self, filepath):
        """Save the regime-aware model."""
        model_data = {
            'caps': self.caps,
            'penalties': self.penalties,
            'manual_mode': self.manual_mode,
            'ml_model': self.ml_model if hasattr(self, 'ml_model') else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load the regime-aware model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.caps = model_data['caps']
        self.penalties = model_data['penalties']
        self.manual_mode = model_data.get('manual_mode', True)
        self.ml_model = model_data.get('ml_model', None)
        return self
    
    def predict(self, days, miles, receipts):
        """Main prediction method (currently uses manual approach)."""
        prediction, _ = self.predict_manual(days, miles, receipts)
        return prediction

def main():
    """Phase 1: Manual cap discovery and tuning."""
    print("Regime-Aware Engine - Phase 1: Manual Hard Cap Discovery")
    print("=" * 60)
    
    engine = RegimeAwareEngine()
    
    # Test with initial caps
    print("Step 1: Testing with initial caps...")
    initial_error = engine.test_edge_cases()
    
    # Manual tuning suggestions
    print("\nStep 2: Manual tuning analysis...")
    suggested_caps, suggested_penalties = engine.tune_caps_manually()
    
    # Apply suggestions and test
    print("\nStep 3: Testing with suggested adjustments...")
    final_error = engine.apply_manual_tuning(suggested_caps, suggested_penalties)
    
    print(f"\nImprovement: ${initial_error:.2f} -> ${final_error:.2f}")
    print(f"Error reduction: ${initial_error - final_error:.2f}")
    
    # Save the manually tuned model
    engine.save('regime_aware_model.pkl')
    print("\nManually tuned model saved as regime_aware_model.pkl")

if __name__ == "__main__":
    main()