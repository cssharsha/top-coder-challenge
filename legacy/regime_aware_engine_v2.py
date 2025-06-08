#!/usr/bin/env python3
"""
Regime-Aware Engine V2 - Incremental Fixes
Fix 1: Add legitimate high-mileage travel bonus
Fix 2: Add mileage bonus for minimal expense cases  
Fix 3: Context-aware penalties (selective application)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import json

class RegimeAwareEngineV2:
    def __init__(self):
        self.caps = None
        self.penalties = None
        self.bonuses = None
        self.ml_model = None
        self.manual_mode = True
        self.fixes_enabled = {
            'legitimate_travel_bonus': True,
            'minimal_expense_mileage_bonus': True,
            'context_aware_penalties': True
        }
        
    def init_manual_caps(self):
        """Initialize manual caps with incremental fix parameters."""
        # Base caps from V1 (the ones that worked well)
        self.caps = {
            'short_trip_cap': 60.0,
            'medium_trip_cap': 80.0,
            'long_trip_cap': 65.0,
            'short_threshold': 2,
            'medium_threshold': 6,
            'base_per_diem': 100.0,
            'mile_rate_tier1': 0.58,
            'mile_rate_tier2': 0.50,
            'mile_rate_tier3': 0.42,
            'mile_tier1': 100,
            'mile_tier2': 300,
            'mile_tier3': 600,
        }
        
        # V1 penalties (these worked well for cases 548, 82)
        self.penalties = {
            'mild_violation': 0.90,
            'moderate_violation': 0.65,
            'severe_violation': 0.45,
            'extreme_violation': 0.30,
            'mild_threshold': 1.2,
            'moderate_threshold': 1.5,
            'severe_threshold': 2.0,
            'extreme_threshold': 3.0,
        }
        
        # NEW: Bonus system for legitimate travel patterns
        self.bonuses = {
            # High-mileage legitimate travel bonus (Fix 1)
            'high_mile_bonus_threshold': 600,     # Miles that trigger bonus consideration
            'high_mile_reasonable_expense': 150,  # $/day that's considered reasonable for high miles
            'high_mile_bonus_factor': 1.8,       # Bonus multiplier for legitimate high-mile travel
            
            # Minimal expense mileage bonus (Fix 2)  
            'minimal_expense_threshold': 30,      # $/day considered "minimal"
            'minimal_expense_mile_bonus': 0.15,   # Extra per mile for minimal expense cases
            'minimal_expense_base_bonus': 1.2,    # Base multiplier for efficient travel
            
            # Context-aware penalty thresholds (Fix 3)
            'receipt_only_penalty': True,         # Apply expense penalties only to receipt portion
            'mileage_protection': True,           # Protect mileage reimbursement from expense penalties
        }
    
    def classify_regime(self, days, miles, receipts):
        """Enhanced regime classification with fix-specific logic."""
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
            elif violation_ratio < self.penalties['moderate_threshold']:
                expense_regime = 'moderate_violation'
            elif violation_ratio < self.penalties['severe_threshold']:
                expense_regime = 'severe_violation'
            else:
                expense_regime = 'extreme_violation'
            violation_factor = violation_ratio
        
        # Enhanced anomaly detection with fix logic
        anomaly_flags = []
        bonus_flags = []
        
        # Fix 1: Detect legitimate high-mileage travel
        if (miles >= self.bonuses['high_mile_bonus_threshold'] and 
            receipts_per_day <= self.bonuses['high_mile_reasonable_expense']):
            bonus_flags.append('legitimate_high_mile')
        
        # Fix 2: Detect minimal expense with high mileage (deserves bonus)
        if (receipts_per_day <= self.bonuses['minimal_expense_threshold'] and
            miles >= 200):
            bonus_flags.append('minimal_expense_high_mile')
        
        # Original anomaly detection
        if days >= 8 and receipts_per_day < 10:
            anomaly_flags.append('long_minimal')
        if days <= 2 and receipts_per_day > 200:
            anomaly_flags.append('short_excessive')
        if miles_per_day > 300 and receipts_per_day < 30:
            anomaly_flags.append('high_mile_low_expense')
        
        return {
            'trip_type': trip_type,
            'expense_regime': expense_regime,
            'violation_factor': violation_factor,
            'anomaly_flags': anomaly_flags,
            'bonus_flags': bonus_flags,
            'expense_cap': expense_cap,
            'receipts_per_day': receipts_per_day,
            'miles_per_day': miles_per_day
        }
    
    def calculate_base_reimbursement(self, days, miles, receipts):
        """Calculate base reimbursement with enhanced mileage handling."""
        # Per diem calculation
        per_diem = days * self.caps['base_per_diem']
        
        # Enhanced mileage calculation
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
        
        # Receipt reimbursement (base rate)
        receipt_reimb = receipts * 0.80
        
        return {
            'per_diem': per_diem,
            'mileage': mileage,
            'receipts': receipt_reimb,
            'total': per_diem + mileage + receipt_reimb
        }
    
    def apply_fix_1_legitimate_travel_bonus(self, base_amounts, regime_info, days, miles, receipts):
        """Fix 1: Add bonus for legitimate high-mileage travel (Case 149)."""
        if not self.fixes_enabled['legitimate_travel_bonus']:
            return base_amounts
        
        if 'legitimate_high_mile' in regime_info['bonus_flags']:
            # This is legitimate business travel - high miles with reasonable expenses
            # Apply significant bonus to reflect the value
            bonus_factor = self.bonuses['high_mile_bonus_factor']
            
            print(f"  [Fix 1] Legitimate high-mileage travel detected - applying {bonus_factor}x bonus")
            
            # Apply bonus to entire amount for legitimate travel
            base_amounts['total'] *= bonus_factor
            base_amounts['mileage'] *= bonus_factor
            base_amounts['per_diem'] *= (bonus_factor - 1) * 0.3 + 1  # Smaller bonus to per diem
            
        return base_amounts
    
    def apply_fix_2_minimal_expense_bonus(self, base_amounts, regime_info, days, miles, receipts):
        """Fix 2: Add mileage bonus for minimal expense cases (Case 522)."""
        if not self.fixes_enabled['minimal_expense_mileage_bonus']:
            return base_amounts
        
        if 'minimal_expense_high_mile' in regime_info['bonus_flags']:
            # High mileage with minimal personal expenses - reward efficiency
            extra_mileage = miles * self.bonuses['minimal_expense_mile_bonus']
            base_multiplier = self.bonuses['minimal_expense_base_bonus']
            
            print(f"  [Fix 2] Minimal expense + high mileage - adding ${extra_mileage:.2f} bonus")
            
            base_amounts['mileage'] += extra_mileage
            base_amounts['total'] *= base_multiplier
            
        return base_amounts
    
    def apply_fix_3_context_aware_penalties(self, base_amounts, regime_info):
        """Fix 3: Apply penalties only to relevant portions (context-aware)."""
        if not self.fixes_enabled['context_aware_penalties']:
            # Fall back to V1 blanket penalty
            if regime_info['expense_regime'] != 'normal':
                penalty_factor = self.penalties[regime_info['expense_regime']]
                base_amounts['total'] *= penalty_factor
            return base_amounts
        
        # Context-aware penalty application
        if regime_info['expense_regime'] != 'normal':
            penalty_factor = self.penalties[regime_info['expense_regime']]
            
            print(f"  [Fix 3] Context-aware {regime_info['expense_regime']} penalty: {penalty_factor:.2f}")
            
            if self.bonuses['receipt_only_penalty']:
                # Apply penalty primarily to receipt portion
                base_amounts['receipts'] *= penalty_factor
                
                # Light penalty to per diem for severe violations
                if regime_info['expense_regime'] in ['severe_violation', 'extreme_violation']:
                    base_amounts['per_diem'] *= (penalty_factor + 0.3)  # Lighter penalty
                
                # Protect mileage for legitimate travel
                if not self.bonuses['mileage_protection'] or 'legitimate_high_mile' not in regime_info['bonus_flags']:
                    base_amounts['mileage'] *= max(0.85, penalty_factor + 0.2)  # Very light penalty
            else:
                # Original blanket penalty
                base_amounts['total'] *= penalty_factor
            
            # Recalculate total
            base_amounts['total'] = base_amounts['per_diem'] + base_amounts['mileage'] + base_amounts['receipts']
        
        return base_amounts
    
    def apply_anomaly_adjustments(self, base_amounts, regime_info):
        """Apply adjustments for anomaly cases."""
        for anomaly in regime_info['anomaly_flags']:
            if anomaly == 'long_minimal':
                # Already handled by Fix 2
                pass
            elif anomaly == 'short_excessive':
                # Additional penalty for suspicious short trips
                base_amounts['receipts'] *= 0.7
                print(f"  [Anomaly] Short excessive trip penalty applied")
            elif anomaly == 'high_mile_low_expense':
                # Small bonus for efficient travel (already handled by Fix 2)
                pass
        
        # Recalculate total after anomaly adjustments
        base_amounts['total'] = base_amounts['per_diem'] + base_amounts['mileage'] + base_amounts['receipts']
        return base_amounts
    
    def predict_manual(self, days, miles, receipts):
        """Enhanced manual prediction with incremental fixes."""
        if self.caps is None:
            self.init_manual_caps()
        
        # Classify the case
        regime_info = self.classify_regime(days, miles, receipts)
        
        # Calculate base reimbursement components
        base_amounts = self.calculate_base_reimbursement(days, miles, receipts)
        
        print(f"\n  Base calculation: Per diem=${base_amounts['per_diem']:.2f}, "
              f"Mileage=${base_amounts['mileage']:.2f}, Receipts=${base_amounts['receipts']:.2f}")
        
        # Apply incremental fixes
        base_amounts = self.apply_fix_1_legitimate_travel_bonus(base_amounts, regime_info, days, miles, receipts)
        base_amounts = self.apply_fix_2_minimal_expense_bonus(base_amounts, regime_info, days, miles, receipts)
        base_amounts = self.apply_fix_3_context_aware_penalties(base_amounts, regime_info)
        base_amounts = self.apply_anomaly_adjustments(base_amounts, regime_info)
        
        return base_amounts['total'], regime_info, base_amounts
    
    def test_edge_cases_with_fixes(self, fix_name="All Fixes"):
        """Test edge cases with current fix configuration."""
        edge_cases = [
            (8, 482, 1411.49, 631.81, "Case 548: Long trip, high expenses"),
            (6, 204, 818.99, 628.40, "Case 82: Medium trip, high expenses"),
            (10, 1192, 23.47, 1157.87, "Case 522: Very long trip, minimal expenses"),
            (7, 1006, 1181.33, 2279.82, "Case 149: Long trip, high miles+expenses"),
            (2, 384, 495.49, 290.36, "Case 89: Short trip, high miles+expenses")
        ]
        
        print(f"\nTesting Edge Cases with {fix_name}:")
        print("=" * 70)
        
        total_error = 0
        for days, miles, receipts, expected, description in edge_cases:
            predicted, regime_info, base_amounts = self.predict_manual(days, miles, receipts)
            error = abs(predicted - expected)
            total_error += error
            
            print(f"\n{description}")
            print(f"  Input: {days}d, {miles}mi, ${receipts:.2f}")
            print(f"  Expected: ${expected:.2f}, Predicted: ${predicted:.2f}")
            print(f"  Error: ${error:.2f}")
            print(f"  Regime: {regime_info['trip_type']} trip, {regime_info['expense_regime']}")
            print(f"  Per-day: ${regime_info['receipts_per_day']:.2f} (cap: ${regime_info['expense_cap']:.2f})")
            if regime_info['bonus_flags']:
                print(f"  Bonuses: {', '.join(regime_info['bonus_flags'])}")
            if regime_info['anomaly_flags']:
                print(f"  Anomalies: {', '.join(regime_info['anomaly_flags'])}")
            print(f"  Final breakdown: Per diem=${base_amounts['per_diem']:.2f}, "
                  f"Mileage=${base_amounts['mileage']:.2f}, Receipts=${base_amounts['receipts']:.2f}")
        
        avg_error = total_error / len(edge_cases)
        print(f"\nAverage Error: ${avg_error:.2f}")
        print(f"Total Error: ${total_error:.2f}")
        
        return avg_error
    
    def test_incremental_fixes(self):
        """Test each fix incrementally to see individual impact."""
        print("INCREMENTAL FIX TESTING")
        print("=" * 50)
        
        # Baseline (no fixes)
        self.fixes_enabled = {'legitimate_travel_bonus': False, 'minimal_expense_mileage_bonus': False, 'context_aware_penalties': False}
        baseline_error = self.test_edge_cases_with_fixes("Baseline (V1)")
        
        # Fix 1 only
        self.fixes_enabled = {'legitimate_travel_bonus': True, 'minimal_expense_mileage_bonus': False, 'context_aware_penalties': False}
        fix1_error = self.test_edge_cases_with_fixes("Fix 1: Legitimate Travel Bonus")
        
        # Fix 1 + 2
        self.fixes_enabled = {'legitimate_travel_bonus': True, 'minimal_expense_mileage_bonus': True, 'context_aware_penalties': False}
        fix12_error = self.test_edge_cases_with_fixes("Fix 1 + 2")
        
        # All fixes
        self.fixes_enabled = {'legitimate_travel_bonus': True, 'minimal_expense_mileage_bonus': True, 'context_aware_penalties': True}
        all_fixes_error = self.test_edge_cases_with_fixes("All Fixes")
        
        print(f"\nINCREMENTAL IMPROVEMENT SUMMARY:")
        print(f"Baseline (V1):     ${baseline_error:.2f}")
        print(f"+ Fix 1:           ${fix1_error:.2f} (improvement: ${baseline_error - fix1_error:.2f})")
        print(f"+ Fix 1+2:         ${fix12_error:.2f} (improvement: ${baseline_error - fix12_error:.2f})")
        print(f"+ All Fixes:       ${all_fixes_error:.2f} (improvement: ${baseline_error - all_fixes_error:.2f})")
        
        return all_fixes_error
    
    def save(self, filepath):
        """Save the enhanced regime-aware model."""
        model_data = {
            'caps': self.caps,
            'penalties': self.penalties,
            'bonuses': self.bonuses,
            'fixes_enabled': self.fixes_enabled,
            'manual_mode': self.manual_mode,
            'ml_model': self.ml_model if hasattr(self, 'ml_model') else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load the enhanced regime-aware model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.caps = model_data['caps']
        self.penalties = model_data['penalties']
        self.bonuses = model_data.get('bonuses', {})
        self.fixes_enabled = model_data.get('fixes_enabled', {})
        self.manual_mode = model_data.get('manual_mode', True)
        self.ml_model = model_data.get('ml_model', None)
        return self
    
    def predict(self, days, miles, receipts):
        """Main prediction method for external use."""
        prediction, _, _ = self.predict_manual(days, miles, receipts)
        return prediction

def main():
    """Test incremental fixes on the regime-aware engine."""
    print("Regime-Aware Engine V2 - Incremental Fix Testing")
    print("=" * 60)
    
    engine = RegimeAwareEngineV2()
    
    # Test incremental fixes
    final_error = engine.test_incremental_fixes()
    
    # Save the enhanced model
    engine.save('regime_aware_v2_model.pkl')
    print(f"\nEnhanced regime-aware model saved with final error: ${final_error:.2f}")

if __name__ == "__main__":
    main()