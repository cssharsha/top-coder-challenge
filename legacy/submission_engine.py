#!/usr/bin/env python3
"""
Final submission engine - the main interface for reimbursement calculations.
This engine will be used to evaluate against private test cases.
"""

import json
from enhanced_engine import EnhancedReimbursementEngine

class SubmissionEngine:
    def __init__(self):
        self.engine = EnhancedReimbursementEngine()
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            self.engine.load('enhanced_model.pkl')
        except FileNotFoundError:
            print("Model file not found, training new model...")
            self.train_model()
    
    def train_model(self):
        """Train the model if it doesn't exist."""
        with open('../public_cases.json', 'r') as f:
            cases = json.load(f)
        self.engine.train(cases)
        self.engine.save('enhanced_model.pkl')
        print("Model trained and saved")
    
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
    
    def process_test_cases(self, test_cases):
        """
        Process a list of test cases and return predictions.
        
        Args:
            test_cases (list): List of test case dictionaries
            
        Returns:
            list: List of prediction results
        """
        results = []
        for case in test_cases:
            input_data = case['input']
            prediction = self.calculate_reimbursement(
                input_data['trip_duration_days'],
                input_data['miles_traveled'],
                input_data['total_receipts_amount']
            )
            results.append({
                'input': input_data,
                'predicted_output': round(prediction, 2)
            })
        return results
    
    def evaluate_public_cases(self):
        """Evaluate performance on public test cases."""
        with open('../public_cases.json', 'r') as f:
            cases = json.load(f)
        
        errors = []
        for case in cases:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            predicted = self.calculate_reimbursement(days, miles, receipts)
            error = abs(predicted - expected)
            errors.append(error)
        
        mae = sum(errors) / len(errors)
        max_error = max(errors)
        within_10 = sum(1 for e in errors if e <= 10) / len(errors) * 100
        within_25 = sum(1 for e in errors if e <= 25) / len(errors) * 100
        within_50 = sum(1 for e in errors if e <= 50) / len(errors) * 100
        
        print(f"Public Test Results:")
        print(f"  Cases: {len(cases)}")
        print(f"  Mean Absolute Error: ${mae:.2f}")
        print(f"  Maximum Error: ${max_error:.2f}")
        print(f"  Within $10: {within_10:.1f}%")
        print(f"  Within $25: {within_25:.1f}%")
        print(f"  Within $50: {within_50:.1f}%")
        
        return mae

def main():
    """Main function for standalone testing."""
    print("Initializing Submission Engine...")
    engine = SubmissionEngine()
    
    # Evaluate on public cases
    engine.evaluate_public_cases()
    
    # Test with some examples
    print("\nExample calculations:")
    examples = [
        (3, 93, 1.42),
        (5, 200, 400),
        (1, 140, 22.71),
        (8, 600, 1200),
        (1, 1082, 1809.49)  # Problematic case
    ]
    
    for days, miles, receipts in examples:
        result = engine.calculate_reimbursement(days, miles, receipts)
        print(f"  {days}d, {miles}mi, ${receipts:.2f} -> ${result:.2f}")

if __name__ == "__main__":
    main()
