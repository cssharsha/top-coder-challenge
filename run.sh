#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>
#
# Optional Environment Variables:
#   MODEL_TYPE     - Model type (default: hybrid_enhanced_clustering)
#   N_CLUSTERS     - Number of clusters for hybrid models (default: optimal)
#   MODEL_PATH     - Specific model file path (overrides MODEL_TYPE and N_CLUSTERS)
#
# Examples:
#   ./run.sh 5 200 300                                    # Use default optimal model
#   MODEL_TYPE=enhanced_parametric ./run.sh 5 200 300    # Use enhanced parametric model
#   N_CLUSTERS=5 ./run.sh 5 200 300                      # Use 5-cluster hybrid model
#   MODEL_PATH=my_model.pkl ./run.sh 5 200 300           # Use specific model file

# Set defaults
MODEL_TYPE=${MODEL_TYPE:-enhanced_parametric}
N_CLUSTERS=${N_CLUSTERS:-}
MODEL_PATH=${MODEL_PATH:-}

# Python implementation using configurable model
cd "$(dirname "$0")/legacy"
python3 -c "
import sys
import os
from final_submission_engine import FinalSubmissionEngine

if len(sys.argv) != 4:
    sys.exit(1)

# Get configuration from environment
model_type = os.environ.get('MODEL_TYPE', 'hybrid_enhanced_clustering')
n_clusters_str = os.environ.get('N_CLUSTERS', '')
model_path = os.environ.get('MODEL_PATH', '')

# Parse n_clusters if provided
n_clusters = None
if n_clusters_str:
    try:
        n_clusters = int(n_clusters_str)
    except ValueError:
        pass

# Create engine with specified configuration
if model_path:
    engine = FinalSubmissionEngine(model_path=model_path)
elif n_clusters:
    engine = FinalSubmissionEngine(model_type=model_type, n_clusters=n_clusters)
else:
    engine = FinalSubmissionEngine(model_type=model_type)

result = engine.calculate_reimbursement(
    int(sys.argv[1]),
    float(sys.argv[2]),
    float(sys.argv[3])
)
print(f'{result:.2f}')
" "$1" "$2" "$3"
