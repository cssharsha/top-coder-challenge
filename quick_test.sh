#!/bin/bash

# Quick test script with better error handling and timeouts
# Usage: ./quick_test.sh [days] [miles] [receipts]

# Default test case (Case 548)
DAYS=${1:-8}
MILES=${2:-482}
RECEIPTS=${3:-1411.49}
EXPECTED=${4:-631.81}

echo "Quick Model Test"
echo "================"
echo "Input: ${DAYS}d, ${MILES}mi, \$${RECEIPTS}"
echo "Expected: \$${EXPECTED}"
echo ""

# Function to test a model with timeout
test_model() {
    local model_type=$1
    local n_clusters=$2
    local timeout_sec=10
    
    # Set environment
    export MODEL_TYPE="$model_type"
    if [ -n "$n_clusters" ]; then
        export N_CLUSTERS="$n_clusters"
        local desc="$model_type ($n_clusters clusters)"
        local model_file="hybrid_enhanced_clustering_${n_clusters}clusters_model.pkl"
    else
        unset N_CLUSTERS
        local desc="$model_type (optimal)"
        local model_file="${model_type}_model.pkl"
    fi
    
    # Check if model file exists (for hybrid models)
    if [[ "$model_type" == "hybrid_enhanced_clustering" && -n "$n_clusters" ]]; then
        if [[ ! -f "legacy/$model_file" ]]; then
            printf "%-35s %s\n" "$desc" "MODEL NOT FOUND"
            return
        fi
    fi
    
    # Run with timeout
    local result
    result=$(timeout $timeout_sec ./run.sh "$DAYS" "$MILES" "$RECEIPTS" 2>/dev/null)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        local error=$(python3 -c "print(f'{abs(float(\"$result\") - float(\"$EXPECTED\")):.2f}')" 2>/dev/null)
        if [ $? -eq 0 ]; then
            printf "%-35s \$%-8s (error: \$%s)\n" "$desc" "$result" "$error"
        else
            printf "%-35s \$%-8s (error: calc failed)\n" "$desc" "$result"
        fi
    elif [ $exit_code -eq 124 ]; then
        printf "%-35s %s\n" "$desc" "TIMEOUT"
    else
        printf "%-35s %s\n" "$desc" "ERROR"
    fi
}

# Test available models (fastest first)
echo "Testing models (10 second timeout each):"
echo "----------------------------------------"

# Test optimal hybrid model first
test_model "hybrid_enhanced_clustering" ""

# Test a few specific cluster sizes if they exist
for clusters in 3 5 7; do
    test_model "hybrid_enhanced_clustering" "$clusters"
done

# Test other model types
test_model "enhanced_parametric" ""
test_model "advanced" ""

echo ""
echo "Done! If any models show TIMEOUT, they may need more time to load."

# Clean up
unset MODEL_TYPE N_CLUSTERS