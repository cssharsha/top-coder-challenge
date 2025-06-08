#!/bin/bash

# Test script for currently available models
# Usage: ./test_available_models.sh [days] [miles] [receipts] [expected]

# Default test case (Case 548)
DAYS=${1:-8}
MILES=${2:-482}
RECEIPTS=${3:-1411.49}
EXPECTED=${4:-631.81}

echo "Available Model Test"
echo "==================="
echo "Input: ${DAYS}d, ${MILES}mi, \$${RECEIPTS}"
if [ -n "$EXPECTED" ]; then
    echo "Expected: \$${EXPECTED}"
fi
echo ""

# Function to test a model with timeout
test_model() {
    local model_type=$1
    local desc=$2
    local timeout_sec=15
    
    export MODEL_TYPE="$model_type"
    unset N_CLUSTERS MODEL_PATH
    
    echo -n "Testing $desc... "
    
    local result
    result=$(timeout $timeout_sec ./run.sh "$DAYS" "$MILES" "$RECEIPTS" 2>/dev/null)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        if [ -n "$EXPECTED" ]; then
            local error=$(python3 -c "print(f'{abs(float(\"$result\") - float(\"$EXPECTED\")):.2f}')" 2>/dev/null)
            echo "\$${result} (error: \$${error})"
        else
            echo "\$${result}"
        fi
    elif [ $exit_code -eq 124 ]; then
        echo "TIMEOUT"
    else
        echo "ERROR"
    fi
}

# Test available models based on existing files
echo "Testing available models:"
echo "------------------------"

if [ -f "legacy/hybrid_enhanced_clustering_model.pkl" ]; then
    test_model "hybrid_enhanced_clustering" "Hybrid Enhanced Clustering (optimal)"
fi

if [ -f "legacy/enhanced_parametric_model.pkl" ]; then
    test_model "enhanced_parametric" "Enhanced Parametric"
fi

if [ -f "legacy/learned_regime_model.pkl" ]; then
    test_model "learned_regime" "Learned Regime"
fi

if [ -f "legacy/advanced_model.pkl" ]; then
    test_model "advanced" "Advanced"
fi

if [ -f "legacy/enhanced_model.pkl" ]; then
    test_model "enhanced" "Enhanced"
fi

if [ -f "legacy/parametric_model.pkl" ]; then
    test_model "parametric" "Parametric"
fi

echo ""
echo "Checking for cluster-specific models..."

# Check for cluster-specific models
cluster_models_found=0
for clusters in {3..11}; do
    if [ -f "legacy/hybrid_enhanced_clustering_${clusters}clusters_model.pkl" ]; then
        echo "Found: ${clusters}-cluster model"
        cluster_models_found=1
        
        # Test this cluster model
        export MODEL_TYPE="hybrid_enhanced_clustering"
        export N_CLUSTERS="$clusters"
        echo -n "  Testing ${clusters} clusters... "
        
        result=$(timeout 15 ./run.sh "$DAYS" "$MILES" "$RECEIPTS" 2>/dev/null)
        if [ $? -eq 0 ]; then
            if [ -n "$EXPECTED" ]; then
                error=$(python3 -c "print(f'{abs(float(\"$result\") - float(\"$EXPECTED\")):.2f}')" 2>/dev/null)
                echo "\$${result} (error: \$${error})"
            else
                echo "\$${result}"
            fi
        else
            echo "ERROR"
        fi
    fi
done

if [ $cluster_models_found -eq 0 ]; then
    echo "No cluster-specific models found yet (training may still be in progress)"
fi

echo ""
echo "Note: You can test specific models with:"
echo "  MODEL_TYPE=enhanced_parametric ./run.sh $DAYS $MILES $RECEIPTS"
echo "  N_CLUSTERS=5 ./run.sh $DAYS $MILES $RECEIPTS  (when cluster models are ready)"

# Clean up
unset MODEL_TYPE N_CLUSTERS MODEL_PATH