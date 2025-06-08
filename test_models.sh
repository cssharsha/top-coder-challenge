#!/bin/bash

# Convenience script for testing different models with run.sh
# Usage: ./test_models.sh [test_case]
#
# Test cases:
#   edge    - Test all edge cases
#   kevin   - Test Kevin's examples
#   custom  - Custom test case (provide days miles receipts)

# Test case data
declare -a EDGE_CASES=(
    "8 482 1411.49 631.81 'Case 548'"
    "6 204 818.99 628.40 'Case 82'"
    "10 1192 23.47 1157.87 'Case 522'"
    "7 1006 1181.33 2279.82 'Case 149'"
    "2 384 495.49 290.36 'Case 89'"
)

declare -a KEVIN_CASES=(
    "5 200 500 'Kevin sweet spot'"
    "3 93 1.42 'Low receipts'"
    "8 600 1200 'Long expensive'"
    "1 1082 1809.49 'Extreme case'"
    "4 180 99.99 'Receipt ending .99'"
)

# Available models
declare -a MODELS=(
    "hybrid_enhanced_clustering:"
    "hybrid_enhanced_clustering:3"
    "hybrid_enhanced_clustering:5" 
    "hybrid_enhanced_clustering:7"
    "hybrid_enhanced_clustering:9"
    "enhanced_parametric:"
    "clustering_regime:"
    "learned_regime:"
    "advanced:"
)

# Function to test a single case with all models
test_case() {
    local days=$1
    local miles=$2
    local receipts=$3
    local expected=$4
    local description=$5
    
    echo "Testing: $description"
    echo "Input: ${days}d, ${miles}mi, \$${receipts}"
    if [ -n "$expected" ]; then
        echo "Expected: \$${expected}"
    fi
    echo "----------------------------------------"
    
    for model_spec in "${MODELS[@]}"; do
        IFS=':' read -r model_type n_clusters <<< "$model_spec"
        
        # Set environment variables
        export MODEL_TYPE="$model_type"
        if [ -n "$n_clusters" ]; then
            export N_CLUSTERS="$n_clusters"
            model_desc="$model_type ($n_clusters clusters)"
        else
            unset N_CLUSTERS
            model_desc="$model_type (optimal)"
        fi
        
        # Run prediction
        result=$(./run.sh "$days" "$miles" "$receipts" 2>/dev/null)
        if [ $? -eq 0 ]; then
            if [ -n "$expected" ]; then
                error=$(python3 -c "print(f'{abs(float(\"$result\") - float(\"$expected\")):.2f}')")
                printf "%-35s \$%-8s (error: \$%s)\n" "$model_desc" "$result" "$error"
            else
                printf "%-35s \$%s\n" "$model_desc" "$result"
            fi
        else
            printf "%-35s ERROR\n" "$model_desc"
        fi
    done
    echo ""
}

# Function to test all cases in an array
test_case_array() {
    local -n cases=$1
    for case_data in "${cases[@]}"; do
        eval "test_case $case_data"
    done
}

# Main script
case "${1:-edge}" in
    "edge")
        echo "Testing Edge Cases"
        echo "=================="
        echo ""
        test_case_array EDGE_CASES
        ;;
    "kevin")
        echo "Testing Kevin's Examples"
        echo "======================="
        echo ""
        test_case_array KEVIN_CASES
        ;;
    "custom")
        if [ $# -eq 4 ]; then
            test_case "$2" "$3" "$4" "" "Custom test case"
        else
            echo "Usage: $0 custom <days> <miles> <receipts>"
            exit 1
        fi
        ;;
    "quick")
        # Quick test with just one edge case
        echo "Quick Test - Case 548"
        echo "===================="
        echo ""
        test_case 8 482 1411.49 631.81 "Case 548"
        ;;
    *)
        echo "Usage: $0 [test_case]"
        echo ""
        echo "Test cases:"
        echo "  edge    - Test all edge cases (default)"
        echo "  kevin   - Test Kevin's examples"  
        echo "  quick   - Quick test with one edge case"
        echo "  custom <days> <miles> <receipts> - Custom test case"
        echo ""
        echo "Available models:"
        for model_spec in "${MODELS[@]}"; do
            IFS=':' read -r model_type n_clusters <<< "$model_spec"
            if [ -n "$n_clusters" ]; then
                echo "  - $model_type ($n_clusters clusters)"
            else
                echo "  - $model_type (optimal)"
            fi
        done
        ;;
esac

# Clean up environment
unset MODEL_TYPE N_CLUSTERS MODEL_PATH