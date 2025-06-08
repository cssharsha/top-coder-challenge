# Run Script Parameters Guide

This document explains how to use the `run.sh` script with different models and configurations for the reimbursement prediction system.

## Basic Usage

```bash
./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>
```

**Example:**
```bash
./run.sh 5 200 300.50
# Output: 425.30
```

## Configuration Parameters

The script accepts configuration through environment variables:

### MODEL_TYPE
Specifies which model to use for predictions.

**Default:** `learned_regime` (best performing model on edge cases)

**Available Models:**

| Model Type | Description | Performance | Use Case |
|------------|-------------|-------------|----------|
| `learned_regime` | ✅ **Recommended** - Optimized parameters with edge case focus | Best on edge cases ($117 avg error) | Production use |
| `advanced` | Original advanced model with interview insights | Good general performance | Baseline comparison |
| `hybrid_enhanced_clustering` | Clustering + enhanced parametric (requires training) | Variable by cluster count | Experimental/research |
| `enhanced_parametric` | Enhanced correlation and pattern detection | High errors on edge cases | Feature testing |
| `parametric` | Basic parametric model | Moderate performance | Simple baseline |
| `clustering_regime` | Pure clustering approach | Moderate performance | Regime analysis |

### N_CLUSTERS
For `hybrid_enhanced_clustering` models, specifies the number of clusters to use.

**Default:** Optimal (auto-selected during training)

**Available Values:** 3-11 (when cluster-specific models are trained)

### MODEL_PATH
Directly specify a model file path, overriding MODEL_TYPE and N_CLUSTERS.

**Format:** Path to `.pkl` file (relative to `legacy/` directory)

## Usage Examples

### 1. Default Usage (Recommended)
```bash
# Uses learned_regime model (best performance)
./run.sh 8 482 1411.49
```

### 2. Specify Model Type
```bash
# Use advanced model
MODEL_TYPE=advanced ./run.sh 8 482 1411.49

# Use enhanced parametric model
MODEL_TYPE=enhanced_parametric ./run.sh 8 482 1411.49

# Use original parametric model
MODEL_TYPE=parametric ./run.sh 8 482 1411.49
```

### 3. Use Cluster-Specific Models
```bash
# Use 5-cluster hybrid model (when available)
N_CLUSTERS=5 ./run.sh 8 482 1411.49

# Use 7-cluster hybrid model
MODEL_TYPE=hybrid_enhanced_clustering N_CLUSTERS=7 ./run.sh 8 482 1411.49

# Use optimal cluster count (auto-selected)
MODEL_TYPE=hybrid_enhanced_clustering ./run.sh 8 482 1411.49
```

### 4. Use Custom Model File
```bash
# Use specific model file
MODEL_PATH=my_custom_model.pkl ./run.sh 8 482 1411.49

# Use specific cluster model directly
MODEL_PATH=hybrid_enhanced_clustering_6clusters_model.pkl ./run.sh 8 482 1411.49
```

### 5. Batch Testing
```bash
# Test multiple models on same input
for model in learned_regime advanced enhanced_parametric; do
    echo "Testing $model:"
    MODEL_TYPE=$model ./run.sh 8 482 1411.49
done
```

## Available Model Files

Based on current training status:

### ✅ Ready to Use
- `learned_regime_model.pkl` - **Recommended** (best edge case performance)
- `advanced_model.pkl` - Original advanced model
- `enhanced_parametric_model.pkl` - Enhanced parametric model
- `parametric_model.pkl` - Basic parametric model
- `hybrid_enhanced_clustering_model.pkl` - Optimal cluster configuration

### 🔄 Training in Progress
- `hybrid_enhanced_clustering_3clusters_model.pkl`
- `hybrid_enhanced_clustering_4clusters_model.pkl`
- `hybrid_enhanced_clustering_5clusters_model.pkl`
- ... (through 11 clusters)

## Performance Comparison

Based on Edge Case 548 (8 days, 482 miles, $1411.49 → expected $631.81):

| Model | Prediction | Error | Rank |
|-------|------------|-------|------|
| `learned_regime` | $749.11 | $117.30 | 🥇 |
| `advanced` | $1087.31 | $455.50 | 🥈 |
| `hybrid_enhanced_clustering` | $1620.63 | $988.82 | 🥉 |
| `enhanced_parametric` | $1782.06 | $1150.25 | 4th |
| `parametric` | $1602.85 | $971.04 | 5th |

## Testing Scripts

### Quick Test Single Case
```bash
./test_available_models.sh
# Tests all available models on Edge Case 548
```

### Test Custom Case
```bash
./test_available_models.sh 5 200 300 425
# Tests all models on custom input with expected output
```

### Evaluate All Cluster Models (when ready)
```bash
cd legacy
python3 final_submission_engine.py --evaluate-clusters
```

## Error Handling

### Model Not Found
If a specified model doesn't exist, the script falls back to the `advanced` model:
```bash
MODEL_TYPE=nonexistent_model ./run.sh 5 200 300
# Will use advanced model instead
```

### Invalid N_CLUSTERS
If N_CLUSTERS is invalid or the specific cluster model doesn't exist:
```bash
N_CLUSTERS=99 ./run.sh 5 200 300
# Will use optimal cluster count instead
```

### Timeout Issues
If a model takes too long to load, use a simpler model:
```bash
# If hybrid models timeout, use faster alternatives
MODEL_TYPE=learned_regime ./run.sh 5 200 300
MODEL_TYPE=advanced ./run.sh 5 200 300
```

## Best Practices

### For Production Use
```bash
# Use the best performing model (default)
./run.sh <days> <miles> <receipts>
```

### For Development/Testing
```bash
# Compare multiple models
for model in learned_regime advanced parametric; do
    echo "$model: $(MODEL_TYPE=$model ./run.sh 8 482 1411.49)"
done
```

### For Research/Experimentation
```bash
# Test different cluster configurations (when available)
for clusters in 3 5 7 9; do
    echo "$clusters clusters: $(N_CLUSTERS=$clusters ./run.sh 8 482 1411.49)"
done
```

## Environment Variable Precedence

1. **MODEL_PATH** (highest priority) - Direct file path
2. **MODEL_TYPE + N_CLUSTERS** - Model type with cluster specification
3. **MODEL_TYPE** - Model type with optimal settings
4. **Default** - `learned_regime` model

## Notes

- All model files are located in the `legacy/` directory
- The script automatically handles model loading and fallbacks
- Output is always formatted to 2 decimal places
- Invalid inputs will cause the script to exit with code 1
- Use the `learned_regime` model for best edge case performance
- Cluster-specific models will become available as training progresses