#!/bin/bash

# Generalized parameter sweep script for crosscoder experiments
# Sweeps over LAM_N values and crosscoder hidden dimensions

# Create logs directory if it doesn't exist
mkdir -p logs

# Define parameter arrays - customize these for your experiments
LAM_N_VALUES=(0 200 1000 10_000)
HIDDEN_DIM_VALUES=(3072 1536)

# Base configuration file to use as template
BASE_YAML="gpt2_medium_openweb.yaml"

# Use a single temporary YAML file
YAML_FILE="temp_sweep.yaml"

# Create a master log file for the entire sweep
MASTER_LOG="logs/parameter_sweep_$(date +%Y%m%d_%H%M%S).log"
echo "Starting parameter sweep at $(date)" >> "$MASTER_LOG"
echo "Sweeping LAM_N: ${LAM_N_VALUES[*]}" >> "$MASTER_LOG"
echo "Sweeping HIDDEN_DIM: ${HIDDEN_DIM_VALUES[*]}" >> "$MASTER_LOG"

# Calculate total number of runs
TOTAL_RUNS=$((${#LAM_N_VALUES[@]} * ${#HIDDEN_DIM_VALUES[@]}))
echo "Total runs planned: $TOTAL_RUNS" >> "$MASTER_LOG"

RUN_COUNT=0

# Loop through all parameter combinations
for LAM_N in "${LAM_N_VALUES[@]}"; do
    for HIDDEN_DIM in "${HIDDEN_DIM_VALUES[@]}"; do
        RUN_COUNT=$((RUN_COUNT + 1))
        
        # Create a descriptive experiment name
        EXPERIMENT_NAME="lam${LAM_N}_dim${HIDDEN_DIM}_sweep_gpt2medium_openweb"
        
        echo "[$RUN_COUNT/$TOTAL_RUNS] Setting up run with lam_n=$LAM_N, hidden_dim=$HIDDEN_DIM" >> "$MASTER_LOG"
        
        # Create a unique log file name
        LOG_FILE="logs/run_lam${LAM_N}_dim${HIDDEN_DIM}.log"
        
        # Copy the base YAML to our temporary file
        cp "$BASE_YAML" "$YAML_FILE"
        
        # Use Python to modify the YAML file with both parameters
        python -c "
import yaml
with open('$YAML_FILE', 'r') as f:
    config = yaml.safe_load(f)
11
# Update parameters
config['train']['lam_n'] = $LAM_N
config['crosscoder']['hidden_dim'] = $HIDDEN_DIM
config['experiment_name'] = '$EXPERIMENT_NAME'
config['train']['num_steps'] = 1000


with open('$YAML_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
        
        echo "[$RUN_COUNT/$TOTAL_RUNS] Starting job for lam_n=$LAM_N, hidden_dim=$HIDDEN_DIM at $(date)" >> "$MASTER_LOG"
        
        # Run the training script
        python run.py "$YAML_FILE" >> "$LOG_FILE" 2>&1
        
        EXIT_STATUS=$?
        echo "[$RUN_COUNT/$TOTAL_RUNS] Finished job for lam_n=$LAM_N, hidden_dim=$HIDDEN_DIM at $(date) with exit status $EXIT_STATUS" >> "$MASTER_LOG"
        
        # Log any failures prominently
        if [ $EXIT_STATUS -ne 0 ]; then
            echo "⚠️  FAILED: lam_n=$LAM_N, hidden_dim=$HIDDEN_DIM (exit code: $EXIT_STATUS)" >> "$MASTER_LOG"
        else
            echo "✅ SUCCESS: lam_n=$LAM_N, hidden_dim=$HIDDEN_DIM" >> "$MASTER_LOG"
        fi

        # Clean up GPU memory between runs
        echo "Cleaning up memory after run..." >> "$MASTER_LOG"
        python -c "import gc, torch; torch.cuda.empty_cache(); gc.collect()" 2>/dev/null || true
        
        # Optional: Add a small delay between runs
        sleep 5
        
        # Show progress
        echo "Progress: $RUN_COUNT/$TOTAL_RUNS runs completed" >> "$MASTER_LOG"
    done
done

# Clean up the temporary YAML file at the end
rm -f "$YAML_FILE"

echo "All jobs completed at $(date). Check logs directory for output." >> "$MASTER_LOG"
echo "Final summary: $RUN_COUNT/$TOTAL_RUNS runs completed" >> "$MASTER_LOG"

# Generate a simple summary
echo "" >> "$MASTER_LOG"
echo "=== EXPERIMENT SUMMARY ===" >> "$MASTER_LOG"
SUCCESS_COUNT=$(grep -c "✅ SUCCESS" "$MASTER_LOG" || echo "0")
FAILED_COUNT=$(grep -c "⚠️  FAILED" "$MASTER_LOG" || echo "0")
echo "Successful runs: $SUCCESS_COUNT" >> "$MASTER_LOG"
echo "Failed runs: $FAILED_COUNT" >> "$MASTER_LOG"

if [ $FAILED_COUNT -gt 0 ]; then
    echo "Failed experiments:" >> "$MASTER_LOG"
    FAILED_LINES=$(grep "⚠️  FAILED" "$MASTER_LOG")
    echo "$FAILED_LINES" >> "$MASTER_LOG"
fi

echo "Parameter sweep completed! Check $MASTER_LOG for full details."