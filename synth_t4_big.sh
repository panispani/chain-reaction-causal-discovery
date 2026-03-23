#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Use the virtual environment's Python
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# Check if virtual environment exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Virtual environment not found at ${PYTHON}"
    echo "Please create a virtual environment: python -m venv .venv"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a logs/main.log
}

# Function to run experiment with logging
run_experiment() {
    local step=$1
    local total=$2
    local script=$3
    shift 3
    local args="$@"

    local logfile="logs/${script%.py}_$(date '+%Y%m%d_%H%M%S').log"

    log_message "================================================"
    log_message "Starting experiment $step/$total: $script"
    log_message "Arguments: $args"
    log_message "Output: $logfile"
    log_message "================================================"

    $PYTHON $script $args > "$logfile" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_message "✓ Completed experiment $step/$total: $script (exit code: $exit_code)"
    else
        log_message "✗ Failed experiment $step/$total: $script (exit code: $exit_code)"
    fi

    return $exit_code
}

# Save PID for tracking
echo $$ > logs/run_em.pid
log_message "Started experiments run (PID: $$)"
log_message "synth_t4_big.sh"

# Latest experiments
run_experiment 1 "synth_t4_big.sh" sample_scaling_experiment.py levels/graphs/t4_big.txt \
        --synthetic \
        --synthetic-graph levels/graphs/t4_big.txt \
        --synthetic-probs 0.1 \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 \
        --max-samples 200 \
        --output-dir synthetic_scaling --evaluate-baselines
run_experiment 2 "synth_t4_big.sh" sample_scaling_experiment.py levels/graphs/t4_big.txt \
        --synthetic \
        --synthetic-graph levels/graphs/t4_big.txt \
        --synthetic-probs 0.2 \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 \
        --max-samples 200 \
        --output-dir synthetic_scaling --evaluate-baselines
run_experiment 3 "synth_t4_big.sh" sample_scaling_experiment.py levels/graphs/t4_big.txt \
        --synthetic \
        --synthetic-graph levels/graphs/t4_big.txt \
        --synthetic-probs 0.3 \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 \
        --max-samples 200 \
        --output-dir synthetic_scaling --evaluate-baselines
run_experiment 4 "synth_t4_big.sh" sample_scaling_experiment.py levels/graphs/t4_big.txt \
        --synthetic \
        --synthetic-graph levels/graphs/t4_big.txt \
        --synthetic-probs 0.4 \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 \
        --max-samples 200 \
        --output-dir synthetic_scaling --evaluate-baselines
run_experiment 5 "synth_t4_big.sh" sample_scaling_experiment.py levels/graphs/t4_big.txt \
        --synthetic \
        --synthetic-graph levels/graphs/t4_big.txt \
        --synthetic-probs 0.5 \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 \
        --max-samples 200 \
        --output-dir synthetic_scaling --evaluate-baselines


# Final summary
log_message "================================================"
log_message "All experiments completed!"
log_message "Check logs/ directory for detailed outputs"
log_message "================================================"

# Remove PID file
rm -f logs/run_em.pid

