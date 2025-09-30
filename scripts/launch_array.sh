#!/bin/bash

# Function to handle Ctrl+C and terminate all child processes
cleanup() {
    echo "Caught Ctrl+C! Terminating all jobs..."
    kill 0  # This will send a kill signal to all child processes in the current process group
    exit 1
}
# Trap Ctrl+C signal (SIGINT) and call the cleanup function
trap cleanup SIGINT

N_WORKERS=$1

for ((i=0; i<N_WORKERS; i++))
do
  GPU_ID=$((i % 8))
  echo Running worker $i on $GPU_ID
  # Limit per-process CPU threads to avoid oversubscription when many workers run.
  # Adjust these if you have many free CPU cores per process.
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
  MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
  NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1} \
  TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1} \
  TORCH_INTEROP_THREADS=${TORCH_INTEROP_THREADS:-1} \
  CUDA_VISIBLE_DEVICES=$GPU_ID WORKER_ID=$i N_WORKERS=$N_WORKERS python research/train.py "$2" &
done

wait
echo "All training jobs completed."
