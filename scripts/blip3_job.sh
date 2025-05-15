#!/bin/bash
#SBATCH --job-name=blip3_eval           # Job name
#SBATCH --output=output_%j.txt      # Output file (%j = job ID)
#SBATCH --error=error_%j.txt        # Error file
#SBATCH --ntasks=1                  # Total number of tasks
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --mem=12G                   # Memory per node
#SBATCH --time=3-00:00:00           # Time limit (hh:mm:ss)
#SBATCH --partition=gpua100     # Partition (queue) name
#SBATCH --gres=gpu:1            # Request GPU (if needed)

# Load modules if needed
# Initialize Conda
source ~/.bashrc  # Important for accessing conda
conda activate vlms_eval

# Your command
python evaluate.py --model_name 'Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5' --device cuda
