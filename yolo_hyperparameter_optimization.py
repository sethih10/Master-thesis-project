"""
Train YOLO Model with Hyperparameter Tuning

This script trains a YOLO model using the Ultralytics YOLO library and tracks the training process with Weights & Biases (wandb). 
It performs hyperparameter tuning with the specified number of iterations.

Dependencies:
- wandb: for experiment tracking
- ultralytics: for YOLO model training
- argparse: for command-line argument parsing
- sys: for handling command-line arguments

Usage:
    python script.py <config_path> <model_name> <epoch> <iteration>
"""

import wandb
from ultralytics import YOLO
import argparse
import sys

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Contains the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Script to train YOLO model with hyperparameter tuning.')
    
    parser.add_argument('config_path', type=str, help='The path to the yml configuration file.')
    parser.add_argument('model_name', type=str, help='Name of the YOLO model to be used. For example - yolov10x.')
    parser.add_argument('epoch', type=int, help='Number of epochs for training.')
    parser.add_argument('iteration', type=int, help='Number of iterations for hyperparameter tuning.')

    # Print help message if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize Wandb
    wandb.login()  # Assuming wandb API key is configured in the environment
    wandb.init(project='your_project_name', name='your_run_name')

    # Load and tune the YOLO model
    model = YOLO(args.model_name)
    model.tune(
        data=args.config_path, 
        epochs=args.epoch, 
        iterations=args.iteration, 
        optimizer='AdamW'
    )

if __name__ == '__main__':
    main()
