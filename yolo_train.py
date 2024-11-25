"""
Train YOLO Model with Hyperparameter Tuning

This script trains a YOLO model using the Ultralytics YOLO library and tracks the training process with Weights & Biases (wandb). 
The script accepts command-line arguments to configure the training process, including the configuration file, model name, 
number of epochs, and wandb project details.

Dependencies:
- wandb: for experiment tracking
- ultralytics: for YOLO model training
- argparse: for command-line argument parsing
- sys: for handling command-line arguments

Usage:
    python script.py <config_path> <model_name> <epoch> <wandb_login_key> <project_name> <run_name>
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
    parser.add_argument('wandb_login_key', type=str, help='API key for Wandb.')
    parser.add_argument('project_name', type=str, help='Name of the project in Wandb.')
    parser.add_argument('run_name', type=str, help='Run name for the given configuration in the project.')

    # Print help message if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize Wandb
    wandb.login(key=args.wandb_login_key)
    wandb.init(project=args.project_name, name=args.run_name)

    # Load and train the YOLO model
    model = YOLO(args.model_name)
    model.train(data=args.config_path, epochs=args.epoch, imgsz=224, lr0=0.001)

if __name__ == '__main__':
    main()
