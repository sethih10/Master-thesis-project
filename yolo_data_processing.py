import os
import shutil
import random
from PIL import Image
import numpy as np
import argparse
import sys

# Argument parser for command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input_labels_dir", help="Directory containing subdirectories of pulses with labels for frames of corresponding pulse videos.")
parser.add_argument("input_image_dir", help="Directory containing subdirectories of pulses with frames of corresponding pulse videos.")
parser.add_argument("yolo_data_dir", help="Directory to save the labels and images in the required YOLO format.", default="./yolo")
parser.add_argument("train_ratio", help="The ratio for training images. For example, 0.75 for 75% training and 25% validation.", default=0.75, type=float)

# Print help message if no arguments are provided
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# Function to copy all label files from subdirectories to a single directory
def select_labels_train(video_label_dir, label_train_directory):
    """
    Copies and selects label files for training from subdirectories.
    
    Args:
        video_label_dir (str): Directory containing subdirectories with label files.
        label_train_directory (str): Directory to copy selected label files into.
    """
    if not os.path.exists(label_train_directory):
        os.makedirs(label_train_directory)
    
    ar_train_labels = []
    count_labelled_files = 0
    background_labels_file = []

    for root, dirs, files in os.walk(video_label_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                src_file = os.path.join(root, file)
                if os.path.getsize(src_file) == 0:
                    background_labels_file.append(src_file)
                    continue
                else:
                    ar_train_labels.append(src_file)
                    count_labelled_files += 1
    
    # Randomly select 10% of background files
    index = random.sample(range(len(background_labels_file)), round(count_labelled_files * 0.1))
    randomly_selected_files = [background_labels_file[i] for i in index]
    
    ar_train_labels.extend(randomly_selected_files)
    
    # Copy selected labels to the training directory
    for orig_label_path in ar_train_labels:
        filename = os.path.basename(orig_label_path)
        label_train_path = os.path.join(label_train_directory, filename)
        shutil.copy(orig_label_path, label_train_path)

# Function to convert floating-point 32-bit grayscale images to PNG format
def convert_float32_grayscale_to_png(image_data, filename):
    """
    Converts a floating-point 32-bit grayscale image to a PNG image.
    
    Args:
        image_data (numpy.ndarray): A NumPy array representing the floating-point grayscale image.
        filename (str): Path to save the converted PNG image.
    """
    # Normalize image data to range 0-1
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # Scale data to range 0-255 and convert to unsigned 8-bit integers
    image_data = (image_data * 255).astype(np.uint8)

    # Convert NumPy array to PIL Image object in grayscale mode
    image = Image.fromarray(image_data, mode='L')

    # Save the image as PNG
    image.save(filename, format="PNG")

# Function to select and convert images for training based on selected labels
def select_images_train(label_train_directory, img_directory, img_train_directory):
    """
    Selects images corresponding to the selected labels and converts them to PNG format.
    
    Args:
        label_train_directory (str): Directory containing selected label files.
        img_directory (str): Directory containing TIFF images.
        img_train_directory (str): Directory to save converted PNG images.
    """
    os.makedirs(img_directory, exist_ok=True)
    os.makedirs(img_train_directory, exist_ok=True)
    
    for filename in os.listdir(label_train_directory):
        filename = os.path.splitext(filename)[0] + '.tif'
        if filename.endswith(".tif"):
            pulse_folder_name = os.listdir(img_directory)[0].split('_')[0] + '_' + filename.split('_')[0]
            tiff_path = os.path.join(img_directory, pulse_folder_name, filename)
            png_filename = os.path.join(img_train_directory, os.path.splitext(filename)[0] + ".png")
                
            try:
                with Image.open(tiff_path) as im:
                    convert_float32_grayscale_to_png(im, png_filename)
            except OSError as e:
                print(f"Error: Could not open or save image files for {tiff_path}")
                print(e)

# Function to move files to YOLO formatted directories
def move_files(img_dir, label_dir, file_list, yolo_data_dir, split):
    """
    Moves image and label files to YOLO formatted directories.
    
    Args:
        img_dir (str): Directory containing image files.
        label_dir (str): Directory containing label files.
        file_list (list): List of filenames to move.
        yolo_data_dir (str): Directory to save YOLO formatted data.
        split (str): The data split ('train' or 'val').
    """
    for file in file_list:
        shutil.move(os.path.join(img_dir, file), os.path.join(yolo_data_dir, f'images/{split}', file))
        label_file = file.replace('.png', '.txt')
        shutil.move(os.path.join(label_dir, label_file), os.path.join(yolo_data_dir, f'labels/{split}', label_file))

# Function to split the dataset into training and validation sets and move files to respective directories
def yolo_split(img_dir, label_dir, yolo_data_dir, train_ratio):
    """
    Splits the dataset into training and validation sets and moves files to respective YOLO formatted directories.
    
    Args:
        img_dir (str): Directory containing image files.
        label_dir (str): Directory containing label files.
        yolo_data_dir (str): Directory to save YOLO formatted data.
        train_ratio (float): Ratio of training images to total images.
    """
    val_ratio = 1 - train_ratio

    # Create necessary directories for YOLO format
    for split in ['train', 'val']:
        os.makedirs(os.path.join(yolo_data_dir, f'images/{split}'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_dir, f'labels/{split}'), exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    # Set a seed for reproducibility
    seed = 42
    random.seed(seed)

    # Shuffle the files
    random.shuffle(image_files)

    # Split the files into training and validation sets
    train_split = int(train_ratio * len(image_files))
    train_files = image_files[:train_split]
    val_files = image_files[train_split:]

    # Move files to respective directories
    move_files(img_dir, label_dir, train_files, yolo_data_dir, 'train')
    move_files(img_dir, label_dir, val_files, yolo_data_dir, 'val')

    print("Dataset split completed successfully.")

# Parse command-line arguments
args = parser.parse_args()

label_directory = args.input_labels_dir 
label_train_dir_temp = "./labels_train"  # Temporary directory to save labels 
yolo_data_dir = args.yolo_data_dir
train_ratio = args.train_ratio

# Select and copy labels for training
select_labels_train(label_directory, label_train_dir_temp)

img_directory = args.input_image_dir  
img_train_dir_temp = "./images_train"  # Temporary directory to save images 

# Convert images in the directory
select_images_train(label_train_dir_temp, img_directory, img_train_dir_temp)  # Uncomment for separate output directory
yolo_split(img_train_dir_temp, label_train_dir_temp, yolo_data_dir, train_ratio)

# Remove the temporary directories
shutil.rmtree(label_train_dir_temp)
shutil.rmtree(img_train_dir_temp)
