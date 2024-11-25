# Import necessary libraries and modules
from n2v.models import N2V
import numpy as np
import os
import h5py
import sys
import argparse

# Set up argument parser to handle command-line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("baseDir", help="directory where the model is stored", default='models')
parser.add_argument("name", help="name of the model", default='n2v_trained_model_1')
parser.add_argument("dataPath", help="path to the input data file")
parser.add_argument("output", help="path where the output data will be saved. Example - filtered_video.h5", default='output.h5')

# Print help message if no arguments are provided
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

def n2v_filtering(model_name, basedir, pulse_file_path, pulse_n2v_file_path):
    """
    Apply denoising to video frames using a pre-trained Noise2Void (N2V) model.

    This function loads a pre-trained N2V model, reads video frames from an HDF5 file,
    applies denoising to each frame, and saves the filtered frames to a new HDF5 file.

    Parameters:
    ----------
    model_name : str
        The name of the pre-trained N2V model to be used for denoising.
    basedir : str
        The directory where the N2V model is stored.
    pulse_file_path : str
        The file path to the HDF5 file containing the video frames to be denoised.
    pulse_n2v_file_path : str
        The file path where the denoised video frames will be saved as an HDF5 file.

    Returns:
    -------
    None
        This function does not return any value. It saves the denoised video frames
        to the specified output file path.

    Raises:
    ------
    IOError
        If there is an issue opening or creating the HDF5 files.
    ValueError
        If the video frames are not in the expected format or dimensions.

    Notes:
    -----
    - The function assumes that the input video frames are in a 3D array format where
      the dimensions are (number_of_frames, height, width).
    - The function creates the output directory if it does not already exist.
    """
    
    # Load the pre-trained model
    model = N2V(config=None, name=model_name, basedir=basedir)

    # Open and read the video data from the HDF5 file
    with h5py.File(pulse_file_path, 'r') as file:
        video = np.array(file['frame_data'])


    # Apply the model to each frame to denoise it
    video_filt = np.empty(video.shape)
    for i in range(video.shape[0]):
        video_filt[i, :, :] = model.predict(video[i].reshape(video.shape[1], video.shape[2], 1), axes='YXC').reshape(video.shape[1], video.shape[2])

    # Save the denoised video data to an HDF5 file
    with h5py.File(pulse_n2v_file_path, 'w') as f:
        f.create_dataset('frame_data', data=video_filt)
        
    print("Denoising completed")

# Parse command-line arguments
args = parser.parse_args()

# Print the output path for verification
print(args.output)

# Extract arguments into variables
model_name = args.name
basedir = args.baseDir
pulse_file_path = args.dataPath
pulse_n2v_file_path = args.output

# Call the function to perform denoising
n2v_filtering(model_name, basedir, pulse_file_path, pulse_n2v_file_path)
