import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import argparse
import h5py
from PIL import Image
import os

from autoencoder import Autoencoder  # Import the Autoencoder model class

# Setup argument parser to accept command-line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("channels", type=int, help="The number of channels for autoencoder architecture")
parser.add_argument("layers", type=int, help="The number of layers in the autoencoder")
parser.add_argument("autoencoder_weights_path", help="The path to the weights of trained autoencoder model")
parser.add_argument("input_file_path", help="The path to the h5 file to be processed")
parser.add_argument("output_file_path", help="The path to which your data is to be saved", default='.')
parser.add_argument("device", help="Optional - Device to be used - cpu or cuda for inference", default='cpu')
parser.add_argument("tif_dir", help="Optional directory to save frames as TIFF files (can be used for labeling)", default='')

# Parse the command-line arguments
args = parser.parse_args()


def normalize(video):
    """
    Normalize each frame in a video to the range [0, 1].

    This function takes a 3D NumPy array representing a video where the first dimension is the frame index,
    and the second and third dimensions are the height and width of the frames, respectively. Each frame
    is normalized independently.

    Parameters:
    -----------
    video : numpy.ndarray
        A 3D array of shape (num_frames, height, width) representing the video frames.

    Returns:
    --------
    numpy.ndarray
        A 3D array of the same shape as the input, where each frame has been normalized to the range [0, 1].
    """
    temp = np.empty_like(video, dtype=np.float64)  # Use float64 for more precise calculations
    for i in range(len(video)):
        frame = video[i]
        temp[i] = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    return temp


def autoencoder_inference(channels, layers, autoencoder_weights_path, input_file_path, output_file_path, device='cpu', tif_dir=''):
    """
    Perform inference using a trained autoencoder on video data stored in an HDF5 file.

    This function loads a trained autoencoder model, processes each video frame by inverting and filtering them,
    and then saves the processed frames back to an HDF5 file. Optionally, the processed frames can also be saved
    as TIFF files for further analysis or labeling.

    Parameters:
    -----------
    channels : int
        The number of channels for the autoencoder architecture.
    layers : int
        The number of layers in the autoencoder model.
    autoencoder_weights_path : str
        The path to the weights of the trained autoencoder model.
    input_file_path : str
        The path to the input HDF5 file containing the video data.
    output_file_path : str
        The path to save the processed video data in an HDF5 file.
    device : str, optional
        The device to be used for inference, either 'cpu' or 'cuda'. Default is 'cpu'.
    tif_dir : str, optional
        The directory to save frames as TIFF files. If empty, TIFF files are not saved. Default is ''.
    """
    
    # Load the trained autoencoder model
    autoencoder_model = Autoencoder(c=channels, layer=layers).to(device)
    autoencoder_model.load_state_dict(torch.load(autoencoder_weights_path, map_location=torch.device(device)))

    # Load the input video data from the HDF5 file
    with h5py.File(input_file_path, 'r') as file:
        video = np.array(file['frame_data'])
        
    # Padding the video so that it can satisfy the requirement of shape (n, 176, 120) where n is the number of frames
    temp = np.zeros((video.shape[0], 176,120))
    temp[:, : video.shape[1], :video.shape[2]] = video
    video = temp

    # Define a transformation pipeline for converting data to PyTorch tensors
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert input data to PyTorch tensor
    ])

    # Normalize the entire video and initialize arrays for different stages of processing
    normalized_video = normalize(video)
    inverted_video = np.zeros(video.shape)
    filtered_video = np.zeros(video.shape)
    processed_video = np.zeros(video.shape)
    processed_normalized = np.zeros(video.shape)
    
    num_frames = len(video)
    print(f"Using device: {device}")
    
    # If the directory to save TIFF files doesn't exist, create it
    if tif_dir:
        os.makedirs(tif_dir, exist_ok=True)
    
    print("Started Running")
    # Perform inference on each frame
    with torch.no_grad():
        for frame_idx in range(num_frames):
            # Invert the normalized frame
            inverted_video[frame_idx] = 1 - normalized_video[frame_idx]
            
            # Apply the transformation to the inverted frame and convert it to a PyTorch tensor
            transformed_frame = transform(inverted_video[frame_idx]).float()
            inverted_tensor = transformed_frame.reshape(1, 1, inverted_video.shape[1], inverted_video.shape[2])
            inverted_tensor = inverted_tensor.to(device)
            
            # Apply the autoencoder model to the inverted tensor
            filtered_tensor = autoencoder_model(inverted_tensor)
            
            # Normalize the filtered tensor to the range [0, 1]
            filtered_tensor_normalized = (filtered_tensor - torch.min(filtered_tensor)) / (torch.max(filtered_tensor) - torch.min(filtered_tensor))
            filtered_video[frame_idx] = filtered_tensor_normalized.cpu().numpy().reshape(video.shape[1], video.shape[2])
            
            # Calculate the difference between the filtered and inverted frames
            processed_video[frame_idx] = filtered_video[frame_idx] - inverted_video[frame_idx]
            
            # Normalize the processed frame
            temp_frame = processed_video[frame_idx]
            temp_frame = (temp_frame - np.min(temp_frame)) / (np.max(temp_frame) - np.min(temp_frame))
            processed_normalized[frame_idx] = temp_frame
            
            # Save the processed frame as a TIFF file if tif_dir is provided
            if tif_dir:
                frame_path = f"{tif_dir}/img_{frame_idx:04d}.tif"
                Image.fromarray((processed_normalized[frame_idx] * 255).astype(np.uint8)).save(frame_path)
                
    
    # Save the processed video data to an HDF5 file
    with h5py.File(output_file_path, 'w') as f:
        f.create_dataset('frame_data', data=processed_normalized)
        
    
    print("Completed")


# Call the inference function with the parsed command-line arguments
autoencoder_inference(
    channels=args.channels, 
    layers=args.layers, 
    autoencoder_weights_path=args.autoencoder_weights_path, 
    input_file_path=args.input_file_path, 
    output_file_path=args.output_file_path, 
    device=args.device,
    tif_dir=args.tif_dir
)
