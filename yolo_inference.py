from ultralytics import YOLO
import argparse
import h5py
import torch
import numpy as np

# Argument parser
parser = argparse.ArgumentParser()

parser.add_argument('model_path', help='Path to the trained model')
parser.add_argument('video_path', help='Path to the .h5 file that contains the video frames')
parser.add_argument('output_path', help='Path to save the .h5 file with tracking data')
parser.add_argument('device', help='The device to be used - cpu or cuda')
args = parser.parse_args()

# Load the YOLO model
device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
model = YOLO(args.model_path).to(device)

# Function to run YOLO model on the video frames and save the tracking data
def track_video(input_h5, output_h5):
    """
    Processes video frames from an input HDF5 file and extracts object tracking data
    using the YOLO model. The results are saved into an output HDF5 file.

    Args:
        input_h5 (str): Path to the input .h5 file containing video frames.
        output_h5 (str): Path to the output .h5 file to save tracking data.
    """
    with h5py.File(input_h5, 'r') as h5_file:
        # Load video frames and timestamps from the .h5 file
        video_frames = h5_file['frame_data'][:]  # Dataset containing frames
        time = h5_file['time']  # Timestamps
        num_frames = video_frames.shape[0]
        
    print('Tracking started')
    
    # Prepare a new .h5 file to store frame data, timestamps, and tracking data
    with h5py.File(output_h5, 'w') as h5_out:
        # Save each frame in the root of the .h5 file
        h5_out.create_dataset('frame_data', data=video_frames)
        # Save the timestamps directly
        h5_out.create_dataset('time', data=time)

        # Process each frame for tracking data
        for i, frame in enumerate(video_frames):
            # Convert grayscale (h, w) to RGB (h, w, 3)
            img = np.stack([frame] * 3, axis=-1)  # Convert to RGB

            # Run YOLO model on the frame
            results = model(img, verbose=False)

            # Extract bounding boxes and object IDs (tracking data)
            boxes = results[0].boxes.xywh.cpu().numpy()  # xywh format: (x_center, y_center, width, height)
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            classes = results[0].boxes.cls.cpu().numpy()  # Class IDs

            # Store tracking data 
            timestamp = time[i]
            h5_out.create_dataset(f'track_boxes_{timestamp}', data=boxes)
            h5_out.create_dataset(f'track_confidences_{timestamp}', data=confidences)
            h5_out.create_dataset(f'track_classes_{timestamp}', data=classes)
        print('Tracking completed')

# Call the tracking function
track_video(args.video_path, args.output_path)
