% Function: saveYOLOLabels
% Purpose: Convert object detection labels from MATLAB format to YOLO format and save them as text files.
%
% Input Parameters:
%   - matFilePath: String. Path to the MATLAB file (.mat) containing labeled data.
%   - outputDir: String. Directory to save the YOLO-formatted label files.
%   - saveBothClasses (optional): Boolean. If true, save both dark and bright island labels.
%                                 If false(default), save only dark island labels.
%
% Details:
% - Image is represented as (width, height).
% - MATLAB labels: [x_vert, y_vert, width, height] where (x_vert, y_vert) are top-left corner vertices.
% - Classes in MATLAB labes : dark_islands, small_dark_islands,
%   mirror_dark, small_bright_islands, mirror_bright
% - YOLO labels: [x_cent_norm, y_cent_norm, width_norm, height_norm] where values are normalized and x_cent_norm and y_cent_norm represents normalized center coordinates of rectangle
% - Classes needed:
%   - Class id 0: All dark islands (dark_islands, small_dark_islands, mirror_dark).
%   - Class id 1: All bright islands (small_bright_islands, mirror_bright).
%
% Workflow:
% 1. Load the labeled data from the MATLAB file.
% 2. Ensure the output directory exists.
% 3. Extract source images and corresponding labels.
% 4. For each image:
%    a. Extract labels for different island types.
%    b. Create a unique filename for each source image.
%    c. Open a text file and write YOLO-formatted labels.
%    d. Close the file.

function saveYOLOLabels(matFilePath, outputDir, saveBothClasses)
    if nargin < 3
        saveBothClasses = false;
    end

    % Load the table
    newData = load(matFilePath, 'gTruth');
    newData = newData.gTruth;

    % Create the output directory if it does not exist
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Extract the source column
    sources = newData.DataSource.Source;

    % Data containing labels of the object
    obj_data = newData.LabelData;

    % Loop through each row of the table
    for i = 1:height(sources)
        % Extract source and labels for the current row
        source = sources{i};
        dark_islands = obj_data.dark_islands{i};
        small_dark_islands = obj_data.small_dark_islands{i};
        mirror_dark = obj_data.mirror_dark{i};
        small_bright_islands = obj_data.small_bright_islands{i};
        mirror_bright = obj_data.mirror_bright{i};

        % Generate a unique file name for each source
        [~, sourceName, ~] = fileparts(source);
        outputFileName = fullfile(outputDir, [sourceName, '.txt']);

        % Open the file for writing
        fid = fopen(outputFileName, 'w');

        % Helper function to write labels
        writeLabels(fid, dark_islands, 0);
        writeLabels(fid, small_dark_islands, 0);
        writeLabels(fid, mirror_dark, 0);
        
        % Optionally write bright island labels
        if saveBothClasses
            writeLabels(fid, small_bright_islands, 1);
            writeLabels(fid, mirror_bright, 1);
        end

        % Close the file
        fclose(fid);

        disp(['Labels saved to ', outputFileName]);
    end
end

function writeLabels(fid, labels, classId)
    [num_rows, ~] = size(labels);

    % Converting the labels containing vertices to center coordinates and then
    % Normalizing the labels
    for i = 1:num_rows
        labels(i,[1]) = labels(i,[1]) + labels(i,[3])/2;
        labels(i,[2]) = labels(i,[2]) + labels(i,[4])/2;
        labels(i,[1,3]) = labels(i,[1,3]) / 120;
        labels(i,[2,4]) = labels(i,[2,4]) / 176;
        label_string = sprintf('%d ', labels(i,:)); % Format row as a string
        fprintf(fid, '%d %s\n', classId, label_string);
    end
end
