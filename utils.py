import numpy as np
import cv2
import scipy.io
import pickle
import re
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from io import BytesIO
from PIL import Image

#######################################################################
# Function to convert .mat to .pkl files
#######################################################################
def mat2pickle(mat_file,output_picklefile):
    """
    Converts a MATLAB .mat file to a Python pickle file.

    Parameters:
    - mat_file: Path to the input .mat file
    - output_picklefile: Path to the output pickle file
    """
    try:
        # Load the .mat file
        mat_data = scipy.io.loadmat(mat_file)
        
        # Remove MATLAB metadata
        mat_data = {key: value for key, value in mat_data.items() if not key.startswith('__')}
        
        # Write to a pickle file
        with open(output_picklefile, 'wb') as pickle_file:
            pickle.dump(mat_data, pickle_file)
        
        print(f"Data successfully saved to {output_picklefile}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

#######################################################################
# Function to convert a NumPy array to an AVI file
#######################################################################
def numpy_to_avi(video_array, output_filename, fps=30):
    """
    Converts a NumPy array representing a video to an AVI file.
    
    Parameters:
    - video_array: NumPy array of shape [height, width, channels, num_frames]
    - output_filename: Name of the output AVI file
    - fps: Frames per second for the output video
    """
    # Check the shape of the input array
    if len(video_array.shape) == 4:
        height, width, channels, num_frames = video_array.shape
    elif len(video_array.shape) == 3:
        height, width, num_frames = video_array.shape
        channels = 1
    else:
        raise ValueError("The video array must have 3 or 4 dimensions")
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    # Write each frame to the output file
    for i in range(num_frames):
        if channels == 3:
            frame = video_array[:, :, :, i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        elif channels == 1:
            frame = video_array[:, :, i]
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
        else:
            raise ValueError("Unsupported number of channels")
        out.write(frame.astype(np.uint8))
    
    # Release the VideoWriter object
    out.release()
    print(f"Video saved as {output_filename}")

#######################################################################
# Function to extract numeric part and suffix
#######################################################################
def extract_parts(patient_id):
    match = re.match(r'(\D+)(\d+)(\D*)', patient_id)
    if match:
        prefix, number, suffix = match.groups()
        return (prefix, int(number), suffix)
    return (patient_id, 0, '')

#######################################################################
# Function to convert mat struct to dict
#######################################################################
def mat_struct_to_dict(mat_struct):
    """
    Recursively converts a MATLAB struct to a Python dictionary.
    
    Parameters:
    - mat_struct: MATLAB struct loaded from a .mat file
    
    Returns:
    - A Python dictionary representation of the MATLAB struct
    """
    def _check_if_struct(item):
        # Check if the item is a MATLAB struct
        return isinstance(item, scipy.io.matlab.mio5_params.mat_struct)
    
    def _convert(item):
        # Convert MATLAB struct to dictionary
        if _check_if_struct(item):
            return {field: _convert(getattr(item, field)) for field in item._fieldnames}
        elif isinstance(item, list):
            return [_convert(sub_item) for sub_item in item]
        elif isinstance(item, dict):
            return {key: _convert(value) for key, value in item.items()}
        else:
            return item
    
    return _convert(mat_struct)

#######################################################################
# Function to convert .avi video to numpy array
#######################################################################
def video_to_binary_numpy(video_path, threshold=127):
    """
    Reads an AVI video and converts it to a binary NumPy array.
    
    Parameters:
    video_path (str): Path to the input video file.
    threshold (int): Threshold value for binary conversion. Default is 127.
    
    Returns:
    np.ndarray: NumPy array of the binary video frames.
                The shape of the array will be (width, height, num_frames).
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Error opening video file.")
    
    # Read the video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create an empty array to store the binary frames
    video_array = np.empty((frame_width, frame_height, frame_count), dtype=np.uint8)
    
    # Read each frame and store it in the array
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply binary threshold
        _, binary_frame = cv2.threshold(gray_frame, threshold, 1, cv2.THRESH_BINARY)
        # Transpose the frame to match the desired output shape (width, height, frame_idx)
        video_array[:, :, frame_idx] = binary_frame
        frame_idx += 1
    
    # Release the video capture object
    cap.release()
    
    return video_array

#######################################################################
# Function to convert .pkl to .mat files
#######################################################################
def pkl_to_mat(pkl_file_path, mat_file_path):
    # Load the .pkl file
    with open(pkl_file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    # Save the data to a .mat file
    scipy.io.savemat(mat_file_path, {'data': data})


#######################################################################
# compute and display trajectories
#######################################################################
def InterX(L1, L2):
    def D(x, y):
        return (x[:, :-1] - y) * (x[:, 1:] - y)

    hF = np.less_equal

    x1 = L1[0, :].reshape(-1, 1)
    y1 = L1[1, :].reshape(-1, 1)
    x2 = L2[0, :]
    y2 = L2[1, :]
    
    dx1 = np.diff(x1, axis=0)
    dy1 = np.diff(y1, axis=0)
    dx2 = np.diff(x2)
    dy2 = np.diff(y2)

    S1 = dx1 * y1[:-1] - dy1 * x1[:-1]
    S2 = dx2 * y2[:-1] - dy2 * x2[:-1]
    
    C1 = hF(D(np.dot(dx1, y2.reshape(1, -1)) - np.dot(dy1, x2.reshape(1, -1)), S1), 0)
    C2 = hF(D((np.dot(y1, dx2.reshape(1, -1)) - np.dot(x1, dy2.reshape(1, -1))).T, S2.reshape(-1, 1)).T, 0)
    
    i, j = np.where(C1 & C2)
    if i.size == 0:
        return np.zeros((2, 0))
    
    i = i.reshape(-1)
    j = j.reshape(-1)
    
    dx2 = dx2.reshape(-1, 1)
    dy2 = dy2.reshape(-1, 1)
    S2 = S2.reshape(-1, 1)
    
    L = dy2[j] * dx1[i] - dy1[i] * dx2[j]
    valid = L != 0
    
    i = i[valid[0][0]]
    j = j[valid[0][0]]
    L = L[valid[0][0]]

    P = np.unique(np.hstack([(dx2[j] * S1[i] - dx1[i] * S2[j]) / L,
                             (dy2[j] * S1[i] - dy1[i] * S2[j]) / L]), axis=0).T
    return P

def interpolate_data(data):
    # Extract indices and non-empty values
    indices = [i for i, val in enumerate(data) if val.size > 0]
    values = np.array([val for val in data if val.size > 0])

    # Separate x and y values
    x_values = values[:, 0] 
    y_values = values[:, 1]

    # Create interpolation functions
    interp_x = interp1d(indices, x_values, kind='linear', fill_value='extrapolate')
    interp_y = interp1d(indices, y_values, kind='linear', fill_value='extrapolate')

    # Interpolate missing values
    for i in range(len(data)):
        if data[i].size == 0:
            data[i] = np.array([interp_x(i), interp_y(i)])
    
    return data

def save_plot_as_array():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)

def display_array_as_plot(image_array):
    image = Image.fromarray(image_array)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def compute_trajectory(edges,VFL_in,VFR_in,Porc_GAxis, sizeVideo, B_plot):
    trajectory = Trajectory(sizeVideo)
    DataD=[edges[i]['Id'] for i in range(sizeVideo.nrframes)]
    DataV=[edges[i]['Iv'] for i in range(sizeVideo.nrframes)]
    DorsalArr=interpolate_data(DataD)
    VentralArr=interpolate_data(DataV)
    frames=[]

    for i in range(100):
        Dorsal=DorsalArr[i]
        Ventral=VentralArr[i]
        D_Vx = abs(Dorsal[0] - Ventral[0])  # 1 is X axis and 0 is Y axis
        rectP_point_1 = np.arange(0, sizeVideo.col + 1, 10)  # line perpendicular to the glottal axis (x coordinates)
        L = np.sqrt((Dorsal[1] - Ventral[1])**2 + (Dorsal[0] - Ventral[0])**2)  # glottal axis length
        g_100 = Dorsal[0] + L
        Angle_rot = np.arcsin(D_Vx / L)  # angle reference for transform coordinates
        m_g = (Dorsal[1] - Ventral[1]) / (Dorsal[0] - Ventral[0])  # slope of glottal axis
        
        g_y = Ventral[1] + (L * (Porc_GAxis / 100)) * np.cos(Angle_rot)  # Y coordinate over glottal axis
        
        if m_g < 0:
            g_x = Dorsal[0] - (np.tan(Angle_rot) * (g_y - Dorsal[1]))  # X coordinate over glottal axis
        else:
            g_x = Dorsal[0] + (np.tan(Angle_rot) * (g_y - Dorsal[1]))
        
        rectP_m = -1 / m_g  # slope of line perpendicular to glottal axis
        rectP_b = g_y - rectP_m * g_x  # b of the line perpendicular to glottal axis
        rectP_point_2 = (rectP_m * rectP_point_1) + rectP_b  # equation of the line points in y
        
        VF_right = np.array([edges[i]['left'][:, 0], edges[i]['left'][:, 1]])  # the left vocal edge is the right for me and vice versa
        VF_left = np.array([edges[i]['right'][:, 0], edges[i]['right'][:, 1]])
        
        #VFL_in = InterX(np.vstack((rectP_point_1, rectP_point_2)), VF_left)  # intersection between the perpendicular to the axis and the VFs
        #VFR_in = InterX(np.vstack((rectP_point_1, rectP_point_2)), VF_right)
        
        
        if B_plot == 1:
            plt.figure(1)
            plt.plot([Dorsal[1], Ventral[1]], [Dorsal[0], Ventral[0]])
            plt.plot([Dorsal[1], Dorsal[1]], [Dorsal[0], g_100])
            plt.plot(g_y,g_x, '*')
            plt.plot(Dorsal[1], g_x, '*')
            plt.plot(rectP_point_2,rectP_point_1)
            plt.plot(VF_left[1], VF_left[0], 'b')  
            plt.plot(VF_right[1], VF_right[0], 'r')  
            plt.plot(VFL_in[i][0], VFL_in[i][1], 'o', markersize=5, markeredgecolor='b')
            plt.plot(VFR_in[i][0], VFR_in[i][1], 'o', markersize=5, markeredgecolor='r')
            plt.title(f'TRAJECTORY - Frame: {i}')
            plt.ylim([0, sizeVideo.row])
            plt.xlim([0, sizeVideo.col])
            plt.gca().invert_yaxis()
            # Save the current frame as a NumPy array
            frame = save_plot_as_array()
            frames.append(frame)
    
    return np.moveaxis(np.array(frames),[1,2,3,0],[0,1,2,3])

class Trajectory:
    def __init__(self, sizeVideo):
        self.video = [None] * sizeVideo.nrframes

class SizeVideo:
    def __init__(self, nrframes, col, row):
        self.nrframes = nrframes
        self.col = col
        self.row = row


if __name__ == "__main__":

    main_folder='/home/gustavo/Code/Git_workspace/Girafe/Girafe/results/Automatic_Segmentation/InP_method'
    PatientID='patient1'
    with open(os.path.join(main_folder,PatientID,PatientID+'_traj.pkl'), 'rb') as pickle_file:
        trajectory=pickle.load(pickle_file)

    main_folder='/home/gustavo/Code/Git_workspace/Girafe/Girafe/results/Playbacks/InP_method'
    PatientID='patient1'
    with open(os.path.join(main_folder,PatientID,PatientID+'.pkl'), 'rb') as pickle_file:
        playbacks=pickle.load(pickle_file)

    main_folder='/home/gustavo/Code/Git_workspace/Girafe/Girafe/results/Automatic_Segmentation/InP_method'
    PatientID='patient1'
    with open(os.path.join(main_folder,PatientID,PatientID+'_segmentation.pkl'), 'rb') as pickle_file:
        segmentation=pickle.load(pickle_file)

    edges=segmentation['edges']
    Dorsal=playbacks['dorsal']
    Ventral=playbacks['ventral']
    Porc_GAxis=50
    sizeVideo=SizeVideo(segmentation['mask'].shape[-1],segmentation['mask'].shape[0],segmentation['mask'].shape[0])
    B_plot=1
    trajectory=compute_trajectory(edges,Porc_GAxis, sizeVideo, B_plot)