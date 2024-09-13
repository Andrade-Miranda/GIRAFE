clear all;
close all;


disp('OF PLAYBACKS');
    disp(' ')
    disp('------------------------------------------------------------')
    disp(' ')
%% Read Segmented video    
% Specify the video file name
videoFile = 'patient1_inpa.avi';

% Create a VideoReader object to read video frames
vidObj = VideoReader(videoFile);

% Get video properties
numFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height;
vidWidth = vidObj.Width;

% Initialize a matrix to hold all frames
% Assuming video is grayscale; use 3 in place of 1 for color videos
videoFrames = zeros(vidHeight, vidWidth, numFrames);

% Read one frame at a time and store it
for k = 1:numFrames
    % Read the k-th frame from the video
    frame = read(vidObj, k);
    
    % If the video is color, convert it to grayscale
    frame = rgb2gray(frame); %Uncomment this if needed
    
    % Store the frame in the matrix
    mask(:, :, k) = frame;
end

%% Playbacks
[lrpvg,lrpvgColor,glpvg,glpvgColor, maxveloc, contours, edges, delta, signdelta,signdeltagl, To, sizepvg, D, V] = getdifpvg(mask,256);
[GAW]=glottal_area(mask);
[GAW]=Normalizacion_imagen(GAW,1);
dorsal=D;
ventral=V;
 

%% Trajectory
sizeVideo.row=vidWidth;
sizeVideo.col=vidHeight;
sizeVideo.nrframes=numFrames;
Inp=uint8(zeros(sizeVideo.row,sizeVideo.col,sizeVideo.nrframes));
[Trajectory_50]=compute_trajectory(edges,D,V,50,sizeVideo);

VFL_dis=Trajectory_50.VFL;
VFR_dis=Trajectory_50.VFR;
VFL_Point=Trajectory_50.VFL_in;
VFR_Point=Trajectory_50.VFR_in;
glottal_center=Trajectory_50.g;
line=Trajectory_50.LineP;
Video_traj=Trajectory_50.video;
    
    
%% save 
segmentation_results ='segmentation.mat'; % path
playbacks_results ='playbacks.mat'; % path
trajectory_results ='trajectory.mat'; % path
save(segmentation_results,'mask','contours','edges');
save(playbacks_results,'lrpvg','lrpvgColor','glpvg','glpvgColor','dorsal','ventral');
save(trajectory_results,'VFL_dis','VFR_dis','VFL_Point','VFR_Point',...
           'glottal_center','line','Video_traj');

