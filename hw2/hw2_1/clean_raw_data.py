import cv2
import os
import tarfile
import glob

''' This code cleans the data downloaded from https://www.cs.utexas.edu/users/ml/clamp/videoDescription/

For this script to work, you must have a dataset directory containing the
YouTubeClips.tar file found at that link as well as the AllVideoDescriptions.txt
file. This script will automatically extract the youtube videos, sample every 10
frames from each youtube video, and sync the sentences with the frames in the
data folder.
'''

# Create a folders for the synced data
data_folder = 'data'
clips_folder = os.path.join(data_folder, 'clips')
captions_folder = os.path.join(data_folder, 'captions')
if not os.path.exists(data_folder):
    os.mkdir(data_folder)
    os.mkdir(clips_folder)
    os.mkdir(captions_folder)
    
# Store every 10 frames from each video in a folder with that video name inside
# the data/clips path
videos = glob.glob('raw_dataset/YouTubeClips/*')
videos.sort()

for video_file in videos:
    cap = cv2.VideoCapture(video_file)
    video_name = os.path.basename(video_file)[:-4]
    frames_folder = os.path.join(clips_folder, video_name)
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    
    # loop through each frame and save it if it is a 10th frame
    count = 0
    
    while True:
        if count % 10 == 0:
            is_read, frame = cap.read()
            if not is_read:
                break
            frame_image_file = os.path.join(frames_folder, f'frame_{count}.jpg')
            cv2.imwrite(frame_image_file, frame)
            
        count += 1
        
# Extract captions into one text file in the format:
# video_name caption

captions = []
with open('raw_dataset/AllVideoDescriptions.txt') as f:
    lines = f.readlines()
    
    for line in lines:
        if line[0] == '#':
            continue
        else:
            captions.append(line)
    del captions[0]

# Define the file path for the captions.txt file
captions_file_path = os.path.join(captions_folder, 'captions.txt')    

# Create cleaned captions file
with open(captions_file_path, 'w') as f:
    f.writelines(captions)


        
    