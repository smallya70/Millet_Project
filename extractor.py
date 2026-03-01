import cv2
import os

def extract_frames(video_file, output_dir, skip_frames=10):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    saved_count = 0
    
    print(f"Opening {video_file} and extracting frames...")
    
    while True:
        success, frame = cap.read()
        if not success:
            break # Video has ended
            
        # Save one image every 'skip_frames'
        if frame_count % skip_frames == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"Success! Extracted {saved_count} images into the '{output_dir}' folder.")

# --- RUN THE SCRIPT ---
# Make sure this matches the exact name of your video file
VIDEO_NAME = "Contaminated_Grain.MOV" 
OUTPUT_FOLDER = "CONTAMINATED_GRAIN_FRAMES"

# skip_frames=10 means it saves 1 out of every 10 frames.
# A standard phone records at 30 frames per second, so this will save 3 images per second.
extract_frames(VIDEO_NAME, OUTPUT_FOLDER, skip_frames=10)

