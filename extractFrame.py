import os
import cv2


def extract_frame(video_path, output_folder):
    # Create a VideoCapture object
    video = cv2.VideoCapture(video_path)

    # Read the first frame
    success, frame = video.read()

    if success:
        # Create output path
        base_name = os.path.basename(video_path)
        file_name = os.path.splitext(base_name)[0]
        output_path = os.path.join(output_folder, f"{file_name}_frame.jpg")

        # Write the first frame to the output path
        cv2.imwrite(output_path, frame)


def main(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        # Check if the file is a video
        if file_name.endswith(
            (".avi", ".mp4", ".mov", ".mkv")
        ):  # add more formats if needed
            # Construct the full file path
            file_path = os.path.join(input_folder, file_name)

            # Extract the first frame of the video
            extract_frame(file_path, output_folder)


if __name__ == "__main__":
    input_folder = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage"
    output_folder = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/frames"
    main(input_folder, output_folder)
