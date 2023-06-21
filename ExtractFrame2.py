import os
import subprocess


def extract_middle_frame(video_file, output_dir):
    # Get duration of the video
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_file}"'
    duration = float(subprocess.check_output(cmd, shell=True).decode("utf-8").strip())

    # Calculate the timestamp for the middle frame
    middle_timestamp = duration / 2

    # Extract the middle frame
    base_name = os.path.basename(video_file).split(".")[0]
    output_file = os.path.join(output_dir, f"{base_name}.jpg")
    cmd = f'ffmpeg -ss {middle_timestamp} -i "{video_file}" -vframes 1 "{output_file}"'
    subprocess.call(cmd, shell=True)


def batch_extract(video_dir, output_dir):
    # Get a list of all video files in the directory
    video_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith((".mp4", ".avi", ".mkv", ".mov"))
    ]

    # Extract the middle frame from each video
    for video_file in video_files:
        extract_middle_frame(video_file, output_dir)


# Specify the input directory containing the videos and the output directory to save the frames
input_directory = (
    "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/videos/DragonInn"
)
output_directory = (
    "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/frames/DragonInn"
)
batch_extract(input_directory, output_directory)
