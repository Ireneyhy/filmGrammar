import os
import subprocess


def extract_shots(video_file, output_dir):
    # Get the base name of the video file
    video_basename = os.path.basename(video_file).split(".")[0]
    timestamps_file = os.path.join(output_dir, f"{video_basename}_timestamps.txt")

    # Generate timestamps
    command = f'ffmpeg -i {video_file} -vf "select=gt(scene\,0.1)" -vsync vfr -f null - 2>&1 | grep pts_time > {timestamps_file}'
    subprocess.call(command, shell=True)

    if not os.path.isfile(timestamps_file):
        print(f"Timestamps file not created: {timestamps_file}")
        return

    with open(timestamps_file, "r") as f:
        lines = f.readlines()

    timestamps = []
    for line in lines:
        split_line = line.split(" ")[-2].split(":")
        if len(split_line) > 1 and split_line[1]:
            try:
                timestamps.append(float(split_line[1]))
            except ValueError:
                print(
                    f"Could not convert string to float: '{split_line[1]}' in line: '{line}'"
                )

    # Print the timestamps
    print(f"Timestamps for shot changes: {timestamps}")

    # Extract shots based on timestamps
    for i in range(1, len(timestamps)):
        output_video_file = os.path.join(output_dir, f"{video_basename}_shot_{i}.mp4")
        command = f"ffmpeg -i {video_file} -ss {timestamps[i-1]} -to {timestamps[i]} -async 1 -c copy {output_video_file}"
        subprocess.call(command, shell=True)
        print(f"Extracted shot {i} to {output_video_file}")


# Call the function
video_file = "/Users/heying/Desktop/test2.mp4"
output_dir = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/videos/TEMP"
extract_shots(video_file, output_dir)
