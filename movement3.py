import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import variance

directory = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/videos/1"

# Create a dataframe for storing data
try:
    data_df = pd.read_csv("motion_data.csv")
except FileNotFoundError:
    data_df = pd.DataFrame(
        columns=[
            "file",
            "movement",
            "border_movement",
            "variance",
            "average_motion_x",
            "average_motion_y",
        ]
    )


# Define the function to track camera movement
def track_movement(file):
    cap = cv2.VideoCapture(file)

    # Params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    color = np.random.randint(0, 255, (100, 3))

    # Take first frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Find corners in the first frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if p0 is not None and len(p0) > 0:
        mask = np.zeros_like(old_frame)

        # Store the points
        magnitudes = []
        motions_x = []
        motions_y = []
        points = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )

            # Select good points if p1 is not None
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                points.append((good_old, good_new))

                # Calculate motion vectors
                motion_vectors = good_new - good_old
                magnitudes.extend(np.sqrt((motion_vectors**2).sum(-1)))
                motions_x.extend(motion_vectors[:, 0])
                motions_y.extend(motion_vectors[:, 1])

                # Update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

            # Draw all the lines on the first frame
            for i, (good_old, good_new) in enumerate(points):
                for new, old in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b, c, d = map(np.intp, [a, b, c, d])
                    mask = cv2.line(mask, (a, b), (c, d), color[i % 100].tolist(), 2)
            img = cv2.add(old_frame, mask)

            # Save the result to an image
            cv2.imwrite(
                os.path.splitext(file)[0] + ".png",
                img,
            )
    else:
        print(f"No good features to track found in file: {file}")
        return None, None, None

    cap.release()

    return (
        np.round(np.var(magnitudes), 3),
        np.round(np.mean(motions_x), 3),
        np.round(np.mean(motions_y), 3),
    )


# Process all videos in the directory
for file in os.listdir(directory):
    file = os.path.join(directory, file)

    if not file.endswith(".mp4"):
        continue

    print(f"Processing video: {os.path.basename(file)}")

    # Check the video border to detect static shot
    cap = cv2.VideoCapture(file)
    _, first_frame = cap.read()

    # Define the border by taking the first and last 5% of the frames
    border_size = int(0.05 * min(first_frame.shape[0], first_frame.shape[1]))
    border = np.concatenate(
        (first_frame[:border_size], first_frame[-border_size:]), axis=0
    )
    border_gray = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)

    _, next_frame = cap.read()
    while next_frame is not None:
        next_border = np.concatenate(
            (next_frame[:border_size], next_frame[-border_size:]), axis=0
        )
        next_border_gray = cv2.cvtColor(next_border, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between consecutive frames
        diff = cv2.absdiff(border_gray, next_border_gray)
        movement = (diff > 25).sum() / diff.size

        if movement > 0.1:
            break

        border_gray = next_border_gray
        _, next_frame = cap.read()

    cap.release()

    filename = os.path.splitext(os.path.basename(file))[0]

    if movement < 0.1:
        if filename in data_df["file"].values:
            data_df.loc[data_df["file"] == filename, "movement"] = "STATIC"
            data_df.loc[data_df["file"] == filename, "variance"] = ""
            data_df.loc[data_df["file"] == filename, "average_motion_x"] = ""
            data_df.loc[data_df["file"] == filename, "average_motion_y"] = ""

        else:
            data_df = pd.concat(
                [
                    data_df,
                    pd.DataFrame(
                        [
                            {
                                "file": filename,
                                "movement": "STATIC",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        continue

    variance, average_motion_x, average_motion_y = track_movement(file)

    row = pd.DataFrame(
        [
            {
                "file": filename,
                "movement": "MOVING",
                "variance": variance,
                "average_motion_x": average_motion_x,
                "average_motion_y": average_motion_y,
            }
        ]
    )

    if filename in data_df["file"].values:
        data_df.loc[data_df["file"] == filename, "movement"] = "MOVING"
        data_df.loc[data_df["file"] == filename, "variance"] = variance
        data_df.loc[data_df["file"] == filename, "average_motion_x"] = average_motion_x
        data_df.loc[data_df["file"] == filename, "average_motion_y"] = average_motion_y

    else:
        data_df = pd.concat([data_df, row], ignore_index=True)

# Save to csv
data_df = data_df.sort_values("file")
data_df.to_csv("motion_data.csv", index=False)
