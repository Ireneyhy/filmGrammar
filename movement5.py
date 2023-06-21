import os
import cv2
import numpy as np
import pandas as pd


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

    if p0 is None:
        print(f"No features found in video: {file}. Skipping...")
        return None, None, None, None

    mask = np.zeros_like(old_frame)

    # Store the points
    magnitudes = []
    border_magnitudes = []
    motions_x = []
    motions_y = []
    points = []

    def is_border_point(point, border_size=20):
        """Checks if a point is within `border_size` pixels from the border of the frame."""
        h, w = old_frame.shape[:2]
        x, y = point.ravel()
        return (
            x < border_size
            or x > w - border_size
            or y < border_size
            or y > h - border_size
        )

    is_border_point_vectorized = np.vectorize(is_border_point, signature="(n)->()")

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

            if good_old.size > 0:
                points.append((good_old, good_new))

                # Calculate motion vectors
                motion_vectors = good_new - good_old
                magnitudes.extend(np.sqrt((motion_vectors**2).sum(-1)))

                border_magnitudes.extend(
                    np.sqrt(
                        (motion_vectors[is_border_point_vectorized(good_old)] ** 2).sum(
                            -1
                        )
                    )
                )
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

    cap.release()

    return (
        np.var(magnitudes),
        np.var(border_magnitudes),
        np.mean(motions_x),
        np.mean(motions_y),
    )


# Create or load existing CSV file
csv_filepath = os.path.join(os.getcwd(), "data.csv")

if os.path.isfile(csv_filepath):
    data_df = pd.read_csv(csv_filepath)
else:
    data_df = pd.DataFrame(
        columns=[
            "file",
            "variance",
            "border_variance",
            "average_motion_x",
            "average_motion_y",
        ]
    )

# Get video files
directory = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/videos/1"
for file in os.listdir(directory):
    if file.endswith(".mp4"):
        file_path = os.path.join(directory, file)
        print(f"Processing {file_path}")

        variance, border_variance, average_motion_x, average_motion_y = track_movement(
            file_path
        )
        # print(variance, border_variance, average_motion_x, average_motion_y)

        file = os.path.splitext(file)[0]  # remove extension

        if file in data_df["file"].values:
            data_df.loc[data_df["file"] == file, "border_variance"] = border_variance
            data_df.loc[data_df["file"] == file, "variance"] = variance
            data_df.loc[data_df["file"] == file, "average_motion_x"] = average_motion_x
            data_df.loc[data_df["file"] == file, "average_motion_y"] = average_motion_y

        else:
            data_df = pd.concat(
                [
                    data_df,
                    pd.DataFrame(
                        [
                            {
                                "file": file,
                                "variance": variance,
                                "border_variance": border_variance,
                                "average_motion_x": average_motion_x,
                                "average_motion_y": average_motion_y,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

# Save CSV
data_df = data_df.sort_values("file")
data_df = data_df.round(3)
data_df.to_csv(csv_filepath, index=False)
