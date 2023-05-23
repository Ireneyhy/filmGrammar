import os
import cv2
import numpy as np
import pandas as pd


def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return contrast


def get_outputs(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net.forward(output_layers)


def get_shot_scale(height, h):
    ratio = h / height

    if ratio < 0.1:
        return "EXTREME LONG SHOT"
    elif ratio < 0.5:
        return "LONG SHOT"
    elif ratio < 0.75:
        return "MEDIUM LONG SHOT"
    elif ratio < 1.2:
        return "MEDIUM CLOSE-UP"
    elif ratio < 1.3:
        return "CLOSE-UP"
    else:
        return "EXTREME CLOSE-UP"


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
        # Checks if a point is within `border_size` pixels from the border of the frame.
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
        print("image saved to: " + os.path.splitext(file)[0] + ".png")

    cap.release()

    return (
        np.var(magnitudes),
        np.var(border_magnitudes),
        np.mean(motions_x),
        np.mean(motions_y),
    )


def main():
    directory = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/frames"
    net = cv2.dnn.readNet(
        "/Users/heying/Documents/Grad_School/慶應/KMD/CREATO/yolov4.weights",
        "/Users/heying/Documents/Grad_School/慶應/KMD/CREATO/yolov4.cfg",
    )
    classes = (
        open("/Users/heying/Documents/Grad_School/慶應/KMD/CREATO/coco.names")
        .read()
        .strip()
        .split("\n")
    )

    data = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filename_without_extension = os.path.splitext(filename)[0].rsplit(
                "_first_frame", 1
            )[0]
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            outputs = get_outputs(image, net)

            max_area = 0
            shot_scale = "UNKNOWN"
            person_detected = False
            object_detected = False

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        box = detection[:4] * np.array([width, height, width, height])
                        (_, _, w, h) = box.astype("int")

                        # Check if this bounding box is larger than the current largest
                        if w * h > max_area:
                            max_area = w * h
                            if classes[class_id] == "person":
                                person_detected = True
                                shot_scale = get_shot_scale(height, h)
                            else:
                                object_detected = True

            # If no person detected, consider it a close-up or extreme close-up shot
            if not person_detected:
                if object_detected:
                    if max_area / float(height * width) > 0.66:
                        shot_scale = "CLOSE-UP"
                    else:
                        shot_scale = "EXTREME LONG SHOT"
                else:
                    shot_scale = "EXTREME CLOSE-UP"

            contrast = calculate_contrast(image)
            data.append([filename_without_extension, contrast, shot_scale])

    df_image = pd.DataFrame(data, columns=["filename", "contrast", "scale"])
    df_image = df_image.sort_values("filename")

    # Get video files
    directory = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/videos/1"
    data = []
    for file in os.listdir(directory):
        if file.endswith(".mp4"):
            file_wo_extension = os.path.splitext(file)[0]
            file_path = os.path.join(directory, file)
            print(f"Processing {os.path.basename(file_path)}")

            (
                variance,
                border_variance,
                average_motion_x,
                average_motion_y,
            ) = track_movement(file_path)

            data.append(
                [
                    file_wo_extension,
                    variance,
                    border_variance,
                    average_motion_x,
                    average_motion_y,
                ]
            )

    df_video = pd.DataFrame(
        data,
        columns=[
            "file",
            "variance",
            "border_variance",
            "average_motion_x",
            "average_motion_y",
        ],
    )
    df_video = df_video.sort_values("file")
    df_video = df_video.round(3)

    # Merge and save CSV
    merged_df = pd.merge(
        df_image, df_video, left_on="filename", right_on="file", how="outer"
    )
    merged_df = merged_df.drop("file", axis=1)
    merged_df.to_csv("combined_values.csv", index=False)


if __name__ == "__main__":
    main()
