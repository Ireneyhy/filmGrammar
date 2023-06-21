import os
import cv2
import numpy as np
import pandas as pd
import fnmatch
from mtcnn import MTCNN
from os.path import isfile


def calculate_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = round(np.std(gray), 3)
    exposure = round(np.mean(gray), 3)
    return contrast, exposure


def get_outputs(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net.forward(output_layers)


def get_face_scale(ratio):
    if ratio < 0.04:
        return "EXTREME LONG SHOT"
    elif ratio < 0.1:
        return "LONG SHOT"
    elif ratio < 0.3:
        return "MEDIUM LONG SHOT"
    elif ratio < 0.6:
        return "MEDIUM CLOSE-UP"
    elif ratio < 0.9:
        return "CLOSE-UP"
    else:
        return "EXTREME CLOSE-UP"


def get_object_scale(ratio):
    if ratio < 0.05:
        return "EXTREME LONG SHOT"
    elif ratio < 0.25:
        return "LONG SHOT"
    elif ratio < 0.35:
        return "MEDIUM LONG SHOT"
    elif ratio < 0.7:
        return "MEDIUM CLOSE-UP"
    elif ratio < 1:
        return "CLOSE-UP"
    else:
        return "EXTREME CLOSE-UP"


def get_figure_scale(ratio):
    if ratio < 0.3:
        return "EXTREME LONG SHOT"
    elif ratio < 0.7:
        return "LONG SHOT"
    else:
        return "MEDIUM LONG SHOT"


def get_face(image, detector):
    h, w, _ = image.shape
    faces = detector.detect_faces(image)
    if faces is not None:
        face = True
        objectClass = "FACE"

    max_area = 0
    shot_scale = "UNKNOWN"
    bounding_box = None
    landmarks = None

    for f in faces:
        (startX, startY, width, height) = f["box"]  # get the bounding box
        face_area = width * height
        ratio = height / h

        if face_area > max_area:
            max_area = face_area
            shot_scale = get_face_scale(ratio)
            bounding_box = (startX, startY, startX + width, startY + height)
            landmarks = f["keypoints"]

    return shot_scale, bounding_box, face, landmarks, objectClass


def get_object(outputs, height, width, classes):
    max_area = 0
    shot_scale = "UNKNOWN"
    bounding_box = None
    object = False

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                object = True
                box = detection[:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - w / 2)
                y = int(centerY - h / 2)

                ratio = h / height

                if classes[class_id] == "person":
                    objectClass = "FIGURE"
                    shot_scale = get_figure_scale(ratio)
                else:
                    objectClass = "OBJECT"
                    shot_scale = get_object_scale(ratio)

                if w * h > max_area:
                    max_area = w * h
                    bounding_box = (x, y, x + w, y + h)

    return shot_scale, bounding_box, object, objectClass


def get_movement(var_magnitudes, var_border_magnitudes, mean_motions_x, mean_motions_y):
    movement = ""
    if var_magnitudes is not None:
        if var_border_magnitudes is None or var_border_magnitudes < 13:
            return "STATIC"
    else:
        if abs(mean_motions_x) > 0.2:
            movement += "PAN"
        if abs(mean_motions_y) > 0.2:
            movement += ", TILT"
    return movement


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

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                color_value = tuple(map(int, color[i]))

                # original vectors
                mask = cv2.line(mask, (a, b), (c, d), color_value, 2)
                old_frame = cv2.circle(old_frame, (a, b), 5, color_value, -1)

            img = cv2.add(old_frame, mask)

            # Calculate motion vectors
            motion_vectors = good_new - good_old
            magnitudes.extend(np.sqrt((motion_vectors**2).sum(-1)))

            if good_old.size > 0:
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

    # Save the result to an image
    cv2.imwrite(os.path.splitext(file)[0] + "_flow.png", img)

    cap.release()

    if magnitudes:
        var_magnitudes = np.var(magnitudes)
    else:
        print("Warning: `magnitudes` is empty.")
        var_magnitudes = np.nan

    # Ensure `border_magnitudes` is not empty before calculating variance
    if border_magnitudes:
        var_border_magnitudes = np.var(border_magnitudes)
    else:
        print("Warning: `border_magnitudes` is empty.")
        var_border_magnitudes = np.nan

    # Ensure `motions_x` is not empty before calculating mean
    if motions_x:
        mean_motions_x = np.mean(motions_x)
    else:
        print("Warning: `motions_x` is empty.")
        mean_motions_x = np.nan

    # Ensure `motions_y` is not empty before calculating mean
    if motions_y:
        mean_motions_y = np.mean(motions_y)
    else:
        print("Warning: `motions_y` is empty.")
        mean_motions_y = np.nan

    return (var_magnitudes, var_border_magnitudes, mean_motions_x, mean_motions_y)


def main():
    directory = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/frames/1"
    detector = MTCNN()
    object_net = cv2.dnn.readNet(
        "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/yolov4.weights",
        "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/yolov4.cfg",
    )
    classes = (
        open("/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/coco.names")
        .read()
        .strip()
        .split("\n")
    )

    data = []

    if not fnmatch.filter(os.listdir(directory), "*.jpg"):
        print("No image files found in the directory.")

    else:
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                filename_without_extension = os.path.splitext(filename)[0].rsplit(
                    "_middle_frame", 1
                )[0]
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path)
                height, width, _ = image.shape
                print(f"Processing {os.path.basename(filename)}")

                shot_scale = "UNKNOWN"
                object = False

                # First try face detection
                shot_scale, bounding_box, face, landmarks, objectClass = get_face(
                    image, detector
                )
                # if shot_scale != "UNKNOWN":
                #     print(f"Shot scale (face detection): {shot_scale}")

                # If face detection did not find anything, use object detection
                if shot_scale == "UNKNOWN":
                    outputs = get_outputs(image, object_net)
                    shot_scale, bounding_box, object, objectClass = get_object(
                        outputs, height, width, classes
                    )
                    # print(f"Shot scale (object detection): {shot_scale}")

                if bounding_box is not None:
                    if face:
                        cv2.rectangle(
                            image,
                            (bounding_box[0], bounding_box[1]),
                            (bounding_box[2], bounding_box[3]),
                            (0, 255, 0),
                            2,
                        )
                        if landmarks is not None:
                            for key, point in landmarks.items():
                                cv2.circle(image, tuple(point), 2, (255, 0, 0), 2)
                    if object:
                        cv2.rectangle(
                            image,
                            (bounding_box[0], bounding_box[1]),
                            (bounding_box[2], bounding_box[3]),
                            (0, 0, 255),
                            2,
                        )

                    cv2.putText(
                        image,
                        shot_scale + " " + objectClass,
                        (bounding_box[0], bounding_box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2,
                    )

                    output_path = (
                        os.path.splitext(image_path)[0]
                        + "_detected"
                        + os.path.splitext(image_path)[1]
                    )
                    cv2.imwrite(output_path, image)

                else:
                    print("Nothing detected.")

                contrast, exposure = calculate_quality(image)
                data.append(
                    [filename_without_extension, contrast, exposure, shot_scale]
                )

    df_image = pd.DataFrame(data, columns=["filename", "contrast", "exposure", "scale"])

    # Get video files
    directory = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/videos/WIP"
    data = []

    if not fnmatch.filter(os.listdir(directory), "*.mp4"):
        print("No video files found in the directory.")

    else:
        for file in os.listdir(directory):
            if file.endswith(".mp4"):
                file_wo_extension = os.path.splitext(file)[0]
                file_path = os.path.join(directory, file)
                print(f"Processing {os.path.basename(file_path)}")

                (
                    var_magnitudes,
                    var_border_magnitudes,
                    mean_motions_x,
                    mean_motions_y,
                ) = track_movement(file_path)

                movement = get_movement(
                    var_magnitudes,
                    var_border_magnitudes,
                    mean_motions_x,
                    mean_motions_y,
                )

                data.append(
                    [
                        file_wo_extension,
                        var_magnitudes,
                        var_border_magnitudes,
                        mean_motions_x,
                        mean_motions_y,
                        movement,
                    ]
                )

    df_video = pd.DataFrame(
        data,
        columns=[
            "filename",
            "variance",
            "border_variance",
            "average_motion_x",
            "average_motion_y",
            "movement",
        ],
    )

    csv = "combined_values.csv"
    new_df = pd.merge(df_image, df_video, on="filename", how="outer")

    if isfile(csv):
        existing_df = pd.read_csv(csv)
        # Iterate over rows in the new DataFrame
        for idx, row in new_df.iterrows():
            # Check if the filename already exists
            if existing_df["filename"].isin([row["filename"]]).any():
                # For each column in the row
                for col in new_df.columns:
                    # round only numeric data
                    if isinstance(row[col], (int, float)):
                        existing_df.loc[
                            existing_df["filename"] == row["filename"], col
                        ] = round(row[col], 3)
                    else:
                        existing_df.loc[
                            existing_df["filename"] == row["filename"], col
                        ] = row[col]
            else:
                # Append the new row to the existing DataFrame
                # Convert row to DataFrame
                row_df = pd.DataFrame(row).T
                # Round only the numeric columns
                row_df = row_df.apply(
                    lambda x: x.round(3)
                    if x.dtypes in ["float64", "float32", "int64", "int32"]
                    else x
                )
                # Convert DataFrame back to Series
                row = row_df.iloc[0]
                existing_df = pd.concat([existing_df, row_df])

        # Write the updated DataFrame back to the csv file
        existing_df.sort_values("filename").to_csv(csv, index=False)

    else:
        new_df = new_df.reindex(
            ["filename"] + list(new_df.columns.drop("filename")), axis=1
        ).sort_values("filename")
        new_df = new_df.round(3)
        new_df.to_csv(csv, index=False)


if __name__ == "__main__":
    main()
