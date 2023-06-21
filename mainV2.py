import os
import cv2
import numpy as np
import pandas as pd
import fnmatch
from mtcnn import MTCNN
from os.path import isfile
from matplotlib import pyplot as plt


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
    elif ratio < 0.23:
        return "MEDIUM LONG SHOT"
    elif ratio < 0.46:
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
    if ratio < 0.25:
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
    objectClass = ""
    object = False
    person_detected = False

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
                    if (w * h > max_area) or (not person_detected):
                        objectClass = "FIGURE"
                        shot_scale = get_figure_scale(ratio)
                        max_area = w * h
                        bounding_box = (x, y, x + w, y + h)
                        person_detected = True
                else:
                    if not person_detected and (w * h > max_area):
                        objectClass = "OBJECT"
                        shot_scale = get_object_scale(ratio)
                        max_area = w * h
                        bounding_box = (x, y, x + w, y + h)

    return shot_scale, bounding_box, object, objectClass


def calculate_mean_or_nan(data):
    return np.mean(data) if data else np.nan


def calculate_variance_or_nan(data):
    if isinstance(data, (list, np.ndarray)):
        if len(data) < 2:
            print(
                "Notice: Data has less than two elements. Assigning np.nan for variance"
            )
            return np.nan
        else:
            return np.var(data)
    else:
        print("Unexpected data type for variance calculation.")
        return np.nan


def get_movement(
    var_magnitudes,
    var_border_magnitudes,
    mean_motions_x,
    mean_motions_y,
    var_orb_magnitudes,
    mean_orb_motions_x,
    mean_orb_motions_y,
):
    if var_magnitudes is None:
        return ""
    movement = ""
    if not np.isnan(var_magnitudes):
        if np.isnan(var_border_magnitudes) or var_border_magnitudes < 13:
            movement = "STATIC"
            return movement
        elif var_orb_magnitudes > 10000000:
            movement = "HANDHELD"
        if abs(mean_motions_x) > 0.2:
            movement += "PAN"
        if abs(mean_motions_y) > 0.2:
            movement += ", TILT"
    elif not np.isnan(var_orb_magnitudes):
        if var_orb_magnitudes > 10000000:
            movement = "HANDHELD"
        if abs(mean_orb_motions_x) > 0.2:
            movement += ", PAN"
        if abs(mean_orb_motions_y) > 0.2:
            movement += ", TILT"
    return movement


def track_movement(file):
    cap = cv2.VideoCapture(file)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Take first frame
    ret, old_frame = cap.read()
    initial_frame = old_frame.copy()
    ransac_frame = old_frame.copy()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Find corners in the first frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if p0 is None:
        print(f"No features found in video: {file}. Skipping...")
        return None, None, None, None, None, None, None, None, None, None, None, None

    color = np.random.randint(0, 255, (100, 3))

    # Initialize Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints in the first frame
    kp1, des1 = orb.detectAndCompute(old_gray, None)

    mask = np.zeros_like(old_frame)
    t_mask = np.zeros_like(old_frame)
    orb_mask = np.zeros_like(old_frame)

    img = np.zeros_like(old_frame)
    t_img = np.zeros_like(old_frame)
    img_ransac = np.zeros_like(old_frame)

    # Store the points
    magnitudes = []
    border_magnitudes = []
    motions_x = []
    motions_y = []

    t_magnitudes = []
    t_border_magnitudes = []
    t_motions_x = []
    t_motions_y = []

    orb_motions_x = []
    orb_motions_y = []
    orb_magnitudes = []
    orb_border_magnitudes = []

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

        # Detect keypoints in the next frame
        kp2, des2 = orb.detectAndCompute(frame_gray, None)

        if des1 is not None and des2 is not None:
            if des1.dtype == des2.dtype and des1.shape[1] == des2.shape[1]:
                # Match descriptors
                matches = bf.match(des1, des2)
            else:
                print("Descriptor mismatch. Skipping this iteration...")
                continue
        else:
            print("One of the descriptors is None. Skipping this iteration...")
            continue

        matches = sorted(matches, key=lambda x: x.distance)

        # Extract the matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate the fundamental matrix
        F, orb_mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

        if orb_mask is None:
            print(
                "Warning: cv2.findFundamentalMat() returned None as orb_mask. Skipping this iteration..."
            )
            continue

        # We select only inlier points
        src_pts = src_pts[orb_mask.ravel() == 1]
        dst_pts = dst_pts[orb_mask.ravel() == 1]

        # Calculation for ORB
        orb_motion_vectors = dst_pts.squeeze() - src_pts.squeeze()

        if len(orb_motion_vectors > 1):
            orb_motions_x.extend(orb_motion_vectors[:, 0])
            orb_motions_y.extend(orb_motion_vectors[:, 1])
            orb_magnitudes.extend(np.var(orb_motion_vectors, axis=1))
            orb_border_motions = orb_motion_vectors[
                is_border_point_vectorized(src_pts.squeeze())
            ]

            if len(orb_border_motions) > 0:
                orb_border_magnitudes.extend(np.sqrt((orb_border_motions**2).sum(-1)))
            else:
                orb_border_magnitudes.append(np.nan)
        else:
            print(
                "Notice: Insufficient orb_motion_vectors. Assigning np.nan for ORB calculations"
            )
            orb_magnitudes = (
                orb_border_magnitudes
            ) = orb_motions_x = orb_motions_y = np.nan

        # Draw the motion vectors
        for i in range(len(src_pts)):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            pt1 = tuple(np.intp(src_pts[i][0]))
            pt2 = tuple(np.intp(dst_pts[i][0]))
            orb_mask = cv2.line(orb_mask, pt1, pt2, color, 2)
            ransac_frame = cv2.line(ransac_frame, pt1, pt2, color, 2)

        kp1, des1 = kp2, des2

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is None:
            print("p1 is None, saving current image and moving to next file...")
            cv2.imwrite(os.path.splitext(file)[0] + "_original.png", img)
            cv2.imwrite(os.path.splitext(file)[0] + "_t_vectors.png", t_img)
            break

        # Select good points if p1 is not None
        else:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            good_old_transformed = good_old
            color = np.random.randint(0, 255, (100, 3))
            if len(good_old) >= 8:
                try:
                    H, _ = cv2.findHomography(good_old, good_new, method=cv2.RANSAC)
                    good_old_transformed = cv2.perspectiveTransform(
                        good_old.reshape(-1, 1, 2), H
                    )
                    good_old_transformed = good_old_transformed.reshape(-1, 2)

                except:
                    print("Not enough points.")

            for i, (new, old, old_transformed) in enumerate(
                zip(good_new, good_old, good_old_transformed)
            ):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                c_t, d_t = old_transformed.ravel().astype(int)
                color_value = tuple(map(int, color[i]))

                # original vectors
                mask = cv2.line(mask, (a, b), (c, d), color_value, 2)
                old_frame = cv2.circle(old_frame, (a, b), 5, color_value, -1)

                # transformed vectors
                t_mask = cv2.line(t_mask, (a, b), (c_t, d_t), color_value, 2)
                initial_frame = cv2.circle(initial_frame, (a, b), 5, color_value, -1)

            img = cv2.add(old_frame, mask)
            t_img = cv2.add(initial_frame, t_mask)

            orb_mask = cv2.resize(orb_mask, (old_frame.shape[1], old_frame.shape[0]))
            if len(orb_mask.shape) == 2:
                orb_mask = cv2.cvtColor(orb_mask, cv2.COLOR_GRAY2BGR)
            img_ransac = cv2.add(ransac_frame, orb_mask)

            # Calculation for Lucas-Kanade Optical Flow
            if good_old.size > 0:
                motion_vectors = good_new - good_old
                magnitudes.extend(np.sqrt((motion_vectors**2).sum(-1)))
                border_motions = motion_vectors[is_border_point_vectorized(good_old)]
                # print(border_motions)
                if len(border_motions) > 0:
                    border_magnitudes.extend(np.sqrt((border_motions**2).sum(-1)))
                else:
                    border_magnitudes.append(np.nan)
                motions_x.extend(motion_vectors[:, 0])
                motions_y.extend(motion_vectors[:, 1])
            else:
                print(
                    "Notice: good_old is empty. Assigning np.nan for Lucas-Kanade calculations"
                )
                magnitudes.append(np.nan)
                border_magnitudes.append(np.nan)
                motions_x.append(np.nan)
                motions_y.append(np.nan)

            # Calculation for Lucas-Kanade Optical Flow with RANSAC
            t_motion_vectors = good_new - good_old_transformed
            if good_old_transformed.size > 0:
                t_magnitudes.extend(np.sqrt((t_motion_vectors**2).sum(-1)))
                t_motions_x.extend(t_motion_vectors[:, 0])
                t_motions_y.extend(t_motion_vectors[:, 1])

                t_border_motions = t_motion_vectors[
                    is_border_point_vectorized(good_old)
                ]
                if len(t_border_motions) > 0:
                    t_border_magnitudes.extend(np.sqrt((t_border_motions**2).sum(-1)))
                else:
                    t_border_magnitudes.append(np.nan)
            else:
                print(
                    "Notice: good_old_transformed is empty. Assigning np.nan for Lucas-Kanade with RANSAC calculations"
                )
                t_magnitudes.append(np.nan)
                t_border_magnitudes.append(np.nan)
                t_motions_x.append(np.nan)
                t_motions_y.append(np.nan)

            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    # Save the result to an image
    cv2.imwrite(os.path.splitext(file)[0] + "_original.png", img)
    cv2.imwrite(os.path.splitext(file)[0] + "_t_vectors.png", t_img)
    cv2.imwrite(os.path.splitext(file)[0] + "_NEWransac.png", img_ransac)

    cap.release()

    var_magnitudes = calculate_variance_or_nan(magnitudes)
    var_border_magnitudes = calculate_variance_or_nan(border_magnitudes)
    mean_motions_x = calculate_mean_or_nan(motions_x)
    mean_motions_y = calculate_mean_or_nan(motions_y)

    var_transformed_magnitudes = calculate_variance_or_nan(t_magnitudes)
    var_t_border_magnitudes = calculate_variance_or_nan(t_border_magnitudes)
    mean_transformed_motions_x = calculate_mean_or_nan(t_motions_x)
    mean_transformed_motions_y = calculate_mean_or_nan(t_motions_y)

    var_orb_magnitudes = calculate_variance_or_nan(orb_magnitudes)
    var_orb_border_magnitudes = calculate_variance_or_nan(orb_border_magnitudes)
    mean_orb_motions_x = calculate_mean_or_nan(orb_motions_x)
    mean_orb_motions_y = calculate_mean_or_nan(orb_motions_y)

    return (
        var_magnitudes,
        var_border_magnitudes,
        mean_motions_x,
        mean_motions_y,
        var_transformed_magnitudes,
        var_t_border_magnitudes,
        mean_transformed_motions_x,
        mean_transformed_motions_y,
        var_orb_magnitudes,
        var_orb_border_magnitudes,
        mean_orb_motions_x,
        mean_orb_motions_y,
    )


def main():
    directory = (
        "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/frames/DragonInn"
    )
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
                filename_without_extension = os.path.splitext(filename)[0]
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

                # If face detection did not find anything, use object detection
                if shot_scale == "UNKNOWN":
                    outputs = get_outputs(image, object_net)
                    shot_scale, bounding_box, object, objectClass = get_object(
                        outputs, height, width, classes
                    )

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
    directory = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/videos"
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
                    var_transformed_magnitudes,
                    var_t_border_magnitudes,
                    mean_transformed_motions_x,
                    mean_transformed_motions_y,
                    var_orb_magnitudes,
                    var_orb_border_magnitudes,
                    mean_orb_motions_x,
                    mean_orb_motions_y,
                ) = track_movement(file_path)

                movement = get_movement(
                    var_magnitudes,
                    var_border_magnitudes,
                    mean_motions_x,
                    mean_motions_y,
                    var_orb_magnitudes,
                    mean_orb_motions_x,
                    mean_orb_motions_y,
                )

                data.append(
                    [
                        file_wo_extension,
                        var_magnitudes,
                        var_border_magnitudes,
                        mean_motions_x,
                        mean_motions_y,
                        var_transformed_magnitudes,
                        var_t_border_magnitudes,
                        mean_transformed_motions_x,
                        mean_transformed_motions_y,
                        var_orb_magnitudes,
                        var_orb_border_magnitudes,
                        mean_orb_motions_x,
                        mean_orb_motions_y,
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
            "t_variance",
            "t_border_variance",
            "t_average_motion_x",
            "t_average_motion_y",
            "orb_variance",
            "orb_border_variance",
            "orb_average_motion_x",
            "orb_average_motion_y",
            "movement",
        ],
    )

    csv = "combined_values.csv"
    if df_image.empty and not df_video.empty:
        new_df = df_video.copy()
    elif not df_image.empty and df_video.empty:
        new_df = df_image.copy()
    elif not df_image.empty and not df_video.empty:
        new_df = pd.merge(df_image, df_video, on="filename", how="outer")
    else:
        print("Both dataframes are empty!")

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
                row_df = row_df.applymap(
                    lambda x: round(x, 3) if isinstance(x, (float, np.float64)) else x
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
