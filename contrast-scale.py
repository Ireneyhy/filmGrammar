import os
import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN


def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return contrast


def get_person_scale(image):
    # Load the model
    net = cv2.dnn.readNetFromCaffe(
        "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/deploy.prototxt",
        "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    )

    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)

    # Set the blob as input to the network
    net.setInput(blob)

    # Perform forward pass to compute output
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Get the confidence of the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array(
                [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
            )
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)


def get_outputs(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net.forward(output_layers)


def get_object_scale(height, h):
    ratio = h / height

    if ratio < 0.1:
        return "EXTREME LONG SHOT"
    elif ratio < 0.5:
        return "LONG SHOT"
    elif ratio < 0.75:
        return "MEDIUM LONG SHOT"
    elif ratio < 0.85:
        return "MEDIUM CLOSE-UP"
    elif ratio < 1:
        return "CLOSE-UP"
    else:
        return "EXTREME CLOSE-UP"


def main():
    directory = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/frames/1"
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
    detector = MTCNN()

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            outputs = get_outputs(image, net)

            result = detector.detect_faces(image)

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
                        (centerX, centerY, w, h) = box.astype("int")
                        x = int(centerX - w / 2)
                        y = int(centerY - h / 2)

                        # Check if this bounding box is larger than the current largest
                        if w * h > max_area:
                            max_area = w * h
                            if classes[class_id] == "person":
                                person_detected = True
                                shot_scale = get_person_scale(image)
                            else:
                                object_detected = True
                                shot_scale = get_object_scale(height, h)

            if person_detected or object_detected:
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    shot_scale,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                output_path = (
                    os.path.splitext(image_path)[0]
                    + "_detected"
                    + os.path.splitext(image_path)[1]
                )
                cv2.imwrite(output_path, image)
                print(f"Processed: {os.path.basename(image_path)}")

            else:
                print("Nothing detected.")

            contrast = calculate_contrast(image)
            data.append([filename, contrast, shot_scale])

    # cv2.imshow("Image with Shot Scale", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    df = pd.DataFrame(data, columns=["filename", "contrast", "scale"])
    df = df.sort_values("filename")
    df.to_csv("contrast_values1.csv", index=False)


if __name__ == "__main__":
    main()
