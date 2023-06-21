import os
import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN


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

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # First try face detection
            shot_scale = get_face_scale(image, image_path, detector)
            if shot_scale != "UNKNOWN":
                print(f"Shot scale (face detection): {shot_scale}")
                continue

            # If face detection did not find anything, use object detection
            outputs = get_outputs(image, object_net)
            shot_scale = get_object_scale(image, outputs, height, width, image_path)
            print(f"Shot scale (object detection): {shot_scale}")


def get_face_scale(image, image_path, detector):
    h, w, _ = image.shape
    result = detector.detect_faces(image)
    max_area = 0
    shot_scale = "UNKNOWN"

    for i in range(len(result)):
        x, y, width, height = result[i]["box"]
        face_area = width * height
        if face_area > max_area:
            max_area = face_area
            shot_scale = get_shot_scale(h, height)

            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            output_path = (
                os.path.splitext(image_path)[0]
                + "_face"
                + os.path.splitext(image_path)[1]
            )
            cv2.imwrite(output_path, image)
    return shot_scale


def get_object_scale(image, outputs, height, width, image_path):
    max_area = 0
    shot_scale = "UNKNOWN"
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

                if w * h > max_area:
                    max_area = w * h
                    shot_scale = get_shot_scale(height, h)

                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # Save the image with bounding box
                    output_path = (
                        os.path.splitext(image_path)[0]
                        + "_object"
                        + os.path.splitext(image_path)[1]
                    )
                    cv2.imwrite(output_path, image)
    return shot_scale


def get_shot_scale(height, h):
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


def get_outputs(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net.forward(output_layers)


if __name__ == "__main__":
    main()
