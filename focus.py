import cv2
import os
import pandas as pd
import numpy as np

# Specify your directory path
directory = "/Users/heying/Documents/Grad_School/慶應/KMD/THESIS/Footage/frames/1"

# Prepare an empty list to store the image names and focus measures
data = []

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

        # Apply Laplacian operator in spatial domain
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

        # Append filename and focus measure to the list
        data.append([os.path.splitext(filename)[0], laplacian_var])

# Convert the list into a DataFrame
df = pd.DataFrame(data, columns=["Image_Name", "Focus_Measure"])

# Save the DataFrame to a CSV file
df = df.sort_values("Image_Name")
df.to_csv("focus_measure.csv", index=False)
