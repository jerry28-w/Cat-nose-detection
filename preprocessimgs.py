import cv2
import pandas as pd
import os

# Path to the folder containing cat images
image_folder_path = r'C:\Users\tejas\OneDrive\Documents\project\archive\dataset-part1\dataset-part1'

# Output CSV file to store annotations
output_csv_path = r'C:\Users\tejas\OneDrive\Documents\project\annotations_manual.csv'

# Create an empty DataFrame to store annotations
annotations_df = pd.DataFrame(columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax'])

# Variables to store bounding box coordinates
rectangles = []
current_rectangle = []

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global current_rectangle

    # Access the resized_image from the parameter
    resized_image = param['resized_image']

    if event == cv2.EVENT_LBUTTONDOWN:
        # Left mouse button clicked - store the starting point of the rectangle
        current_rectangle = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        # Left mouse button released - store the ending point of the rectangle
        current_rectangle.append((x, y))
        
        # Draw the rectangle on the image
        cv2.rectangle(resized_image, current_rectangle[0], current_rectangle[1], (0, 255, 0), 2)
        cv2.imshow('Manual Annotation - Click on the cat nose and press "c" to continue', resized_image)

# Function to annotate an image
def annotate_image(filename, image_path):
    global annotations_df, rectangles

    # Read the image
    original_image = cv2.imread(image_path)

    # Resize the image (adjust the dimensions as needed)
    resized_image = cv2.resize(original_image, (800, 600))

    # Set up the window and the callback
    cv2.namedWindow('Manual Annotation - Click on the cat nose and press "c" to continue')
    cv2.setMouseCallback('Manual Annotation - Click on the cat nose and press "c" to continue', mouse_callback, {'resized_image': resized_image})

    # Display the image
    cv2.imshow('Manual Annotation - Click on the cat nose and press "c" to continue', resized_image)

    # Wait for key press
    key = cv2.waitKey(0) & 0xFF

    # If 'c' is pressed, continue with annotation
    if key == ord('c') and len(current_rectangle) == 2:
        # Get the bounding box coordinates
        xmin = min(current_rectangle[0][0], current_rectangle[1][0])
        ymin = min(current_rectangle[0][1], current_rectangle[1][1])
        xmax = max(current_rectangle[0][0], current_rectangle[1][0])
        ymax = max(current_rectangle[0][1], current_rectangle[1][1])

        # Save the annotation to the DataFrame
        annotation = {
            'filename': filename,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
        }
        annotations_df = pd.concat([annotations_df, pd.DataFrame([annotation])], ignore_index=True)

        # Save the annotations to a CSV file after each image
        annotations_df.to_csv(output_csv_path, index=False)

        # Print the DataFrame
        print(annotations_df)

        # Clear the output to remove the displayed image
        cv2.destroyAllWindows()

# Run the function for each image in the folder
for filename in os.listdir(image_folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder_path, filename)
        annotate_image(filename, image_path)
