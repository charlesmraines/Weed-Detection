import cv2
import numpy as np

def is_inside(inner, outer):
    x1_min, y1_min, x1_max, y1_max = inner
    x2_min, y2_min, x2_max, y2_max = outer
    return x1_min >= x2_min and y1_min >= y2_min and x1_max <= x2_max and y1_max <= y2_max

# Load the sample image
image = cv2.imread("/home/charles/Desktop/weed_detection/CotWeed/IMG_3869.JPG")
original = image.copy()

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV ranges for green (adjust based on your crop row color)
lower_green = np.array([35, 40, 40])  # Lower bound of green
upper_green = np.array([90, 255, 255])  # Upper bound of green

# Create a binary mask where green colors are white, and the rest are black
mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Apply the mask to the original image
segmented = cv2.bitwise_and(image, image, mask=mask)

# Perform morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
dilated_mask = cv2.dilate(mask, kernel, iterations=2)

cleaned_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)  # Remove noise

# Find contours in the cleaned mask
contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bboxes = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 200:  # Filter out very small noise
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, x + w, y + h))

merged_bboxes = []
for bbox in bboxes:
    absorbed = False
    for i, merged_bbox in enumerate(merged_bboxes):
        x_min, y_min, x_max, y_max = merged_bbox
        x1_min, y1_min, x1_max, y1_max = bbox
        # Check if the bounding boxes overlap or are very close
        if (x1_min <= x_max + 20 and x1_max >= x_min - 20 and
                y1_min <= y_max + 20 and y1_max >= y_min - 20):  # Threshold = 20 pixels
            # Update the merged bounding box to encompass both
            merged_bboxes[i] = (
                min(x_min, x1_min), min(y_min, y1_min),
                max(x_max, x1_max), max(y_max, y1_max)
            )
            absorbed = True
            break
        elif is_inside(bbox, merged_bbox):
            absorbed = True
            break
    if not absorbed:
        merged_bboxes.append(bbox)

# segmented = original.copy()
for i, (x_min, y_min, x_max, y_max) in enumerate(merged_bboxes):
    cv2.rectangle(segmented, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(segmented, f"Cluster {i+1}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# Combine original and segmented images side-by-side
combined = np.hstack((original, segmented))
resized = cv2.resize(combined, (1700, 700), interpolation=cv2.INTER_AREA)

# Display the result
cv2.imshow("Original and Segmented", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()