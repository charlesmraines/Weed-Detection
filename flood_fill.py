import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy.spatial import KDTree

def preprocess(image):
    image = cv2.resize(image, (512, 512))
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return image

def plot(axes, images, titles):
    for ax, img, title in zip(axes.flatten(), images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)  # Set title
        ax.axis("off")  # Hide axes

    plt.tight_layout()  # Adjust layout
    plt.show()

def compute_bounding_box(cluster):
    """Compute the bounding box of the given cluster."""
    x_min, y_min = np.min(cluster, axis=0)
    x_max, y_max = np.max(cluster, axis=0)
    return (x_min, y_min, x_max, y_max)

def remove_edge_connected_regions(binary_mask):
    """
    Removes regions that are connected to the edges using flood fill.
    
    :param binary_mask: Binary image where white represents detected regions.
    :return: Binary image with edge-connected regions removed.
    """
    h, w = binary_mask.shape
    mask = binary_mask.copy()

    # Define a mask for flood filling (2 pixels larger to prevent boundary errors)
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill from the edges
    for y in range(h):
        if mask[y, 0] == 255:  # Left edge
            cv2.floodFill(mask, flood_mask, (0, y), 0)
        if mask[y, w-1] == 255:  # Right edge
            cv2.floodFill(mask, flood_mask, (w-1, y), 0)
    for x in range(w):
        if mask[0, x] == 255:  # Top edge
            cv2.floodFill(mask, flood_mask, (x, 0), 0)
        if mask[h-1, x] == 255:  # Bottom edge
            cv2.floodFill(mask, flood_mask, (x, h-1), 0)

    return mask


def euclidean_clustering(dilation_mask, distance_threshold=80, min_cluster_size=5, image=None):

    # Find all nonzero points (white pixels)
    points = np.column_stack(np.where(dilation_mask > 0))  # (y, x) format

    if len(points) == 0:
        return dilation_mask, [], []  # No clusters found

    # Build KD-Tree for efficient neighbor search
    tree = KDTree(points)
    clusters = []
    visited = set()

    # Perform clustering
    for i, point in enumerate(points):
        if i in visited:
            continue  # Skip already clustered points

        # Find neighbors within the threshold distance
        neighbors = tree.query_ball_point(point, distance_threshold)
        
        # Create a new cluster
        cluster = [tuple(points[j]) for j in neighbors]
        visited.update(neighbors)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    # Create an output image to visualize clusters
    clustered_image = image.copy()

    # Bounding boxes for each cluster
    bounding_boxes = []
    for cluster in clusters:
        cluster_points = np.array(cluster)
        x, y, w, h = cv2.boundingRect(cluster_points[:, ::-1])  # Reverse (y, x) to (x, y)
        bounding_boxes.append((x, y, w, h))

        # Draw bounding box
        cv2.rectangle(clustered_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return clustered_image, clusters, bounding_boxes


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
titles = ["Original", "Pre-Processed", "Masked", "Result"]

img_num = "3887"#"3886""3869"
start_time = time.time()
image = cv2.imread(f"/home/charles/Downloads/CotWeedZip/CotWeed/IMG_{img_num}.JPG")
original = image.copy()
image = preprocess(image)



hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([200, 255, 255])


mask = cv2.inRange(hsv_image, lower_green, upper_green)
segmented = cv2.bitwise_and(image, image, mask=mask)

kernel = np.ones((5, 5), np.uint8)
dilated_mask = cv2.dilate(mask, kernel, iterations=2)
final_mask = remove_edge_connected_regions(dilated_mask)
clustered_img, clusters, bboxes = euclidean_clustering(final_mask, image=image)

elapsed_time = (time.time() - start_time) * 1000

images = [original, image, dilated_mask, clustered_img]
fig.suptitle(f"Weed Detection Results - Computed in {elapsed_time:.2f} ms")
plot(axes, images, titles)
# cv2.imshow(f"Weed Image - Computed in {elapsed_time:.4f} seconds", clustered_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
