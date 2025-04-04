import depthai as dai
import cv2
import numpy as np
import time
from scipy.spatial import KDTree

# Set which camera to use
CAMERA_IP = "10.0.0.5"
CAM_NAME = "Camera 1"

def process_frame(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 60, 60])
    upper_green = np.array([80, 220, 220])


    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    segmented = cv2.bitwise_and(image, image, mask=mask)

    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    final_mask = remove_edge_connected_regions(dilated_mask)
    clustered_img, clusters, bboxes = euclidean_clustering(final_mask, image=image)
    return clustered_img, clusters, bboxes

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

# Initialize pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(6)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# Connect to device
with dai.Device(pipeline, dai.DeviceInfo(CAMERA_IP)) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        in_frame = video_queue.get()
        frame = in_frame.getCvFrame()

        start_time = time.time()

        preprocessed_img = preprocess(frame)
        output, _, _ = process_frame(preprocessed_img)

        elapsed_time = (time.time() - start_time) * 1000  # ms
        cv2.imshow(f"{CAM_NAME} - CPU Dilation", output)
        print(f"{CAM_NAME} - CPU Dilation Time: {elapsed_time:.2f} ms")



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

