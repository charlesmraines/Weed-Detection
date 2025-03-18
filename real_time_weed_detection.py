import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def preprocess(image):
    """Performs GPU-accelerated preprocessing: resize and bilateral filter."""
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)
    
    # Resize
    gpu_image = cv2.cuda.resize(gpu_image, (512, 512))
    
    # Bilateral filtering
    gpu_filtered = cv2.cuda.bilateralFilter(gpu_image, d=9, sigmaColor=75, sigmaSpace=75)
    
    return gpu_filtered.download()

def plot(axes, images, titles):
    """Plots images with titles."""
    for ax, img, title in zip(axes.flatten(), images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def remove_edge_connected_regions(binary_mask):
    """Removes regions touching the edges using GPU flood fill."""
    h, w = binary_mask.shape
    gpu_mask = cv2.cuda_GpuMat()
    gpu_mask.upload(binary_mask)

    # Create a mask for flood filling (2 pixels larger)
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill from edges
    mask_cpu = gpu_mask.download()  # Needs CPU access for flood fill
    for y in range(h):
        if mask_cpu[y, 0] == 255:
            cv2.floodFill(mask_cpu, flood_mask, (0, y), 0)
        if mask_cpu[y, w-1] == 255:
            cv2.floodFill(mask_cpu, flood_mask, (w-1, y), 0)
    for x in range(w):
        if mask_cpu[0, x] == 255:
            cv2.floodFill(mask_cpu, flood_mask, (x, 0), 0)
        if mask_cpu[h-1, x] == 255:
            cv2.floodFill(mask_cpu, flood_mask, (x, h-1), 0)

    gpu_mask.upload(mask_cpu)
    return gpu_mask.download()

def gpu_connected_components(binary_mask):
    """Performs connected component analysis using GPU acceleration."""
    gpu_mask = cv2.cuda_GpuMat()
    gpu_mask.upload(binary_mask)

    num_labels, labels, stats, centroids = cv2.cuda.connectedComponentsWithStats(
        gpu_mask, 8, cv2.CV_32S
    )

    labels_cpu = labels.download()
    stats_cpu = stats

    # Filter out small clusters
    min_cluster_size = 5
    valid_clusters = [i for i in range(1, num_labels) if stats_cpu[i, cv2.CC_STAT_AREA] >= min_cluster_size]

    return labels_cpu, stats_cpu, valid_clusters

# --- Main Processing Pipeline ---
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
titles = ["Original", "Pre-Processed", "Masked", "Result"]

img_num = "3887"
start_time = time.time()

# Load image
image = cv2.imread(f"/home/charles/Desktop/Weed-Detection/camera_images/IMG_{img_num}.JPG")
original = image.copy()
image = preprocess(image)

# Convert to HSV and create mask (GPU)
gpu_image = cv2.cuda_GpuMat()
gpu_image.upload(image)

hsv_image = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([200, 255, 255])
mask = cv2.cuda.inRange(hsv_image, lower_green, upper_green)

# Dilation (GPU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
gpu_kernel = cv2.cuda_GpuMat()
gpu_kernel.upload(kernel)

gpu_dilated = cv2.cuda.dilate(mask, gpu_kernel, iterations=2)
dilated_mask = gpu_dilated.download()

# Remove edge-connected regions (GPU)
final_mask = remove_edge_connected_regions(dilated_mask)

# Connected Components (GPU)
labels, stats, valid_clusters = gpu_connected_components(final_mask)

# Draw bounding boxes
clustered_img = image.copy()
for i in valid_clusters:
    x, y, w, h, _ = stats[i]
    cv2.rectangle(clustered_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

# Plot results
images = [original, image, dilated_mask, clustered_img]
fig.suptitle(f"Weed Detection Results - Computed in {elapsed_time:.2f} ms")
plot(axes, images, titles)
