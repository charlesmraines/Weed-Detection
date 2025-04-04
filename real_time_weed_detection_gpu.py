import depthai as dai
import cv2
import numpy as np
import threading
import queue
import time

# Camera IPs
CAMERA_1_IP = "192.168.1.100"
CAMERA_2_IP = "192.168.1.101"

# Shared queue for frame processing
frame_queue = queue.Queue()

# CUDA Kernel (cv2.cuda) for Dilation
def apply_cuda_dilation(image):
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gpu_kernel = cv2.cuda_GpuMat()
    gpu_kernel.upload(kernel)

    dilated_gpu = cv2.cuda.dilate(gpu_mat, gpu_kernel)
    return dilated_gpu.download()  # Bring result back to CPU

# Camera Capture Thread
def capture_camera(ip, queue, cam_name):
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(6)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam_rgb.preview.link(xout.input)

    with dai.Device(pipeline, ip) as device:
        video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

        while True:
            frame = video_queue.get().getCvFrame()
            queue.put((cam_name, frame))  # Add frame to processing queue

            if cv2.waitKey(1) == ord('q'):
                break

# CUDA Processing Thread
def process_frames(queue):
    while True:
        if not queue.empty():
            cam_name, frame = queue.get()

            start_time = time.time()
            processed_frame = apply_cuda_dilation(frame)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

            cv2.imshow(f"{cam_name} - Processed (CUDA Dilation) - {elapsed_time:.2f} ms", processed_frame)

        if cv2.waitKey(1) == ord('q'):
            break

# Start capture threads
thread1 = threading.Thread(target=capture_camera, args=(CAMERA_1_IP, frame_queue, "Camera 1"))
thread2 = threading.Thread(target=capture_camera, args=(CAMERA_2_IP, frame_queue, "Camera 2"))
processing_thread = threading.Thread(target=process_frames, args=(frame_queue,))

thread1.start()
thread2.start()
processing_thread.start()

thread1.join()
thread2.join()
processing_thread.join()
