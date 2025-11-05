import cv2
import numpy as np

width, height = 640, 480
fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 编码器
out = cv2.VideoWriter("test.mp4", fourcc, 30, (width, height))

# 写入10帧纯色图像
for _ in range(10):
    frame = np.ones((height, width, 3), dtype=np.uint8) * 100
    out.write(frame)

out.release()
