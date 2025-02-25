import cv2
import numpy as np

# Load the image
image = cv2.imread("./images/twitter.png", cv2.IMREAD_UNCHANGED)  # Preserve alpha if present
h, w = image.shape[:2]

# Compute new dimensions while maintaining aspect ratio
if w > h:
    new_w, new_h = 16, int(h * (16 / w))
else:
    new_h, new_w = 16, int(w * (16 / h))

# Resize using nearest-neighbor interpolation
resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

# Create a 16x16 black canvas (supports 3 or 4 channels)
channels = image.shape[2] if len(image.shape) == 3 else 1
canvas = np.zeros((16, 16, channels), dtype=np.uint8)

# Compute top-left position to center the image
x_offset = (16 - new_w) // 2
y_offset = (16 - new_h) // 2

# Place resized image onto the canvas
canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

# Save or display the final image
cv2.imwrite("output_16x16.jpg", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
