# %%
import cv2
import numpy as np
import random
# %%
# Define image dimensions
patch_size = 16
canvas_size = 384
num_patches = (canvas_size // patch_size) ** 2

# Load the 16x16 RGB image
patch = cv2.imread("./shrinked_images/mercedes.jpg")
patch = cv2.resize(patch, (patch_size, patch_size))  # Ensure correct size

# Create a blank 384x384 RGB image
canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

# Select a random patch index
patch_idx = random.randint(0, num_patches - 1)

# Compute row and column position of the patch
row = (patch_idx // (canvas_size // patch_size)) * patch_size
col = (patch_idx % (canvas_size // patch_size)) * patch_size

# Place the patch on the canvas
canvas[row:row+patch_size, col:col+patch_size] = patch

# Save the image with the patch index as filename
filename = f"./crafted_images/mercedes_random_positions/mercedes_{patch_idx}_crafted.png"
cv2.imwrite(filename, canvas)

print(f"Image saved as {filename}")

#%%
#Code that will generate an image full of the same patches except for one patch.
# Define image dimensions
patch_size = 16
canvas_size = 384
num_patches = (canvas_size // patch_size) ** 2

# Load the 16x16 RGB images
patch = cv2.imread("./shrinked_images/mercedes.jpg")
patch = cv2.resize(patch, (patch_size, patch_size))  # Ensure correct size

alt_patch = cv2.imread("./shrinked_images/white.jpg")
alt_patch = cv2.resize(alt_patch, (patch_size, patch_size))  # Ensure correct size

# Create a white 384x384 RGB image
canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

# Select a random patch index for the alternate patch
alt_patch_idx = random.randint(0, num_patches - 1)

# Fill all patches with the main patch
for i in range(num_patches):
    row = (i // (canvas_size // patch_size)) * patch_size
    col = (i % (canvas_size // patch_size)) * patch_size
    
    if i == alt_patch_idx:
        canvas[row:row+patch_size, col:col+patch_size] = alt_patch
    else:
        canvas[row:row+patch_size, col:col+patch_size] = patch

# Save the image with the alternate patch index as filename
filename = f"./crafted_images/mercedes_except_one_position/mercedes_{alt_patch_idx}_crafted.png"
cv2.imwrite(filename, canvas)

print(f"Image saved as {filename}")
# %%
