# An Underwater Image Enhancement Benchmark Dataset and Beyond

"""
References:

An Underwater Image Enhancement Benchmark Dataset and Beyond, IEEE TIP 2019
Paper: https://arxiv.org/abs/1901.05495
Code: https://github.com/tnwei/waternet
"""

import torch
import cv2
import matplotlib.pyplot as plt

# Load from torchhub
preprocess, postprocess, model = torch.hub.load('tnwei/waternet', 'waternet')
model.eval()

# Load one image using OpenCV
im = cv2.imread("images/image1.jpg")
#im = image_1.copy()
rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# Resize image
rgb_im = cv2.resize(rgb_im, (720, 480))

# Inference -> return numpy array (1, 3, H, W)
rgb_ten, wb_ten, he_ten, gc_ten = preprocess(rgb_im)
out_ten = model(rgb_ten, wb_ten, he_ten, gc_ten)
out_im = postprocess(out_ten)

fig, ax = plt.subplots(ncols=2, figsize=(14, 5))
ax[0].imshow(out_im[0])
ax[0].axis("off")
ax[0].set_title("WaterNet output")

ax[1].imshow(rgb_im)
ax[1].axis("off")
ax[1].set_title("Original image")

plt.show()
