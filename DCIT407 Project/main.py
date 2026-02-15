import cv2
import matplotlib.pyplot as plt
from filters import mean_filter, gaussian_filter

# load image
img = cv2.imread("407-img-1.jpg", cv2.IMREAD_GRAYSCALE)

# apply filters
mean_img = mean_filter(img)
gauss_img = gaussian_filter(img)

# display
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1,3,2)
plt.title("Mean")
plt.imshow(mean_img, cmap='gray')

plt.subplot(1,3,3)
plt.title("Gaussian")
plt.imshow(gauss_img, cmap='gray')

plt.show()
