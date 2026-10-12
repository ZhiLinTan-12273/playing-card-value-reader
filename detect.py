import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, 'test_images', '1_simple', 'card1.jpg')
image = cv2.imread(image_path)

if image is None:
    print("Cant find image")
    exit()

lower_bound = np.array([0, 0, 0])
upper_bound = np.array([150, 150, 150])

mask = cv2.inRange(image, lower_bound, upper_bound)

cv2.imshow("Original", image)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()