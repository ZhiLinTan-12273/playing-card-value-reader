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

card_mask = cv2.bitwise_not(mask)
contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
card_contours = []

for cnt in contours:
    if cv2.contourArea(cnt) > 2000:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            card_contours.append(approx)
output = image.copy()
cv2.drawContours(output, card_contours, -1, (0, 255, 0), 3)

for card in card_contours:
    for point in card:
        x, y = point[0]
        cv2.circle(output, (x, y), 8, (0, 0, 255), -1)

cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Detected Cards", output)

cv2.waitKey(0)
cv2.destroyAllWindows()