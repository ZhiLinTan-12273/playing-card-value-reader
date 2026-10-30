import os
import cv2
import numpy as np

def order_point(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_point(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def find_cards(image):
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([150, 150, 150])
    mask = cv2.inRange(image, lower_bound, upper_bound)

    card_mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    warped_cards = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 2000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                warped = four_point_transform(image, pts)

                h, w = warped.shape[:2]
                if w > h:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                card_200x300 = cv2.resize(warped, (200, 300))
                warped_cards.append(card_200x300)

    return warped_cards

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, 'test_images', '1_simple', 'card1.jpg')
image = cv2.imread(image_path)

if image is None:
    print("Cant find image")
    exit()

cards = find_cards(image)

for idx, card in enumerate(cards):
    filename = f"card_{idx + 1}.jpg"
    cv2.imwrite(filename, card)
    cv2.imshow(f"Card {idx + 1}", card)

cv2.waitKey(0)
cv2.destroyAllWindows()