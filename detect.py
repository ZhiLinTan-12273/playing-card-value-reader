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
    h, w, d = image.shape
    scale = 800.0 / w
    I_resized = cv2.resize(image, dsize=(800, int(h * scale)))

    G = cv2.cvtColor(I_resized, cv2.COLOR_BGR2GRAY)
    G_blur = cv2.GaussianBlur(G, (5, 5), 0)

    E = cv2.Canny(G_blur, threshold1=20, threshold2=80)

    shape = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    Closed = cv2.morphologyEx(E, cv2.MORPH_CLOSE, shape)

    contours, _ = cv2.findContours(Closed, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    warped_cards = []
    img_area = I_resized.shape[0] * I_resized.shape[1]

    for c in contours:
        A = cv2.contourArea(c)
        if img_area * 0.03 < A < img_area * 0.90:
            rect = cv2.minAreaRect(c)
            (w_rect, h_rect) = rect[1]

            if w_rect == 0 or h_rect == 0:
                continue

            aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
            if 1.15 < aspect_ratio < 1.85:
                box = cv2.boxPoints(rect)
                pts = (box / scale).astype("float32")
                warped = four_point_transform(image, pts)

                wh, ww = warped.shape[:2]
                if ww > wh:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                card_200x300 = cv2.resize(warped, dsize=(200, 300))
                warped_cards.append(card_200x300)

    return warped_cards

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, 'test_images', 'deck', 'templates' , '10_spades.jpg')
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