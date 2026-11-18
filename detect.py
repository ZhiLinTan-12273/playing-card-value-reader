import os
import glob
import cv2
import numpy as np
from recognise import identify_card

def order_point(pts):
    rect = np.zeros((4, 2), dtype = "float32") #ready for 4 space
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)] #x + y top left
    rect[2] = pts[np.argmax(s)] #x + y botton right
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)] # y - x top right
    rect[3] = pts[np.argmax(diff)] # y - x botton left

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
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def find_cards(image, use_minrect=False, inset=0.0):
    h, w, d = image.shape
    scale = 800.0 / w # Shrink the image cuz processing will bit fast
    I_resized = cv2.resize(image, dsize=(800, int(h * scale)))

    G = cv2.cvtColor(I_resized, cv2.COLOR_BGR2GRAY)
    G_blur = cv2.GaussianBlur(G, (5, 5), 0)

    warped_cards = []
    img_area = I_resized.shape[0] * I_resized.shape[1]

    attempts = [(15, 70, 7), (8, 35, 9), (4, 18, 11)]

    for t1, t2, k in attempts:
        E = cv2.Canny(G_blur, threshold1=t1, threshold2=t2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        Closed = cv2.morphologyEx(E, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(Closed, mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            A = cv2.contourArea(c)
            if not (img_area * 0.03 < A < img_area * 0.90):
                continue

            peri = cv2.arcLength(c, True)
            pts = None

            if not use_minrect:
                for eps in [0.02, 0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(c, eps * peri, True)
                    if len(approx) == 4:
                        pts = approx.reshape(4, 2)
                        break

            if pts is None:
                rect = cv2.minAreaRect(c)
                (w_rect, h_rect) = rect[1]
                if w_rect == 0 or h_rect == 0:
                    continue
                aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                if 1.15 < aspect_ratio < 1.85:
                    pts = cv2.boxPoints(rect)

            if pts is None:
                continue

            pts_orig = (pts / scale).astype("float32")
            warped = four_point_transform(image, pts_orig)

            wh, ww = warped.shape[:2]
            if ww > wh:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            hh, ww = warped.shape[:2]
            my, mx = int(hh * inset), int(ww * inset)
            if my > 0 and mx > 0:
                warped = warped[my:hh - my, mx:ww - mx]

            warped_cards.append(cv2.resize(warped, dsize=(200, 300)))

        if warped_cards:  
            break

    return warped_cards


def score_of(card):
    best = (0, 0.0)
    for img in (card, cv2.rotate(card, cv2.ROTATE_180)):
        r, s, rs, ss = identify_card(img, with_scores=True)
        known = (r != "unknown") + (s != "unknown")
        total = (rs if r != "unknown" else 0) + (ss if s != "unknown" else 0)
        best = max(best, (known, total))
    return best


def identify_any_orientation(card):
    best = None
    for img in (card, cv2.rotate(card, cv2.ROTATE_180)):
        rank, suit = identify_card(img)
        score = (rank != "unknown") + (suit != "unknown")
        if best is None or score > best[0]:
            best = (score, rank, suit)
    if best[0] == 2:
        return best[1], best[2]

    best = None
    for base in (card, cv2.rotate(card, cv2.ROTATE_180)):
        for s in (1.0, 0.96, 1.04, 0.92, 1.08, 0.88, 1.12):
            M = cv2.getRotationMatrix2D((100, 150), 0, s)
            img = cv2.warpAffine(base, M, (200, 300),
                                 borderMode=cv2.BORDER_REPLICATE)
            rank, suit, rs, ss = identify_card(img, with_scores=True)
            known = (rank != "unknown") + (suit != "unknown")
            total = (rs if rank != "unknown" else 0) + (ss if suit != "unknown" else 0)
            key = (known, total)
            if best is None or key > best[0]:
                best = (key, rank, suit)
    return best[1], best[2]


def get_best_card(image):
    first = find_cards(image)
    if len(first) == 1 and score_of(first[0])[0] == 2:
        return first[0]

    candidates = list(first) if len(first) == 1 else []
    for use_minrect in (False, True):
        for inset in (0.0, 0.02, 0.04, 0.06):
            c = find_cards(image, use_minrect=use_minrect, inset=inset)
            if len(c) == 1:
                candidates.append(c[0])
    if not candidates:
        return None
    return max(candidates, key=score_of)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    test_images = sorted(glob.glob(os.path.join(BASE_DIR, 'test_images', 'deck', 'test', '*.jpg')))

    correct = 0
    for img_path in test_images:
        filename = os.path.basename(img_path)
        expected_rank, expected_suit = filename.replace('.jpg', '').split('_')

        image = cv2.imread(img_path)
        card = get_best_card(image)

        if card is None:
            print(f"[detect fail] {filename}: can find")
            continue

        rank, suit = identify_any_orientation(card)
        if rank == expected_rank and suit == expected_suit:
            correct += 1
        else:
            print(f"[detect error] {filename}: the card it thought ({rank}, {suit})")
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite(f"debug/{filename}", card)
            corner = card[0:110, 0:45]
            cv2.imwrite(f"debug/{filename}_corner.png",
                        cv2.resize(corner, None, fx=4, fy=4,
                                   interpolation=cv2.INTER_NEAREST))

    print(f"\nfinal result: {correct}/{len(test_images)} correct ({correct/len(test_images)*100:.0f}%)")