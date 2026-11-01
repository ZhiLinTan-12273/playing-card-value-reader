# recognise.py - Member B: read the rank and suit of one upright card.
# Methods: Otsu thresholding (Segmentation lab), contours (Features lab),
# cv2.matchTemplate (OpenCV docs). Contour-based symbol splitting suggested
# by Claude (Anthropic, 2026); tested and adjusted by me.

import os
import cv2

CORNER = (110, 45)        # top-left region to search: (height, width)
RANK_SIZE = (30, 45)      # fixed sizes so symbols can be compared with templates
SUIT_SIZE = (30, 30)

# Names must match answers.csv exactly
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUITS = ['hearts', 'diamonds', 'clubs', 'spades']
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')


def crop_boxes(binary, boxes, size):
    """Crop one rectangle around all the given boxes and resize it."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return cv2.resize(binary[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST)


def extract_symbols(card_img):
    """Find the rank and suit in the top-left corner.
    Returns (rank_symbol, suit_symbol), or (None, None) if not found."""
    corner = card_img[0:CORNER[0], 0:CORNER[1]]
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Keep symbol shapes; drop specks and shapes cut by the edge of the corner
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        touches_edge = x == 0 or y == 0 or x + bw == w or y + bh == h
        if cv2.contourArea(c) > 15 and not touches_edge:
            boxes.append((x, y, bw, bh))
    if len(boxes) < 2:
        return None, None

    # Top row = rank (both digits of "10" share a row); largest shape below = suit
    boxes.sort(key=lambda b: b[1])
    top = boxes[0]
    rank_boxes = [b for b in boxes if b[1] < top[1] + top[3] / 2]
    below = [b for b in boxes if b not in rank_boxes]
    if not below:
        return None, None
    suit_box = max(below, key=lambda b: b[2] * b[3])

    return crop_boxes(binary, rank_boxes, RANK_SIZE), crop_boxes(binary, [suit_box], SUIT_SIZE)


def load_templates(names, prefix):
    """Load templates like templates/rank_K.png into a dictionary."""
    templates = {}
    for name in names:
        img = cv2.imread(os.path.join(TEMPLATE_DIR, f'{prefix}_{name}.png'), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates[name] = img
    return templates


def best_match(symbol, templates, min_score=0.5):
    """Return the name of the most similar template, or 'unknown'."""
    best_name, best_score = 'unknown', min_score
    for name, tmpl in templates.items():
        score = cv2.matchTemplate(symbol, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
        if score > best_score:
            best_name, best_score = name, score
    return best_name


def identify_card(card_img):
    """Input: upright 200 x 300 card. Output: (rank, suit), e.g. ('K', 'hearts')."""
    rank_sym, suit_sym = extract_symbols(card_img)
    if rank_sym is None:
        return ('unknown', 'unknown')
    return (best_match(rank_sym, load_templates(RANKS, 'rank')),
            best_match(suit_sym, load_templates(SUITS, 'suit')))


def save_templates(card_img, rank, suit):
    """Make templates from a card I know, e.g. save_templates(card, 'K', 'hearts')."""
    rank_sym, suit_sym = extract_symbols(card_img)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(TEMPLATE_DIR, f'rank_{rank}.png'), rank_sym)
    cv2.imwrite(os.path.join(TEMPLATE_DIR, f'suit_{suit}.png'), suit_sym)


if __name__ == '__main__':
    tests = [('test_images/single_cards/kh.jpg', 'K', 'hearts'),
             ('test_images/warped/ah.jpg', 'A', 'hearts')]
    for path, rank, suit in tests:
        card = cv2.resize(cv2.imread(path), (200, 300))
        save_templates(card, rank, suit)
    for path, _, _ in tests:
        card = cv2.resize(cv2.imread(path), (200, 300))
        print(path, '->', identify_card(card))