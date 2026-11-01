# recognise.py - Member B: read the rank and suit of one upright card.
# Methods: Otsu thresholding (Segmentation lab), contours (Features lab),
# cv2.matchTemplate (OpenCV docs). Contour-based symbol splitting, template
# caching, match scores and the green-channel threshold suggested by
# Claude (Anthropic, 2026); tested and adjusted by me on our own deck.

import os
import cv2

CORNER = (110, 45)        # top-left region to search: (height, width)
RANK_SIZE = (30, 45)      # fixed sizes so symbols can be compared with templates
SUIT_SIZE = (30, 30)
MIN_SCORE = 0.5           # below this, the card is reported as 'unknown'

# Names must match answers.csv exactly
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUITS = ['hearts', 'diamonds', 'clubs', 'spades']
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

_templates = None         # loaded once, then reused for every card


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
    # Green channel: red and black ink are both dark in it, while the white
    # card stays bright, so red symbols get much better contrast than in
    # plain greyscale (tested on our deck: fewer failures than greyscale)
    green = corner[:, :, 1]
    _, binary = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
    """Read templates like templates/rank_K.png into a dictionary,
    skipping any that haven't been made yet."""
    templates = {}
    for name in names:
        path = os.path.join(TEMPLATE_DIR, f'{prefix}_{name}.png')
        if os.path.exists(path):
            templates[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return templates


def get_templates():
    """Load all templates the first time they are needed, then reuse them."""
    global _templates
    if _templates is None:
        _templates = {'rank': load_templates(RANKS, 'rank'),
                      'suit': load_templates(SUITS, 'suit')}
    return _templates


def best_match(symbol, templates, method=cv2.TM_CCOEFF_NORMED):
    """Compare a symbol with every template.
    Returns (name, score); score is 0-1 where higher means more similar."""
    best_name, best_score = 'unknown', 0.0
    for name, tmpl in templates.items():
        score = cv2.matchTemplate(symbol, tmpl, method)[0][0]
        if method == cv2.TM_SQDIFF_NORMED:   # this method: lower = better
            score = 1 - score
        if score > best_score:
            best_name, best_score = name, score
    if best_score < MIN_SCORE:
        best_name = 'unknown'
    return best_name, best_score


def identify_card(card_img, method=cv2.TM_CCOEFF_NORMED, with_scores=False):
    """Input: upright 200 x 300 card.
    Output: (rank, suit), e.g. ('K', 'hearts').
    With with_scores=True: (rank, suit, rank_score, suit_score)."""
    rank_sym, suit_sym = extract_symbols(card_img)
    if rank_sym is None:
        return ('unknown', 'unknown', 0.0, 0.0) if with_scores else ('unknown', 'unknown')

    t = get_templates()
    rank, rank_score = best_match(rank_sym, t['rank'], method)
    suit, suit_score = best_match(suit_sym, t['suit'], method)
    if with_scores:
        return rank, suit, rank_score, suit_score
    return rank, suit


def save_templates(card_img, rank, suit):
    """Make templates from a card I know, e.g. save_templates(card, 'K', 'hearts')."""
    global _templates
    rank_sym, suit_sym = extract_symbols(card_img)
    if rank_sym is None:
        print(f'Could not find symbols for {rank} of {suit}')
        return
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(TEMPLATE_DIR, f'rank_{rank}.png'), rank_sym)
    cv2.imwrite(os.path.join(TEMPLATE_DIR, f'suit_{suit}.png'), suit_sym)
    _templates = None        # force a reload next time


if __name__ == '__main__':
    # Templates now come from our own deck: run  python test_deck.py
    # This quick check only reads the cards, it does not change any templates.
    for path in ['test_images/single_cards/kh.jpg', 'test_images/warped/ah.jpg']:
        card = cv2.resize(cv2.imread(path), (200, 300))
        rank, suit, rs, ss = identify_card(card, with_scores=True)
        print(f'{path} -> ({rank}, {suit})  scores: rank {rs:.2f}, suit {ss:.2f}')
