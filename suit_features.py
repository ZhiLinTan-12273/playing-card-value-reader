"""suit_features.py - red/black detection and pip counting.
Input: a BGR card (resized to 200x300 internally, like recognise.py).
Uses the green channel + Otsu, the same as recognise.extract_symbols."""

import sys
from pathlib import Path

import cv2
import numpy as np

CARD_SIZE = (200, 300)          # (width, height)
INDEX_W, INDEX_H = 45, 110      # corner index area, same as CORNER in recognise.py

# ---- colour ----
RED_RATIO_THRESHOLD = 0.5       # fraction of ink pixels that must be red
MIN_INK_PIXELS = 20             # fewer than this = nothing to judge
RED_SAT, RED_VAL = 90, 50       # minimum saturation/brightness to count as red

# ---- pips (all tunable; check with debug images) ----
INSET = 8                       # ignore the outer border of the card
MIN_PIP_AREA = 150              # smaller blobs are noise
PIP_AREA = 700                  # typical area of ONE pip at 200x300
FACE_BOX = 110                  # a blob taller/wider than this = court card art
MAX_INK_FRACTION = 0.30         # more ink than this in the pip zone = court card
MAX_BLOBS = 12


def _ink_mask(img, drop_edge_shapes):
    """White where there is ink (green channel + Otsu)."""
    green = img[:, :, 1]
    _, binary = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape
    keep = np.zeros_like(binary)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        touches_edge = x == 0 or y == 0 or x + bw == w or y + bh == h
        if cv2.contourArea(c) > 15 and not (drop_edge_shapes and touches_edge):
            cv2.drawContours(keep, [c], -1, 255, cv2.FILLED)
    return cv2.bitwise_and(keep, binary)


# ------------------------------ red / black ------------------------------
def red_ratio(img, is_card=True):
    """Fraction of ink pixels that are red (HSV inRange).
    is_card=True : img is a whole card; only the top-left index is examined.
    is_card=False: img is a tight COLOUR crop of a symbol (not a binary template).
    Returns None if there is no ink to judge."""
    if is_card:
        img = cv2.resize(img, CARD_SIZE)[0:INDEX_H, 0:INDEX_W]
    ink = _ink_mask(img, drop_edge_shapes=is_card)
    n = cv2.countNonZero(ink)
    if n < MIN_INK_PIXELS:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # red wraps around the hue axis, so two ranges
    low = cv2.inRange(hsv, (0, RED_SAT, RED_VAL), (10, 255, 255))
    high = cv2.inRange(hsv, (170, RED_SAT, RED_VAL), (180, 255, 255))
    red = cv2.bitwise_or(low, high)
    return cv2.countNonZero(cv2.bitwise_and(red, ink)) / n


def is_red(img, is_card=True):
    """True = red, False = black, None = can't tell."""
    r = red_ratio(img, is_card)
    return None if r is None else r > RED_RATIO_THRESHOLD


def allowed_suits(img, is_card=True):
    """Suits that match the colour (all four if the colour is unclear)."""
    red = is_red(img, is_card)
    if red is None:
        return ['hearts', 'diamonds', 'clubs', 'spades']
    return ['hearts', 'diamonds'] if red else ['clubs', 'spades']


# ------------------------------ pip counting ------------------------------
def count_pips(card_img, debug=False):
    """Counts the suit symbols in the middle of a number card.
    Returns 1..10, or None for court cards / unreadable cards.
    debug=True returns (count, annotated_card)."""
    card = cv2.resize(card_img, CARD_SIZE)
    inner = card[INSET:CARD_SIZE[1] - INSET, INSET:CARD_SIZE[0] - INSET]
    ink = _ink_mask(inner, drop_edge_shapes=False)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pips = []
    for c in contours:
        if cv2.contourArea(c) < MIN_PIP_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cx, cy = x + bw / 2 + INSET, y + bh / 2 + INSET
        in_top_left = cx < INDEX_W and cy < INDEX_H                  # corner index
        in_bottom_right = (cx > CARD_SIZE[0] - INDEX_W and
                           cy > CARD_SIZE[1] - INDEX_H)              # upside-down index
        if not (in_top_left or in_bottom_right):
            pips.append(c)

    annotated = card.copy()
    for c in pips:
        cv2.drawContours(annotated, [c + [INSET, INSET]], -1, (0, 255, 0), 2)

    count = None
    if pips:
        boxes = [cv2.boundingRect(c) for c in pips]
        zone_area = (CARD_SIZE[0] - 2 * INSET) * (CARD_SIZE[1] - 2 * INSET)
        ink_fraction = cv2.countNonZero(ink) / zone_area
        is_court = (any(b[3] > FACE_BOX or b[2] > FACE_BOX for b in boxes)
                    or ink_fraction > MAX_INK_FRACTION or len(pips) > MAX_BLOBS)

        if not is_court:
            areas = sorted(cv2.contourArea(c) for c in pips)
            if len(areas) == 1:
                count = 1                                   # a lone pip = ace
            elif areas[-1] < 1.5 * areas[0]:
                count = len(areas)                          # all alike: none merged
            else:                                           # some pips touch: split by area
                unit = float(np.median(areas)) if len(areas) >= 3 else PIP_AREA
                count = sum(max(1, round(a / unit)) for a in areas)
            if not 1 <= count <= 10:
                count = None

    return (count, annotated) if debug else count


def rank_from_pips(card_img):
    """'A', '2'..'10', or None (court card / unsure)."""
    n = count_pips(card_img)
    if n is None:
        return None
    return 'A' if n == 1 else str(n)


def pips_agree(rank, card_img):
    """Second opinion on a rank. True/False, or None when pips can't judge."""
    if rank in ('J', 'Q', 'K'):
        return None
    guess = rank_from_pips(card_img)
    return None if guess is None else guess == rank


# ------------------------- plug into recognise.py -------------------------
def refine_card(card_img, rank_min_score=0.7):
    """identify_card + colour check + pip check.
    Returns (rank, suit, notes); notes lists anything that was changed or disputed."""
    from recognise import identify_card, extract_symbols, get_templates, best_match

    card = cv2.resize(card_img, CARD_SIZE)
    rank, suit, rank_score, _ = identify_card(card, with_scores=True)
    notes = []

    # 1. colour: the suit must match red/black, otherwise re-match among allowed suits
    allowed = allowed_suits(card)
    if len(allowed) == 2 and suit not in allowed:
        _, suit_sym = extract_symbols(card)
        if suit_sym is not None:
            templates = {k: v for k, v in get_templates()['suit'].items() if k in allowed}
            new_suit, _ = best_match(suit_sym, templates)
            notes.append(f'suit {suit} -> {new_suit} (colour)')
            suit = new_suit

    # 2. pips: only overrule a weak or unknown rank, otherwise just flag it
    guess = rank_from_pips(card)
    if guess is not None and guess != rank and rank not in ('J', 'Q', 'K'):
        if rank == 'unknown' or rank_score < rank_min_score:
            notes.append(f'rank {rank} -> {guess} (pips)')
            rank = guess
        else:
            notes.append(f'pips say {guess}, kept {rank}')
    return rank, suit, notes


# ------------------------------ quick test runner ------------------------------
def _run(mode, folder, is_card):
    files = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if not files:
        print(f'No images found in {folder}')
        return
    Path('pip_debug').mkdir(exist_ok=True)
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f'{path.name}: could not load')
        elif mode == 'colour':
            r = red_ratio(img, is_card)
            label = '?' if r is None else ('RED' if r > RED_RATIO_THRESHOLD else 'BLACK')
            print(f'{path.name:<24} red_ratio={r if r is None else round(r, 2)}  ->  {label}')
        else:
            n, annotated = count_pips(img, debug=True)
            cv2.imwrite(str(Path('pip_debug') / path.name), annotated)
            print(f'{path.name:<24} pips={n}  rank guess={rank_from_pips(img)}')


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('colour', 'pips'):
        print('Usage: python suit_features.py colour|pips <folder> [crop]')
    else:
        _run(sys.argv[1], sys.argv[2], is_card='crop' not in sys.argv[3:])