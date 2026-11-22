# pipeline.py - connects the three parts of the Playing Card Value Reader:
#   detect.py    find_cards(image)   -> straightened 200 x 300 cards + outlines
#   recognise.py identify_card(card) -> (rank, suit)
#   score.py     card_value(rank)    -> points
#
# Robustness: edge detection can find a slightly different card outline when
# the photo changes by only a pixel value or two (another computer, another
# JPEG decoder, a tiny change in light). Some photos sit right on that edge.
# So each photo is read three times - as it is, and lightly smoothed with a
# 3x3 and a 5x5 Gaussian blur (Filtering lab) - and the reading that the
# recogniser is most confident about is kept.
#
# Written with help from Claude (Anthropic, 2026).

import cv2

from detect import find_cards, identify_any_orientation, rescue_card, score_of
from score import card_value

VIEWS = [lambda image: image,
         lambda image: cv2.GaussianBlur(image, (3, 3), 0),
         lambda image: cv2.GaussianBlur(image, (5, 5), 0)]


def read_cards(image):
    """Find and read every card in one version of the photo.
    Returns (results, confidence), where results = [(rank, suit, outline), ...]
    and confidence = (rank/suit symbols read, average match score)."""
    results, known, score = [], 0, 0.0
    for card, outline in find_cards(image, return_outlines=True):
        rank, suit = identify_any_orientation(card)
        if rank == 'unknown' or suit == 'unknown':
            # maybe only the picture frame of a face card was found
            rescued = rescue_card(image, outline)
            if rescued is not None:
                card, outline = rescued
                rank, suit = identify_any_orientation(card)
        k, s = max(score_of(card), score_of(cv2.rotate(card, cv2.ROTATE_180)))
        known += k
        score += s
        results.append((rank, suit, outline))
    return results, (known, len(results) > 0, score / max(len(results), 1))


def run_pipeline(image):
    """Input: a BGR photo with one or more cards.
    Output: (results, total)
      results = [(rank, suit, outline), ...]  one entry per card found;
                outline = the card's 4 corners in the photo (for draw_results)
      total   = sum of the card values (unrecognised cards count 0)"""
    best_results, best_confidence = [], None
    for view in VIEWS:
        results, confidence = read_cards(view(image))
        if best_confidence is None or confidence > best_confidence:
            best_results, best_confidence = results, confidence
    total = sum(card_value(rank) for rank, _, _ in best_results)
    return best_results, total
