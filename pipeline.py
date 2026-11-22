# pipeline.py - connects the three parts of the Playing Card Value Reader:
#   detect.py    find_cards(image)  -> straightened 200 x 300 cards + outlines
#   recognise.py identify_card(card) -> (rank, suit)
#   score.py     card_value(rank)    -> points
# Each card is read the right way up and upside down (detect.identify_any_orientation).
# If a card still can't be read, detect.rescue_card grows its outline in case
# only the picture frame of a face card was found.
#
# Written with help from Claude (Anthropic, 2026).

from detect import find_cards, identify_any_orientation, rescue_card
from score import card_value


def run_pipeline(image):
    """Input: a BGR photo with one or more cards.
    Output: (results, total)
      results = [(rank, suit, outline), ...]  one entry per card found;
                outline = the card's 4 corners in the photo (for draw_results)
      total   = sum of the card values (unrecognised cards count 0)"""
    results = []
    for card, outline in find_cards(image, return_outlines=True):
        rank, suit = identify_any_orientation(card)
        if rank == 'unknown' or suit == 'unknown':
            rescued = rescue_card(image, outline)
            if rescued is not None:
                card, outline = rescued
                rank, suit = identify_any_orientation(card)
        results.append((rank, suit, outline))
    total = sum(card_value(rank) for rank, _, _ in results)
    return results, total
