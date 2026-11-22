# Imports

import cv2

# The scoring rule: A = 1, 2-10 = face value, J/Q/K = 10

CARD_VALUES = { 
    "A": 1,
    "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10,
    "J": 10, "Q": 10, "K": 10,
}

def card_value(rank):
    """Input: rank string (e.g. "K", "A", "5").
    Returns: the points as an int. Returns 0 for an unrecognised rank
    (e.g. "unknown" from identify_card)."""
    return CARD_VALUES.get(rank,0)

def draw_results(image, cards, total):
    """Draws each card outline, a label above it and the total score.
    Input: BGR image, cards = list of (rank, suit, contour), total = int
    Returns a new annoted image (the original is left untouched)"""
    output = image.copy()
    h, w = output.shape[:2]

    # scale line and text size to the photo so it's readable on big images
    scale = max(w, h) / 1000
    thickness = max(2, int(3 * scale))
    font_scale = max(0.6, scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)

    for rank, suit, contour in cards:
        # outline around the card
        cv2.drawContours(output, [contour], -1, green, thickness)

        # label for the card (typically above but inside it if the card touches the top edge)
        x, y, bw, bh = cv2.boundingRect(contour)
        label_y = y - 10 if y > 40 * scale else y + int(30 * scale)
        cv2.putText(output, f"{rank} {suit}", (x, label_y),
                    font, font_scale, green, thickness)

    # total for the top left: black outline first so its readable on any bg
    total_text = f"Total: {total}"
    pos = (int(15 * scale), int(50 * scale))
    cv2.putText(output, total_text, pos, font, font_scale * 1.3, (0, 0, 0), thickness + 3)
    cv2.putText(output, total_text, pos, font, font_scale * 1.3, (255, 255, 255), thickness)

    return output


if __name__ == "__main__":
    # Running the tests
    assert card_value("K") == 10
    assert card_value("5") == 5
    assert card_value("10") == 10
    assert card_value("unknown") == 0

    # Confirm tests have passed
    print("All card_value tests passed!")
