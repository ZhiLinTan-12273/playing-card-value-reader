# The scoring rule: A = 1, 2-10 = face value, J/Q/K = 10

CARD_VALUES = { 
    "A": 1
    "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, 
    "J": 10, "Q" 10, "K": 10,
}

def card_value(rank):
    """Input: rank string (e.g. "K", "A", "5").
    Returns: the points as an int. Returns 0 for an unrecognised rank
    (e.g. "unknown" from identify_card)."""
    return CARD_VALUES.get(rank,0)

assert card_value("K") == 10
assert card_value("5") == 5
assert card_value("10") == 10
assert card_value("unknown") == 0

print("All card_value tests passed!")