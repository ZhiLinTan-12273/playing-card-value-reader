# main.py - Playing Card Value Reader
# Usage:  python main.py <photo>          e.g.  python main.py test_images/4card.jpg
# Finds every card, reads its rank and suit, adds up the score, and saves
# result.png with each card outlined and labelled.

import sys

import cv2
import matplotlib.pyplot as plt

from pipeline import run_pipeline
from score import card_value, draw_results


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "test_images/4card.jpg"
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Could not load {path}. Check the path.")

    results, total = run_pipeline(image)
    print(f"{path}: {len(results)} card(s) found")
    for rank, suit, _ in results:
        print(f"  {rank} of {suit}: {card_value(rank)} points")
    print(f"Total score: {total}")

    annotated = draw_results(image, results, total)
    cv2.imwrite("result.png", annotated)
    print("Saved result.png")

    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
