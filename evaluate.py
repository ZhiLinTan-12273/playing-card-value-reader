"""
There are placeholder values in this file that do not represent the final version of this code. 
Once other team members have finished their work this will be updated accordingly
"""

import numpy as np
from itertools import cycle
from score import card_value

import csv
from collections import Counter, defaultdict

import cv2

from pipeline import run_pipeline
from score import card_value

TEST_ROOT = "test_images"
ANSWERS = "answers.csv"

def parse_cards(text):
    """'K-hearts;7-spades' -> [('K', 'hearts'), ('7','spades')]"""
    return [tuple(c.strip().split("-") for c in text.split(";"))]

def main():
    cards_right = defaultdict(int)          # identified cards per folder
    cards_total = defaultdict(int)          # cards that exist per folder 
    totals_right = defaultdict(int)         # photos with the correct total per folder
    photos = defaultdict(int)               # photos per folder
    failures = []


    with open(ANSWERS, newline="") as f:
        for row in csv.DictReader(f):
            file = row["file"]
            folder = file.split("/")[0]
            truth = parse_cards(row["cards"])
            true_total = int(row["total"])

            # sanity check on my own answers
            expected = sum(card_value(rank) for rank, _ in truth)
            if expected != true_total:
                print(f"WARNING {file}: csv total {true_total} but cards add up to {expected}")

            image = cv2.imread(f"{TEST_ROOT}/{file}")
            if image is None:
                print(f"SKIPPED {file}: could not load image")
                continue

            results, total = run_pipeline(image)
            predicted = [(rank, suit) for rank, suit, _ in results]

            # a card counts as correct if its (rank, suit) was found
            # (order doesn't matter, duplicates handled by Counter)
            matched = sum((Counter(predicted) & Counter(truth)).values())

            photos[folder] += 1
            cards_total[folder] += len(truth)
            cards_right[folder] += matched
            if total == true_total:
                totals_right[folder] += 1
            else:
                failures.append((file, true_total, total))

    print(f"\n{'Folder':<14}{'Photos':>7}{'Card accuracy':>16}{'Correct totals':>17}")
    print("-" * 54)
    for folder in sorted(photos):
        card_acc = 100 * cards_right[folder] / cards_total[folder]
        total_acc = 100 * totals_right[folder] / photos[folder]
        print(f"{folder:<14}{photos[folder]:>7}{card_acc:>15.1f}%{total_acc:>16.1f}%")

    if failures:
        print("\nPhotos with the wrong total (send these to A and B):")
        for file, expected, got in failures:
            print(f"  {file}: expected {expected}, got {got}")


if __name__ == "__main__":
    main()