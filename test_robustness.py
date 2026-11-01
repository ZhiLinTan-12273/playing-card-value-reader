# test_robustness.py - Member B: how well does identify_card cope with
# imperfect cards? Applies distortions from the Transformation lab
# (brightness, blur, rotation, noise) and compares three matching methods.
# Written with help from Claude (Anthropic, 2026).

import cv2
import numpy as np
from recognise import identify_card

CARDS = [('test_images/warped/ah.jpg', 'A', 'hearts'),
         ('test_images/single_cards/kh.jpg', 'K', 'hearts')]

METHODS = {'CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
           'CCORR_NORMED': cv2.TM_CCORR_NORMED,
           'SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED}


def rotate(card, angle):
    """Rotate around the centre, filling the gaps with white like the card."""
    h, w = card.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(card, M, (w, h), borderValue=(255, 255, 255))


def add_noise(card, sigma):
    """Add random Gaussian noise, like a photo taken in poor light."""
    noise = np.random.default_rng(0).normal(0, sigma, card.shape)
    return np.clip(card + noise, 0, 255).astype(np.uint8)


DISTORTIONS = {
    'original':   lambda c: c,
    'darker':     lambda c: cv2.convertScaleAbs(c, alpha=0.6),
    'brighter':   lambda c: cv2.convertScaleAbs(c, alpha=1.3, beta=30),
    'blur 5x5':   lambda c: cv2.GaussianBlur(c, (5, 5), 0),
    'blur 9x9':   lambda c: cv2.GaussianBlur(c, (9, 9), 0),
    'noise':      lambda c: add_noise(c, 20),
    'rotate 3':   lambda c: rotate(c, 3),
    'rotate 6':   lambda c: rotate(c, 6),
    'rotate 10':  lambda c: rotate(c, 10),
}


def main():
    cards = [(cv2.resize(cv2.imread(p), (200, 300)), r, s) for p, r, s in CARDS]
    correct = {m: 0 for m in METHODS}
    tests = 0

    for d_name, distort in DISTORTIONS.items():
        line = f'{d_name:<11}'
        for card, true_rank, true_suit in cards:
            tests += 1
            test_card = distort(card)
            for m_name, method in METHODS.items():
                rank, suit, rs, ss = identify_card(test_card, method, with_scores=True)
                ok = (rank, suit) == (true_rank, true_suit)
                correct[m_name] += ok
                if m_name == 'CCOEFF_NORMED':      # show details for the method I use
                    mark = 'OK  ' if ok else 'FAIL'
                    line += f'  {true_rank}: {rank:>7} {suit:<8} {rs:.2f}/{ss:.2f} {mark}'
        print(line)

    print('\nAccuracy by matching method:')
    for m_name in METHODS:
        print(f'  {m_name:<14} {correct[m_name]}/{tests}  ({100 * correct[m_name] / tests:.0f}%)')


if __name__ == '__main__':
    main()
