# app.py - web demo for the Playing Card Value Reader.
# Run it with:   streamlit run app.py
#
# Two tabs:
#   "Try it"  - upload a photo (or pick a sample) and see every card found,
#               read and scored
#   "Results" - the whole system run on our test set (answers.csv)
#
# All the image processing is in detect.py, recognise.py and score.py, joined
# by pipeline.py. This file only shows the results.
# Written with help from Claude (Anthropic, 2026).

import csv
import os
from collections import Counter

import cv2
import numpy as np
import streamlit as st

from pipeline import run_pipeline
from recognise import extract_symbols, identify_card
from score import card_value, draw_results

SAMPLES = {'Four fives (4 cards)': 'test_images/4card.jpg',
           'King of diamonds': 'test_images/deck/test/K_diamonds.jpg',
           'Ten of hearts': 'test_images/deck/test/10_hearts.jpg',
           'Seven of clubs': 'test_images/deck/test/7_clubs.jpg',
           'Queen of hearts': 'test_images/deck/test/Q_hearts.jpg'}
SAMPLES = {name: path for name, path in SAMPLES.items() if os.path.exists(path)}
NONE = 'None'


# ---------- helpers ----------

def read_upload(uploaded):
    """Uploaded file -> BGR image (JPG, PNG or iPhone HEIC). Big photos are shrunk."""
    data = uploaded.getvalue()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:                                   # probably HEIC
        try:
            from io import BytesIO
            from PIL import Image, ImageOps
            from pillow_heif import register_heif_opener
            register_heif_opener()
            pil = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert('RGB')
            image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    scale = 2000 / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def order_corners(pts):
    """4 corners -> top-left, top-right, bottom-right, bottom-left."""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def read_quality(card):
    """How well a card can be read: (symbols found, total match score)."""
    rank, suit, rs, ss = identify_card(card, with_scores=True)
    return ((rank != 'unknown') + (suit != 'unknown'), rs + ss)


def straighten(image, outline):
    """Cut one card out of the photo, upright and the right way up, 200 x 300.
    Only used to SHOW the cards - the reading itself happens in pipeline.py."""
    src = order_corners(np.float32(outline).reshape(4, 2))
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    card = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h))
    if card.shape[1] > card.shape[0]:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    card = cv2.resize(card, (200, 300))
    flipped = cv2.rotate(card, cv2.ROTATE_180)
    return flipped if read_quality(flipped) > read_quality(card) else card


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def card_name(rank, suit):
    """('K', 'hearts') -> 'K of hearts'; anything not read is shown as '?'."""
    rank = '?' if rank == 'unknown' else rank
    suit = '?' if suit == 'unknown' else suit
    return f'{rank} of {suit}'


# ---------- page ----------

st.set_page_config(page_title='Playing Card Value Reader', layout='centered')
st.title('Playing Card Value Reader')
st.caption('Finds the cards in a photo, reads each rank and suit, and adds up the score. '
           'A = 1, 2-10 = face value, J/Q/K = 10.')

tab_try, tab_results = st.tabs(['Try it', 'Results'])

# ---------- tab 1: try it ----------

with tab_try:
    uploaded = st.file_uploader('Upload a photo', type=['jpg', 'jpeg', 'png', 'heic'])
    sample = st.selectbox('Or choose a sample photo', [NONE] + list(SAMPLES))

    if uploaded is not None:
        photo = read_upload(uploaded)
    elif sample != NONE:
        photo = cv2.imread(SAMPLES[sample])
    else:
        photo = None
        st.caption('Best results: cards lying flat on a plain surface, photographed from above.')

    if photo is None and (uploaded is not None or sample != NONE):
        st.error('This photo could not be read.')
    elif photo is not None:
        with st.spinner('Reading cards...'):
            results, total = run_pipeline(photo)

        if not results:
            st.warning('No cards found.')
        else:
            unknown = sum('unknown' in (r, s) for r, s, _ in results)
            c1, c2, c3 = st.columns(3)
            c1.metric('Total score', total)
            c2.metric('Cards found', len(results))
            c3.metric('Not fully read', unknown)

            st.image(rgb(draw_results(photo, results, total)), use_container_width=True)

            cards = [straighten(photo, outline) for _, _, outline in results]
            cols = st.columns(6)
            for i, ((rank, suit, _), card) in enumerate(zip(results, cards)):
                cols[i % 6].image(rgb(card), use_container_width=True,
                                  caption=f'{card_name(rank, suit)}, {card_value(rank)} pts')

            with st.expander('Show how each card was read'):
                st.caption('Top-left corner of the straightened card, then the rank and '
                           'suit shapes cut out of it and matched against the templates.')
                for (rank, suit, _), card in zip(results, cards):
                    rank_sym, suit_sym = extract_symbols(card)
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
                    c1.image(rgb(card[0:110, 0:45]), caption='corner', width=60)
                    if rank_sym is not None:
                        c2.image(rank_sym, caption='rank', width=45)
                        c3.image(suit_sym, caption='suit', width=45)
                    c4.write(card_name(rank, suit))

# ---------- tab 2: results on the test set ----------

@st.cache_data(show_spinner=False)
def evaluate_all():
    """Run the whole pipeline on every photo in answers.csv that exists."""
    rows = []
    with open('answers.csv', newline='') as f:
        for row in csv.DictReader(f):
            image = cv2.imread(os.path.join('test_images', row['file']))
            if image is None:
                continue
            truth = [tuple(c.strip().split('-')) for c in row['cards'].split(';')]
            results, total = run_pipeline(image)
            found = [(r, s) for r, s, _ in results]
            rows.append({'file': row['file'], 'truth': truth, 'found': found,
                         'cards_right': sum((Counter(found) & Counter(truth)).values()),
                         'true_total': int(row['total']), 'total': total})
    return rows


with tab_results:
    st.write('The whole system run on every photo in answers.csv. '
             'None of these photos were used to make the templates.')
    if st.button('Run evaluation'):
        with st.spinner('Reading every test photo...'):
            st.session_state['rows'] = evaluate_all()

    rows = st.session_state.get('rows')
    if rows:
        n_cards = sum(len(r['truth']) for r in rows)
        cards_right = sum(r['cards_right'] for r in rows)
        totals_right = sum(r['total'] == r['true_total'] for r in rows)
        c1, c2, c3 = st.columns(3)
        c1.metric('Photos', len(rows))
        c2.metric('Cards correct', f'{100 * cards_right / n_cards:.1f}%',
                  help=f'{cards_right} of {n_cards} cards')
        c3.metric('Totals correct', f'{100 * totals_right / len(rows):.1f}%',
                  help=f'{totals_right} of {len(rows)} photos')

        table = [{'Photo': r['file'],
                  'Expected': ', '.join(card_name(*c) for c in r['truth']),
                  'Read as': ', '.join(card_name(*c) for c in r['found']) or '-',
                  'Total': f"{r['total']} / {r['true_total']}",
                  'Correct': 'Yes' if r['found'] == r['truth'] else 'No'}
                 for r in rows]
        only_wrong = st.checkbox('Show only mistakes')
        if only_wrong:
            table = [t for t in table if t['Correct'] == 'No']
        st.dataframe(table, use_container_width=True, hide_index=True)
