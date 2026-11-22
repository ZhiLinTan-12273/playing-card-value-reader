# app.py - web demo for the Playing Card Value Reader.
# Run it with:   streamlit run app.py
#
# Two tabs:
#   "Read your cards" - upload a photo (or pick a sample) and see every card
#                       found, read and scored
#   "Results"         - the whole system run on our test set (answers.csv)
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

SUIT_SYMBOL = {'hearts': '♥', 'diamonds': '♦', 'clubs': '♣', 'spades': '♠'}
RED = ('hearts', 'diamonds')
SAMPLES = {'Four fives (4 cards in one photo)': 'test_images/4card.jpg',
           'King of diamonds (face card)': 'test_images/deck/test/K_diamonds.jpg',
           'Ten of hearts': 'test_images/deck/test/10_hearts.jpg',
           'Seven of clubs': 'test_images/deck/test/7_clubs.jpg',
           'Queen of hearts': 'test_images/deck/test/Q_hearts.jpg'}
SAMPLES = {name: path for name, path in SAMPLES.items() if os.path.exists(path)}


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
    """How well a card can be read: (rank and suit both found, total match score)."""
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


def card_label(rank, suit):
    if 'unknown' in (rank, suit):
        return '❓ not recognised'
    return f'{rank} {SUIT_SYMBOL[suit]}'


# ---------- page ----------

st.set_page_config(page_title='Playing Card Value Reader', page_icon='🃏', layout='wide')
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1100px;}
  .card-label {font-size: 1.4rem; font-weight: 700; text-align: center; margin: 0;}
  .red {color: #d62828;} .black {color: #222;}
  .pts {text-align: center; color: #888; margin: 0 0 1rem 0;}
</style>""", unsafe_allow_html=True)

st.title('🃏 Playing Card Value Reader')
st.caption('TU Dublin Image Processing project · find the cards in a photo, '
           'read each rank and suit, and add up the score (A = 1, 2–10 = face value, J/Q/K = 10)')

tab_try, tab_results = st.tabs(['Read your cards', 'Results'])

# ---------- tab 1: try it ----------

with tab_try:
    left, right = st.columns([2, 1])
    with left:
        uploaded = st.file_uploader('Upload a photo of playing cards',
                                    type=['jpg', 'jpeg', 'png', 'heic'])
    with right:
        sample = st.selectbox('…or try a sample photo', ['—'] + list(SAMPLES))

    if uploaded is not None:
        photo = read_upload(uploaded)
    elif sample != '—':
        photo = cv2.imread(SAMPLES[sample])
    else:
        photo = None
        st.info('Tip: take the photo from above, in good light, with a little '
                'space between the cards.')

    if photo is None and (uploaded is not None or sample != '—'):
        st.error('Sorry, that photo could not be read as an image.')
    elif photo is not None:
        with st.spinner('Finding and reading cards…'):
            results, total = run_pipeline(photo)

        if not results:
            st.warning('No cards found. Try a plainer background and a photo from directly above.')
        else:
            unknown = sum('unknown' in (r, s) for r, s, _ in results)
            m1, m2, m3 = st.columns(3)
            m1.metric('Total score', total)
            m2.metric('Cards found', len(results))
            m3.metric('Not recognised', unknown)

            st.image(rgb(draw_results(photo, results, total)), use_container_width=True)

            st.subheader('Cards')
            cards = [straighten(photo, outline) for _, _, outline in results]
            cols = st.columns(min(len(results), 6))
            for i, ((rank, suit, _), card) in enumerate(zip(results, cards)):
                with cols[i % len(cols)]:
                    st.image(rgb(card), use_container_width=True)
                    colour = 'red' if suit in RED else 'black'
                    st.markdown(f'<p class="card-label {colour}">{card_label(rank, suit)}</p>'
                                f'<p class="pts">{card_value(rank)} points</p>',
                                unsafe_allow_html=True)

            with st.expander('How each card was read'):
                st.caption('Top-left corner of the straightened card → threshold → '
                           'rank and suit shapes that are matched against our templates.')
                for (rank, suit, _), card in zip(results, cards):
                    rank_sym, suit_sym = extract_symbols(card)
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
                    c1.image(rgb(card[0:110, 0:45]), caption='corner', width=70)
                    if rank_sym is not None:
                        c2.image(rank_sym, caption='rank', width=50)
                        c3.image(suit_sym, caption='suit', width=50)
                    c4.markdown(f'**{card_label(rank, suit)}**')

# ---------- tab 2: results on the test set ----------

@st.cache_data(show_spinner=False)
def evaluate_all():
    """Run the whole pipeline on every photo in answers.csv that exists."""
    rows = []
    with open('answers.csv', newline='') as f:
        for row in csv.DictReader(f):
            path = os.path.join('test_images', row['file'])
            image = cv2.imread(path)
            if image is None:
                continue
            truth = [tuple(c.strip().split('-')) for c in row['cards'].split(';')]
            results, total = run_pipeline(image)
            found = [(r, s) for r, s, _ in results]
            rows.append({'file': row['file'], 'path': path,
                         'truth': truth, 'found': found,
                         'cards_right': sum((Counter(found) & Counter(truth)).values()),
                         'true_total': int(row['total']), 'total': total})
    return rows


with tab_results:
    st.write('The whole system run on every photo in `answers.csv`. None of these '
             'photos were used to make the templates.')
    if st.button('Run the evaluation', type='primary'):
        with st.spinner('Reading every test photo…'):
            st.session_state['rows'] = evaluate_all()

    rows = st.session_state.get('rows')
    if rows:
        n_cards = sum(len(r['truth']) for r in rows)
        cards_right = sum(r['cards_right'] for r in rows)
        totals_right = sum(r['total'] == r['true_total'] for r in rows)
        m1, m2, m3 = st.columns(3)
        m1.metric('Photos tested', len(rows))
        m2.metric(f'Cards read correctly ({cards_right} of {n_cards})',
                  f'{100 * cards_right / n_cards:.1f}%')
        m3.metric(f'Correct total score ({totals_right} of {len(rows)})',
                  f'{100 * totals_right / len(rows):.1f}%')

        wrong = [r for r in rows if r['found'] != r['truth']]
        st.subheader('Mistakes' if wrong else 'No mistakes 🎉')
        for r in wrong:
            c1, c2 = st.columns([1, 3])
            c1.image(rgb(cv2.imread(r['path'])), use_container_width=True)
            c2.markdown(f"**{r['file']}**  \n"
                        f"Expected: {', '.join(card_label(*c) for c in r['truth'])} "
                        f"(total {r['true_total']})  \n"
                        f"Read as: {', '.join(card_label(*c) for c in r['found']) or 'nothing'} "
                        f"(total {r['total']})")

        st.subheader('Every test photo')
        cols = st.columns(6)
        for i, r in enumerate(rows):
            ok = r['found'] == r['truth']
            with cols[i % 6]:
                st.image(rgb(cv2.resize(cv2.imread(r['path']), (150, 200))),
                         caption=('✅ ' if ok else '❌ ') + ', '.join(card_label(*c) for c in r['found']),
                         use_container_width=True)
