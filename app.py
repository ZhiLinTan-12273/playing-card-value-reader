# app.py - web demo for the Playing Card Value Reader.
# Run it with:   streamlit run app.py
#
# Two tabs:
#   "Try it"  - build a hand from our deck photos, or upload your own photos
#               (several at once, each with one or more cards); every card is
#               found, read and scored, and the scores are added up
#   "Results" - the whole system run on our test set (answers.csv)
#
# All the image processing is in detect.py, recognise.py and score.py, joined
# by pipeline.py. This file only shows the results.
# Written with help from Claude (Anthropic, 2026).

import csv
import os
import random
from collections import Counter

import cv2
import numpy as np
import streamlit as st

from pipeline import run_pipeline
from recognise import extract_symbols, identify_card
from score import card_value, draw_results

DECK_DIRS = {'test': 'test_images/deck/test',            # never used for templates
             'template': 'test_images/deck/templates'}   # used to build the templates
RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUIT_ORDER = ['spades', 'hearts', 'diamonds', 'clubs']


# ---------- helpers ----------

def deck_photos():
    """All card photos we have: {'K of hearts': path}, sorted by suit then rank."""
    photos = {}
    for kind, folder in DECK_DIRS.items():
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith('.jpg') and '_' in f:
                rank, suit = f[:-4].split('_')
                photos[(rank, suit)] = (os.path.join(folder, f), kind)
    keys = sorted(photos, key=lambda k: (SUIT_ORDER.index(k[1]), RANK_ORDER.index(k[0])))
    return {f'{r} of {s}': photos[(r, s)] for r, s in keys}


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


def same_size(img, w=300, h=400):
    """Fit an image into a w x h frame (white bars), so thumbnails line up."""
    scale = min(w / img.shape[1], h / img.shape[0])
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame = np.full((h, w, 3), 255, np.uint8)
    y, x = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    frame[y:y + img.shape[0], x:x + img.shape[1]] = img
    return frame


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def card_name(rank, suit):
    """('K', 'hearts') -> 'K of hearts'; anything not read is shown as '?'."""
    rank = '?' if rank == 'unknown' else rank
    suit = '?' if suit == 'unknown' else suit
    return f'{rank} of {suit}'


def read_photo(image):
    """Run the whole pipeline on one photo. Returns a dict with the cards read,
    the photo's score, a thumbnail with the outlines, and each straightened card."""
    results, total = run_pipeline(image)
    scale = 600 / max(image.shape[:2])
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_results = [(r, s, (np.float32(o) * scale).astype(np.int32)) for r, s, o in results]
    return {'cards': [(r, s) for r, s, _ in results],
            'total': total,
            'picture': rgb(same_size(draw_results(small, small_results, total))),
            'straight': [straighten(image, o) for _, _, o in results]}


@st.cache_data(show_spinner=False)
def read_deck_photo(path):
    """Same as read_photo, for a photo from our deck (cached, so it is read once)."""
    return read_photo(cv2.imread(path))


def show_hand(photos):
    """Show the cards read from a list of photos and add up the score.
    photos = [(title, result of read_photo), ...]"""
    cards = [c for _, p in photos for c in p['cards']]
    total = sum(p['total'] for _, p in photos)
    unread = sum('unknown' in c for c in cards)

    c1, c2, c3 = st.columns(3)
    c1.metric('Total score', total)
    c2.metric('Cards found', len(cards))
    c3.metric('Not fully read', unread)

    if cards:
        st.write('Read as: ' + ', '.join(f'{card_name(r, s)} ({card_value(r)})' for r, s in cards))

    cols = st.columns(4)
    for i, (title, p) in enumerate(photos):
        found = ', '.join(card_name(*c) for c in p['cards']) or 'no card found'
        cols[i % 4].image(p['picture'], use_container_width=True,
                          caption=f'{title} - read as: {found}')

    with st.expander('Show how each card was read'):
        st.caption('Top-left corner of each straightened card, then the rank and suit '
                   'shapes cut out of it and matched against the templates.')
        for _, p in photos:
            for (rank, suit), card in zip(p['cards'], p['straight']):
                rank_sym, suit_sym = extract_symbols(card)
                c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 3])
                c1.image(rgb(card), width=60)
                c2.image(rgb(card[0:110, 0:45]), caption='corner', width=40)
                if rank_sym is not None:
                    c3.image(rank_sym, caption='rank', width=40)
                    c4.image(suit_sym, caption='suit', width=40)
                c5.write(card_name(rank, suit))


# ---------- page ----------

st.set_page_config(page_title='Playing Card Value Reader', layout='centered')
st.title('Playing Card Value Reader')
st.caption('Finds the cards in a photo, reads each rank and suit, and adds up the score. '
           'A = 1, 2-10 = face value, J/Q/K = 10.')

tab_try, tab_results = st.tabs(['Try it', 'Results'])

# ---------- tab 1: try it ----------

with tab_try:
    mode = st.radio('Choose how to test', ['Pick cards from our deck', 'Upload your own photos'],
                    horizontal=True, label_visibility='collapsed')

    if mode == 'Pick cards from our deck':
        deck = deck_photos()
        if 'hand' not in st.session_state:
            st.session_state['hand'] = []
        c1, c2 = st.columns([3, 1])
        with c2:
            st.write('')
            if st.button('Deal 5 random cards', use_container_width=True):
                st.session_state['hand'] = random.sample(list(deck), 5)
        with c1:
            hand = st.multiselect('Choose cards to make a hand', list(deck), key='hand')
        st.caption('Each card is a separate photo of our own deck; every photo is read '
                   'by the system and the scores are added up.')

        if hand:
            with st.spinner('Reading cards...'):
                photos = [(f'Real card: {name}', read_deck_photo(deck[name][0])) for name in hand]
            show_hand(photos)
            wrong = [name for name, (_, p) in zip(hand, photos) if [tuple(name.split(' of '))] != p['cards']]
            if wrong:
                st.warning('Read differently from the real card: ' + ', '.join(wrong))

    else:
        uploads = st.file_uploader('Upload one or more photos (each may contain several cards)',
                                   type=['jpg', 'jpeg', 'png', 'heic'], accept_multiple_files=True)
        st.caption('Best results: cards lying flat on a plain surface, photographed from above, '
                   'with a little space between them.')
        if uploads:
            photos, failed = [], []
            with st.spinner('Reading cards...'):
                for up in uploads:
                    image = read_upload(up)
                    if image is None:
                        failed.append(up.name)
                    else:
                        photos.append((up.name, read_photo(image)))
            if failed:
                st.error('Could not read: ' + ', '.join(failed))
            if photos:
                show_hand(photos)

# ---------- tab 2: results on the test set ----------

@st.cache_data(show_spinner=False)
def evaluate_all():
    """Run the whole pipeline on every photo in answers.csv that exists."""
    rows = []
    with open('answers.csv', newline='') as f:
        for row in csv.DictReader(f):
            path = os.path.join('test_images', row['file'])
            if cv2.imread(path) is None:
                continue
            truth = [tuple(c.strip().split('-')) for c in row['cards'].split(';')]
            p = read_deck_photo(path)
            rows.append({'file': row['file'], 'truth': truth, 'found': p['cards'],
                         'cards_right': sum((Counter(p['cards']) & Counter(truth)).values()),
                         'true_total': int(row['total']), 'total': p['total'],
                         'correct': p['cards'] == truth, 'picture': p['picture']})
    return rows


with tab_results:
    st.write('The whole system run on every photo in answers.csv. '
             'None of these photos were used to make the templates.')
    if st.button('Run evaluation'):
        with st.spinner('Reading every test photo (about 30 seconds)...'):
            st.session_state['rows'] = evaluate_all()

    rows = st.session_state.get('rows')
    if rows:
        n_cards = sum(len(r['truth']) for r in rows)
        cards_right = sum(r['cards_right'] for r in rows)
        totals_right = sum(r['total'] == r['true_total'] for r in rows)
        wrong = [r for r in rows if not r['correct']]

        c1, c2, c3 = st.columns(3)
        c1.metric('Cards read correctly', f'{cards_right} / {n_cards}')
        c2.metric('Correct total score', f'{totals_right} / {len(rows)}')
        c3.metric('Accuracy', f'{100 * cards_right / n_cards:.1f}%')

        if wrong:
            st.subheader(f'Mistakes ({len(wrong)})')
            for r in wrong:
                c1, c2 = st.columns([1, 2])
                c1.image(r['picture'], use_container_width=True)
                c2.markdown(f"**{r['file']}**")
                c2.write(f"Expected: {', '.join(card_name(*c) for c in r['truth'])} "
                         f"(total {r['true_total']})")
                c2.write(f"Read as: {', '.join(card_name(*c) for c in r['found']) or 'no card found'} "
                         f"(total {r['total']})")
        else:
            st.success(f'All {len(rows)} test photos were read correctly.')

        st.subheader('All test photos')
        st.caption('Each photo with the card outline and the reading the system produced.')
        cols = st.columns(4)
        for i, r in enumerate(rows):
            label = ', '.join(card_name(*c) for c in r['found']) or 'no card found'
            cols[i % 4].image(r['picture'], use_container_width=True,
                              caption=label if r['correct'] else f'WRONG: {label}')

        with st.expander('Show as a table'):
            st.dataframe([{'Photo': r['file'],
                           'Expected': ', '.join(card_name(*c) for c in r['truth']),
                           'Read as': ', '.join(card_name(*c) for c in r['found']) or '-',
                           'Total': f"{r['total']} / {r['true_total']}",
                           'Correct': 'Yes' if r['correct'] else 'No'} for r in rows],
                         use_container_width=True, hide_index=True)
