import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Dependent vowel signs (Kars) and diacritics not present as standalone classes in BanglaLekha-Isolated
# (the virama ্ is included because it only joins letters into conjuncts and is never a CNN class)
BANGLA_KARS = set("ািীুূৃেৈোৌৗ্")

# Curated Bangla lexicon for whole-word & vowel-sign recovery
BANGLA_LEXICON = [
    "জল", "মানুষ", "বাংলা", "বাংলাদেশ", "কলম", "বই", "নদী", "ফুল", "আকাশ", "মাটি",
    "দেশ", "শিক্ষা", "স্কুল", "কলেজ", "বিশ্ববিদ্যালয়", "ছাত্র", "শিক্ষক", "বন্ধু",
    "পরিবার", "মা", "বাবা", "ভাই", "বোন", "ছেলে", "মেয়ে", "বাড়ি", "ঘর", "রাস্তা",
    "গাড়ি", "ট্রেন", "সময়", "দিন", "রাত", "সকাল", "বিকেল", "সন্ধ্যা", "সূর্য", "চাঁদ",
    "তারা", "বৃষ্টি", "বাতাস", "আগুন", "পাখি", "মাছ", "গাছ", "পাতা", "ফল", "ভাত",
    "রুটি", "দুধ", "চা", "চিনি", "লবণ", "কাজ", "কথা", "গান", "কবিতা", "গল্প",
    "ছবি", "রং", "লাল", "নীল", "সবুজ", "হলুদ", "সাদা", "কালো", "ভালো", "মন্দ",
    "সুন্দর", "বড়", "ছোট", "নতুন", "পুরাতন", "সহজ", "কঠিন", "সত্য", "মিথ্যা",
    "শান্তি", "স্বাধীনতা", "বিজয়", "ভাষা", "সাহিত্য", "বিজ্ঞান", "প্রযুক্তি", "কম্পিউটার",
]


# Maximum normalized edit distance for accepting a lexicon word (0 = perfect, ~0.85 = nothing matches)
MAX_LEXICON_SCORE = 0.55

# Edit-distance costs
DELETE_COST = 0.85        # an extra segmented piece
BASE_INSERT_COST = 0.85   # a base letter the segmenter missed
KAR_INSERT_COST = 0.15    # a vowel sign (never recognized by the CNN)
MERGE_PENALTY = 0.12      # one segment covering several characters
MAX_GROUP = 3

# Class indices 0-49 are the letters (folders 1-50); 50-59 are digits and 60-83 are compounds.
# Words are made of letters, so the app only lets the CNN pick among the 50 letters.
NUM_LETTER_CLASSES = 50


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("models/model.keras")
    with open("labels.json", "r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return model, labels


def restrict_to_letters(preds: np.ndarray) -> np.ndarray:
    """Zero out digit/compound classes and renormalize so each row sums to 1 over letters."""
    out = preds.copy()
    out[:, NUM_LETTER_CLASSES:] = 0.0
    sums = out.sum(axis=1, keepdims=True)
    return out / np.maximum(sums, 1e-9)


def strip_kars(word: str) -> str:
    """Remove dependent vowel signs to obtain the base character skeleton."""
    return "".join(ch for ch in word if ch not in BANGLA_KARS)


def split_wide_box(working: np.ndarray, box, depth: int = 0):
    """
    Split a box that is too wide to be one letter (touching letters) at the column with the
    least ink near its middle (vertical projection valley). Returns a list of boxes.
    """
    x, y, w, h = box
    if depth >= 2 or w < 1.6 * h:
        return [box]

    cols = np.sum(working[y : y + h, x : x + w] > 0, axis=0).astype(np.float32)
    lo, hi = int(w * 0.35), int(w * 0.65)
    if hi <= lo:
        return [box]
    cut = lo + int(np.argmin(cols[lo:hi]))

    # Only split if the valley is clearly thinner than the typical stroke density
    if cols[cut] > 0.5 * max(float(cols.mean()), 1.0):
        return [box]

    left, right = (x, y, cut, h), (x + cut, y, w - cut, h)
    return split_wide_box(working, left, depth + 1) + split_wide_box(working, right, depth + 1)


def preprocess_and_segment(rgba_image: np.ndarray, img_size: int = 64):
    """
    1. Convert RGBA canvas (black ink on white bg) -> binary mask (white ink on black bg).
    2. Detect top matra line via horizontal projection and suppress it to separate joined letters.
    3. Extract connected components left-to-right and crop from the un-erased image.
    """
    gray = cv2.cvtColor(rgba_image.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    # Invert & threshold: strokes become 255, background becomes 0
    _, binary = cv2.threshold(255 - gray, 40, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return []

    # Tight bounding box around the entire drawn word
    wx, wy, ww, wh = cv2.boundingRect(coords)
    word_crop = binary[wy : wy + wh, wx : wx + ww]

    # Locate top headline (matra) in upper 35% of word height
    upper_limit = max(1, int(wh * 0.35))
    horiz_proj = np.sum(word_crop[:upper_limit, :] > 0, axis=1)
    matra_row = int(np.argmax(horiz_proj))

    # Suppress matra band on a working copy only if a prominent horizontal bar exists
    working = word_crop.copy()
    if horiz_proj[matra_row] > ww * 0.25:
        r_start = max(0, matra_row - 2)
        r_end = min(wh, matra_row + 3)
        working[r_start:r_end, :] = 0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(working, connectivity=8)

    boxes = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        # Filter tiny stroke specks
        if area < 30 or (w < 6 and h < 6):
            continue
        boxes.append((x, y, w, h))

    # Merge vertically overlapping fragments (e.g., upper/lower parts of the same letter)
    boxes.sort(key=lambda b: b[0])
    merged = []
    for box in boxes:
        if not merged:
            merged.append(list(box))
            continue
        px, py, pw, ph = merged[-1]
        cx, cy, cw, ch = box
        overlap_x = min(px + pw, cx + cw) - max(px, cx)
        if overlap_x > 0.45 * min(pw, cw):
            nx = min(px, cx)
            ny = min(py, cy)
            nw = max(px + pw, cx + cw) - nx
            nh = max(py + ph, cy + ch) - ny
            merged[-1] = [nx, ny, nw, nh]
        else:
            merged.append([cx, cy, cw, ch])

    # Split blobs that contain several touching letters
    split_boxes = []
    for box in merged:
        split_boxes.extend(split_wide_box(working, tuple(box)))

    segments = []
    for x, y, w, h in split_boxes:
        # Expand crop slightly vertically to restore the matra row for CNN classification
        y0 = max(0, y - 4)
        y1 = min(wh, y + h + 2)
        char_crop = word_crop[y0:y1, x : x + w]

        side = max(char_crop.shape[0], char_crop.shape[1]) + 8
        padded = np.zeros((side, side), dtype=np.uint8)
        yo = (side - char_crop.shape[0]) // 2
        xo = (side - char_crop.shape[1]) // 2
        padded[yo : yo + char_crop.shape[0], xo : xo + char_crop.shape[1]] = char_crop

        resized = cv2.resize(padded, (img_size, img_size), interpolation=cv2.INTER_AREA)
        norm = (resized.astype(np.float32) / 255.0)[..., np.newaxis]
        segments.append((resized, norm))

    return segments


def weighted_edit_distance(topk_preds: list[dict[str, float]], target_str: str) -> float:
    """
    Compute edit distance between CNN top-k distributions and a target character sequence.
    Matching any character in a segment's top-k incurs low cost (1 - prob).
    """
    m = len(topk_preds)
    n = len(target_str)
    # Dependent vowel signs are never recognized by the CNN, so missing one is cheap
    insert_cost = [KAR_INSERT_COST if ch in BANGLA_KARS else BASE_INSERT_COST for ch in target_str]

    dp = np.full((m + 1, n + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for j in range(1, n + 1):
        dp[0, j] = dp[0, j - 1] + insert_cost[j - 1]
    for i in range(1, m + 1):
        dp[i, 0] = dp[i - 1, 0] + DELETE_COST

    for i in range(1, m + 1):
        probs = topk_preds[i - 1]
        for j in range(1, n + 1):
            best = min(
                dp[i - 1, j] + DELETE_COST,          # extra segmented piece
                dp[i, j - 1] + insert_cost[j - 1],   # missed character / vowel sign
            )
            # One segment may cover 1-3 consecutive target characters (touching letters
            # or a letter with its vowel sign drawn as one blob)
            for k in range(1, min(MAX_GROUP, j) + 1):
                bases = [c for c in target_str[j - k : j] if c not in BANGLA_KARS]
                if not bases:
                    continue
                p = max(probs.get(c, 0.0) for c in bases)
                match_cost = (1.0 - p) * 0.5 if p > 0 else 1.0
                best = min(best, dp[i - 1, j - k] + match_cost + MERGE_PENALTY * (k - 1))
            dp[i, j] = best
    return float(dp[m, n])


def recover_word_from_lexicon(topk_preds: list[dict[str, float]], lexicon: list[str]) -> str:
    """
    Match predicted sequence against both full lexicon words and kar-stripped skeletons
    so dependent vowel signs (া, ি, ী, ু, etc.) are restored into valid Unicode words.
    """
    raw_word = "".join(max(p.items(), key=lambda kv: kv[1])[0] for p in topk_preds)
    best_word = raw_word
    best_score = float("inf")
    m = len(topk_preds)

    for word in lexicon:
        # Vowel signs are cheap to insert, so one comparison against the full word is enough
        score = weighted_edit_distance(topk_preds, word)

        # Normalize by the longer of (segments, base letters) so neither long nor short
        # lexicon words get an unfair advantage
        norm_score = score / max(m, len(strip_kars(word)), 1)
        if norm_score < best_score:
            best_score = norm_score
            best_word = word

    # If nothing in the lexicon is a reasonable match, show the raw CNN reading instead
    if best_score > MAX_LEXICON_SCORE:
        return raw_word
    return best_word


def main():
    st.set_page_config(page_title="Bangla Handwritten Word OCR", layout="centered")
    st.title("Bangla Handwritten Word Recognition")
    st.caption("Draw a single Bangla word (e.g., জল or মানুষ) in the box below and click Predict.")

    if not Path("models/model.keras").exists():
        st.error("Trained model not found at `models/model.keras`. Run `python train.py` first.")
        return

    model, labels = load_assets()
    img_size = model.input_shape[1]

    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=7,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=220,
        width=650,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

    col1, col2 = st.columns([1, 1])
    predict_clicked = col1.button("Predict Word", type="primary", use_container_width=True)
    clear_clicked = col2.button("Clear Canvas", use_container_width=True)

    if clear_clicked:
        st.session_state.canvas_key += 1
        st.rerun()

    if predict_clicked:
        if canvas_result.image_data is None:
            st.warning("Canvas is empty. Please draw a Bangla word first.")
            return

        segments = preprocess_and_segment(canvas_result.image_data, img_size=img_size)
        if not segments:
            st.warning("Canvas is empty! Please draw a Bangla word before clicking Predict.")
            return

        batch = np.stack([seg[1] for seg in segments], axis=0)
        preds = restrict_to_letters(model.predict(batch, verbose=0))

        topk_list = []
        raw_chars = []
        for prob_vec in preds:
            top_indices = np.argsort(prob_vec)[::-1][:5]
            topk_dict = {labels[int(idx)]: float(prob_vec[idx]) for idx in top_indices}
            topk_list.append(topk_dict)
            raw_chars.append(labels[int(top_indices[0])])

        raw_joined = "".join(raw_chars)
        final_word = recover_word_from_lexicon(topk_list, BANGLA_LEXICON)

        st.subheader("Predicted Bangla Word")
        st.markdown(
            f"<div style='padding:16px; background-color:#f0f2f6; border-radius:8px; text-align:center;'>"
            f"<span style='font-size:64px; font-weight:bold; color:#111;'>{final_word}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("Segmentation & Per-Character Debug Info")
        st.write(f"**Raw CNN Top-1 Sequence (before lexicon vowel recovery):** `{raw_joined}`")

        cols = st.columns(len(segments))
        for idx, (col, (vis_img, _), prob_vec) in enumerate(zip(cols, segments, preds)):
            top1_idx = int(np.argmax(prob_vec))
            conf = float(prob_vec[top1_idx]) * 100.0
            col.image(vis_img, caption=f"#{idx+1}: {labels[top1_idx]} ({conf:.1f}%)", width=80)


if __name__ == "__main__":
    main()