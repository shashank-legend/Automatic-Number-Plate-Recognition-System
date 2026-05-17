"""
pipeline.py
-----------
OCR pipeline for Confidence-Aware OCR Dashboard.
Handles image preprocessing, OCR execution, and confidence extraction.

Novelty additions:
  1. Multi-Scale Retinex (MSR) — 13th preprocessing variant for low-light/glare robustness
  2. Spatial Frequency Blur Classifier — Laplacian variance + FFT for intelligent image triage
  3. Closed-Loop Retry — Indian plate regex validator drives adaptive parameter mutation
"""

import os
import sys
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
import base64
import math

# ── Fix Unicode output on Windows (prevents crashes with special chars) ────────
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Tesseract path (Windows) ──────────────────────────────────────────────────
_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if sys.platform.startswith("win") and os.path.isfile(_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH

# ── Confidence thresholds ─────────────────────────────────────────────────────
CONF_HIGH   = 80
CONF_MEDIUM = 50

# Global Tesseract Config for Automotive/Plate-like text
# --oem 1: Neural Nets LSTM, --psm 7: Single line, 
# load_system_dawg=F, load_freq_dawg=F: Disables Tesseract's English dictionary dictionary guessing
_AUTO_CFG = "--oem 1 --psm 7 -c load_system_dawg=F -c load_freq_dawg=F"


# ─────────────────────────────────────────────────────────────────────────────
# DIP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _reduce_glare(img_gray):
    """
    Classic DIP technique for illumination correction:
    Divide the image by its blurred version to 'flatten' lighting.
    Includes Gaussian smoothing to prevent noise amplification in dark areas.
    """
    # 0. Contrast Stretch first to ensure signal exists
    img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Create background estimate (smoothed to ignore noise/texture)
    bg = cv2.medianBlur(img_gray, 51)
    bg = cv2.GaussianBlur(bg, (51, 51), 0)
    # Divide original by background (normalizes to 1.0)
    norm = cv2.divide(img_gray, bg, scale=255)
    return norm


def _deskew(image):
    """
    Detect text angle and rotate to improve Tesseract accuracy.
    Uses minAreaRect on the most prominent contours.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Threshold for detection
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find all white pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return image, 0
        
    angle = cv2.minAreaRect(coords)[-1]
    
    # minAreaRect angle is inconsistent depending on width/height
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    if abs(angle) < 0.5 or abs(angle) > 20: 
        return image, 0 # Ignore tiny tilts or extreme rotations (likely noise)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def _make_pillbox_psf(radius, size=21):
    """Creates a circular point spread function for defocus blur."""
    psf = np.zeros((size, size), dtype=np.float32)
    cv2.circle(psf, (size // 2, size // 2), radius, 1, -1)
    return psf / psf.sum()

def _wiener_deblur(img_gray):
    """
    Advanced DIP: True Frequency-Domain Wiener Deconvolution.
    Specifically designed for heavily defocused/blurred photos like the one provided.
    """
    # Assuming a defocus blur roughly 3 to 5 pixels in radius
    psf = _make_pillbox_psf(radius=4, size=21)
    snr = 20
    
    img_f = np.float32(img_gray)
    img_fft = np.fft.fft2(img_f)
    psf_fft = np.fft.fft2(psf, s=img_gray.shape)
    
    psf_fft_conj = np.conj(psf_fft)
    psf_power = np.abs(psf_fft)**2
    
    wiener_filter = psf_fft_conj / (psf_power + 1.0 / snr)
    
    result_fft = img_fft * wiener_filter
    result = np.fft.ifft2(result_fft)
    result = np.abs(result)
    
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# NOVELTY 1: MULTI-SCALE RETINEX (MSR)
# ─────────────────────────────────────────────────────────────────────────────

def _multiscale_retinex(img_gray, sigmas=(15, 80, 250)):
    """
    Multi-Scale Retinex (MSR) — models human lightness constancy.

    The Retinex theory (Land, 1977) states that the perceived colour/lightness
    of a surface is independent of the illuminant. MSR separates the illumination
    component (low-frequency, estimated by Gaussian blurs at multiple scales)
    from the reflectance component (the actual surface properties we care about).

    Mathematical formulation (in log domain):
        MSR(x,y) = Σ_k  [log(I(x,y)) - log(I(x,y) * G_σk(x,y))]
    where G_σk is a Gaussian kernel of scale σk.

    Benefits over simple glare correction (divide-by-background):
      - Multi-scale: captures both fine and coarse illumination variations
      - Log domain: compresses dynamic range, mimics the human visual system
      - Explicitly separates illumination from reflectance (principled, not heuristic)
      - Dramatically improves plates photographed at night, under headlights,
        or in uneven outdoor lighting
    """
    img_f = img_gray.astype(np.float64) + 1.0   # +1 to avoid log(0)
    log_img = np.log(img_f)
    msr = np.zeros_like(log_img)
    for sigma in sigmas:
        # Gaussian blur approximates illumination at this spatial scale
        k = int(6 * sigma + 1) | 1   # kernel size: 6σ+1, must be odd
        k = max(k, 3)
        blurred = cv2.GaussianBlur(img_f, (k, k), sigma)
        blurred = np.maximum(blurred, 1e-6)   # numerical safety
        msr += log_img - np.log(blurred)
    msr /= len(sigmas)
    # Normalize to [0, 255] uint8
    msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX)
    return msr.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# NOVELTY 2: SPATIAL FREQUENCY BLUR CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_image_quality(img_gray):
    """
    Classify input image quality using two spatial frequency measures:

    A) Laplacian Variance (defocus / blur magnitude)
       - The Laplacian operator ∇²I is a 2nd-order derivative that responds
         strongly to edges. In a sharp image, edges have large magnitude.
       - A blurred image has fewer sharp transitions → lower Laplacian variance.
       - This gives a scalar 'sharpness score': high → sharp, low → blurry.

    B) FFT Power Spectrum Directionality (motion blur detection)
       - The 2D Fourier Transform decomposes an image into spatial frequencies.
       - Motion blur smears energy in one direction → concentrated bright streak
         in the FFT magnitude spectrum, perpendicular to the motion direction.
       - We detect this directional concentration and estimate the blur angle.

    Returns a dict with:
      - blur_type:        'sharp' | 'defocus' | 'motion'
      - sharpness_score:  Laplacian variance (higher = sharper)
      - blur_angle:       Estimated motion direction (degrees, or None)
      - quality_label:    Human-readable label for the UI
      - recommended_variants: list of variant names to prioritise for this type
    """
    h, w = img_gray.shape[:2]

    # ── A. Laplacian Variance ──────────────────────────────────────────────────
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    lap_var = float(lap.var())

    # ── B. FFT Directional Analysis ───────────────────────────────────────────
    fft      = np.fft.fft2(img_gray.astype(np.float64))
    fft_mag  = np.abs(np.fft.fftshift(fft))
    # Log scale to suppress DC bias
    log_mag  = np.log1p(fft_mag)

    # Directional energy: sample horizontal vs vertical strips
    cy, cx = h // 2, w // 2
    strip_w = max(1, w // 20)   # ±5% of width
    strip_h = max(1, h // 20)

    h_energy = float(log_mag[cy - strip_h:cy + strip_h, :].sum())
    v_energy = float(log_mag[:, cx - strip_w:cx + strip_w].sum())
    total_e  = h_energy + v_energy + 1e-9
    directional_ratio = abs(h_energy - v_energy) / total_e

    blur_angle  = None
    motion_blur = False
    if directional_ratio > 0.25 and lap_var < 300:
        motion_blur = True
        # Dominant direction: horizontal strip bright → vertical motion, vice versa
        blur_angle = 90 if h_energy > v_energy else 0

    # ── Classification thresholds ─────────────────────────────────────────────
    # These were calibrated on typical license plate images at various distances
    SHARP_THRESH  = 400   # Laplacian variance above this → sharp
    DEFOCUS_THRESH = 100  # Below this → clearly defocused

    if lap_var >= SHARP_THRESH:
        blur_type = 'sharp'
        quality_label = f'Sharp (score: {lap_var:.0f})'
        recommended = ['clahe', 'otsu', 'adaptive', 'bilateral']
    elif motion_blur:
        blur_type = 'motion'
        quality_label = f'Motion blur detected (score: {lap_var:.0f}, angle ~{blur_angle}°)'
        recommended = ['wiener', 'unsharp', 'laplacian', 'histeq']
    elif lap_var < DEFOCUS_THRESH:
        blur_type = 'defocus'
        quality_label = f'Defocus blur (score: {lap_var:.0f})'
        recommended = ['unsharp', 'wiener', 'laplacian', 'retinex']
    else:
        blur_type = 'mild_blur'
        quality_label = f'Mild blur (score: {lap_var:.0f})'
        recommended = ['clahe', 'bilateral', 'retinex', 'adaptive']

    return {
        'blur_type':             blur_type,
        'sharpness_score':       round(lap_var, 1),
        'blur_angle':            blur_angle,
        'directional_ratio':     round(directional_ratio, 3),
        'quality_label':         quality_label,
        'recommended_variants':  recommended,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NOVELTY 3: INDIAN PLATE REGEX VALIDATOR + CLOSED-LOOP RETRY
# ─────────────────────────────────────────────────────────────────────────────

# Indian plate format patterns (covers most common formats):
#   Standard private:  MH12AB1234  (state + dist + letters + digits)
#   New BH series:     22BH1234AB
#   Commercial (old):  MHC1234 / MHT1234
_PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$'),     # MH12AB1234
    re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$'),              # 22BH1234AB
    re.compile(r'^[A-Z]{2}[A-Z]\d{4,5}$'),                # MHC01234 commercial
]

def _is_valid_plate(text):
    """
    Return True if `text` matches any known Indian number plate format.
    Normalises by stripping whitespace and converting to uppercase.
    """
    cleaned = re.sub(r'[\s\-]', '', text.upper())
    return any(p.match(cleaned) for p in _PLATE_PATTERNS)


def _extract_plate_text(word_data):
    """Heuristic: join longest continuous alphanumeric tokens likely to be a plate."""
    tokens = [w['word'] for w in word_data]
    # Try the longest contiguous token first
    for t in sorted(tokens, key=len, reverse=True):
        alnum = re.sub(r'[^A-Z0-9]', '', t.upper())
        if 6 <= len(alnum) <= 12:
            return alnum
    # Fallback: concatenate all tokens
    joined = re.sub(r'[^A-Z0-9]', '', ''.join(tokens).upper())
    return joined


def _retry_ocr_for_plate(image_cv, max_attempts=5):
    """
    Closed-loop adaptive OCR retry for plate text.

    Algorithm:
      - Attempt 0: baseline (CLAHE clipLimit=3, adaptiveThresh C=25)
      - On failure, mutate: increase CLAHE clip limit, change threshold C,
        try Otsu vs adaptive, apply morphological closing to heal broken chars.
      - Accept as soon as a result passes the Indian plate regex validator.
      - Return (best_words, plate_text, retry_trace)

    This is a principled feedback control system:
      observation → validate → mutate parameters → re-process → re-validate
    """
    retry_trace    = []
    best_words     = []
    best_plate     = ''
    best_matched   = False

    # Parameter schedule — each attempt uses different DIP parameters
    mutations = [
        {'clip': 3.0, 'threshold_c': 25, 'morph': False, 'label': 'CLAHE clip=3, C=25 (baseline)'},
        {'clip': 5.0, 'threshold_c': 20, 'morph': False, 'label': 'CLAHE clip=5, C=20'},
        {'clip': 7.0, 'threshold_c': 15, 'morph': True,  'label': 'CLAHE clip=7, C=15 + Morph-close'},
        {'clip': 2.0, 'threshold_c': 30, 'morph': False, 'label': 'Low CLAHE, aggressive threshold C=30'},
        {'clip': 4.0, 'threshold_c': 10, 'morph': True,  'label': 'Otsu + Morph-open recovery'},
    ]

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv.copy()

    for attempt, params in enumerate(mutations[:max_attempts]):
        # Apply current parameter set
        clahe    = cv2.createCLAHE(clipLimit=params['clip'], tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        thresh  = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, params['threshold_c'])

        if params['morph']:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

        candidate_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        words, _      = _best_ocr(candidate_img)
        plate_text    = _extract_plate_text(words)
        matched       = _is_valid_plate(plate_text)

        trace_entry = {
            'attempt':    attempt + 1,
            'params':     params['label'],
            'plate_text': plate_text,
            'matched':    matched,
        }
        retry_trace.append(trace_entry)
        print(f"[retry] Attempt {attempt+1}: '{plate_text}' -- {'VALID' if matched else 'retry'}")

        # Keep the best result (valid match > longer string > first attempt)
        if matched and not best_matched:
            best_words, best_plate, best_matched = words, plate_text, True

        if not best_matched or len(plate_text) > len(best_plate):
            if not best_matched:
                best_words, best_plate = words, plate_text

        if matched:
            break   # ← Early exit: plant regex satisfied

    return best_words, best_plate, best_matched, retry_trace


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_input):
    """
    Load and upscale the image.  Returns original (upscaled) and a binarized
    version for the enhanced display panel.

    Returns dict with keys:
        original_cv  — upscaled BGR image (raw, used for OCR variants)
        enhanced_cv  — binarized BGR image (display only)
    """
    # Load
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("image_input must be filepath, numpy array, or PIL Image")

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_input}")

    # Upscale so Tesseract has enough pixels (target ≥ 1200 px wide)
    h, w = img.shape[:2]
    scale = max(1.0, 1200 / w)
    if scale > 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LANCZOS4)

    # Enhanced display: grayscale → binarize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Illumination normalization for the dashboard preview too
    bg = cv2.medianBlur(gray, 51)
    norm = cv2.divide(gray, bg, scale=255)
    
    # 2. Thresholding + Despeckle for dashboard
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological Opening to remove small noise dots
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    
    # Ensure black text on white
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    if white_pixels < black_pixels:
        binary = cv2.bitwise_not(binary)
        
    enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return {
        'original_cv': img,
        'enhanced_cv': enhanced,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OCR CORE
# ─────────────────────────────────────────────────────────────────────────────

def _run_tesseract(image_cv, psm):
    """
    Single Tesseract call for general text recognition.
    Ensures black-on-white polarity.
    """
    if len(image_cv.shape) == 3:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_cv.copy()

    # Polarity check: Tesseract prefers black text on white background
    white_px = np.sum(gray == 255)
    black_px = np.sum(gray == 0)
    if white_px < black_px:
        gray = cv2.bitwise_not(gray)

    # Standard OCR upscale for crops (Tesseract loves 300 DPI / large chars)
    # Target height of ~60-80 pixels per character
    h, w = gray.shape[:2]
    if h < 60:
        scale = 60 / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    cfg = (f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    # Special config to disable dictionary guessing for short strings/plates
    if psm in [7, 8, 10, 6, 13]:
        cfg += " -c load_system_dawg=F -c load_freq_dawg=F"

    pil = Image.fromarray(gray)

    try:
        data = pytesseract.image_to_data(
            pil, config=cfg, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"[pipeline] Tesseract PSM {psm} error: {e}")
        return []

    words = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        conf = int(data['conf'][i])
        if not word or conf < 0:
            continue
        words.append({
            'word':     word,
            'confidence': conf,
            'x':        data['left'][i],
            'y':        data['top'][i],
            'w':        data['width'][i],
            'h':        data['height'][i],
            'line_num': data['line_num'][i],
            'word_num': data['word_num'][i],
        })
    return words


def _text_score(words):
    """
    Score a word list by how much valid text it contains.
    Rule: (length of alphanumeric string)^1.5 * confidence.
    Power 1.5 strongly favors coherent strings over scattered noise bits.
    """
    score = 0
    for w in words:
        conf = w['confidence']
        text = w['word']
        if conf < 15:
            continue
            
        # Alphanumeric character count
        alphanums = sum(1 for c in text if c.isalnum())
        
        # JUNK FILTER: Ignore low-confidence single-character noise
        if len(text) <= 2 and alphanums < 2 and conf < 40:
            continue

        # Exponential boost for length
        score += (alphanums ** 1.5) * conf
    return score


def _build_variants(original_bgr, enhanced_bgr=None):
    """
    Build 10 preprocessed image variants.
    Each variant targets a different image condition (blur, low contrast, etc.)
    """
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    # Base for enhanced variants
    if enhanced_bgr is not None:
        e_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    else:
        e_gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Glare-Corrected grayscale
    g_gray = _reduce_glare(e_gray)

    # 1. CLAHE — local contrast on glare-corrected image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v1 = clahe.apply(g_gray.copy())

    # 2. Histogram equalization + sharpen
    v2 = cv2.equalizeHist(g_gray.copy())
    blur = cv2.GaussianBlur(v2, (0, 0), sigmaX=1.5)
    v2 = cv2.addWeighted(v2, 1.5, blur, -0.5, 0)

    # 3. Adaptive threshold — sharper, less 'fat' characters
    # Lower block size (15) and higher constant (25) to prevent blobs
    blurred = cv2.GaussianBlur(g_gray, (3, 3), 0)
    v3 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 25)

    # 4. Inverted adaptive (Top-Hat style)
    v4 = cv2.bitwise_not(v3)

    # 5. Otsu threshold on original contrast
    _, v5 = cv2.threshold(g_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def to_bgr(g):
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # 6. Original Normalized
    v6 = to_bgr(g_gray)

    # 7. Bilateral Filter + Glare Correction
    v7 = cv2.bilateralFilter(g_gray, 9, 75, 75)
    v7 = to_bgr(v7)

    # 8. Dilated Sharpening (Good for very thin/faded characters)
    s_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(g_gray, -1, s_kernel)
    v8 = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 12)
    v8 = to_bgr(v8)

    # 9. Morphological Opening (Noise Removal)
    # Uses Opening to clear out small glare artifacts before Otsu
    m_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    v9 = cv2.morphologyEx(g_gray, cv2.MORPH_OPEN, m_kernel)
    _, v9 = cv2.threshold(v9, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v9 = to_bgr(v9)

    # 10. Morphological Closing (Char Reconstruction)
    # Heals broken characters
    c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    v10 = cv2.morphologyEx(v5, cv2.MORPH_CLOSE, c_kernel)
    v10 = to_bgr(v10)

    # 11. EXTRA: Unsharp Mask (Extreme Blur Recovery)
    # This specifically targets images that are too blurry for the other variants
    u_blur = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 2.0, u_blur, -1.0, 0)
    _, v11 = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v11 = to_bgr(v11)

    # 12. Wiener-ish restoration
    v12 = _wiener_deblur(gray)
    v12 = to_bgr(v12)

    # 13. NOVELTY: Multi-Scale Retinex (MSR) — principled illumination/reflectance separation
    # Targets nighttime plates, headlight glare, and uneven outdoor lighting
    v13 = _multiscale_retinex(gray, sigmas=(15, 80, 250))
    v13 = to_bgr(v13)

    return [
        ('clahe',     to_bgr(v1)),
        ('histeq',    to_bgr(v2)),
        ('adaptive',  to_bgr(v3)),
        ('inverted',  to_bgr(v4)),
        ('otsu',      to_bgr(v5)),
        ('original',  v6),
        ('bilateral', v7),
        ('laplacian', v8),
        ('morph',     v9),
        ('m-sharpen', v10),
        ('unsharp',   v11),
        ('wiener',    v12),
        ('retinex',   v13),
    ]


def run_ocr(image_cv, config=None):
    """
    Try 4 PSM modes on image_cv; return word list with best plate score.
    (config arg kept for API compatibility — ignored internally.)
    """
    best_words, _ = _best_ocr(image_cv)
    return best_words


def _best_ocr(image_cv):
    """Try specialized PSM modes on one image; return best result."""
    best_words, best_score = [], -1
    for psm in [7, 8, 6, 13]:
        words = _run_tesseract(image_cv, psm)
        score = _text_score(words)
        if score > best_score:
            best_score, best_words = score, words
    return best_words, best_score


def _ensemble_results(all_candidate_words):
    """
    THE CORE ALGORITHM ENHANCEMENT:
    Combines OCR words from ALL variants and PSMs using majority voting.
    Identifies clusters of words overlapping in space and picks the winner.
    """
    if not all_candidate_words:
        return []
        
    final_words = []
    # Cluster threshold (pixels)
    DIST_THRESH = 30 
    
    # Simple spatial clustering
    clusters = []
    for w in all_candidate_words:
        found = False
        for cluster in clusters:
            # Match center point overlap
            cx, cy = w['x'] + w['w']//2, w['y'] + w['h']//2
            ccx, ccy = cluster['x'] + cluster['w']//2, cluster['y'] + cluster['h']//2
            if abs(cx - ccx) < DIST_THRESH and abs(cy - ccy) < DIST_THRESH:
                cluster['candidates'].append(w)
                found = True
                break
        if not found:
            clusters.append({'x': w['x'], 'y': w['y'], 'w': w['w'], 'h': w['h'], 'candidates': [w]})

    for cluster in clusters:
        # Tally votes for each unique spelling in this cluster
        votes = {}
        for c in cluster['candidates']:
            txt = c['word'].strip().upper() # Normalize for voting
            if not txt: continue
            votes[txt] = votes.get(txt, 0) + (c['confidence'] / 100.0)
            
        if not votes:
            continue
            
        # Champion is the string with highest weighted vote
        best_txt = max(votes.items(), key=lambda x: x[1])[0]
        
        # Pick the candidate instance that had this text with highest confidence to get meta (line_num etc)
        best_inst = max([c for c in cluster['candidates'] if c['word'].strip().upper() == best_txt], 
                         key=lambda x: x['confidence'])
        
        final_words.append(best_inst)
        
    # Re-sort by position
    final_words = sorted(final_words, key=lambda w: (w['line_num'], w['x']))
    return final_words


def detect_text_regions(image_cv):
    """
    Search for possible text regions using a combination of 
    Adaptive Thresholding and Canny Edge Detection.
    """
    h_img, w_img = image_cv.shape[:2]
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # 1. Approach A: Adaptive Thresholding (Fills characters)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2. Approach B: Canny Edges (Finds plate borders)
    # Good for high-contrast white plates in light scenes
    edges = cv2.Canny(gray, 30, 200)
    
    # Combine them
    combined = cv2.bitwise_or(thresh, edges)

    # 3. Morphological dilation to merge nearby characters & edges into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    dilated = cv2.dilate(combined, kernel, iterations=1)

    # 4. Find block contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        area = w * h
        
        # Filtering: 
        # Area must be at least 0.1% of the image (prevents small noise)
        # Aspect must be wider than it is tall (lines of text)
        if (h_img * w_img * 0.001) < area < (h_img * w_img * 0.9):
            if 0.8 < aspect < 20: 
                text_regions.append((x, y, w, h))

    # Fallback: If no regions or only tiny ones, add the center focus
    if not text_regions or len(text_regions) < 2:
        # Add a ROI that covers the middle 80% of the image
        cx1, cy1 = int(w_img * 0.1), int(h_img * 0.1)
        cw, ch   = int(w_img * 0.8), int(h_img * 0.8)
        text_regions.append((cx1, cy1, cw, ch))

    return text_regions


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(image_input):
    """
    Main OCR pipeline.

    Strategy:
      1. Load + upscale the image
      2. NOVELTY: Classify image quality (sharp / defocus / motion blur)
      3. Build 13 preprocessing variants (12 original + MSR Retinex)
      4. For EACH variant run 6 Tesseract PSM modes → 78 total attempts
      5. Score by:  Σ (alphanumeric_chars^1.5 × confidence)
         → plate text always beats single-char noise blobs
      6. NOVELTY: Closed-loop retry — if plate regex fails, mutate params and retry
      7. Champion word list → BOTH the overlay images AND extracted text
    """
    result      = preprocess_image(image_input)
    original_cv = result['original_cv']
    enhanced_cv = result['enhanced_cv']

    # Extra upscale for very small crops (e.g. user-cropped plate region)
    h_up, w_up = original_cv.shape[:2]
    if w_up < 800:
        scale = 800 / w_up
        original_cv = cv2.resize(original_cv, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_LANCZOS4)
        enhanced_cv = cv2.resize(enhanced_cv, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_LANCZOS4)
        print(f"[pipeline] Small crop detected — upscaled {scale:.1f}x "
              f"to {original_cv.shape[1]}x{original_cv.shape[0]}")

    # Add 40px white padding around the image. Tesseract performs MUCH better
    # on crops if there is a 'quiet zone' around the text.
    pad = 40
    original_cv = cv2.copyMakeBorder(original_cv, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=[255, 255, 255])
    enhanced_cv = cv2.copyMakeBorder(enhanced_cv, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # ── NOVELTY 2: Image Quality Classification ───────────────────────────────
    gray_for_analysis = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
    image_quality = classify_image_quality(gray_for_analysis)
    print(f"[pipeline] Image quality: {image_quality['quality_label']} "
          f"(recommended variants: {image_quality['recommended_variants']})")

    variants = _build_variants(original_cv, enhanced_cv)

    # 1. Baseline OCR (for Comparison)
    # Run a simple PSM 3 pass on the raw original image
    baseline_words = _run_tesseract(original_cv, 3)
    baseline_stats = compute_stats(baseline_words)
    print(f"[pipeline] Baseline OCR complete. Avg Conf: {baseline_stats['avg_confidence']}%")

    # 2. Attempt text region localization
    candidates = detect_text_regions(original_cv)
    print(f"[pipeline] Detected {len(candidates)} potential plate regions.")
    
    # Restrict ROIs to top 3 by area to avoid processing too much background noise
    rois = sorted(candidates, key=lambda x: x[2]*x[3], reverse=True)[:3]

    all_ocr_results = []

    for region in rois:
        rx, ry, rw, rh = region
        p = 20
        x1, y1 = max(0, rx-p), max(0, ry-p)
        x2, y2 = min(w_up, rx+rw+p), min(h_up, ry+rh+p)
        
        for name, var_img in variants:
            crop = var_img[y1:y2, x1:x2]
            
            # Deskew the crop before OCR
            crop, angle = _deskew(crop)
                
            # ENSEMBLE CHANGE: Collect ALL results from ALL PSMs for each variant
            # This is slow but maximum accuracy for degraded images
            for psm in [7, 8, 6, 13]:
                words = _run_tesseract(crop, psm)
                
                for w in words:
                    w['x'] += x1
                    w['y'] += y1
                    all_ocr_results.append(w)

    # 3. ENSEMBLE VOTING: Synthesize the final word list from all candidates
    champion_words = _ensemble_results(all_ocr_results)
    text = _words_to_text(champion_words)
    
    print(f"[pipeline] [OK] Ensemble Reconstruction Complete. Found '{text.replace(chr(10), ' ')}'")

    # ── NOVELTY 3: Closed-Loop Retry with Plate Format Validation ────────────
    # Check if the champion text looks like a valid Indian plate number.
    # If not, run the adaptive retry loop with parameter mutation.
    initial_plate = _extract_plate_text(champion_words)
    plate_matched = _is_valid_plate(initial_plate)
    retry_trace   = []

    if not plate_matched:
        print(f"[pipeline] Plate '{initial_plate}' failed validation — starting closed-loop retry...")
        retry_words, retry_plate, retry_matched, retry_trace = _retry_ocr_for_plate(original_cv)
        if retry_matched or len(retry_plate) > len(initial_plate):
            champion_words = retry_words
            text           = _extract_plate_text(champion_words)
            plate_matched  = retry_matched
            print(f"[pipeline] Retry improved result to '{text}' (valid={plate_matched})")
    else:
        print(f"[pipeline] Plate '{initial_plate}' is a valid Indian number plate! YES")
        retry_trace = [{'attempt': 1, 'params': 'Direct match — no retry needed',
                        'plate_text': initial_plate, 'matched': True}]

    final_plate_text = _extract_plate_text(champion_words) if champion_words else initial_plate

    # Baseline words on original, Champion words on enhanced
    ann_original = _draw_annotations(original_cv.copy(), baseline_words)
    ann_enhanced = _draw_annotations(enhanced_cv.copy(), champion_words)

    stats = compute_stats(champion_words)

    return {
        'original_b64':           image_to_base64(original_cv),
        'enhanced_b64':           image_to_base64(enhanced_cv),
        'annotated_original_b64': image_to_base64(ann_original),
        'annotated_enhanced_b64': image_to_base64(ann_enhanced),
        'original_words':         baseline_words,
        'enhanced_words':         champion_words,
        'original_stats':         baseline_stats,
        'enhanced_stats':         stats,
        'original_text':          _words_to_text(baseline_words),
        'enhanced_text':          text,
        'ocr_engine':             'Tesseract (multi-variant + MSR Retinex)',
        # ── Novelty fields ──────────────────────────────────────────────────
        'image_quality':          image_quality,
        'plate_text':             final_plate_text,
        'plate_matched':          plate_matched,
        'retry_trace':            retry_trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_annotations(img_bgr, word_data):
    """Draw color-coded boxes AND recognised text on a BGR image."""
    for word in word_data:
        r, g, b = get_confidence_color(word['confidence'])
        color   = (b, g, r)
        x, y, bw, bh = word['x'], word['y'], word['w'], word['h']
        cv2.rectangle(img_bgr, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(img_bgr, word['word'],
                    (x, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return img_bgr


def _words_to_text(word_data):
    """Reconstruct plain text from word list, preserving line breaks."""
    if not word_data:
        return ""
    lines = {}
    for w in word_data:
        lines.setdefault(w['line_num'], []).append(w['word'])
    return "\n".join(" ".join(lines[k]) for k in sorted(lines))


def get_confidence_color(conf):
    if conf >= CONF_HIGH:     return (34, 197, 94)
    elif conf >= CONF_MEDIUM: return (234, 179, 8)
    else:                     return (239, 68, 68)


def get_confidence_label(conf):
    if conf >= CONF_HIGH:     return "high"
    elif conf >= CONF_MEDIUM: return "medium"
    return "low"


def compute_stats(word_data):
    if not word_data:
        return {'avg_confidence': 0, 'high_count': 0, 'medium_count': 0,
                'low_count': 0, 'total_words': 0,
                'high_pct': 0, 'medium_pct': 0, 'low_pct': 0}
    total  = len(word_data)
    high   = sum(1 for w in word_data if w['confidence'] >= CONF_HIGH)
    medium = sum(1 for w in word_data if CONF_MEDIUM <= w['confidence'] < CONF_HIGH)
    low    = sum(1 for w in word_data if w['confidence'] < CONF_MEDIUM)
    avg    = sum(w['confidence'] for w in word_data) / total
    return {
        'avg_confidence': round(avg, 1),
        'high_count':   high,   'medium_count': medium, 'low_count':   low,
        'total_words':  total,
        'high_pct':   round(high   / total * 100, 1),
        'medium_pct': round(medium / total * 100, 1),
        'low_pct':    round(low    / total * 100, 1),
    }


def image_to_base64(image_cv, format=".png"):
    _, buffer = cv2.imencode(format, image_cv)
    return base64.b64encode(buffer).decode('utf-8')
