import numpy as np
import cv2
import os
import re
from PIL import Image
import pytesseract as tess
import argparse

# ── NOVELTY 1: Frequency Domain Wiener Deconvolution for Blurry Images
def make_pillbox_psf(radius, size=21):
    """Creates a circular point spread function for defocus blur."""
    psf = np.zeros((size, size), dtype=np.float32)
    cv2.circle(psf, (size // 2, size // 2), radius, 1, -1)
    return psf / psf.sum()

def wiener_deconvolution(img, psf, snr=25):
    """Applies Frequency-Domain Wiener Deconvolution to a grayscale image."""
    img_f = np.float32(img)
    img_fft = np.fft.fft2(img_f)
    psf_fft = np.fft.fft2(psf, s=img.shape)
    
    psf_fft_conj = np.conj(psf_fft)
    psf_power = np.abs(psf_fft)**2
    
    wiener_filter = psf_fft_conj / (psf_power + 1.0 / snr)
    
    result_fft = img_fft * wiener_filter
    result = np.fft.ifft2(result_fft)
    result = np.abs(result)
    
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ── NOVELTY 2: Multi-Scale Retinex for Dark/Glare Images
def multiscale_retinex(img_gray, sigmas=(15, 80, 250)):
    """MSR to fix low-light or glare images based on Retinex theory."""
    img_f = img_gray.astype(np.float64) + 1.0
    log_img = np.log(img_f)
    msr = np.zeros_like(log_img)
    for sigma in sigmas:
        k = int(6 * sigma + 1) | 1
        k = max(k, 3)
        blurred = cv2.GaussianBlur(img_f, (k, k), sigma)
        msr += log_img - np.log(np.maximum(blurred, 1e-6))
    msr /= len(sigmas)
    return cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ── NOVELTY 3: Image Quality Classification
def classify_blur(img_gray):
    """Returns Laplacian variance (sharpness score)."""
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(lap.var())

def preprocess(img):
    if img is None:
        raise ValueError("Error: Could not load image.")
    cv2.imshow("1. Input", img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Analyze Image Quality
    sharpness = classify_blur(gray)
    print(f"\n[Analyze] Image Sharpness Score: {sharpness:.1f}")
    
    enhanced_gray = gray.copy()
    if sharpness < 250:
        print("[Analyze] Severe blur detected! Applying Frequency-Domain Wiener Deconvolution...")
        # Larger blur radius for lower sharpness
        radius = 5 if sharpness < 100 else 3
        psf = make_pillbox_psf(radius)
        enhanced_gray = wiener_deconvolution(gray, psf, snr=20)
        cv2.imshow("2. Wiener Deblurred", enhanced_gray)
    else:
        print("[Analyze] Image sharpness is acceptable. Applying Multi-Scale Retinex...")
        enhanced_gray = multiscale_retinex(gray)
        cv2.imshow("2. Retinex Enhanced", enhanced_gray)

    imgBlurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    sobelx = cv2.Sobel(imgBlurred, cv2.CV_8U, 1, 0, ksize=3)
    
    ret2, threshold_img = cv2.threshold(
        sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cv2.imshow("3. Sobel Threshold", threshold_img)
    return threshold_img, enhanced_gray

def cleanPlate(plate_gray):
    """Takes enhanced grayscale plate and binarizes it."""
    # Adaptive threshold works better for varying illumination on plates
    thresh = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours_clean, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_clean:
        areas = [cv2.contourArea(c) for c in contours_clean]
        max_index = np.argmax(areas)
        max_cnt = contours_clean[max_index]
        max_cntArea = areas[max_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        if not ratioCheck(max_cntArea, w, h):
            return plate_gray, None

        cleaned_final = thresh[y: y + h, x: x + w]
        return cleaned_final, [x, y, w, h]
    else:
        return plate_gray, None

def extract_contours(threshold_img):
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = threshold_img.copy()
    cv2.morphologyEx(
        src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold
    )
    cv2.imshow("4. Morphed Regions", morph_img_threshold)

    contours, _ = cv2.findContours(
        morph_img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return contours

def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio

    aspect = 4.7272
    min = 5 * aspect * 5
    max = 200 * aspect * 200

    rmin = 1.5
    rmax = 10

    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
    return True

def validateRotationAndRatio(rect):
    (x, y), (width, height), rect_angle = rect
    if width > height:
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle > 30:
        return False
    if height == 0 or width == 0:
        return False

    area = height * width
    if not ratioCheck(area, width, height):
        return False
    return True

def isMaxWhite(plate_gray):
    avg = np.mean(plate_gray)
    return avg >= 90

# ── NOVELTY 4: Closed-Loop Regex Validation ──
PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$')

def validate_plate_text(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return PLATE_REGEX.match(cleaned) is not None, cleaned

def cleanAndRead(img, contours, enhanced_gray):
    print("\n[OCR] Evaluating potential plate regions...")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for i, cnt in enumerate(contours):
        min_rect = cv2.minAreaRect(cnt)
        x, y, w, h_temp = cv2.boundingRect(cnt)
        
        if validateRotationAndRatio(min_rect):
            plate_gray = enhanced_gray[y: y + h_temp, x: x + w]

            if isMaxWhite(plate_gray):
                clean_plate, rect = cleanPlate(plate_gray)

                if rect:
                    x1, y1, w1, h1 = rect
                    x_final, y_final, w_final, h_final = x + x1, y + y1, w1, h1
                    
                    plate_im = Image.fromarray(clean_plate)
                    
                    text = ""
                    valid = False
                    cleaned_text = ""
                    
                    # Retry OCR with different configurations (mutations) until regex matches
                    psms_to_try = [
                        "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        "--psm 6"
                    ]
                    
                    print(f"\n[OCR] Processing Candidate {i+1}...")
                    for config in psms_to_try:
                        text = tess.image_to_string(plate_im, lang="eng", config=config)
                        valid, cleaned_text = validate_plate_text(text)
                        
                        if valid:
                            print(f"      -> SUCCESS: Format matched using config: {config.split()[1]}")
                            break
                        else:
                            print(f"      -> {config.split()[1]} extracted '{text.strip()}' - format mismatch. Retrying...")

                    cv2.rectangle(img, (x_final, y_final), (x_final + w_final, y_final + h_final), (0, 255, 0), 2)
                    cv2.putText(img, cleaned_text or text.strip(), (x_final, y_final - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    print("\n" + "="*35)
                    print("*** FINAL DETECTED PLATE TEXT ***")
                    print(f"    {cleaned_text if cleaned_text else text.strip()}")
                    if valid:
                        print("✅ Valid Indian Plate Format!")
                    else:
                        print("⚠️ Format deviation detected.")
                    print("="*35 + "\n")
                    
                    cv2.imshow("Detected Plate", img)
                    cv2.imshow("Cleaned Plate", clean_plate)
                    cv2.waitKey(1)
                    return # Exit once we successfully find & process a plate

    print("[OCR] No valid plate could be read from the image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian Number Plate Recognition")
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "testData", "Final.JPG"),
        help="Path to input image",
    )
    args = parser.parse_args()

    print("DETECTING PLATE . . .")
    img_path = args.image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Error loading image: {img_path}")

    threshold_img, enhanced_gray = preprocess(img)
    contours = extract_contours(threshold_img)
    cleanAndRead(img, contours, enhanced_gray)
    
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
