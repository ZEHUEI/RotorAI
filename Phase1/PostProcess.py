import cv2
import numpy as np

#-----------------------
#POST PROCESSING: RUST AND CRACKS
#------------------------
def detect_rust_and_cracks(image,corrosion_mask, confidence_map=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    mask0 = cv2.inRange(hsv, np.array([0, 75, 70]), np.array([19, 190, 120]))
    mask1 = cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 200, 120]))
    mask2 = cv2.inRange(hsv, np.array([0, 40, 50]), np.array([25, 100, 80]))

    # Combined rust color mask
    rust_color_mask = cv2.bitwise_or(cv2.bitwise_or(mask0, mask1), mask2)

    # Only keep AI mask pixels that are also rust-colored
    corrosion_mask = corrosion_mask.copy()
    corrosion_mask[rust_color_mask == 0] = 0

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rust_mask = corrosion_mask.copy()
    rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel_close)
    rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel_open)

    # Optional: use color as a BOOST not a GATE
    # If color matches, keep it. If AI says rust but color doesn't match, still keep it.
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    # Wider range to catch dry/dark rust too
    lower_rust_orange = np.array([20, 134, 132])
    upper_rust_orange = np.array([95, 168, 188])

    # Dark dry rust (like IMG 1) — low lightness, slight warm tone
    lower_rust_dark = np.array([0, 150, 134])
    upper_rust_dark = np.array([40, 155, 150])

    color_mask_orange = cv2.inRange(lab, lower_rust_orange, upper_rust_orange)
    color_mask_dark = cv2.inRange(lab, lower_rust_dark, upper_rust_dark)
    color_mask_any = cv2.bitwise_or(color_mask_orange, color_mask_dark)

    # AI mask OR color match — union instead of intersection
    rust_mask = cv2.bitwise_or(rust_mask, cv2.bitwise_and(
        color_mask_any, color_mask_any, mask=corrosion_mask
    ))

    # Connected component filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rust_mask, connectivity=8)
    clean_rust = np.zeros_like(rust_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        if area > 200 and aspect_ratio < 5.0:  # loosened from 350 & 4.0
            clean_rust[labels == i] = 255
    rust_mask = clean_rust

    #CRACKS-----------------
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6, 6))
    enhanced_gray = clahe.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # kernel = np.ones((5, 5), np.uint8)
    # rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)
    # rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
    blackhat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_BLACKHAT, kernel)

    blackhat_in_rust = cv2.bitwise_and(blackhat, blackhat, mask=rust_mask)

    _, cracks_thresh = cv2.threshold(blackhat_in_rust, 50, 255, cv2.THRESH_BINARY)

    #Filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cracks_thresh, connectivity=8)
    final_cracks = np.zeros_like(cracks_thresh)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # Calculate aspect ratio (Length / Width)
        # Cracks are long (high aspect ratio). Rust pits are round (low ratio).
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

        #Solidity: Real cracks vs Rust pits
        mask_component = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        convex_hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = float(area) / (convex_hull_area + 1e-5)
        complexity = (perimeter ** 2) / (area + 1e-5)

        is_straight_crack = aspect_ratio > 1.7
        is_fine_crack = complexity > 25 and solidity < 0.6
        is_micro_crack = area > 5 and aspect_ratio > 2.2

        if is_straight_crack or is_fine_crack or is_micro_crack:
            final_cracks[labels == i] = 255

    return rust_mask, final_cracks