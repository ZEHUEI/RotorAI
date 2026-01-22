import cv2
import numpy as np

#-----------------------
#POST PROCESSING: RUST AND CRACKS
#------------------------
def detect_rust_and_cracks(image,corrosion_mask):
    lab = cv2.cvtColor(image,cv2.COLOR_RGB2Lab)

    #L = lightness, a channel= X G/Y b channel= X Y/Dark brown
    rust_mask = cv2.inRange(lab, np.array([40,135,140]), np.array([200, 180, 190]))
    rust_mask = cv2.bitwise_and(rust_mask, rust_mask, mask=corrosion_mask)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6, 6))
    enhanced_gray = clahe.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
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

        if contours:
            cnt = contours[0]
            # Calculate Perimeter
            perimeter = cv2.arcLength(cnt, True)

            # Convex Hull and Solidity
            convex_hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = float(area) / (convex_hull_area + 1e-5)

            # --- NEW TUNING LOGIC ---
            # 1. High aspect ratio (clear straight lines)
            is_straight_crack = aspect_ratio > 1.7

            # 2. High perimeter complexity (curved/diagonal fine cracks)
            # A circle is ~12.5; higher values mean more 'line-like'
            complexity = (perimeter ** 2) / (area + 1e-5)
            is_fine_crack = complexity > 25 and solidity < 0.6

            # 3. Small but very thin (micro-cracks)
            is_micro_crack = area > 5 and aspect_ratio > 2.2

            if is_straight_crack or is_fine_crack or is_micro_crack:
                final_cracks[labels == i] = 255

    return rust_mask, final_cracks