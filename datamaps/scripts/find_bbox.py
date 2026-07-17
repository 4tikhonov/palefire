import cv2
import numpy as np

def find_map_bbox(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to HSV to filter out white/gray background and blue ocean
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Honduras is colored in greens, yellows, oranges, reds.
    # We can mask out white (low saturation, high value), gray, and blue (hue 100-140)
    # Actually, a simpler way is to find all pixels that are NOT white/gray/blue/black
    
    # Mask for colored areas (Saturation > 40)
    _, sat_mask = cv2.threshold(hsv[:, :, 1], 40, 255, cv2.THRESH_BINARY)
    
    # Ignore blue (ocean/lakes) which is Hue roughly 90-130
    hue = hsv[:, :, 0]
    non_blue_mask = cv2.bitwise_not(cv2.inRange(hue, 90, 140))
    
    # Ignore very dark colors (Value < 50) e.g., text
    _, val_mask = cv2.threshold(hsv[:, :, 2], 50, 255, cv2.THRESH_BINARY)
    
    # Combine masks
    combined_mask = cv2.bitwise_and(sat_mask, non_blue_mask)
    combined_mask = cv2.bitwise_and(combined_mask, val_mask)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Get the bounding box of all significant contours
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Draw rectangle for debugging
    debug_img = img.copy()
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.imwrite("debug_bbox.png", debug_img)
    
    return (x, y, w, h)

print(find_map_bbox("report/page_4.png"))
