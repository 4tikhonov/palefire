import cv2
import numpy as np

def detect_maps(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find non-white areas
    # Most of the background is white (255). 
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to group map components together
    kernel = np.ones((50, 50), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Image shape: {img.shape}")
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # A map will be a large area, e.g., > 5% of the image
        if area > (img.shape[0] * img.shape[1] * 0.05):
            bboxes.append((x, y, w, h))
            
    print(f"Found {len(bboxes)} maps:")
    for i, bbox in enumerate(bboxes):
        print(f"Map {i+1}: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")

if __name__ == "__main__":
    print("Testing on page_4.png (2K)")
    detect_maps("report/page_4.png")
    print("\nTesting on page_4_400.png (4K)")
    detect_maps("report/page_4_400.png")
