import cv2
import numpy as np

def dist(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

known_colors = [
    (139, 219, 255), (50, 220, 230), (182, 240, 255), (47, 207, 216),
    (201, 239, 219), (46, 190, 230), (158, 216, 179), (120, 194, 145),
    (66, 151, 93), (55, 217, 227), (181, 232, 253), (144, 212, 253),
    (116, 197, 253), (96, 185, 253)
]

def find_legend(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # We'll create a mask of pixels that match ANY of our known colors within threshold 40
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # This is slow in pure python, so we use numpy broadcasting
    img_bgr = img.astype(np.int32)
    
    for c_bgr in known_colors:
        diff = img_bgr - np.array(c_bgr)
        dist_map = np.sqrt(np.sum(diff**2, axis=-1))
        mask[dist_map < 40] = 255
        
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    patches = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        # Legend patches are small rectangles
        if 10 < w_box < 100 and 10 < h_box < 100:
            aspect = w_box / float(h_box)
            if 0.5 < aspect < 3.0:
                patch = img[y:y+h_box, x:x+w_box]
                bgr = patch.mean(axis=0).mean(axis=0).astype(int)
                patches.append({
                    "box": (x, y, w_box, h_box),
                    "bgr": bgr.tolist()
                })
                
    # Sort patches by Y coordinate to get them in legend order
    patches.sort(key=lambda p: p["box"][1])
    
    for i, p in enumerate(patches):
        print(f"Patch {i}: BGR {p['bgr']} at {p['box']}")

if __name__ == "__main__":
    print("Testing on Page 4 Left...")
    find_legend("report/debug_full_page-04_left.png")
    print("Testing on Page 5 Left...")
    find_legend("report/debug_full_page-05_left.png")
