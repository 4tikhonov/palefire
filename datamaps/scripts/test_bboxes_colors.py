import cv2
import json

bboxes = [
  {
    "color": "Green",
    "value": "0% to 20%",
    "box": [358, 832, 422, 848]
  },
  {
    "color": "Yellow",
    "value": "-20% to 0%",
    "box": [422, 832, 485, 848]
  },
  {
    "color": "Orange",
    "value": "-40% to -20%",
    "box": [485, 832, 548, 848]
  },
  {
    "color": "Red",
    "value": "-65% to -40%",
    "box": [548, 832, 611, 848]
  }
]

def test_crop():
    img = cv2.imread("report/debug_full_page-04_left.png")
    h, w, _ = img.shape
    
    for item in bboxes:
        ymin, xmin, ymax, xmax = item["box"]
        y1 = int(ymin * h / 1000)
        y2 = int(ymax * h / 1000)
        x1 = int(xmin * w / 1000)
        x2 = int(xmax * w / 1000)
        
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            print(f"Empty patch for {item['color']}")
            continue
            
        mean_bgr = patch.mean(axis=0).mean(axis=0).astype(int)
        print(f"Color {item['color']} ({item['value']}): BGR {mean_bgr}")

if __name__ == "__main__":
    test_crop()
