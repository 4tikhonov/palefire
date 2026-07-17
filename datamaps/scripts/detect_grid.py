import cv2
import numpy as np

def detect_grid(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200, minLineLength=300, maxLineGap=20)
    
    if lines is None:
        print("No lines detected.")
        return
        
    horizontals = []
    verticals = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # if horizontal
        if abs(y1 - y2) < 5:
            horizontals.append(y1)
        # if vertical
        elif abs(x1 - x2) < 5:
            verticals.append(x1)
            
    # Cluster the lines
    def cluster(vals, threshold=20):
        if not vals: return []
        vals.sort()
        clusters = []
        current = [vals[0]]
        for v in vals[1:]:
            if v - current[-1] < threshold:
                current.append(v)
            else:
                clusters.append(int(np.mean(current)))
                current = [v]
        clusters.append(int(np.mean(current)))
        return clusters

    h_lines = cluster(horizontals)
    v_lines = cluster(verticals)
    
    print("Horizontal lines (Y coordinates):", h_lines)
    print("Vertical lines (X coordinates):", v_lines)
    
    # Draw them for visual debug
    for y in h_lines:
        cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 2)
    for x in v_lines:
        cv2.line(img, (x, 0), (x, img.shape[0]), (0, 0, 255), 2)
        
    cv2.imwrite("report/grid_debug.png", img)
    print("Saved grid debug to report/grid_debug.png")

if __name__ == "__main__":
    detect_grid("report/page_4_400_left.png")
