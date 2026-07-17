import cv2
import numpy as np
import geopandas as gpd

def get_geo_extremes(geojson_path):
    gdf = gpd.read_file(geojson_path)
    
    # Get all coordinates from all polygons
    all_coords = []
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            all_coords.extend(list(geom.exterior.coords))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                all_coords.extend(list(poly.exterior.coords))
                
    # Find West (min X), East (max X), South (min Y)
    # coords are (lon, lat)
    west_pt = min(all_coords, key=lambda p: p[0])
    east_pt = max(all_coords, key=lambda p: p[0])
    south_pt = min(all_coords, key=lambda p: p[1])
    
    return west_pt, east_pt, south_pt

def get_pixel_extremes(image_path):
    img = cv2.imread(image_path)
    # Convert to HSV to find landmass (ignoring white background and black text)
    # The land is colored, so saturation > something and value < 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    
    # Mask where saturation is high enough, or it's not white
    # White is low saturation, high value
    mask = cv2.inRange(hsv, (0, 10, 0), (180, 255, 245))
    
    # Morphological operations to merge map components
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the map
    c = max(contours, key=cv2.contourArea)
    
    # Find extremes in the contour
    # c is shape (N, 1, 2) where 2 is (x,y)
    pts = c.reshape(-1, 2)
    west_px = tuple(pts[pts[:, 0].argmin()])
    east_px = tuple(pts[pts[:, 0].argmax()])
    south_px = tuple(pts[pts[:, 1].argmax()]) # y increases downwards
    
    # Draw for visual debug
    debug = img.copy()
    cv2.drawContours(debug, [c], -1, (0, 255, 0), 2)
    cv2.circle(debug, west_px, 10, (255, 0, 0), -1)
    cv2.circle(debug, east_px, 10, (255, 0, 0), -1)
    cv2.circle(debug, south_px, 10, (255, 0, 0), -1)
    cv2.imwrite("report/extremes_debug.png", debug)
    
    return west_px, east_px, south_px

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "report/page_4_400_left.png"
    
    g_w, g_e, g_s = get_geo_extremes("honduras_departments.geojson")
    print("Geo Extremes (Lon, Lat):")
    print(f"  West : {g_w}")
    print(f"  East : {g_e}")
    print(f"  South: {g_s}")
    
    p_w, p_e, p_s = get_pixel_extremes(img_path)
    print("\nPixel Extremes (X, Y):")
    print(f"  West : {p_w}")
    print(f"  East : {p_e}")
    print(f"  South: {p_s}")
    
    # Calculate Affine Matrix
    src_pts = np.float32([g_w, g_e, g_s])
    dst_pts = np.float32([p_w, p_e, p_s])
    
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    print("\nComputed Affine Matrix:")
    print(matrix)
    print("Saved visual debug to report/extremes_debug.png")
