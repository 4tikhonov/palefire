import pytest
import numpy as np
from datamaps.extractor import geo_to_pixel_affine

def test_geo_to_pixel_affine():
    # Test affine matrix transformation logic
    # Identity matrix (just translates to see if matrix math works)
    matrix = np.array([
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 20.0]
    ])
    
    px, py = geo_to_pixel_affine(5.0, 5.0, matrix)
    
    assert px == 15
    assert py == 25
