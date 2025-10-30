import numpy as np

def bounding_box_creator(all_pts, original_image, label, margin=0.10):

    """
    Compute a square bounding box (with optional margin) 
    around all given tumour points.

    Parameters:
        all_pts (np.ndarray): Nx2 array of [x, y] coordinates.
        margin (float): Fractional margin (e.g., 0.10 = 10% larger box).

    Returns:
        (x1, y1, x2, y2): Coordinates of the square bounding box.
    """

    x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
    y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])

    # Compute tumour region before margin
    w_tumour_before_margin, h_tumour_before_margin = x_max - x_min, y_max - y_min

    # Expand with margin 
    size = max(w_tumour_before_margin, h_tumour_before_margin) * (1 + margin)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    side = int(round(size))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = x1 + side
    y2 = y1 + side
    
    
    # Boundary correction
    H, W = original_image.shape[:2]
    out_of_bounds = x1 < 0 or y1 < 0 or x2 > W or y2 > H

    if out_of_bounds:
        if label != "multiple osteochondromas":
            # Shift inward to keep tumour inside
            # Horizontal adjustment
            if x1 < 0:
                shift = -x1
                x1 += shift
                x2 += shift
            elif x2 > W:
                shift = x2 - W
                x1 -= shift
                x2 -= shift

            # Vertical adjustment
            if y1 < 0:
                shift = -y1
                y1 += shift
                y2 += shift
            elif y2 > H:
                shift = y2 - H
                y1 -= shift
                y2 -= shift

    return x1,y1,x2,y2
