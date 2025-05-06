# mlc_pipeline/utils.py
import logging
from pathlib import Path
import cv2
import numpy as np
import matplotlib as plt

def auto_utm_epsg(bbox):
    """
    Given a bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84,
    compute the center and return the corresponding UTM EPSG code.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    utm_zone = int((center_lon + 180) / 6) + 1
    if center_lat >= 0:
        return 32600 + utm_zone
    else:
        return 32700 + utm_zone

def split_bbox(bbox, n):
    """
    Split the overall bounding box along longitude into n equal parts.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    dx = (max_lon - min_lon) / n
    return [[min_lon + i * dx, min_lat, min_lon + (i + 1) * dx, max_lat] for i in range(n)]

def setup_logger(log_file="pipeline.log", level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear any existing handlers.
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler.
    from pathlib import Path
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def laplacian_smoothing(mask, iterations=5, lambda_val=0.2):
    """
    Iteratively smooth the image using a Laplacian operator.
    
    Parameters:
      mask: Input image (in 0-255, float32).
      iterations: Number of iterations.
      lambda_val: Smoothing factor (controls the amount subtracted per iteration).
      
    Returns:
      smoothed: The smoothed image.
    """
    # Ensure the mask is float32
    smooth = mask.astype(np.float32)
    
    for i in range(iterations):
        # Compute the Laplacian—the second derivative (difference to local average)
        lap = cv2.Laplacian(smooth, cv2.CV_32F)
        # Subtract a fraction of the Laplacian to diffuse high-frequency details.
        smooth = smooth - lambda_val * lap
        # Clip to maintain valid range.
        smooth = np.clip(smooth, 0, 255)
    return smooth

def smooth_mask_with_laplacian(mask, iterations=5, lambda_val=0.2, threshold_val=127):
    """
    Smooth a binary mask using iterative Laplacian smoothing and re-thresholding.
    
    Parameters:
      mask: Input binary mask (numpy array). It can be either in the 0-1 range or 0-255.
      iterations: How many iterations to run the Laplacian smoothing.
      lambda_val: The smoothing factor (a higher lambda means more smoothing per iteration).
      threshold_val: The threshold value for binarization (in the 0-255 range).
      
    Returns:
      smoothed_mask: A binary mask (0 or 255) with smoother boundaries.
    """
    # If the mask is normalized (0-1), scale it up to 0-255.
    if mask.max() <= 1:
        mask_scaled = (mask * 255).astype(np.uint8)
    else:
        mask_scaled = mask.astype(np.uint8)
    
    # Apply Laplacian-based smoothing.
    smoothed = laplacian_smoothing(mask_scaled, iterations=iterations, lambda_val=lambda_val)
    
    # Convert the smoothed image to 8-bit for thresholding.
    smoothed_uint8 = smoothed.astype(np.uint8)
    
    # Threshold to get a binary mask. Adjust the threshold if necessary.
    _, binary_smoothed = cv2.threshold(smoothed_uint8, threshold_val, 255, cv2.THRESH_BINARY)
    
    return binary_smoothed

def cluster_boundary_segments(segs):
    """
    Given an array of segments [[n0,n1],...], return a list of lists of
    segment‐indices where each sub‐list is one connected boundary chain.
    """
    # build adjacency: segment i connects to j if they share a node
    N = len(segs)
    adj = {i:set() for i in range(N)}
    for i,(a,b) in enumerate(segs):
        for j,(c,d) in enumerate(segs):
            if j <= i: continue
            if a in (c,d) or b in (c,d):
                adj[i].add(j)
                adj[j].add(i)

    seen = set()
    clusters = []
    for i in range(N):
        if i in seen: continue
        # flood‐fill from i
        stack = [i]
        comp  = []
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u)
            comp.append(u)
            stack.extend(adj[u] - seen)
        clusters.append(comp)
    return clusters


def get_boundary_nodes(boundary_segments):
    """
    Return sorted unique node indices on the boundary.

    Parameters:
    - boundary_segments: np.ndarray of shape (M,2) of node index pairs
    """
    if boundary_segments is None:
        raise ValueError("boundary_segments must be provided.")
    return np.unique(boundary_segments.flatten())


def get_boundary_elements(boundary_segments, mesh):
    """
    Return element indices that share at least one boundary edge.

    Parameters:
    - boundary_segments: np.ndarray of shape (M,2)
    - mesh: object with .faces attribute, array of shape (E,3)
    """
    if boundary_segments is None:
        raise ValueError("boundary_segments must be provided.")
    # build set of edges
    b_edges = set(map(lambda e: tuple(sorted(e)), boundary_segments.tolist()))
    boundary_elems = []
    for eid, face in enumerate(mesh.faces):
        edges = [tuple(sorted((face[i], face[(i+1)%3]))) for i in range(3)]
        if any(edge in b_edges for edge in edges):
            boundary_elems.append(eid)
    return np.array(boundary_elems, dtype=int)


def get_all_boundary_loops(boundary_segments: np.ndarray, mesh):
    loops = []
    # 1) extract just the node‐chains
    node_chains = extract_boundary_loops(boundary_segments)

    # 2) for each node‐chain, find its elements
    edge_set = { tuple(sorted(e)) for e in boundary_segments.tolist() }
    face_edges = [
        [ tuple(sorted((face[i], face[(i+1)%3]))) for i in range(3) ]
        for face in mesh.faces
    ]

    for chain in node_chains:
        # build a set of edges along this chain
        chain_edges = {
            tuple(sorted((chain[i], chain[i+1])))
            for i in range(len(chain)-1)
        }
        # find all elements that use any of those edges
        elems = [
            eid
            for eid, edges in enumerate(face_edges)
            if any(e in chain_edges for e in edges)
        ]
        loops.append((np.array(chain, dtype=int), np.array(elems, dtype=int)))

    # debug
    print(f"Found {len(loops)} boundary loop{'s' if len(loops)!=1 else ''}.")
    for i, (nodes, elems) in enumerate(loops, 1):
        print(f" Loop {i}: {len(nodes)} nodes, {len(elems)} elems")

    return loops


def extract_boundary_loops(boundary_segments):
    """
    Turn a flat list of segments [(n1,n2),…] into a list of node‐chains [[…],[…]…].
    Each chain is walked in order and closed; disjoint loops stay separate.
    """
    # work on a mutable copy
    segs = [tuple(map(int,s)) for s in boundary_segments.tolist()]
    loops = []

    while segs:
        # start a new loop with the first segment
        a, b = segs.pop(0)
        chain = [a, b]
        current = b

        # walk forward until we get back to 'a' or run out
        while True:
            for idx, (c, d) in enumerate(segs):
                if c == current:
                    next_node = d
                elif d == current:
                    next_node = c
                else:
                    continue

                chain.append(next_node)
                segs.pop(idx)
                current = next_node
                break
            else:
                # no connecting segment found
                break

            if current == a:
                # we’ve closed the loop
                break

        loops.append(chain)

    return loops




