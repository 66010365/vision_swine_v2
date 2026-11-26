# ======================================================
# Import Libraries
# ======================================================

import heapq

import cv2
import numpy as np
from skimage.morphology import skeletonize


# ======================================================
# Feature Extraction Constants
# ======================================================

SQRT2 = float(np.sqrt(2.0))

NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
]


# ======================================================
# OpenCV Shape Features
# ======================================================

def extract_opencv_features(mask):
    """
    Extract five shape features from a pig segmentation mask.

    Returns:
        list: Area, perimeter, bounding-box length,
              bounding-box width, and normalized deviation.
        None: If no contour is found.
    """
    binary_mask = (np.asarray(mask) > 0).astype(np.uint8) * 255

    if binary_mask.ndim != 2:
        raise ValueError("Expected a 2D segmentation mask")

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    area = float(cv2.countNonZero(binary_mask))
    perimeter = float(cv2.arcLength(contour, True))

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]

    bounding_box_length = float(max(width, height))
    bounding_box_width = float(min(width, height))

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        deviation = 0.0
    else:
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]

        image_height, image_width = binary_mask.shape

        image_center_x = image_width / 2.0
        image_center_y = image_height / 2.0

        distance = np.sqrt(
            (centroid_x - image_center_x) ** 2
            + (centroid_y - image_center_y) ** 2
        )

        max_distance = np.sqrt(
            image_center_x ** 2 + image_center_y ** 2
        )

        deviation = (
            float(distance / max_distance)
            if max_distance > 0
            else 0.0
        )

    return [
        area,
        perimeter,
        bounding_box_length,
        bounding_box_width,
        deviation,
    ]


# ======================================================
# Skeleton Pixel Utilities
# ======================================================

def skeleton_neighbors(skeleton, pixel):
    """Yield neighboring skeleton pixels and their distances."""
    y, x = pixel
    height, width = skeleton.shape

    for dy, dx in NEIGHBOR_OFFSETS:
        ny = y + dy
        nx = x + dx

        if (
            0 <= ny < height
            and 0 <= nx < width
            and skeleton[ny, nx]
        ):
            cost = SQRT2 if dy != 0 and dx != 0 else 1.0
            yield (ny, nx), cost


def get_skeleton_endpoints(skeleton):
    """Find skeleton pixels with exactly one neighboring pixel."""
    ys, xs = np.nonzero(skeleton)
    endpoints = []

    for y, x in zip(ys, xs):
        y0 = max(0, y - 1)
        y1 = min(skeleton.shape[0], y + 2)
        x0 = max(0, x - 1)
        x1 = min(skeleton.shape[1], x + 2)

        degree = int(
            np.count_nonzero(skeleton[y0:y1, x0:x1])
        ) - 1

        if degree == 1:
            endpoints.append((int(y), int(x)))

    return endpoints


# ======================================================
# Skeleton Path Calculation
# ======================================================

def farthest_skeleton_point(skeleton, start):
    """Find the farthest reachable pixel using Dijkstra's algorithm."""
    distances = {start: 0.0}
    queue = [(0.0, start)]

    while queue:
        current_distance, current = heapq.heappop(queue)

        if current_distance != distances.get(current):
            continue

        for neighbor, cost in skeleton_neighbors(skeleton, current):
            new_distance = current_distance + cost

            if new_distance < distances.get(
                neighbor, float("inf")
            ):
                distances[neighbor] = new_distance
                heapq.heappush(
                    queue, (new_distance, neighbor)
                )

    if not distances:
        return None, distances

    farthest = max(distances, key=distances.get)
    return farthest, distances


def longest_skeleton_path(skeleton):
    """
    Estimate skeleton path length and straight distance
    between the selected endpoints.
    """
    pixels = list(zip(*np.nonzero(skeleton)))

    if len(pixels) < 2:
        return 0.0, 0.0

    endpoints = get_skeleton_endpoints(skeleton)
    start = endpoints[0] if endpoints else (
        int(pixels[0][0]), int(pixels[0][1])
    )

    first, _ = farthest_skeleton_point(skeleton, start)

    if first is None:
        return 0.0, 0.0

    second, distances = farthest_skeleton_point(skeleton, first)

    if second is None:
        return 0.0, 0.0

    skeleton_length = float(distances.get(second, 0.0))

    y1, x1 = first
    y2, x2 = second

    straight_distance = float(
        np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    )

    return skeleton_length, straight_distance


# ======================================================
# Skeleton Features
# ======================================================

def extract_skeleton_features(mask):
    """
    Extract three features from the centerline of the mask.

    Returns:
        list: Skeleton length, straight distance, and straightness.
        None: If a usable skeleton cannot be extracted.
    """
    mask = np.asarray(mask)
    ys, xs = np.nonzero(mask > 0)

    if len(xs) < 2:
        return None

    # Crop around the pig to avoid skeletonizing the entire frame.
    y_min = max(0, int(ys.min()) - 2)
    y_max = min(mask.shape[0], int(ys.max()) + 3)
    x_min = max(0, int(xs.min()) - 2)
    x_max = min(mask.shape[1], int(xs.max()) + 3)

    cropped = (mask[y_min:y_max, x_min:x_max] > 0).astype(np.uint8)
    skeleton = skeletonize(cropped > 0)

    if np.count_nonzero(skeleton) < 2:
        return None

    skeleton_length, straight_distance = longest_skeleton_path(
        skeleton
    )

    if skeleton_length <= 0:
        return None

    straightness = float(straight_distance / skeleton_length)

    return [
        float(skeleton_length),
        float(straight_distance),
        straightness,
    ]


# ======================================================
# Combine All Features
# ======================================================

def extract_features(mask):
    """
    Return the eight features in the original model's order.

    Feature order:
        1. Area
        2. Perimeter
        3. Bounding-box length
        4. Bounding-box width
        5. Normalized deviation from image center
        6. Skeleton length
        7. Straight distance
        8. Skeleton straightness
    """
    opencv_features = extract_opencv_features(mask)

    if opencv_features is None:
        return None

    skeleton_features = extract_skeleton_features(mask)

    if skeleton_features is None:
        return None

    return opencv_features + skeleton_features
