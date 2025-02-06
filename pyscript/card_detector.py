import cv2
import numpy as np
from typing import Callable

EXPECTED_ASPECT_RATIO = 2.5 / 3.5


def calculate_card_dimensions(canvas_height: int) -> tuple[int, int]:
    """Calculate the target dimensions for the card based on canvas height.

    Args:
        canvas_height: Height of the target canvas

    Returns:
        tuple: (card_width, card_height)
    """
    card_height = int(canvas_height * 0.8)
    card_width = int(card_height * EXPECTED_ASPECT_RATIO)
    return card_width, card_height


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Basic image preprocessing for better edge detection.

    Args:
        img: Source image in RGBA format

    Returns:
        np.ndarray: Preprocessed grayscale image
    """
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=0)
    return blurred


def detect_edges(img: np.ndarray) -> np.ndarray:
    """Detect and enhance edges in the image.

    Args:
        img: Preprocessed grayscale image

    Returns:
        np.ndarray: Binary image with enhanced edges
    """
    edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
    kernel = np.ones(shape=(5, 5))
    dilated = cv2.dilate(src=edges, kernel=kernel, iterations=2)
    closed_edges = cv2.erode(src=dilated, kernel=kernel, iterations=1)
    return closed_edges


def find_card_contour(
    enhanced_edges: np.ndarray, debug_callback: Callable | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Find the best card contour in the image.

    Args:
        enhanced_edges: Preprocessed image
        debug_callback: Optional callback function for debug information

    Returns:
        tuple: (best_contour, best_approx) or (None, None) if no card found
    """
    contours, _ = cv2.findContours(
        image=enhanced_edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_approx = None

    ASPECT_RATIO_TOLERANCE = 0.2  # Allow 20% deviation from perfect ratio

    # Card should typically occupy between 10-50% of the image
    MIN_CARD_AREA_PERCENT = 0.10  # 10% of image area
    MAX_CARD_AREA_PERCENT = 0.50  # 50% of image area

    image_area = enhanced_edges.shape[0] * enhanced_edges.shape[1]
    min_card_area = image_area * MIN_CARD_AREA_PERCENT
    max_card_area = image_area * MAX_CARD_AREA_PERCENT

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.05 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(contour)
            area_percentage = area / image_area

            # Check if area percentage is within expected range
            if area < min_card_area or area > max_card_area:
                if debug_callback:
                    debug_callback(
                        f"Rejected contour - area {area_percentage:.1%} of image outside expected range"
                    )
                continue

            # Check aspect ratio
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width == 0 or height == 0:  # Avoid division by zero
                continue
            aspect_ratio = min(width, height) / max(width, height)
            expected_ratio = min(EXPECTED_ASPECT_RATIO, 1 / EXPECTED_ASPECT_RATIO)

            if abs(aspect_ratio - expected_ratio) > ASPECT_RATIO_TOLERANCE:
                if debug_callback:
                    debug_callback(
                        f"Rejected contour - aspect ratio {aspect_ratio:.2f} outside tolerance"
                    )
                continue

            if best_contour is None or area > cv2.contourArea(best_contour):
                best_contour = contour
                best_approx = approx
                if debug_callback:
                    debug_callback(
                        f"Found potential card (area: {area_percentage:.1%} of image, aspect ratio: {aspect_ratio:.2f})"
                    )

    return best_contour, best_approx


def create_debug_image(img: np.ndarray, approx: np.ndarray) -> np.ndarray:
    """Create a debug image showing the detected card outline.

    Args:
        img: Original image
        approx: Approximated contour points

    Returns:
        np.ndarray: Debug image with card outline
    """
    debug_img = img.copy()
    cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
    for point in approx:
        x, y = point[0]
        cv2.circle(debug_img, (x, y), 3, (0, 255, 0), 2)
    return debug_img


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order the corners of the detected card for perspective transform.

    Args:
        corners: Array of corner points

    Returns:
        np.ndarray: Ordered corner points
    """
    # Calculate the sum of each corner's coordinates along axis 1
    coord_sums = corners.sum(axis=1)

    # Get the indices that would sort the coordinate sums
    sorted_indices = np.argsort(coord_sums)

    top_left = corners[sorted_indices[0]]
    bottom_right = corners[sorted_indices[3]]

    is_titled_right = corners[sorted_indices[1]][1] > corners[sorted_indices[2]][1]
    if is_titled_right:
        top_right = corners[sorted_indices[2]]
        bottom_left = corners[sorted_indices[1]]
    else:
        top_right = corners[sorted_indices[1]]
        bottom_left = corners[sorted_indices[2]]
    return np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ],
        dtype="float32",
    )


def detect_card(
    img: np.ndarray, canvas_height: int, debug_callback: Callable | None = None
) -> tuple[bool, np.ndarray, np.ndarray | None]:
    """Detect and extract Pokemon card from image.

    Args:
        img: Source image in RGBA format
        canvas_height: Height of the target canvas
        debug_callback: Optional callback function for debug information

    Returns:
        tuple: (success: bool, processed_image: np.array, card_image: np.array)
            success: Whether a card was successfully detected
            processed_image: Debug image showing detected card outline
            card_image: Extracted and perspective-corrected card image (None if no card found)
    """
    # Calculate dimensions and preprocess image
    card_width, card_height = calculate_card_dimensions(canvas_height)
    preprocessed = preprocess_image(img)
    frame_threshold = detect_edges(preprocessed)

    if debug_callback:
        debug_callback("Image preprocessed")

    # Find card contour
    best_contour, best_approx = find_card_contour(frame_threshold, debug_callback)

    if best_contour is None:
        if debug_callback:
            debug_callback("No card detected")
        return False, img, None

    # Create debug image
    debug_img = create_debug_image(img, best_approx)

    # Order corners and create destination points
    corners = np.array(best_approx).reshape(4, 2)
    ordered_corners = order_corners(corners)

    dst = np.array(
        [[0, 0], [card_width, 0], [card_width, card_height], [0, card_height]],
        dtype="float32",
    )

    # Warp perspective to get straight-on view
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst)
    card_img = cv2.warpPerspective(img, matrix, (card_width, card_height))

    if debug_callback:
        debug_callback("Card successfully extracted")
    return True, debug_img, card_img
