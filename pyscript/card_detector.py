from typing import Callable

import cv2
import numpy as np


def calculate_card_dimensions(canvas_height: int) -> tuple[int, int]:
    """Calculate the target dimensions for the card based on canvas height.

    Args:
        canvas_height: Height of the target canvas

    Returns:
        tuple: (card_width, card_height)
    """
    card_ratio = 2.5 / 3.5  # Pokemon card ratio
    card_height = int(canvas_height * 0.8)
    card_width = int(card_height * card_ratio)
    return card_width, card_height


def preprocess_image(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess the image for contour detection.

    Args:
        img: Source image in RGBA format

    Returns:
        tuple: (edges, blurred) preprocessed images
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=0)

    edges = cv2.Canny(blurred, 140, 250)
    kernel = np.ones(shape=(5, 5))
    frame_dial = cv2.dilate(edges, kernel, iterations=2)
    frame_threshold = cv2.erode(frame_dial, kernel, iterations=1)

    return frame_threshold, blurred


def find_card_contour(
    frame_threshold: np.ndarray, debug_callback: Callable | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Find the best card contour in the image.

    Args:
        frame_threshold: Preprocessed image
        debug_callback: Optional callback function for debug information

    Returns:
        tuple: (best_contour, best_approx) or (None, None) if no card found
    """
    contours, _ = cv2.findContours(
        frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if debug_callback:
        debug_callback(f"Found {len(contours)} contours")

    best_contour = None
    best_approx = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.05 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and 300 < perimeter < 2000:
            if best_contour is None or cv2.contourArea(contour) > cv2.contourArea(
                best_contour
            ):
                best_contour = contour
                best_approx = approx
                if debug_callback:
                    debug_callback(f"Found potential card (perimeter: {perimeter:.1f})")

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
    sums = corners.sum(axis=1)
    sorted_idx = np.argsort(sums)

    # Handle different card orientations
    if corners[sorted_idx[1]][1] > corners[sorted_idx[2]][1]:  # Tilted Right
        return np.array(
            [
                corners[sorted_idx[0]],  # Top-left
                corners[sorted_idx[2]],  # Top-right
                corners[sorted_idx[3]],  # Bottom-right
                corners[sorted_idx[1]],  # Bottom-left
            ],
            dtype="float32",
        )
    else:  # Tilted Left
        return np.array(
            [
                corners[sorted_idx[0]],  # Top-left
                corners[sorted_idx[1]],  # Top-right
                corners[sorted_idx[3]],  # Bottom-right
                corners[sorted_idx[2]],  # Bottom-left
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
    frame_threshold, _ = preprocess_image(img)

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
