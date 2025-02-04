import cv2
import numpy as np


class CardDetector:
    def __init__(self, canvas_width, canvas_height):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.card_ratio = 2.5 / 3.5  # Pokemon card ratio

        # Calculate card dimensions
        self.card_height = int(canvas_height * 0.8)  # 80% of canvas height
        self.card_width = int(self.card_height * self.card_ratio)

    def detect(self, img, debug_callback=None):
        """
        Detect and extract Pokemon card from image
        Returns: (success, processed_image, card_image)
        """
        # Convert to grayscale and blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=0)
        if debug_callback:
            debug_callback("Image preprocessed")

        # Edge detection and contour preparation
        edges = cv2.Canny(blurred, 140, 250)
        kernel = np.ones(shape=(5, 5))
        frame_dial = cv2.dilate(edges, kernel, iterations=2)
        frame_threshold = cv2.erode(frame_dial, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if debug_callback:
            debug_callback(f"Found {len(contours)} contours")

        # Find best card contour
        best_contour = None
        best_approx = None

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.05 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Look for card-sized 4-sided shapes
            if len(approx) == 4 and 300 < perimeter < 2000:
                if best_contour is None or cv2.contourArea(contour) > cv2.contourArea(
                    best_contour
                ):
                    best_contour = contour
                    best_approx = approx
                    if debug_callback:
                        debug_callback(
                            f"Found potential card (perimeter: {perimeter:.1f})"
                        )

        if best_contour is None:
            if debug_callback:
                debug_callback("No card detected")
            return False, img, None

        # Draw card outline on the debug image
        debug_img = img.copy()
        cv2.drawContours(debug_img, [best_approx], -1, (0, 255, 0), 3)
        for point in best_approx:
            x, y = point[0]
            cv2.circle(debug_img, (x, y), 3, (0, 255, 0), 2)

        # Order corners
        corners = np.array(best_approx).reshape(4, 2)
        sums = corners.sum(axis=1)
        sorted_idx = np.argsort(sums)

        # Handle different card orientations
        if corners[sorted_idx[1]][1] > corners[sorted_idx[2]][1]:  # Tilted Right
            ordered_corners = np.array(
                [
                    corners[sorted_idx[0]],  # Top-left
                    corners[sorted_idx[2]],  # Top-right
                    corners[sorted_idx[3]],  # Bottom-right
                    corners[sorted_idx[1]],  # Bottom-left
                ],
                dtype="float32",
            )
        else:  # Tilted Left
            ordered_corners = np.array(
                [
                    corners[sorted_idx[0]],  # Top-left
                    corners[sorted_idx[1]],  # Top-right
                    corners[sorted_idx[3]],  # Bottom-right
                    corners[sorted_idx[2]],  # Bottom-left
                ],
                dtype="float32",
            )

        # Create destination points for the card
        dst = np.array(
            [
                [0, 0],
                [self.card_width, 0],
                [self.card_width, self.card_height],
                [0, self.card_height],
            ],
            dtype="float32",
        )

        # Warp perspective to get straight-on view
        matrix = cv2.getPerspectiveTransform(ordered_corners, dst)
        card_img = cv2.warpPerspective(img, matrix, (self.card_width, self.card_height))

        if debug_callback:
            debug_callback("Card successfully extracted")
        return True, debug_img, card_img
