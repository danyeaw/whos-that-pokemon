import numpy as np
import cv2


WIDTH, HEIGHT = 330, 440

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    video = False
else:
    video = True


def stack_images(image_array, labels: list[str]):
    rows = len(image_array)  # Get number of rows of images
    cols = len(image_array[0])  # Get numbers of images in a row

    # Loop through the images
    # OpenCV stores grayscale images as 2D arrays, so we need to convert them to 3D arrays to be able to combine them
    # with the colored images
    for x in range(0, rows):
        for y in range(0, cols):
            if len(image_array[x][y].shape) == 2:
                image_array[x][y] = cv2.cvtColor(image_array[x][y], cv2.COLOR_GRAY2BGR)

    # Create a black image
    black_image = np.zeros((WIDTH, HEIGHT, 3), np.uint8)

    # Stack the images
    hor = [black_image] * rows
    for row in range(0, rows):
        hor[row] = np.hstack(image_array[row])
    stacked = np.vstack(hor)

    # Add labels via white rectangles and text
    for d in range(0, rows):
        for c in range(0, cols):
            cv2.rectangle(
                stacked,
                (c * WIDTH, d * HEIGHT),
                (c * WIDTH + WIDTH, d * HEIGHT + 32),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                stacked,
                labels[d][c],
                (WIDTH * c + 10, HEIGHT * d + 23),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 0),
                2,
            )

    return stacked


while True:
    if not video:
        # Step 1: Load the image
        orig_image = cv2.imread("test_images/tiltleft2.jpg")
        orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    else:
        # Capture frame-by-frame
        ret, orig_image = cap.read()
        orig_image_rgb = orig_image
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if cv2.waitKey(1) == ord("q"):
            break

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=0)

    # Step 3: Edge detection
    edges = cv2.Canny(blurred, 140, 250)
    kernel = np.ones(shape=(5, 5))
    frame_dial = cv2.dilate(edges, kernel, iterations=2)
    frame_threshold = cv2.erode(frame_dial, kernel, iterations=1)

    contours, _ = cv2.findContours(
        frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_image = orig_image.copy()
    cv2.drawContours(
        image=contour_image,
        contours=contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=2,
    )
    if not video:
        contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
    else:
        contour_image_rgb = contour_image

    # Step 4: Find contours

    best_contour = None
    best_approx = None

    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.05 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and 300 < perimeter < 2000:
            if best_contour is None or cv2.contourArea(contour) > cv2.contourArea(
                best_contour
            ):
                best_contour = contour
                best_approx = approx
                print(perimeter)
    if best_contour is not None:
        print("Found a 4 Sided Contour")
    else:
        print("No 4 Sided Contour Found")

    # Step 5: Draw the 4 Sided Card Approximation
    if best_contour is not None:
        approx_image = orig_image.copy()

        for point in best_approx:
            x, y = point[0]
            cv2.circle(approx_image, (x, y), 3, (0, 255, 0), 2)

        cv2.drawContours(approx_image, [best_approx], -1, (0, 255, 0), 1)
        if not video:
            approx_image_rgb = cv2.cvtColor(approx_image, cv2.COLOR_BGR2RGB)
        else:
            approx_image_rgb = approx_image

    # Step 5: Reorder the corners

    # Order needs to be top-left, top-right, bottom-right, bottom-left

    if best_approx is not None and len(best_approx) > 1:
        corners = np.array(best_approx).reshape(4, 2)
    else:
        continue

    # Calculate the sum of each corner's coordinates along axis 1
    sums = corners.sum(axis=1)

    # Get the indices that would sort the sums
    sorted_idx = np.argsort(sums)

    if corners[sorted_idx[1]][1] > corners[sorted_idx[2]][1]:  # Tilted Right

        ordered_corners = np.array(
            [
                corners[sorted_idx[0]],
                corners[sorted_idx[2]],
                corners[sorted_idx[3]],
                corners[sorted_idx[1]],
            ],
            dtype="float32",
        )

    else:  # Tilted Left
        ordered_corners = np.array(
            [
                corners[sorted_idx[0]],
                corners[sorted_idx[1]],
                corners[sorted_idx[3]],
                corners[sorted_idx[2]],
            ],
            dtype="float32",
        )

    # Step 6: Warp the image to a Rectangle

    dst = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype="float32")

    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(orig_image, M, (WIDTH, HEIGHT))

    if not video:
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    else:
        warped_rgb = warped

    # Step 7: Resize and Display Images
    orig_image_rgb = cv2.resize(orig_image_rgb, (WIDTH, HEIGHT))
    gray = cv2.resize(gray, (WIDTH, HEIGHT))
    blurred = cv2.resize(blurred, (WIDTH, HEIGHT))
    contour_image_rgb = cv2.resize(contour_image_rgb, (WIDTH, HEIGHT))
    approx_image_rgb = cv2.resize(approx_image_rgb, (WIDTH, HEIGHT))
    warped_rgb = cv2.resize(warped_rgb, (WIDTH, HEIGHT))

    images = (
        [orig_image_rgb, gray, blurred],
        [contour_image_rgb, approx_image_rgb, warped],
    )
    labels = (["Original", "Gray", "Blurred"], ["Contours", "Approximate", "Warped"])

    image_stack = stack_images(images, labels)

    # Display the result using Matplotlib
    cv2.imshow("Card Finder", image_stack)

    if not video:  # If reading image file, display image until key is pressed
        cv2.waitKey(0)  # Keeps window open until any key is pressed
        break
    elif cv2.waitKey(1) & 0xFF == ord(
        "q"
    ):  # If reading from video, quit if 'q' is pressed
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
