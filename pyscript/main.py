import asyncio
import js
import numpy as np
import cv2
from pyodide.ffi import to_js
from js import Uint8Array, Object
from js.ffi.wrappers import add_event_listeners

active_stream = None
available_cameras = []
current_camera_index = 0

video = js.document.querySelector("#video")
click_button = js.document.querySelector("#click-photo")
camera_toggle = js.document.querySelector("#camera-toggle")
camera_switch = js.document.querySelector("#camera-switch")
canvas = js.document.querySelector("#canvas")
result_div = js.document.querySelector("#result")
match_info = js.document.querySelector("#match-info")


async def stop_camera():
    """Stop the current camera stream"""
    global active_stream
    if active_stream:
        tracks = active_stream.getTracks()
        for track in tracks:
            track.stop()
        active_stream = None
        if video:
            video.srcObject = None


async def start_camera(camera_id=None):
    """Start the camera with optional camera_id"""
    try:
        global active_stream
        await stop_camera()

        # Set up constraints
        constraints = Object.new()
        constraints.audio = False
        video_constraints = Object.new()

        if camera_id:
            video_constraints.deviceId = camera_id
        else:
            video_constraints.facingMode = "environment"

        constraints.video = video_constraints

        # Get camera stream
        stream = await js.navigator.mediaDevices.getUserMedia(constraints)
        active_stream = stream
        if video:
            video.srcObject = stream
            video.style.display = "block"

        js.console.log("Camera started successfully")
    except Exception as e:
        js.console.log(f"Camera error: {str(e)}")
        import traceback

        js.console.log(traceback.format_exc())


async def toggle_camera(e):
    """Toggle the camera on/off"""
    if active_stream:
        await stop_camera()
        if camera_toggle:
            camera_toggle.innerHTML = "📷"
    else:
        await start_camera()
        if camera_toggle:
            camera_toggle.innerHTML = "⏹️"


async def switch_camera(e):
    """Switch between available cameras"""
    global current_camera_index, available_cameras

    try:
        # Get list of cameras if we haven't already
        if not available_cameras:
            devices = await js.navigator.mediaDevices.enumerateDevices()
            available_cameras = [d for d in devices if d.kind == "videoinput"]
            js.console.log(f"Found {len(available_cameras)} cameras")

        if len(available_cameras) > 1:
            current_camera_index = (current_camera_index + 1) % len(available_cameras)
            await start_camera(available_cameras[current_camera_index].deviceId)
    except Exception as e:
        js.console.log(f"Error switching camera: {str(e)}")


def process_match_result(match_result):
    print("Processing match result:", match_result)  # Debug print

    if match_result:
        # Ensure subtypes is a list
        subtypes = match_result.get("subtypes", [])
        if isinstance(subtypes, str):
            subtypes = [subtypes]

        # Prepare market prices
        market_prices = match_result.get("market_prices", {})
        if isinstance(market_prices, str):
            market_prices = {"error": market_prices}

        # Prepare the data dictionary
        data = {
            "number": match_result.get("number", ""),
            "supertype": match_result.get("supertype", ""),
            "rarity": match_result.get("rarity", ""),
            "subtypes": subtypes,
            "images": match_result.get("images", {}),
            "market_prices": market_prices,
            "confidence": float(match_result.get("confidence", 0)),
            "match_quality": match_result.get("match_quality", ""),
        }

        js.console.log("Pokemon card detected! ✅")

        print("Processed data:", data)
        js_data = to_js(data, dict_converter=js.Object.fromEntries)
        js.showResultScreen(str(match_result.get("name", "")), js_data)
    else:
        print("No match result to process")


def click_button_click(e):
    """Handle the capture button click"""
    try:
        js.console.log("Processing image...")

        if not video or not canvas:
            js.console.log("Video or canvas not ready")
            return

        # Update canvas size to match video
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight

        # Get the canvas context and draw video frame
        ctx = canvas.getContext("2d")
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        # Get image data
        image_data = ctx.getImageData(0, 0, canvas.width, canvas.height)
        js.console.log("Got image data")

        # Convert image data to bytes
        js_array = Uint8Array.new(image_data.data)
        bytes_data = js_array.to_bytes()
        js.console.log("Converted to bytes")

        # Convert to numpy array
        pixels_flat = np.frombuffer(bytes_data, dtype=np.uint8)
        js.console.log(f"Converted to array with shape: {pixels_flat.shape}")

        # Reshape the array
        try:
            frame = pixels_flat.reshape((canvas.height, canvas.width, 4))
            js.console.log(f"Reshaped array to: {frame.shape}")
        except Exception as e:
            js.console.log(f"Reshape error: {str(e)}")
            return

        # Convert RGBA to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        js.console.log("Converted to BGR")

        # Initialize detector and find card
        from card_detector import CardDetector

        detector = CardDetector(canvas.width, canvas.height)
        card_found, debug_img, card_img = detector.detect(frame_bgr, js.console.log)

        if card_found and card_img is not None:
            # Initialize matcher and find matching card
            from card_matcher import CardMatcher

            matcher = CardMatcher(js.console.log)
            match_result = matcher.find_matching_card(card_img)
            process_match_result(match_result)

        else:
            js.console.log(
                '<span style="color: red; font-size: 20px;">No Pokemon card found ❌</span>'
            )

    except Exception as e:
        js.console.log(f"Error processing frame: {str(e)}")
        import traceback

        js.console.log(traceback.format_exc())


# Set up event listeners
add_event_listeners(click_button, "click", click_button_click)
add_event_listeners(camera_toggle, "click", toggle_camera)
add_event_listeners(camera_switch, "click", switch_camera)

# Start camera and show main screen
start_camera()
js.document.getElementById("loading-screen").style.display = "none"
js.document.getElementById("main-container").style.display = "block"
js.console.log("Initialization complete")
