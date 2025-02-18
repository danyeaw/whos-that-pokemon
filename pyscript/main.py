import asyncio
import base64

import cv2
import js
import numpy as np
from card_detector import detect_card
from card_matcher import CardMatcher
from pyodide.ffi import to_js
from pyodide.ffi.wrappers import add_event_listener
from pyscript import document, window


class PokemonCardApp:
    def __init__(self):
        self.active_stream = None
        self.available_cameras = []
        self.current_camera_index = 0

        # Get DOM elements
        self.video = document.querySelector("#video")
        self.click_button = document.querySelector("#click-photo")
        self.camera_toggle = document.querySelector("#camera-toggle")
        self.camera_switch = document.querySelector("#camera-switch")
        self.result_div = document.querySelector("#result")
        self.match_info = document.querySelector("#match-info")

        # Bind event handlers
        add_event_listener(self.click_button, "click", self.click_button_click)
        add_event_listener(self.camera_toggle, "click", self.toggle_camera)
        add_event_listener(self.camera_switch, "click", self.switch_camera)

        # Show main container
        document.getElementById("loading-screen").style.display = "none"
        document.getElementById("main-container").style.display = "block"

        # Start camera automatically
        asyncio.create_task(self.start_camera())

    async def stop_camera(self):
        tracks = self.active_stream.getTracks()
        for track in tracks:
            track.stop()
        self.active_stream = None
        if self.video:
            self.video.srcObject = None

    async def start_camera(self, camera_id=None):
        try:
            if self.active_stream:
                await self.stop_camera()
            constraints = js.Object.new()
            constraints.audio = False
            video_constraints = js.Object.new()
            if camera_id:
                video_constraints.deviceId = camera_id
            else:
                video_constraints.facingMode = "environment"
            constraints.video = video_constraints
            stream = await js.navigator.mediaDevices.getUserMedia(constraints)
            self.active_stream = stream
            if self.video:
                self.video.srcObject = stream
                self.video.style.display = "block"
            window.console.log("Camera started successfully")
        except Exception as e:
            window.console.log(f"Camera error: {str(e)}")

    async def toggle_camera(self, e):
        if self.active_stream:
            await self.stop_camera()
            if self.camera_toggle:
                self.camera_toggle.innerHTML = "📷"
        else:
            await self.start_camera()
            if self.camera_toggle:
                self.camera_toggle.innerHTML = "⏹️"

    async def switch_camera(self, e):
        try:
            if not self.available_cameras:
                devices = await js.navigator.mediaDevices.enumerateDevices()
                self.available_cameras = [d for d in devices if d.kind == "videoinput"]
                window.console.log(f"Found {len(self.available_cameras)} cameras")

            if len(self.available_cameras) > 1:
                self.current_camera_index = (self.current_camera_index + 1) % len(
                    self.available_cameras
                )
                await self.start_camera(
                    self.available_cameras[self.current_camera_index].deviceId
                )
        except Exception as e:
            window.console.log(f"Error switching camera: {str(e)}")

    def process_match_result(self, match_result):
        if not match_result:
            print("No match result to process")
            return

        subtypes = match_result.get("subtypes", [])
        if isinstance(subtypes, str):
            subtypes = [subtypes]

        market_prices = match_result.get("market_prices", {})
        if isinstance(market_prices, str):
            market_prices = {"error": market_prices}

        data = {
            "number": match_result.get("number", ""),
            "supertype": match_result.get("supertype", ""),
            "rarity": match_result.get("rarity", ""),
            "subtypes": subtypes,
            "images": match_result.get("images", {}),
            "market_prices": market_prices,
            "confidence": float(match_result.get("confidence", 0)),
            "match_quality": match_result.get("match_quality", ""),
            "detected_card_image": match_result.get("detected_card_image", None)
        }

        window.console.log("Pokemon card detected! ✅")
        js_data = to_js(data, dict_converter=js.Object.fromEntries)
        js.showResultScreen(str(match_result.get("name", "")), js_data)

    def click_button_click(self, e):
        width = self.video.videoWidth
        height = self.video.videoHeight
        canvas = js.OffscreenCanvas.new(width, height)
        ctx = canvas.getContext("2d")

        ctx.drawImage(self.video, 0, 0, width, height)
        image_data = ctx.getImageData(0, 0, width, height).data

        frame = np.asarray(image_data, dtype=np.uint8).reshape((height, width, 4))
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        card_found, debug_img, card_img = detect_card(frame_rgba, height, window.console.log)

        # Convert detected card image to base64 if found
        detected_card_base64 = None
        if card_found and card_img is not None:
            # Convert the OpenCV image to base64
            _, buffer = cv2.imencode('.png', card_img)
            detected_card_base64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

            matcher = CardMatcher(window.console.log)
            match_result = matcher.find_matching_card(card_img)

            # Add detected card image to match result
            if match_result:
                match_result['detected_card_image'] = detected_card_base64

            self.process_match_result(match_result)
        else:
            window.console.log('<span style="color: red; font-size: 20px;">No Pokemon card found ❌</span>')
            # Show the no-card message in the debug section
            document.getElementById('no-card-message').classList.remove('hidden')


# Initialize the app
app = PokemonCardApp()
