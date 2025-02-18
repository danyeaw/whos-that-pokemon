import asyncio
import base64
import cv2
import js
import numpy as np
from card_detector import detect_card
from card_matcher import CardMatcher
from pyodide.ffi import to_js
from pyscript import document, window, when

def create_pokemon_card_app():
    app = PokemonCardApp()

    @when("click", "#click-photo")
    def handle_photo_click(event):
        width = app.video.videoWidth
        height = app.video.videoHeight
        canvas = js.OffscreenCanvas.new(width, height)
        ctx = canvas.getContext("2d")

        ctx.drawImage(app.video, 0, 0, width, height)
        image_data = ctx.getImageData(0, 0, width, height).data

        frame = np.asarray(image_data, dtype=np.uint8).reshape((height, width, 4))
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        card_found, debug_img, card_img = detect_card(frame_rgba, height, window.console.log)

        detected_card_base64 = None
        if card_found and card_img is not None:
            _, buffer = cv2.imencode('.png', card_img)
            detected_card_base64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

            matcher = CardMatcher(window.console.log)
            match_result = matcher.find_matching_card(card_img)

            if match_result:
                match_result['detected_card_image'] = detected_card_base64

            app.process_match_result(match_result)
        else:
            window.console.log('<span style="color: red; font-size: 20px;">No Pokemon card found ❌</span>')
            document.getElementById('no-card-message').classList.remove('hidden')

    @when("click", "#camera-toggle")
    async def handle_camera_toggle(event):
        if app.active_stream:
            await app.stop_camera()
            if app.camera_toggle:
                app.camera_toggle.innerHTML = "📷"
        else:
            await app.start_camera()
            if app.camera_toggle:
                app.camera_toggle.innerHTML = "⏹️"

    @when("click", "#camera-switch")
    async def handle_camera_switch(event):
        try:
            if not app.available_cameras:
                devices = await js.navigator.mediaDevices.enumerateDevices()
                app.available_cameras = [d for d in devices if d.kind == "videoinput"]
                window.console.log(f"Found {len(app.available_cameras)} cameras")

            if len(app.available_cameras) > 1:
                app.current_camera_index = (app.current_camera_index + 1) % len(app.available_cameras)
                await app.start_camera(app.available_cameras[app.current_camera_index].deviceId)
        except Exception as e:
            window.console.log(f"Error switching camera: {str(e)}")

    return app

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

app = create_pokemon_card_app()
