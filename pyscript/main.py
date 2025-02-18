import asyncio
import base64
import cv2
import js
import numpy as np
from card_detector import detect_card
from card_matcher import CardMatcher
from pyscript import when, window
from pyscript.media import Device, list_devices
from pyscript.web import page
from pyscript.ffi import to_js


class PokemonCardApp:
    def __init__(self):
        self.active_stream = None
        self.available_cameras = []
        self.current_camera_index = 0

        self.video = page["#video"][0]
        self.video_element = self.video._dom_element
        self.click_button = page["#click-photo"]
        self.camera_toggle = page["#camera-toggle"]
        self.camera_switch = page["#camera-switch"]
        self.result_div = page["#result"]

        # Hide loading screen, show main container
        page["#loading-screen"].style["display"] = "none"
        page["#main-container"].style["display"] = "block"

        # Start camera
        asyncio.create_task(self.start_camera())

    async def stop_camera(self):
        """Stop the active camera stream"""
        if self.active_stream:
            tracks = self.active_stream.getTracks()
            for track in tracks:
                track.stop()
            self.active_stream = None
            if self.video_element:
                self.video_element.srcObject = None

    async def start_camera(self, device_id=None):
        """Start the camera with optional device ID"""
        try:
            if self.active_stream:
                await self.stop_camera()

            video_options = (
                {"facingMode": "environment"}
                if not device_id
                else {"deviceId": {"exact": device_id}}
            )
            self.active_stream = await Device.load(audio=False, video=video_options)

            if self.video_element:
                self.video_element.srcObject = self.active_stream
                self.video.style["display"] = "block"
        except Exception as e:
            window.console.log(f"Camera error: {str(e)}")

    def process_match_result(self, match_result):
        """Process and display the card matching results"""
        if not match_result:
            return

        data = {
            "number": match_result.get("number", ""),
            "supertype": match_result.get("supertype", ""),
            "rarity": match_result.get("rarity", ""),
            "subtypes": (
                [match_result.get("subtypes")]
                if isinstance(match_result.get("subtypes"), str)
                else match_result.get("subtypes", [])
            ),
            "images": match_result.get("images", {}),
            "market_prices": (
                {"error": match_result.get("market_prices")}
                if isinstance(match_result.get("market_prices"), str)
                else match_result.get("market_prices", {})
            ),
            "confidence": float(match_result.get("confidence", 0)),
            "match_quality": match_result.get("match_quality", ""),
            "detected_card_image": match_result.get("detected_card_image", None),
        }

        js.showResultScreen(
            str(match_result.get("name", "")),
            to_js(data, dict_converter=js.Object.fromEntries),
        )

    def handle_photo_click(self, event):
        """Handle the photo capture button click"""
        width = self.video_element.videoWidth
        height = self.video_element.videoHeight
        canvas = window.OffscreenCanvas.new(width, height)
        ctx = canvas.getContext("2d")
        ctx.drawImage(self.video_element, 0, 0, width, height)

        frame = np.asarray(
            ctx.getImageData(0, 0, width, height).data, dtype=np.uint8
        ).reshape((height, width, 4))

        card_found, debug_img, card_img = detect_card(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA), height, window.console.log
        )

        if card_found and card_img is not None:
            _, buffer = cv2.imencode(".png", card_img)
            match_result = CardMatcher(window.console.log).find_matching_card(card_img)

            if match_result:
                match_result["detected_card_image"] = (
                    f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
                )
                self.process_match_result(match_result)
        else:
            page["#no-card-message"][0].classes.remove("hidden")

    async def toggle_camera(self, event):
        """Toggle the camera on/off"""
        if self.active_stream:
            await self.stop_camera()
            self.camera_toggle.innerHTML = "📷"
        else:
            await self.start_camera()
            self.camera_toggle.innerHTML = "⏹️"

    async def switch_camera(self, event):
        """Switch between available cameras"""
        try:
            if not self.available_cameras:
                devices = await list_devices()
                self.available_cameras = [d for d in devices if d.kind == "videoinput"]

            if len(self.available_cameras) > 1:
                self.current_camera_index = (self.current_camera_index + 1) % len(
                    self.available_cameras
                )
                await self.start_camera(
                    self.available_cameras[self.current_camera_index].id
                )
        except Exception as e:
            window.console.log(f"Error switching camera: {str(e)}")


# Initialize app and set up event handlers
app = PokemonCardApp()


# Event handlers using @when decorator
@when("click", app.click_button)
def handle_click(event):
    app.handle_photo_click(event)


@when("click", app.camera_toggle)
async def handle_toggle(event):
    await app.toggle_camera(event)


@when("click", app.camera_switch)
async def handle_switch(event):
    await app.switch_camera(event)
