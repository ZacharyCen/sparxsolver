"""Chrome session that captures and sends input without desktop input devices."""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np


class BrowserSessionError(RuntimeError):
    pass


class BrowserSession:
    def __init__(self, port, allowed_host="sparx-learning.com"):
        self.port = port
        self.allowed_host = allowed_host
        self.socket = None
        self.target_id = None
        self.command_timeout = 10.0
        self.sequence = 0
        self.lock = threading.RLock()
        self.scale = (1.0, 1.0)
        self.image_size = (1280, 800)
        self.pressed_point = None

    @staticmethod
    def open(initial_url="https://maths.sparx-learning.com/", profile_dir=None, allowed_host="sparx-learning.com"):
        profile = Path(profile_dir or Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local/share"))) / "AutoAnswer" / "Browser")
        profile.mkdir(parents=True, exist_ok=True)
        port_file = profile / "DevToolsActivePort"
        def existing():
            port = int(port_file.read_text().splitlines()[0])
            session = BrowserSession(port, allowed_host)
            session._http("/json/version")
            return session
        try:
            return existing()
        except (OSError, ValueError, IndexError, BrowserSessionError):
            pass
        chrome = shutil.which("chrome") or shutil.which("google-chrome")
        for root in (os.environ.get("PROGRAMFILES", "C:/Program Files"), os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")):
            candidate = Path(root) / "Google/Chrome/Application/chrome.exe"
            if candidate.exists():
                chrome = str(candidate)
                break
        if not chrome:
            raise BrowserSessionError("Chrome not found. Install Chrome to use the separate browser.")
        subprocess.Popen([chrome, f"--user-data-dir={profile}", "--remote-debugging-address=127.0.0.1",
                          "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check",
                          "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
                          "--disable-background-timer-throttling", "--force-device-scale-factor=1",
                          "--window-size=1280,900", initial_url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                return existing()
            except (OSError, ValueError, IndexError, BrowserSessionError):
                time.sleep(.15)
        raise BrowserSessionError("The separate browser did not start. Close its window and retry Open browser.")

    def _http(self, path, method="GET"):
        try:
            with urlopen(Request(f"http://127.0.0.1:{self.port}{path}", method=method), timeout=2) as response:
                return json.load(response)
        except (OSError, ValueError) as error:
            raise BrowserSessionError("The separate browser is closed or unavailable. Reopen it before starting.") from error

    def _allowed(self, url):
        host = urlparse(url).hostname or ""
        return host == self.allowed_host or host.endswith("." + self.allowed_host)

    def select_homework_tab(self):
        tabs = [tab for tab in self._http("/json/list") if tab.get("type") == "page" and self._allowed(tab.get("url", ""))]
        if not tabs:
            raise BrowserSessionError("Open Sparx in the separate browser and sign in before starting.")
        self._connect(tabs[0])

    def open_homework(self):
        tabs = [tab for tab in self._http("/json/list") if tab.get("type") == "page" and self._allowed(tab.get("url", ""))]
        if not tabs:
            self._http("/json/new?" + quote("https://maths.sparx-learning.com/", safe=""), "PUT")

    def _connect(self, tab):
        import websocket
        with self.lock:
            self.close()
            self.socket = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=10, suppress_origin=True)
            self.target_id = tab["id"]
            self.sequence = 0
            # Page.enable only subscribes to events; capture/input do not need it.

    def command(self, method, params=None):
        with self.lock:
            if self.socket is None:
                raise BrowserSessionError("Separate browser is not connected. Open browser and select your homework.")
            try:
                self.sequence += 1
                request_id = self.sequence
                self.socket.settimeout(self.command_timeout)
                self.socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
                deadline = time.monotonic() + self.command_timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Browser command deadline exceeded")
                    self.socket.settimeout(remaining)
                    response = json.loads(self.socket.recv())
                    if response.get("id") == request_id:
                        if "error" in response:
                            raise BrowserSessionError(response["error"].get("message", "Browser command failed"))
                        return response.get("result", {})
            except BrowserSessionError:
                raise
            except Exception as error:
                broken = self.socket
                self.socket = None
                self.pressed_point = None
                try:
                    broken.close()
                except Exception:
                    pass
                operation = {"Page.captureScreenshot": "capturing its page", "Page.getLayoutMetrics": "reading its page size"}.get(method, "responding to a command")
                if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
                    message = f"The separate browser timed out while {operation}. To use your open Chrome window, choose 'Use existing Chrome window' in Configuration."
                else:
                    message = "The separate browser disconnected. Reopen it, or choose 'Use existing Chrome window' in Configuration."
                raise BrowserSessionError(message) from error

    def ensure_homework(self):
        tabs = self._http("/json/list")
        if not any(tab.get("id") == self.target_id and self._allowed(tab.get("url", "")) for tab in tabs):
            raise BrowserSessionError("The selected homework tab was closed or left Sparx. Stop and select homework again.")

    def capture(self):
        with self.lock:
            self.ensure_homework()
            metrics = self.command("Page.getLayoutMetrics")["cssLayoutViewport"]
            encoded = self.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})["data"]
            image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise BrowserSessionError("The separate browser returned an empty screenshot")
            self.image_size = (image.shape[1], image.shape[0])
            self.scale = (image.shape[1] / metrics["clientWidth"], image.shape[0] / metrics["clientHeight"])
            return image

    def mouse(self, event, point, **params):
        if event == "mousePressed" or params.get("buttons") == 1:
            self.pressed_point = point
        self.command("Input.dispatchMouseEvent", {"type": event, "x": point[0] / self.scale[0], "y": point[1] / self.scale[1], **params})
        if event == "mouseReleased":
            self.pressed_point = None

    def click(self, point):
        self.ensure_homework()
        self.mouse("mousePressed", point, button="left", clickCount=1)
        self.mouse("mouseReleased", point, button="left", clickCount=1)

    def key(self, key, code, vk, text="", modifiers=0):
        self.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": key, "code": code,
                     "windowsVirtualKeyCode": vk, "text": text, "modifiers": modifiers})
        self.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": key, "code": code,
                     "windowsVirtualKeyCode": vk, "modifiers": modifiers})

    def type_text(self, point, text, cancelled=lambda: False):
        self.click(point)
        try:
            for _ in range(32):
                if cancelled():
                    return
                self.key("Backspace", "Backspace", 8)
            for char in text:
                if cancelled():
                    return
                code = "Digit" + char if char.isdigit() else {"-": "Minus", ".": "Period"}.get(char, "")
                self.key(char, code, ord(char.upper()) if char.isalnum() else {"-": 189, ".": 190}.get(char, 0), char)
        finally:
            self.release_inputs()

    def fill(self, point, text):
        self.click(point)
        self.key("a", "KeyA", 65, modifiers=2)
        self.command("Input.insertText", {"text": text})

    def release_inputs(self):
        if self.socket is not None:
            try:
                if self.pressed_point is not None:
                    self.mouse("mouseReleased", self.pressed_point, button="left", clickCount=1)
                for key, code, vk in (("Control", "ControlLeft", 17), ("Shift", "ShiftLeft", 16), ("Alt", "AltLeft", 18)):
                    self.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": key, "code": code, "windowsVirtualKeyCode": vk})
            except Exception:
                pass

    def close(self):
        with self.lock:
            if self.socket is not None:
                self.socket.close()
                self.socket = None
