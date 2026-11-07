"""
executor.py - Desktop and Separate-Browser Input Execution Layer
Implements:
- Pynput desktop input or browser-targeted input without desktop focus
- Atomic clipboard text injection with safe clipboard restoration (pyperclip)
- 6-step physical smooth drag with guaranteed mouse/modifier release in finally
- Platform wheel scroll step conversion
- Emergency stop cancellation checks and forceful input state release
"""

import sys
import time
from typing import List, Tuple, Optional
import pynput
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController
import pyperclip


class UnsupportedInputMethod(ValueError):
    """The selected input shortcut cannot enter this answer; use other controls."""


class ExecutionPaused(RuntimeError):
    """An action was blocked before dispatch because the user paused input."""


class ActionExecutor:
    """
    Executes single-step atomic actions planned by the model.
    Guarantees clean release of all physical mouse buttons and keys on cancellation or error.
    """

    def __init__(self, roi_offset: Tuple[int, int] = (0, 0)):
        """
        :param roi_offset: (offset_x, offset_y) to transform ROI-relative coordinates
                           to global desktop screen coordinates.
        """
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.roi_offset_x, self.roi_offset_y = roi_offset
        self.browser_session = None
        self.input_delays = False
        self.smooth_pointer = False
        self.keyboard_entry_enabled = True
        self._cancelled = False
        self._paused = False

    def pause(self):
        """Block new actions while allowing the current atomic input to release normally."""
        self._paused = True

    def resume(self):
        self._paused = False

    def set_roi_offset(self, offset_x: int, offset_y: int) -> None:
        """Updates the ROI offset for translating image coordinates to screen coordinates."""
        self.roi_offset_x = offset_x
        self.roi_offset_y = offset_y

    def cancel(self) -> None:
        """Flags cancellation and triggers input cleanup."""
        self._cancelled = True
        self.release_all_inputs()

    def reset_cancellation(self) -> None:
        """Resets cancellation flag for the next cycle."""
        self._cancelled = False
        self._paused = False

    def is_cancelled(self) -> bool:
        """Checks if current execution has been cancelled."""
        return self._cancelled

    def release_all_inputs(self) -> None:
        """
        Forcefully releases any pressed mouse buttons and common modifier keys
        to guarantee fail-safe state.
        """
        if getattr(self, "browser_session", None):
            self.browser_session.release_inputs()
            return
        try:
            for button, virtual_key in ((Button.left,0x01),(Button.right,0x02)):
                # A synthetic right-button UP without a held button opens Chrome's
                # context menu on Windows. Release held inputs, not idle buttons.
                if sys.platform == 'win32':
                    import ctypes
                    if not ctypes.windll.user32.GetAsyncKeyState(virtual_key) & 0x8000:
                        continue
                self.mouse.release(button)
        except Exception:
            pass

        try:
            self.keyboard.release(Key.ctrl)
            self.keyboard.release(Key.cmd)
            self.keyboard.release(Key.alt)
            self.keyboard.release(Key.shift)
        except Exception:
            pass

    def _to_screen_coords(self, coords: List[int]) -> Tuple[int, int]:
        """Translates frame [x, y] coordinates to screen [x, y] coordinates."""
        return (coords[0] + self.roi_offset_x, coords[1] + self.roi_offset_y)

    def execute(self, action: dict) -> None:
        """
        Executes a validated single-step action dict.
        """
        if getattr(self, '_paused', False):
            raise ExecutionPaused('Input is paused')
        if self._cancelled:
            print("[Executor] Action skipped: cancelled.")
            return

        action_type = action.get("type")
        if action_type == 'press_enter':
            if getattr(self,'browser_session',None):
                self.browser_session.key('Enter','Enter',13)
            else:
                try:self.keyboard.press(Key.enter)
                finally:self.keyboard.release(Key.enter)
        elif action_type == "type_text":
            self.type_text(action["target"], action["text"])
        elif action_type == "fill":
            self.fill(action["target"], action["text"])
        elif action_type == "click":
            self.click(action["target"])
        elif action_type == "drag":
            self.drag(action["start"], action["end"])
        elif action_type == "scroll":
            self.scroll(action["direction"], action["steps"])
        elif action_type == "wait":
            self.wait(action.get("wait_ms", 500))
        elif action_type == "stop":
            print(f"[Executor] Stop action received: {action.get('reason')}")
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def _input_pause(self, seconds: float) -> None:
        """Optional input pacing; browser state verification is separate."""
        if self.input_delays:
            self.wait(int(seconds * 1000))

    def _move_pointer(self, x: int, y: int) -> None:
        if self.smooth_pointer:
            old_x, old_y = self.mouse.position
            for step in range(1, 7):
                if self._cancelled:
                    return
                self.mouse.position = (round(old_x + (x-old_x)*step/6), round(old_y + (y-old_y)*step/6))
                self.wait(10)
        else:
            self.mouse.position = (x, y)

    @staticmethod
    def numeric_keyboard_text(text: str) -> Optional[str]:
        """Normalize plain decimal input; expressions require other entry controls."""
        import re
        value=str(text).strip().replace('−','-').replace('–','-')
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',value):
            return None
        value=value.lstrip('+')
        if value.startswith('.'):value='0'+value
        if value.startswith('-.'):value='-0'+value[1:]
        return value if len(value)<=32 else None

    def type_text(self, target: List[int], text: str) -> None:
        """Replace a focused numeric answer using real keyboard events."""
        if self._cancelled:
            return
        if not self.keyboard_entry_enabled:
            raise UnsupportedInputMethod("Numeric keyboard entry is disabled; use the visible answer controls")
        normalized=self.numeric_keyboard_text(text)
        if normalized is None:
            raise UnsupportedInputMethod("This answer requires fraction/expression controls or a text field; the numeric shortcut cannot enter it")
        text=normalized
        if getattr(self, "browser_session", None):
            self.browser_session.type_text(target, text, self.is_cancelled)
            return
        self.click(target)
        if self._cancelled:
            return
        held_key = None
        try:
            # Custom keypads often ignore Ctrl+A/paste, but handle Backspace.
            for _ in range(32):
                if self._cancelled:
                    return
                held_key = Key.backspace
                self.keyboard.press(held_key)
                self.keyboard.release(held_key)
                held_key = None
            for char in text:
                if self._cancelled:
                    return
                key = char
                if sys.platform == "win32":
                    vk = {"-": 0xBD, ".": 0xBE}.get(char, ord(char))
                    key = pynput.keyboard.KeyCode.from_vk(vk)
                held_key = key
                self.keyboard.press(held_key)
                self.keyboard.release(held_key)
                held_key = None
                self._input_pause(0.02)
        finally:
            if held_key is not None:
                self.keyboard.release(held_key)

    def fill(self, target: List[int], text: str) -> None:
        """
        Injects text using atomic clipboard paste:
        1. Backs up original clipboard content
        2. Copies text to clipboard
        3. Clicks input field target
        4. Simulates Ctrl+V (or Cmd+V on macOS)
        5. Restores original clipboard content
        """
        if self._cancelled:
            return

        if getattr(self, "browser_session", None):
            self.browser_session.fill(target, text)
            return
        screen_x, screen_y = self._to_screen_coords(target)
        print(f"[Executor] Text FILL '{text}' at screen ({screen_x}, {screen_y}) [ROI: {target}, Offset: ({self.roi_offset_x}, {self.roi_offset_y})]")

        # 1. Backup clipboard
        orig_clipboard = ""
        try:
            orig_clipboard = pyperclip.paste()
        except Exception:
            pass

        try:
            # 2. Put text into clipboard
            pyperclip.copy(text)
            self._input_pause(0.05)

            # 3. Focus input target
            self.click(target)
            self._input_pause(0.08)

            if self._cancelled:
                return

            # Select existing text to overwrite (Ctrl+A / Cmd+A)
            mod_key = Key.cmd if sys.platform == "darwin" else Key.ctrl
            with self.keyboard.pressed(mod_key):
                self.keyboard.press('a')
                self.keyboard.release('a')
            self._input_pause(0.05)

            # 4. Paste
            with self.keyboard.pressed(mod_key):
                self.keyboard.press('v')
                self.keyboard.release('v')
            self._input_pause(0.08)

        finally:
            # 5. Restore clipboard
            try:
                if orig_clipboard:
                    pyperclip.copy(orig_clipboard)
            except Exception:
                pass

    def click(self, target: List[int]) -> None:
        """Moves mouse to target and performs a single physical click."""
        if self._cancelled:
            return

        if getattr(self, "browser_session", None):
            self.browser_session.click(target)
            return
        screen_x, screen_y = self._to_screen_coords(target)
        print(f"[Executor] Mouse CLICK at screen ({screen_x}, {screen_y}) [ROI: {target}, Offset: ({self.roi_offset_x}, {self.roi_offset_y})]")
        self._move_pointer(screen_x, screen_y)
        self._input_pause(0.06)
        if not self._cancelled:
            self.mouse.click(Button.left, 1)
            self._input_pause(0.05)

    def drag(self, start: List[int], end: List[int], steps: int = 6) -> None:
        """
        Performs 6-step physical smooth drag from start to end with guaranteed release.
        """
        if self._cancelled:
            return

        if getattr(self, "browser_session", None):
            browser = self.browser_session
            browser.ensure_homework()
            try:
                browser.mouse("mousePressed", start, button="left", clickCount=1)
                for step in range(1, steps+1):
                    if self._cancelled:
                        break
                    point = [start[i] + (end[i]-start[i])*step/steps for i in (0, 1)]
                    browser.mouse("mouseMoved", point, buttons=1)
            finally:
                browser.release_inputs()
            return
        start_x, start_y = self._to_screen_coords(start)
        end_x, end_y = self._to_screen_coords(end)
        print(f"[Executor] Smooth DRAG from ({start_x}, {start_y}) to ({end_x}, {end_y}) [Offset: ({self.roi_offset_x}, {self.roi_offset_y})] ({steps} steps)")

        try:
            # Move to start and press
            self.mouse.position = (start_x, start_y)
            time.sleep(0.05)
            self.mouse.press(Button.left)
            time.sleep(0.05)

            # Smooth intermediate steps
            for step in range(1, steps + 1):
                if self._cancelled:
                    break
                ratio = step / float(steps)
                curr_x = int(start_x + (end_x - start_x) * ratio)
                curr_y = int(start_y + (end_y - start_y) * ratio)
                self.mouse.position = (curr_x, curr_y)
                time.sleep(0.02)

            time.sleep(0.05)
        finally:
            # Forceful release in finally block
            self.mouse.release(Button.left)
            time.sleep(0.05)

    def scroll(self, direction: str, steps: int) -> None:
        """
        Simulates mouse wheel scrolling.
        """
        if self._cancelled:
            return

        if getattr(self, "browser_session", None):
            browser = self.browser_session
            browser.ensure_homework()
            browser.mouse("mouseWheel", [browser.image_size[0]//2, browser.image_size[1]//2], deltaX=0, deltaY=(1 if direction == "down" else -1)*abs(steps)*120)
            return
        dy = -abs(steps) if direction == "down" else abs(steps)
        self.mouse.scroll(0, dy)
        time.sleep(0.1)

    def wait(self, wait_ms: int) -> None:
        """Waits for specified duration, checking cancellation periodically."""
        elapsed = 0
        step_wait = 50
        while elapsed < wait_ms:
            if self._cancelled:
                break
            time.sleep(step_wait / 1000.0)
            elapsed += step_wait
