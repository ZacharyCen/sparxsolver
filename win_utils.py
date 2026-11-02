"""
win_utils.py - Windows Window Enumeration & Target Window Tracker
Provides DPI-aware window listing, bounds tracking, and window activation
for seamless homework website window selection without manual ROI dragging.
"""

import sys
from typing import List, Dict, Optional, Any

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long)
        ]


def get_candidate_windows() -> List[Dict[str, Any]]:
    """
    Enumerates visible top-level application windows on Windows desktop.
    Filters out system utilities, overlays, and the assistant itself.
    """
    if sys.platform != "win32":
        # Fallback for Linux / WSL dev environment
        return [
            {"hwnd": 1001, "title": "Sparx Maths - Google Chrome (Mock)", "left": 0, "top": 0, "width": 1920, "height": 1080}
        ]

    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi

    # Per-Monitor DPI Aware v2 (-4)
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        pass

    results = []
    IGNORE_TITLES = {
        "Program Manager", "Windows Input Experience", "Settings",
        "NVIDIA GeForce Overlay", "Microsoft Text Input Application"
    }
    IGNORE_SUBSTRINGS = ["autoanswer", "windows terminal", "visual studio code", "powershell", "cmd.exe"]

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.strip()
        tl = title.lower()
        if ("@" in title and ":" in title) or not title or title in IGNORE_TITLES or any(s in tl for s in IGNORE_SUBSTRINGS):
            return True

        # Query true visual bounds via DWM to omit drop-shadow borders
        rect = RECT()
        hr = dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            ctypes.c_uint(9), # DWMWA_EXTENDED_FRAME_BOUNDS
            ctypes.byref(rect),
            ctypes.sizeof(rect)
        )
        if hr != 0:
            user32.GetWindowRect(hwnd, ctypes.byref(rect))

        w = rect.right - rect.left
        h = rect.bottom - rect.top

        # Filter out tray icons and tiny utility windows
        if w >= 250 and h >= 200:
            results.append({
                "hwnd": int(hwnd),
                "title": title,
                "left": int(rect.left),
                "top": int(rect.top),
                "width": int(w),
                "height": int(h)
            })
        return True

    cb = WNDENUMPROC(callback)
    user32.EnumWindows(cb, 0)
    return results


def get_window_bounds(hwnd: int) -> Optional[Dict[str, int]]:
    """Retrieves current visual bounds (left, top, width, height) for an HWND."""
    if sys.platform != "win32":
        return {"left": 0, "top": 0, "width": 1920, "height": 1080}

    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi

    if not user32.IsWindow(wintypes.HWND(hwnd)):
        return None

    rect = RECT()
    hr = dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        ctypes.c_uint(9),
        ctypes.byref(rect),
        ctypes.sizeof(rect)
    )
    if hr != 0:
        if not user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None

    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None

    return {
        "left": int(rect.left),
        "top": int(rect.top),
        "width": int(w),
        "height": int(h)
    }


def activate_target_window(hwnd: int) -> bool:
    """Un-minimizes and foregrounds the target window."""
    if sys.platform != "win32":
        return True

    user32 = ctypes.windll.user32
    try:
        # Restore if minimized
        if user32.IsIconic(wintypes.HWND(hwnd)):
            user32.ShowWindow(wintypes.HWND(hwnd), 9) # SW_RESTORE

        if user32.GetForegroundWindow() == hwnd:
            return True

        # Tap Alt only when a foreground change is actually required.
        VK_MENU = 0x12
        KEYEVENTF_KEYUP = 0x0002
        user32.keybd_event(VK_MENU, 0, 0, 0)
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
        user32.SetForegroundWindow(wintypes.HWND(hwnd))
        return True
    except Exception:
        return False
