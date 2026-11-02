"""
main.py - Application Entry Point & Floating Control Panel
Implements:
- Windows Per-Monitor DPI Aware v2 initialization prior to UI creation
- Modern CustomTkinter floating control bar
- Mode switching (Key-free codex_dev vs External API remote)
- Interactive screen region selection overlay
- Global Esc emergency stop listener via pynput daemon thread
"""

import datetime
import sys
import threading
import tkinter as tk
from typing import Optional, Dict, Any, List

import customtkinter as ctk
import pynput.keyboard
import pyperclip

from homework_store import HomeworkStore
from bookwork import BookworkSession
from solver_ui import SolverUI
from observer import init_dpi_awareness, Observer
from planner import CodexCliPlanner, RemoteApiPlanner, BasePlanner
from executor import ActionExecutor
from controller import AutoAnswerController
from win_utils import get_candidate_windows, get_window_bounds, activate_target_window


class StdoutRedirector:
    """Redirects stdout and stderr to the Tkinter console log textbox in a thread-safe manner."""
    def __init__(self, log_callback, original_stream):
        self.log_callback = log_callback
        self.original_stream = original_stream

    def write(self, text):
        if self.original_stream:
            try:
                self.original_stream.write(text)
                self.original_stream.flush()
            except Exception:
                pass
        stripped = text.strip()
        if stripped:
            self.log_callback(stripped)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass


class RegionSelectorDialog:
    """
    High-contrast fullscreen overlay for framing the question ROI.
    Supports multi-monitor virtual desktop, high-visibility neon outline + dimension badge,
    and guaranteed cleanup without lingering alpha compositing artifacts.
    """

    def __init__(self, master, on_selected_callback):
        self.master = master
        self.callback = on_selected_callback
        self.start_x = 0
        self.start_y = 0
        self.current_rect_outer = None
        self.current_rect_inner = None
        self.dim_bg = None
        self.dim_text = None
        self.instruction_text = None
        self._closed = False

        # Query virtual screen bounds
        import mss
        with mss.MSS() as sct:
            v_mon = sct.monitors[0]
            self.v_left = v_mon["left"]
            self.v_top = v_mon["top"]
            self.v_width = v_mon["width"]
            self.v_height = v_mon["height"]

        self.top = tk.Toplevel(master)
        self.top.overrideredirect(True)
        self.top.geometry(f"{self.v_width}x{self.v_height}+{self.v_left}+{self.v_top}")
        self.top.attributes("-topmost", True)
        self.top.attributes("-alpha", 0.22)
        self.top.config(cursor="cross")

        self.canvas = tk.Canvas(
            self.top,
            cursor="cross",
            bg="#0f172a",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        center_x = self.v_width // 2
        self.instruction_text = self.canvas.create_text(
            center_x, 40,
            text="Drag mouse to select Question ROI | Right-Click or ESC to Cancel",
            fill="#ffffff",
            font=("Arial", 14, "bold")
        )

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<ButtonPress-3>", lambda e: self.close())
        self.top.bind("<Escape>", lambda e: self.close())

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect_outer = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#000000", width=4
        )
        self.current_rect_inner = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#00ffff", width=2
        )
        self.dim_bg = self.canvas.create_rectangle(
            self.start_x, max(10, self.start_y - 24), self.start_x + 120, max(32, self.start_y),
            fill="#000000", outline="#00ffff", width=1
        )
        self.dim_text = self.canvas.create_text(
            self.start_x + 60, max(21, self.start_y - 12),
            text="0 x 0",
            fill="#00ffff",
            font=("Arial", 10, "bold")
        )

    def on_move_press(self, event):
        cur_x, cur_y = event.x, event.y
        if self.current_rect_outer:
            self.canvas.coords(self.current_rect_outer, self.start_x, self.start_y, cur_x, cur_y)
        if self.current_rect_inner:
            self.canvas.coords(self.current_rect_inner, self.start_x, self.start_y, cur_x, cur_y)

        w = abs(cur_x - self.start_x)
        h = abs(cur_y - self.start_y)
        bx = max(cur_x, self.start_x)
        by = min(cur_y, self.start_y)
        badge_y = max(10, by - 24)
        if self.dim_bg and self.dim_text:
            self.canvas.coords(self.dim_bg, max(0, bx - 130), badge_y, bx, badge_y + 22)
            self.canvas.coords(self.dim_text, max(65, bx - 65), badge_y + 11)
            self.canvas.itemconfig(self.dim_text, text=f"{w} x {h}")

    def on_button_release(self, event):
        end_x, end_y = event.x, event.y
        rel_left = min(self.start_x, end_x)
        rel_top = min(self.start_y, end_y)
        width = abs(end_x - self.start_x)
        height = abs(end_y - self.start_y)

        abs_left = self.v_left + rel_left
        abs_top = self.v_top + rel_top

        self.close()

        if width > 15 and height > 15:
            self.callback({"left": abs_left, "top": abs_top, "width": width, "height": height})

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.top.withdraw()
            self.top.update_idletasks()
            self.top.destroy()
            self.master.update()
        except Exception:
            pass


class WindowSelectorDialog(ctk.CTkToplevel):
    """
    Dedicated modal window picker that remains visible until explicitly selected or dismissed.
    Provides instant search/filtering, high-contrast visual cards, and automatic prioritization
    of Sparx Maths and web browser windows. Immune to OS popup menu auto-dismissal.
    """
    def __init__(self, master, on_selected_callback, on_roi_callback):
        super().__init__(master)
        self.master = master
        self.callback = on_selected_callback
        self.roi_callback = on_roi_callback

        self.title("Select Target Homework Window")
        self.geometry("560x520")
        self.minsize(500, 420)
        self.transient(master)
        self.attributes("-topmost", True)

        try:
            m_x = self.master.winfo_x()
            m_y = self.master.winfo_y()
            m_w = self.master.winfo_width()
            s_w = self.winfo_screenwidth()
            if m_x + m_w + 580 <= s_w:
                pos_x = m_x + m_w + 12
            elif m_x - 580 >= 0:
                pos_x = m_x - 580
            else:
                pos_x = max(10, m_x + 30)
            self.geometry(f"+{pos_x}+{max(10, m_y)}")
        except Exception:
            pass

        self._windows_cache = []
        self._build_dialog_ui()
        self.refresh_windows()

        self.bind("<Escape>", lambda e: self.destroy())
        self.lift()
        self.focus_force()

    def _build_dialog_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 6))

        title = ctk.CTkLabel(
            header,
            text="🪟 Select Homework Window",
            font=("Arial", 15, "bold"),
            text_color="#38bdf8"
        )
        title.pack(side="left")

        close_btn = ctk.CTkButton(
            header,
            text="✕ Close",
            width=65,
            height=26,
            font=("Arial", 11),
            fg_color="#334155",
            hover_color="#475569",
            command=self.destroy
        )
        close_btn.pack(side="right")

        # Instruction subtitle
        sub = ctk.CTkLabel(
            self,
            text="Click the window below where Sparx Maths or your homework is running:",
            font=("Arial", 11),
            text_color="#94a3b8"
        )
        sub.pack(anchor="w", padx=16, pady=(0, 8))

        # Filter and Refresh Row
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=16, pady=(0, 8))

        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="🔍 Filter windows (e.g. Sparx, Chrome, Edge)...",
            font=("Arial", 11)
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.search_entry.bind("<KeyRelease>", self._apply_filter)

        ref_btn = ctk.CTkButton(
            search_frame,
            text="🔄 Refresh",
            width=80,
            height=28,
            font=("Arial", 11, "bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            border_width=1,
            border_color="#475569",
            command=self.refresh_windows
        )
        ref_btn.pack(side="right")

        # Scrollable list of windows
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="#0a0f1d",
            corner_radius=8,
            border_width=1,
            border_color="#1e293b"
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        # Footer with fallback ROI drag button
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(0, 12))

        roi_fallback = ctk.CTkButton(
            footer,
            text="🎯 Drag Custom Screen Region (ROI) Instead",
            font=("Arial", 11, "bold"),
            fg_color="#1e293b",
            hover_color="#334155",
            border_width=1,
            border_color="#0284c7",
            height=32,
            command=self._on_roi_clicked
        )
        roi_fallback.pack(side="left", fill="x", expand=True)

    def refresh_windows(self):
        self._windows_cache = get_candidate_windows()
        self._apply_filter()

    def _apply_filter(self, event=None):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        query = self.search_entry.get().strip().lower()

        def sort_key(w):
            tl = w["title"].lower()
            is_browser = any(b in tl for b in ["chrome", "edge", "firefox", "brave", "opera"])
            is_sparx = "sparx" in tl
            if is_sparx and is_browser:
                return 0
            if is_browser:
                return 1
            if is_sparx:
                return 2
            return 3

        sorted_windows = sorted(self._windows_cache, key=sort_key)
        matched_count = 0

        for w in sorted_windows:
            title = w["title"]
            if query and query not in title.lower():
                continue

            matched_count += 1
            t_lower = title.lower()
            is_sparx = "sparx" in t_lower
            is_browser = any(b in t_lower for b in ["chrome", "edge", "firefox", "brave", "opera"])

            card_border = "#38bdf8" if is_sparx else ("#3b82f6" if is_browser else "#334155")
            card_fg = "#1e293b" if is_sparx else "#131c2e"
            hover_fg = "#0369a1" if is_sparx else "#1e293b"

            card = ctk.CTkFrame(
                self.scroll_frame,
                fg_color=card_fg,
                border_width=2 if is_sparx else 1,
                border_color=card_border,
                corner_radius=6
            )
            card.pack(fill="x", pady=4, padx=4)

            icon_char = "⭐ 🌐" if is_sparx else ("🌐" if is_browser else "💻")
            header_text = f"{icon_char}  {title[:48]}" + ("..." if len(title) > 48 else "")

            lbl_title = ctk.CTkLabel(
                card,
                text=header_text,
                font=("Arial", 11, "bold"),
                text_color="#f8fafc" if not is_sparx else "#38bdf8",
                anchor="w"
            )
            lbl_title.pack(anchor="w", padx=10, pady=(6, 2))

            sub_text = f"Bounds: {w['width']}x{w['height']} at ({w['left']}, {w['top']})  |  HWND: {w['hwnd']}"
            if is_sparx:
                sub_text = "★ Sparx Maths Detected!  |  " + sub_text

            lbl_sub = ctk.CTkLabel(
                card,
                text=sub_text,
                font=("Arial", 9),
                text_color="#94a3b8" if not is_sparx else "#4ade80",
                anchor="w"
            )
            lbl_sub.pack(anchor="w", padx=10, pady=(0, 6))

            def make_handler(target_w=w):
                return lambda event=None: self._select_window(target_w)

            card.bind("<Button-1>", make_handler())
            lbl_title.bind("<Button-1>", make_handler())
            lbl_sub.bind("<Button-1>", make_handler())

            def on_enter(e, c=card, h=hover_fg):
                c.configure(fg_color=h)
            def on_leave(e, c=card, orig=card_fg):
                c.configure(fg_color=orig)

            card.bind("<Enter>", on_enter)
            card.bind("<Leave>", on_leave)
            lbl_title.bind("<Enter>", on_enter)
            lbl_sub.bind("<Enter>", on_enter)

        if matched_count == 0:
            empty_lbl = ctk.CTkLabel(
                self.scroll_frame,
                text="No matching windows found.\nTry clicking '🔄 Refresh' or launch your homework browser.",
                font=("Arial", 11),
                text_color="#94a3b8",
                justify="center"
            )
            empty_lbl.pack(pady=30)

    def _select_window(self, window_dict):
        self.destroy()
        self.callback(window_dict)

    def _on_roi_clicked(self):
        self.destroy()
        self.roi_callback()


class AutoAnswerApp(SolverUI, ctk.CTk):
    """Floating control panel for AutoAnswer Assistant with live activity log HUD."""

    def __init__(self):
        super().__init__()

        self.title("AutoAnswer Assistant")
        self.geometry("920x740")
        self.minsize(840, 660)
        self.attributes("-topmost", True)
        self.resizable(True, True)

        # State flags
        self._is_running = False
        self._is_hidden = False
        self._is_topmost = True

        # Core Components
        self.observer = Observer()
        self.executor = ActionExecutor()
        self.homework_store = None
        self.browser_session = None
        self._opening_browser = False
        self.planner: Optional[BasePlanner] = None
        self.controller: Optional[AutoAnswerController] = None
        self.selected_region: Optional[Dict[str, int]] = None
        self.selected_hwnd: Optional[int] = None
        self.selected_window_title: str = ""
        self._detected_windows: list = []

        # Build UI
        self._build_ui()

        # Initial Window Auto-Detection
        self._quick_refresh_windows()

        # Start Global Hotkey Listener (Esc to Stop, F8 to Hide/Unhide)
        self._start_global_listeners()

        # Hover translucency bindings
        # Hovering does not change window opacity or steal focus.

        # Redirect standard streams to in-app Console Log
        self._setup_logging()

    def destroy(self):
        """Finish queued local exports and release input when the panel closes."""
        if getattr(self, "_is_running", False):
            if getattr(self, "controller", None):
                try: self.controller.stop("Application closed")
                except Exception: pass
            if getattr(self, "executor", None):
                try: self.executor.cancel()
                except Exception: pass
        exporter = getattr(self, "_workings_exporter", None)
        if exporter:
            if getattr(self, "homework_store", None):
                self.homework_store.on_change = None
            exporter.close(wait=False)
        super().destroy()

    def _build_ui(self):
        self.build_workspace()

    def _setup_logging(self):
        """Redirects stdout and stderr to the in-app console textbox."""
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StdoutRedirector(self.log, self._orig_stdout)
        sys.stderr = StdoutRedirector(self.log, self._orig_stderr)
        self.log("AutoAnswer Assistant initialized. Front layer active.")

    def log(self, message: str):
        """Thread-safe logging to the UI console textbox."""
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{now_str}] {message}"
        self.after(0, self._append_log_text, formatted)

    def _append_log_text(self, line: str):
        if hasattr(self, "log_box"):
            try:
                self.log_box.insert("end", line + "\n")
                self.log_box.see("end")
                # Prune old logs if exceeding 600 lines
                line_count = int(float(self.log_box.index("end-1c").split('.')[0]))
                if line_count > 600:
                    self.log_box.delete("1.0", "50.0")
            except Exception:
                pass

    def _clear_log(self):
        self.log_box.delete("1.0", "end")

    def _copy_log(self):
        content = self.log_box.get("1.0", "end-1c")
        pyperclip.copy(content)
        self.log("Copied all activity log lines to clipboard.")

    def hide_window(self):
        """Hides the control panel window from screen."""
        self._is_hidden = True
        self.withdraw()
        self.log("Window hidden. Press F8 on keyboard to restore.")

    def show_window(self):
        """Restores window to the front layer."""
        self._is_hidden = False
        self.deiconify()
        self.lift()
        self.attributes("-topmost", True)
        self.focus_force()
        self.log("Window restored to front layer.")

    def toggle_visibility(self):
        """Toggles between hidden and visible."""
        if self._is_hidden:
            self.show_window()
        else:
            self.hide_window()

    def _on_mouse_enter(self, event):
        """While running, hover increases visibility to 95%."""
        if self._is_running:
            self.attributes("-alpha", 0.95)

    def _on_mouse_leave(self, event):
        """While running, leaving cursor returns to slightly transparent (65%)."""
        if self._is_running:
            self.attributes("-alpha", 0.65)

    def _choose_existing_window(self):
        if self._is_running or self._opening_browser:
            self.log("Stop the run and wait for browser setup before changing the target.")
            return
        self.browser_var.set(False)
        self.executor.browser_session = self.observer.browser_session = None
        self._apply_preferences()
        self._open_window_selector()

    def _place_solver_beside_homework(self, region):
        """Use a spare display for the pinned panel, leaving desktop input unobstructed."""
        if sys.platform != "win32" or not self.winfo_viewable():
            return
        left, top = self.winfo_rootx(), self.winfo_rooty()
        if not (left < region["left"]+region["width"] and left+self.winfo_width() > region["left"]
                and top < region["top"]+region["height"] and top+self.winfo_height() > region["top"]):
            return
        for monitor in self.observer._sct.monitors[1:]:
            overlaps = (monitor["left"] < region["left"]+region["width"] and monitor["left"]+monitor["width"] > region["left"]
                        and monitor["top"] < region["top"]+region["height"] and monitor["top"]+monitor["height"] > region["top"])
            if not overlaps and monitor["width"] >= self.winfo_width() and monitor["height"] >= self.winfo_height():
                import ctypes
                from ctypes import wintypes
                user32 = ctypes.windll.user32
                user32.GetAncestor.restype = wintypes.HWND
                hwnd = user32.GetAncestor(wintypes.HWND(self.winfo_id()), 2)
                user32.SetWindowPos(wintypes.HWND(hwnd), None, monitor["left"]+30, monitor["top"]+30, 0, 0, 0x15)
                self.log("Moved the solver to the other display so it does not cover homework.")
                break

    def _open_window_selector(self):
        """Opens the persistent modal dialog to select target homework window."""
        WindowSelectorDialog(self, self._on_window_selected_dict, self._select_region)

    def _quick_refresh_windows(self):
        """Scans open application windows and auto-selects Sparx Maths in Chrome/Browser."""
        self._detected_windows = get_candidate_windows()
        sparx_browser_w = None
        sparx_any_w = None
        browser_w = None

        for w in self._detected_windows:
            tl = w["title"].lower()
            is_browser = any(b in tl for b in ["chrome", "edge", "firefox", "brave", "opera"])
            is_sparx = "sparx" in tl
            if is_sparx and is_browser:
                sparx_browser_w = w
                break
            elif is_sparx and not sparx_any_w:
                sparx_any_w = w
            elif is_browser and not browser_w:
                browser_w = w

        chosen = sparx_browser_w or sparx_any_w or browser_w or (self._detected_windows[0] if self._detected_windows else None)
        if chosen:
            self._on_window_selected_dict(chosen)
        else:
            self.window_select_btn.configure(
                text="🔍 Click to Select Homework Window...  ▼",
                border_color="#38bdf8"
            )
            self.log("Window scan refreshed: No target window auto-matched.")

    def _on_window_selected_dict(self, w: Dict[str, Any]):
        """Callback when a window is chosen from WindowSelectorDialog."""
        self.selected_hwnd = w["hwnd"]
        self.selected_window_title = w["title"]
        self.selected_region = {
            "left": w["left"],
            "top": w["top"],
            "width": w["width"],
            "height": w["height"]
        }
        self.executor.set_roi_offset(w["left"], w["top"])
        self.window_select_btn.configure(border_color="#22c55e")
        self.status_label.configure(
            text="Open your homework browser above, then press Start." if self.browser_var.get() else "Ready to solve in the selected window.",
            text_color="#4ade80"
        )
        self._refresh_browser_controls()
        self.log(f"Target Window selected: '{w['title']}' (HWND: {w['hwnd']}) at ({w['left']}, {w['top']}) {w['width']}x{w['height']}")
        if not self.browser_var.get():
            self.after(100, self._place_solver_beside_homework, self.selected_region)

    def _select_region(self):
        RegionSelectorDialog(self, self._on_region_selected)

    def _on_region_selected(self, region: Dict[str, int]):
        self.selected_hwnd = None  # Using custom manual drag region
        self.selected_window_title = ""
        self.selected_region = region
        self.executor.set_roi_offset(region["left"], region["top"])
        self.window_select_btn.configure(
            text=f"🎯 Custom ROI: ({region['left']},{region['top']}) {region['width']}x{region['height']}  ▼",
            border_color="#eab308"
        )
        self.status_label.configure(
            text="Ready to solve the selected screen area.",
            text_color="lightgreen"
        )
        self._refresh_browser_controls()
        self.log(f"Custom ROI selected: ({region['left']}, {region['top']}) {region['width']}x{region['height']}")

    def _start_global_listeners(self):
        def on_press(key):
            try:
                if key == pynput.keyboard.Key.esc and self._is_running:
                    self.log("[Global Hotkey] Esc pressed -> Emergency Stop!")
                    self.after(0, self._on_stop)
                elif key == pynput.keyboard.Key.f8:
                    self.after(0, self.toggle_visibility)
            except Exception:
                pass

        self.hotkey_listener = pynput.keyboard.Listener(on_press=on_press)
        self.hotkey_listener.daemon = True
        self.hotkey_listener.start()

    def _set_waiting_time(self, value):
        self.waiting_time = round(float(value), 1)
        self._apply_preferences()

    def _apply_preferences(self):
        if self._is_running and self.browser_var.get() != bool(getattr(self.executor, "browser_session", None)):
            self.browser_var.set(bool(self.executor.browser_session))
            self.log("Stop the run before switching browser mode.")
        if hasattr(self, "option_switches") and len(self.option_switches) > 2:
            self.option_switches[2].configure(state="disabled" if self.browser_var.get() else "normal")
        if hasattr(self, "topmost_badge"):
            self.topmost_badge.configure(text="Separate browser" if self.browser_var.get() else "Existing window")
        if hasattr(self, "browser_mode_switch"):
            self._refresh_browser_controls()
        enabled = self.delay_var.get()
        self.wait_slider.configure(state="normal" if enabled else "disabled")
        self.wait_label.configure(text=f"Waiting Time: {self.waiting_time:.1f} s" if enabled else "Waiting Time: off")
        self.executor.input_delays = enabled
        self.executor.smooth_pointer = self.pointer_var.get()
        self.executor.keyboard_entry_enabled = self.keyboard_var.get()
        if self.planner:
            self.planner.keyboard_entry_enabled = self.keyboard_var.get()
            self.planner.generate_workings = self.workings_var.get()
        if self.controller:
            self.controller.waiting_time = self.waiting_time if enabled else 0
            self.controller.fast_submit = self.submit_var.get()
            self.controller.require_bookwork = self.bookwork_var.get()
            if self.controller.bookwork:
                self.controller.bookwork.collect_workings = self.workings_var.get()
        if hasattr(self, "human_var"):
            self.human_var.set("On" if self.delay_var.get() or (self.pointer_var.get() and not self.browser_var.get()) else "Off")
        if hasattr(self, "_workings_store"):
            self._watch_workings_store()
        self._save_workspace()

    def _play_finish_sound(self, success):
        try:
            if sys.platform == "win32":
                import winsound
                name = "SystemAsterisk" if success else "SystemExclamation"
                winsound.PlaySound(name, winsound.SND_ALIAS | winsound.SND_ASYNC)
            else:
                self.bell()
            self.log("Completion sound played." if success else "Stopped-run alert played.")
        except Exception as error:
            self.log(f"Sound unavailable: {error}")

    def _open_background_browser(self):
        if self._is_running or self._opening_browser:
            self.log("Stop the run before opening or changing browser sessions.")
            return
        self._opening_browser = True
        self.browser_open_btn.configure(state="disabled", text="Opening browser...")
        def opened(session=None, error=None):
            self._opening_browser = False
            self.browser_open_btn.configure(state="normal", text="Open separate browser · sign in once")
            if error:
                self.log(f"Browser error: {error}")
                self.status_label.configure(text=str(error), text_color="orange", wraplength=520)
                return
            self.browser_session = session
            self.browser_var.set(True)
            self._apply_preferences()
            self.attributes("-topmost", True)
            self.topmost_badge.configure(text="Separate browser")
            self.status_label.configure(text="Open homework in the separate browser (sign in if needed), then Start.", text_color="#4ade80", wraplength=520)
            self.log("Separate browser opened. Sign into Sparx once; its session is saved separately. Desktop mouse and keyboard stay free while it runs.")
        def launch():
            try:
                from browser_session import BrowserSession
                session = BrowserSession.open()
                session.open_homework()
                self.after(0, lambda: opened(session=session))
            except Exception as error:
                self.after(0, lambda message=str(error): opened(error=message))
        threading.Thread(target=launch, daemon=True).start()

    def _ensure_homework_record(self):
        """Never start a run that silently drops requested answer/workings records."""
        if self.homework_store:
            return True
        self._new_homework_record()
        if not self.homework_store:
            self.status_label.configure(text="Create or open a homework record before starting, so answers and workings can be saved.", text_color="orange")
            return False
        return True

    def _on_start(self):
        if self._is_running:
            return
        max_q = None  # Discover the complete visible set instead of asking for a count.
        active_region, active_hwnd = self.selected_region, self.selected_hwnd
        if self.browser_var.get():
            if not self.browser_session:
                if not self._opening_browser:
                    self._open_background_browser()
                self.status_label.configure(text="Opening your browser. Choose homework there, then press Start.", text_color="orange", wraplength=520)
                return
            try:
                self.browser_session.select_homework_tab()
                frame = self.browser_session.capture()
                active_region = {"left": 0, "top": 0, "width": frame.shape[1], "height": frame.shape[0]}
                active_hwnd = None
                self.executor.browser_session = self.observer.browser_session = self.browser_session
            except Exception as error:
                self.status_label.configure(text=f"Browser: {error}", text_color="orange", wraplength=520)
                return
        else:
            self.executor.browser_session = self.observer.browser_session = None
        if not self.browser_var.get() and not self.selected_hwnd and not self.selected_region:
            self.status_label.configure(text="Please select target homework window!", text_color="orange")
            self.log("Error: Please select target homework window before starting.")
            return

        if not self.browser_var.get() and active_region:
            if active_hwnd:
                active_region = get_window_bounds(active_hwnd) or active_region
            left, top = self.winfo_rootx(), self.winfo_rooty()
            right, bottom = left + self.winfo_width(), top + self.winfo_height()
            if (left < active_region["left"] + active_region["width"] and right > active_region["left"]
                    and top < active_region["top"] + active_region["height"] and bottom > active_region["top"]):
                self.status_label.configure(text="Move the solver outside the homework area, or select Separate browser on the Solver page.", text_color="orange", wraplength=520)
                return

        if not self._ensure_homework_record():
            return

        # Initialize Planner according to chosen mode
        mode = self.mode_var.get()
        self.log(f"Initializing planner in '{mode}' mode...")
        try:
            if mode == "codex_dev":
                self.planner = CodexCliPlanner()
            else:
                self.planner = RemoteApiPlanner(
                    base_url=self.base_url_entry.get().strip(),
                    model=self.model_entry.get().strip()
                )
                self.planner.usage_callback = self._usage_queue.put

            self.controller = AutoAnswerController(
                observer=self.observer,
                planner=self.planner,
                executor=self.executor,
                max_questions=max_q,
                region=active_region,
                target_hwnd=active_hwnd,
                waiting_time=self.waiting_time if self.delay_var.get() else 0,
                fast_submit=self.submit_var.get(),
                bookwork=BookworkSession(self.observer, self.executor, self.homework_store, collect_workings=self.workings_var.get()),
                require_bookwork=self.bookwork_var.get(),
                submit_with_enter=True
            )
        except Exception as e:
            self.log(f"Planner Initialization Error: {e}")
            self.status_label.configure(text=f"Error: {e}", text_color="red")
            self.start_btn.configure(state="normal")
            return

        self._apply_preferences()
        self._notified_controller = None
        self._is_running = True
        self._set_working_visuals(True)
        self.status_label.configure(text="Running • desktop input free" if self.browser_var.get() else "Running • Esc stops", text_color="#4ade80")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.log("Run started in separate browser; desktop mouse and keyboard remain free." if self.browser_var.get() else "Run started. Solver remains pinned and dimmed; Esc stops it.")

        # Run loop in background thread to avoid freezing UI
        self._loop_thread = threading.Thread(target=self._run_controller_thread, daemon=True)
        self._loop_thread.start()

    def _run_controller_thread(self):
        try:
            self.controller.start()
        except Exception as e:
            self.log(f"Controller thread error: {e}")
        finally:
            self.after(0, self._on_finish)

    def _on_finish(self):
        self._is_running = False
        if self._is_hidden and not getattr(self.executor, "browser_session", None):
            self.show_window()
        self._set_working_visuals(False)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        reason = self.controller.finish_reason if self.controller else "Stopped"
        self.status_label.configure(text=reason, text_color="gray", wraplength=520)
        if self.controller and getattr(self, "_notified_controller", None) is not self.controller:
            self._notified_controller = self.controller
            if self.sound_var.get() and reason != "Stopped by user":
                self._play_finish_sound(not reason.startswith('Needs Human intervention') and (getattr(self.controller, '_bookwork_finish_passed', False) or reason.startswith('Finished all') or (not getattr(self.controller, 'auto_complete', False) and self.controller.current_question_count >= self.controller.max_questions) or reason.startswith('Page state: completed')))
        self.log(f"Run ended: {reason}")
        self.log("Autonomous answering loop finished. Full opacity restored.")

    def _on_stop(self):
        if self.controller:
            self.controller.stop()
        self.executor.cancel()
        worker = getattr(self, "_loop_thread", None)
        if worker and worker.is_alive():
            self.status_label.configure(text="Stopping • waiting for the active request...", text_color="orange")
        else:
            self.after(0, self._on_finish)


def main():
    # Single instance guard on Windows to prevent conflicting duplicate processes
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        mutex = kernel32.CreateMutexW(None, False, "AutoAnswerAssistant_SingleInstance_Mutex")
        if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            user32 = ctypes.windll.user32
            existing_hwnd = user32.FindWindowW(None, "AutoAnswer Assistant")
            if existing_hwnd:
                user32.SetForegroundWindow(existing_hwnd)
            sys.exit(0)

    # Step 1: DPI awareness must be set prior to creating UI
    init_dpi_awareness()

    # Step 2: Launch Floating Panel
    app = AutoAnswerApp()
    if "--background-browser" in sys.argv:
        app.after(250, app._open_background_browser)
    app.mainloop()


if __name__ == "__main__":
    main()
