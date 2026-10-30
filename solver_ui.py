"""Sidebar workspace, appearance preferences, and reported token metering."""
import datetime
import json
import math
import os
from pathlib import Path
import queue
import time

import customtkinter as ctk
import tkinter as tk

from planner import load_api_config
from retry_policy import QuestionRetries


class SolverUI:
    ACCENTS = {"Mint": "#66e4bc", "Violet": "#a78bfa", "Blue": "#70b8ff", "Amber": "#f2c76b"}
    BG = ("#f3f5fa", "#0a101b")
    PANEL = ("#ffffff", "#111c2b")
    RAISED = ("#e9edf5", "#1b293e")
    TEXT = ("#172337", "#edf3fc")
    MUTED = ("#586a83", "#8b9eb9")

    def build_workspace(self):
        self._settings_path = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".config"))) / "AutoAnswer" / "settings.json"
        try:
            self._prefs = json.loads(self._settings_path.read_text())
            if not isinstance(self._prefs, dict):
                self._prefs = {}
        except (OSError, ValueError):
            self._prefs = {}
        self._ui_ready = False
        self._usage_queue = queue.SimpleQueue()
        self._token_totals = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "unreported": 0}
        self._usage_rows = []
        self._active_page = "Solver"
        self._anim_phase = 0
        self._nav_generation = 0
        self._accent_widgets = []
        self._choice_widgets = []
        prefs = self._prefs
        self.theme_var = ctk.StringVar(value=prefs.get("theme", "Midnight"))
        self.accent_var = ctk.StringVar(value=prefs.get("accent", "Mint"))
        self.animation_var = ctk.BooleanVar(value=prefs.get("animations", True))
        self.log_visible_var = ctk.BooleanVar(value=prefs.get("live_log", False))
        self.pin_var = ctk.BooleanVar(value=True)
        self.human_var = ctk.StringVar(value="Off")
        self.mode_var = ctk.StringVar(value=prefs.get("provider", "remote"))
        for name, default in (("sound", True), ("delay", False), ("pointer", False), ("keyboard", True), ("submit", True), ("browser", True)):
            setattr(self, name + "_var", ctk.BooleanVar(value=prefs.get(name, default)))
        self.waiting_time = max(0., min(5., float(prefs.get("waiting_time", .3))))
        self.configure(fg_color=self.BG)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        sidebar = ctk.CTkFrame(self, width=175, corner_radius=0, fg_color=self.PANEL)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        self._label(sidebar, "S / A", 30, bold=True).pack(anchor="w", padx=24, pady=(30, 0))
        self._label(sidebar, "SPARX ASSISTANT", 10, muted=True).pack(anchor="w", padx=24, pady=(0, 34))
        self._nav_buttons = {}
        for name, icon in (("Solver", "◈"), ("Configuration", "⚙"), ("Appearance", "◐"), ("Usage", "↗")):
            button = ctk.CTkButton(sidebar, text=f"{icon}   {name}", anchor="w", height=43,
                                  fg_color="transparent", text_color=self.MUTED, hover_color=self.RAISED,
                                  font=("Segoe UI", 12), command=lambda page=name: self._show_page(page))
            button.pack(fill="x", padx=12, pady=4)
            self._nav_buttons[name] = button
        self._nav_marker = ctk.CTkFrame(sidebar, width=3, height=25, corner_radius=2)
        self._nav_marker.place(x=0, y=149)
        footer = ctk.CTkFrame(sidebar, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=23, pady=24)
        self._label(footer, "DESKTOP / v3.3", 10, muted=True).pack(anchor="w")
        self.topmost_badge = self._label(footer, "Ready to focus", 10, muted=True)
        self.topmost_badge.pack(anchor="w", pady=(5, 0))
        self._content = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self._content.grid(row=0, column=1, sticky="nsew", padx=26, pady=22)
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_rowconfigure(0, weight=1)
        self._pages = {}
        for name in self._nav_buttons:
            page = ctk.CTkFrame(self._content, fg_color="transparent", corner_radius=0)
            page.grid(row=0, column=0, sticky="nsew")
            self._pages[name] = page
        self._build_solver()
        self._build_configuration()
        self._build_appearance()
        self._build_usage()
        self._ui_ready = True
        self._apply_preferences()
        self._apply_appearance(save=False)
        self._toggle_log(save=False)
        self._show_page("Solver", animate=False)
        self.after(100, self._workspace_tick)

    def _label(self, parent, text, size=12, bold=False, muted=False):
        return ctk.CTkLabel(parent, text=text, anchor="w", justify="left",
                           text_color=self.MUTED if muted else self.TEXT,
                           font=("Segoe UI", size, "bold" if bold else "normal"))

    def _heading(self, parent, title, subtitle):
        self._label(parent, title, 27, True).pack(anchor="w")
        self._label(parent, subtitle, 12, muted=True).pack(anchor="w", pady=(2, 20))

    def _card(self, parent, title=None):
        card = ctk.CTkFrame(parent, fg_color=self.PANEL, corner_radius=16)
        card.pack(fill="x", pady=(0, 12))
        if title:
            self._label(card, title.upper(), 10, True, muted=True).pack(anchor="w", padx=18, pady=(15, 7))
        return card

    def _build_solver(self):
        page = self._pages["Solver"]
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.pack(fill="x", pady=(0, 8))
        self._label(header, "Your focus. Your time.", 27, True).pack(side="left")
        self.hide_btn = ctk.CTkButton(header, text="Hide · F8", width=88, height=30, fg_color=self.RAISED,
                                     text_color=self.MUTED, command=self.hide_window)
        self.hide_btn.pack(side="right")
        self._label(page, "A quieter space to get things done.", 12, muted=True).pack(anchor="w", pady=(0, 16))
        hero = self._card(page)
        self._orb = tk.Canvas(hero, width=90, height=90, highlightthickness=0, bd=0)
        self._orb.pack(side="left", padx=22, pady=18)
        mode = ctk.CTkFrame(hero, fg_color="transparent")
        mode.pack(side="left", fill="both", expand=True, padx=(0, 20), pady=18)
        self._label(mode, "HUMAN MODE", 10, True, True).pack(anchor="w")
        self._label(mode, "Set your pace", 21, True).pack(anchor="w", pady=(0, 4))
        self.human_switch = ctk.CTkSegmentedButton(mode, values=["Off", "On"], variable=self.human_var,
                                                height=32, command=self._set_human_mode)
        self.human_switch.pack(anchor="w")
        self._label(mode, "Off is fast. On adds gentle pacing.", 11, muted=True).pack(anchor="w", pady=(5, 0))
        controls = ctk.CTkFrame(page, fg_color="transparent")
        controls.pack(fill="x", pady=(0, 14))
        controls.grid_columnconfigure(0, weight=3)
        controls.grid_columnconfigure(1, weight=1)
        self.start_btn = ctk.CTkButton(controls, text="Start solver  →", height=48, corner_radius=12,
                                      font=("Segoe UI", 14, "bold"), command=self._on_start)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 9))
        self.stop_btn = ctk.CTkButton(controls, text="Stop · Esc", height=48, corner_radius=12,
                                     fg_color=self.RAISED, hover_color="#8b3346", text_color=self.TEXT, command=self._on_stop)
        self.stop_btn.grid(row=0, column=1, sticky="ew")
        self.stop_btn.configure(state="disabled")
        self._accent_widgets.extend([self.start_btn])
        self.status_label = self._label(page, "Choose your homework in Configuration, then press Start.", 12, muted=True)
        self.status_label.configure(wraplength=520)
        self.status_label.pack(fill="x", pady=(0, 16))
        stats = ctk.CTkFrame(page, fg_color="transparent")
        stats.pack(fill="x")
        stats.grid_columnconfigure((0, 1), weight=1)
        self._home_input = self._token_card(stats, "INPUT TOKENS", 0)
        self._home_output = self._token_card(stats, "OUTPUT TOKENS", 1)
        self._meter_note = self._label(page, "THIS SESSION  ·  Waiting for the first API response", 10, muted=True)
        self._meter_note.pack(anchor="w", pady=(8, 12))
        self._log_panel = ctk.CTkFrame(page, fg_color=self.PANEL, corner_radius=14)
        bar = ctk.CTkFrame(self._log_panel, fg_color="transparent")
        bar.pack(fill="x", padx=14, pady=(8, 3))
        self._label(bar, "LIVE ACTIVITY", 10, True, True).pack(side="left")
        self.copy_btn = ctk.CTkButton(bar, text="Copy", width=55, height=24, fg_color=self.RAISED, text_color=self.TEXT, command=self._copy_log)
        self.copy_btn.pack(side="right")
        self.clear_btn = ctk.CTkButton(bar, text="Clear", width=55, height=24, fg_color=self.RAISED, text_color=self.TEXT, command=self._clear_log)
        self.clear_btn.pack(side="right", padx=5)
        self.log_box = ctk.CTkTextbox(self._log_panel, height=130, fg_color="transparent", text_color=self.MUTED,
                                      font=("Consolas", 10), activate_scrollbars=True)
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _token_card(self, parent, title, column):
        frame = ctk.CTkFrame(parent, fg_color=self.PANEL, corner_radius=14)
        frame.grid(row=0, column=column, sticky="ew", padx=(0, 6) if column == 0 else (6, 0))
        self._label(frame, title, 10, True, True).pack(anchor="w", padx=18, pady=(13, 0))
        label = self._label(frame, "0", 28, True)
        label.pack(anchor="w", padx=18, pady=(0, 13))
        return label

    def _build_configuration(self):
        page = self._pages["Configuration"]
        self._heading(page, "Configuration", "Run limits, pacing, and your homework connection.")
        scroll = ctk.CTkScrollableFrame(page, fg_color="transparent", corner_radius=0)
        scroll.pack(fill="both", expand=True)
        run = self._card(scroll, "Run setup")
        row = ctk.CTkFrame(run, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(4, 14))
        self._label(row, "New questions this run").pack(side="left")
        validation = (self.register(lambda value: value == "" or value.isdecimal()), "%P")
        self.limit_entry = ctk.CTkEntry(row, width=70, validate="key", validatecommand=validation)
        self.limit_entry.insert(0, str(self._prefs.get("new_questions", 3)))
        self.limit_entry.pack(side="right")
        self.limit_entry.bind("<FocusOut>", lambda event: self._save_workspace())
        display = self._card(scroll, "Visibility")
        self.log_switch = ctk.CTkSwitch(display, text="Show live activity log on Solver", variable=self.log_visible_var, command=self._toggle_log)
        self.log_switch.pack(anchor="w", padx=18, pady=16)

        review = self._card(scroll, "Questions needing help")
        self._label(review, "2 wrong → high reasoning · 4 wrong → save & skip", 11, muted=True).pack(anchor="w", padx=18, pady=10)
        ctk.CTkButton(review, text="Open human review logs", command=self._open_review_logs,
                      fg_color=self.RAISED, text_color=self.TEXT).pack(fill="x", padx=18, pady=(0, 14))

        options = self._card(scroll, "Automation")
        self.option_switches = []
        for text, name in (("Finish sound", "sound"), ("Input delays", "delay"), ("Smooth pointer", "pointer"),
                           ("Numeric keyboard entry", "keyboard"), ("Fast submit", "submit"), ("Use separate browser · not your existing Chrome", "browser")):
            switch = ctk.CTkSwitch(options, text=text, variable=getattr(self, name + "_var"), command=self._apply_preferences)
            switch.pack(anchor="w", padx=18, pady=8)
            self.option_switches.append(switch)
        self.wait_label = self._label(options, "Waiting Time: off", 11, muted=True)
        self.wait_label.pack(anchor="w", padx=18, pady=(8, 0))
        self.wait_slider = ctk.CTkSlider(options, from_=0, to=5, number_of_steps=50, command=self._set_waiting_time)
        self.wait_slider.set(self.waiting_time)
        self.wait_slider.pack(fill="x", padx=18, pady=(8, 20))
        target = self._card(scroll, "Homework browser")
        self.browser_open_btn = ctk.CTkButton(target, text="Open background browser · sign in once", height=38, command=self._open_background_browser)
        self.browser_open_btn.pack(fill="x", padx=18, pady=8)
        self._accent_widgets.append(self.browser_open_btn)
        self.window_select_btn = ctk.CTkButton(target, text="Use existing Chrome window", fg_color=self.RAISED, text_color=self.TEXT,
                                             command=self._choose_existing_window, height=36)
        self.window_select_btn.pack(fill="x", padx=18, pady=7)
        row = ctk.CTkFrame(target, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(0, 14))
        self.refresh_btn = ctk.CTkButton(row, text="Refresh windows", width=130, height=30, fg_color=self.RAISED, text_color=self.TEXT, command=self._quick_refresh_windows)
        self.refresh_btn.pack(side="left")
        self.roi_quick_btn = ctk.CTkButton(row, text="Select screen area", width=140, height=30, fg_color=self.RAISED, text_color=self.TEXT, command=self._select_region)
        self.roi_quick_btn.pack(side="left", padx=8)
        provider = self._card(scroll, "Model connection")
        self.provider_var = ctk.StringVar(value="Remote API" if self.mode_var.get() == "remote" else "Local CLI")
        self.mode_switch = ctk.CTkOptionMenu(provider, values=["Remote API", "Local CLI"], variable=self.provider_var, command=self._set_provider)
        self.mode_switch.pack(anchor="w", padx=18, pady=8)
        self._choice_widgets.append(self.mode_switch)
        config = load_api_config()
        self._label(provider, "Server URL", 11, muted=True).pack(anchor="w", padx=18)
        self.base_url_entry = ctk.CTkEntry(provider, height=34)
        self.base_url_entry.insert(0, self._prefs.get("base_url", config["base_url"]))
        self.base_url_entry.pack(fill="x", padx=18, pady=(2, 10))
        self._label(provider, "Model · low effort, high after 2 wrong answers", 11, muted=True).pack(anchor="w", padx=18)
        self.model_entry = ctk.CTkEntry(provider, height=34)
        self.model_entry.insert(0, self._prefs.get("model", config["model"]))
        self.model_entry.pack(fill="x", padx=18, pady=(2, 8))
        for entry in (self.base_url_entry, self.model_entry):
            entry.bind("<FocusOut>", lambda event: self._save_workspace())
        self._label(provider, "Credentials: OpenAI-KEY.txt · never displayed", 10, muted=True).pack(anchor="w", padx=18, pady=(0, 15))

    def _open_review_logs(self):
        """Open persistent question records and screenshots in the file manager."""
        folder = QuestionRetries().log_dir
        folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(folder))
        else:
            import subprocess
            import sys
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])

    def _build_appearance(self):
        page = self._pages["Appearance"]
        self._heading(page, "Appearance", "Choose your colors, contrast, and motion.")
        card = self._card(page, "Color & contrast")
        for label, variable, values in (("Theme", self.theme_var, ["Midnight", "Light", "System"]),
                                         ("Accent", self.accent_var, list(self.ACCENTS))):
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=14)
            self._label(row, label, 13).pack(side="left")
            menu = ctk.CTkOptionMenu(row, values=values, variable=variable, command=lambda value: self._apply_appearance())
            menu.pack(side="right")
            self._choice_widgets.append(menu)
        card = self._card(page, "Comfort")
        self.animation_switch = ctk.CTkSwitch(card, text="Animations · turn off for reduced motion", variable=self.animation_var, command=self._save_workspace)
        self.animation_switch.pack(anchor="w", padx=20, pady=18)
        self.pin_switch = ctk.CTkSwitch(card, text="Always on top · enabled", variable=self.pin_var, state="disabled")
        self.pin_switch.pack(anchor="w", padx=20, pady=(0, 18))
        self._label(page, "Your preferences are saved on this device.", 11, muted=True).pack(anchor="w", pady=10)

    def _build_usage(self):
        page = self._pages["Usage"]
        self._heading(page, "Token usage", "Input and output tokens reported by the API for this session.")
        stats = ctk.CTkFrame(page, fg_color="transparent")
        stats.pack(fill="x")
        stats.grid_columnconfigure((0, 1), weight=1)
        self._usage_input = self._token_card(stats, "INPUT TOKENS", 0)
        self._usage_output = self._token_card(stats, "OUTPUT TOKENS", 1)
        self._usage_note = self._label(page, "No responses yet.", 12, muted=True)
        self._usage_note.pack(anchor="w", pady=15)
        self.usage_history = ctk.CTkTextbox(page, fg_color=self.PANEL, text_color=self.MUTED, corner_radius=14, font=("Consolas", 11))
        self.usage_history.pack(fill="both", expand=True)
        self.usage_history.insert("end", "Your requests will appear here.\n")
        self.usage_history.configure(state="disabled")
        self.reset_usage_btn = ctk.CTkButton(page, text="Reset session counters", fg_color=self.RAISED, text_color=self.TEXT, command=self._reset_usage)
        self.reset_usage_btn.pack(anchor="w", pady=(12, 0))

    def _show_page(self, name, animate=True):
        self._active_page = name
        self._pages[name].tkraise()
        self._pages[name].focus_set()
        accent = self.ACCENTS.get(self.accent_var.get(), self.ACCENTS["Mint"])
        light_ink = "#" + "".join(f"{int(int(accent[i:i+2], 16)*.48):02x}" for i in (1, 3, 5))
        for page, button in self._nav_buttons.items():
            button.configure(fg_color=self.RAISED if page == name else "transparent", text_color=(light_ink, accent) if page == name else self.MUTED)
        self._nav_generation += 1
        generation = self._nav_generation
        target = 149 + list(self._nav_buttons).index(name)*51
        start = self._nav_marker.winfo_y()
        def step(frame=0):
            if generation != self._nav_generation:
                return
            fraction = 1 - (1-frame/12)**3
            self._nav_marker.place(y=round(start+(target-start)*fraction))
            if frame < 12:
                self.after(16, step, frame+1)
        if animate and self.animation_var.get():
            step()
        else:
            self._nav_marker.place(y=target)

    def _set_human_mode(self, value):
        on = value == "On"
        self.delay_var.set(on)
        self.pointer_var.set(on)
        if on and self.waiting_time == 0:
            self.waiting_time = .3
            self.wait_slider.set(.3)
        self._apply_preferences()

    def _set_provider(self, value):
        self.mode_var.set("remote" if value == "Remote API" else "codex_dev")
        self._save_workspace()

    def _toggle_log(self, save=True):
        if self.log_visible_var.get():
            self._log_panel.pack(fill="both", expand=True, pady=(0, 0))
        else:
            self._log_panel.pack_forget()
        if save:
            self._save_workspace()

    def _apply_appearance(self, save=True):
        theme = {"Midnight": "Dark", "Light": "Light", "System": "System"}.get(self.theme_var.get(), "Dark")
        ctk.set_appearance_mode(theme)
        accent = self.ACCENTS.get(self.accent_var.get(), self.ACCENTS["Mint"])
        text = "#08141a"  # Every offered accent is light enough to need dark button text.
        for button in self._accent_widgets:
            button.configure(fg_color=accent, hover_color=accent, text_color=text)
        for switch in self.option_switches + [self.log_switch, self.animation_switch, self.pin_switch]:
            switch.configure(progress_color=accent)
        shade = "#" + "".join(f"{int(int(accent[i:i+2], 16)*.35):02x}" for i in (1, 3, 5))
        self.human_switch.configure(selected_color=(accent, shade), selected_hover_color=(accent, shade),
                                    unselected_color=self.RAISED, unselected_hover_color=self.RAISED, text_color=self.TEXT)
        for menu in self._choice_widgets:
            menu.configure(fg_color=self.RAISED, button_color=self.RAISED, button_hover_color=self.RAISED,
                           text_color=self.TEXT, dropdown_fg_color=self.PANEL,
                           dropdown_text_color=self.TEXT, dropdown_hover_color=self.RAISED)
        self.wait_slider.configure(button_color=accent, button_hover_color=accent, progress_color=accent)
        self._nav_marker.configure(fg_color=accent)
        light_ink = "#" + "".join(f"{int(int(accent[i:i+2], 16)*.48):02x}" for i in (1, 3, 5))
        for label in (self._home_input, self._home_output, self._usage_input, self._usage_output):
            label.configure(text_color=(light_ink, accent))
        self.pin_var.set(True)
        self.attributes("-topmost", True)
        self._is_topmost = True
        self._set_working_visuals(self._is_running)
        self._show_page(self._active_page, animate=False)
        if save:
            self._save_workspace()

    def _set_working_visuals(self, working):
        """Dim the running workspace while leaving Stop and navigation usable."""
        self.attributes("-topmost", True)
        self.attributes("-alpha", .76 if working else 1.0)
        accent = self.ACCENTS.get(self.accent_var.get(), self.ACCENTS["Mint"])
        ink = "#" + "".join(f"{int(int(accent[i:i+2], 16)*.48):02x}" for i in (1, 3, 5))
        for button in self._accent_widgets:
            button.configure(fg_color="#64748b" if working else accent,
                             hover_color="#64748b" if working else accent,
                             text_color="#ffffff" if working else "#08141a")
        for label in (self._home_input, self._home_output, self._usage_input, self._usage_output):
            label.configure(text_color=self.MUTED if working else (ink, accent))

    def _keep_in_front(self):
        # Reassert z-order without activating the window or stealing keyboard focus.
        if self._is_hidden or not self.winfo_viewable() or self.grab_current():
            return
        self.attributes("-topmost", True)
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            user32.GetAncestor.restype = wintypes.HWND
            hwnd = user32.GetAncestor(wintypes.HWND(self.winfo_id()), 2)
            user32.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0, 0x0013)

    def _save_workspace(self):
        if not getattr(self, "_ui_ready", False):
            return
        settings = {name: getattr(self, name + "_var").get() for name in ("sound", "delay", "pointer", "keyboard", "submit", "browser")}
        settings.update(theme=self.theme_var.get(), accent=self.accent_var.get(), animations=self.animation_var.get(),
                        live_log=self.log_visible_var.get(), pin=self.pin_var.get(), waiting_time=self.waiting_time,
                        provider=self.mode_var.get(), new_questions=int(self.limit_entry.get() or 3),
                        base_url=self.base_url_entry.get().strip(), model=self.model_entry.get().strip())
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._settings_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(settings, indent=2))
            temporary.replace(self._settings_path)
        except OSError:
            pass

    def _record_usage(self, record):
        totals = self._token_totals
        totals["requests"] += 1
        if record.get("available"):
            totals["input_tokens"] += record["input_tokens"]
            totals["output_tokens"] += record["output_tokens"]
        else:
            totals["unreported"] += 1
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        row = f"{stamp}  {record['input_tokens']:,} in / {record['output_tokens']:,} out" if record.get("available") else f"{stamp}  Usage not reported"
        self._usage_rows.append(row)
        self._usage_rows = self._usage_rows[-100:]
        self._refresh_usage()

    def _refresh_usage(self):
        totals = self._token_totals
        for label in (self._home_input, self._usage_input):
            label.configure(text=f"{totals['input_tokens']:,}")
        for label in (self._home_output, self._usage_output):
            label.configure(text=f"{totals['output_tokens']:,}")
        count = totals["requests"]
        note = f"{count} {'response' if count == 1 else 'responses'}"
        if totals["unreported"]:
            note += f" · usage unavailable for {totals['unreported']}"
        self._meter_note.configure(text="THIS SESSION  ·  " + note)
        self._usage_note.configure(text=note)
        self.usage_history.configure(state="normal")
        self.usage_history.delete("1.0", "end")
        self.usage_history.insert("end", "\n".join(self._usage_rows) or "Your requests will appear here.")
        self.usage_history.configure(state="disabled")

    def _reset_usage(self):
        if self._is_running:
            return
        try:
            while True:
                self._usage_queue.get_nowait()
        except queue.Empty:
            pass
        self._token_totals = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "unreported": 0}
        self._usage_rows = []
        self._refresh_usage()

    def _workspace_tick(self):
        now = time.monotonic()
        if now - getattr(self, "_last_front_check", 0) > 1:
            self._last_front_check = now
            self._keep_in_front()
        try:
            while True:
                self._record_usage(self._usage_queue.get_nowait())
        except queue.Empty:
            pass
        self.reset_usage_btn.configure(state="disabled" if self._is_running else "normal")
        if self._active_page == "Solver":
            dark = ctk.get_appearance_mode() == "Dark"
            self._orb.configure(bg="#111c2b" if dark else "#ffffff")
            self._orb.delete("all")
            accent = self.ACCENTS.get(self.accent_var.get(), self.ACCENTS["Mint"])
            if self._is_running:
                accent = "#91a0b6"
            elif not dark:
                accent = "#" + "".join(f"{int(int(accent[i:i+2], 16)*.48):02x}" for i in (1, 3, 5))
            self._orb.create_oval(8, 8, 82, 82, outline="#2b405a" if dark else "#dce4ef", width=1)
            self._orb.create_oval(21, 21, 69, 69, outline=accent, width=2)
            self._orb.create_text(45, 45, text="S", fill=accent, font=("Segoe UI", 22, "bold"))
            phase = self._anim_phase if self.animation_var.get() else -.7
            for offset in (0, math.pi):
                x, y = 45+37*math.cos(phase+offset), 45+37*math.sin(phase+offset)
                self._orb.create_oval(x-3, y-3, x+3, y+3, fill=accent, outline="")
            self._anim_phase += .055 if self._is_running else .015
        self.after(40 if self.animation_var.get() and self._active_page == "Solver" else 180, self._workspace_tick)
