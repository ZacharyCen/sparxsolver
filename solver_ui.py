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
from homework_store import HomeworkStore
from workings_export import WorkingsExporter, build_report
from app_version import VERSION


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
        self._workings_queue = queue.SimpleQueue()
        self._workings_exporter = WorkingsExporter(self._workings_queue.put)
        self._workings_store = None
        self._workings_saving = False
        self._homework_error = ""
        saved_record = self._prefs.get("homework_record")
        if saved_record:
            try:
                self.homework_store = HomeworkStore(saved_record)
            except (OSError, ValueError, TypeError) as error:
                self._homework_error = f"Could not open homework record: {error}"
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
        for name, default in (("sound", True), ("delay", False), ("pointer", False), ("keyboard", True), ("submit", True), ("browser", True), ("bookwork", True), ("workings", True)):
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
        for name, icon in (("Solver", "◈"), ("Workings", "∑"), ("Configuration", "⚙"), ("Appearance", "◐"), ("Usage", "↗")):
            button = ctk.CTkButton(sidebar, text=f"{icon}   {name}", anchor="w", height=43,
                                  fg_color="transparent", text_color=self.MUTED, hover_color=self.RAISED,
                                  font=("Segoe UI", 12), command=lambda page=name: self._show_page(page))
            button.pack(fill="x", padx=12, pady=4)
            self._nav_buttons[name] = button
        self._nav_marker = ctk.CTkFrame(sidebar, width=3, height=25, corner_radius=2)
        self._nav_marker.place(x=0, y=149)
        footer = ctk.CTkFrame(sidebar, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=23, pady=24)
        self._label(footer, f"DESKTOP / v{VERSION}", 10, muted=True).pack(anchor="w")
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
        self._build_workings()
        self._ui_ready = True
        self._apply_preferences()
        self._apply_appearance(save=False)
        self._toggle_log(save=False)
        self._show_page("Solver", animate=False)
        self.after(100, self._workspace_tick)
        self.after(60, self._opening_reveal)

    @staticmethod
    def _blend_color(first, second, amount):
        return '#' + ''.join(f'{round(int(first[i:i+2],16)*(1-amount)+int(second[i:i+2],16)*amount):02x}' for i in (1,3,5))

    def _opening_reveal(self):
        """A short native reveal; never delays the solver or changes window opacity."""
        if not self.animation_var.get():return
        overlay=ctk.CTkFrame(self._content,fg_color=self.BG,corner_radius=0)
        overlay.place(x=0,y=0,relwidth=1,relheight=1)
        center=ctk.CTkFrame(overlay,fg_color="transparent")
        center.place(relx=.5,rely=.43,anchor="center")
        title=self._label(center,'S / A',48,True);title.pack()
        self._label(center,'A clearer way through.',14,muted=True).pack(pady=(6,24))
        bar=ctk.CTkProgressBar(center,width=200,height=3,corner_radius=2)
        bar.pack();bar.set(0)
        started=time.monotonic()
        def frame():
            if not overlay.winfo_exists():return
            p=min(1.,(time.monotonic()-started)/.85)
            accent=self.ACCENTS.get(self.accent_var.get(),self.ACCENTS['Mint'])
            color=self._blend_color(accent,'#70b8ff',.5-.5*math.cos(p*math.pi))
            title.configure(text_color=color);bar.configure(progress_color=color);bar.set(1-(1-p)**3)
            center.place_configure(rely=.43-.025*(1-(1-p)**3))
            if p<1:self.after(16,frame)
            else:overlay.destroy()
        frame()

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
        page = ctk.CTkScrollableFrame(self._pages["Solver"], fg_color="transparent", corner_radius=0)
        page.pack(fill="both", expand=True)
        self._solver_body = page
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.pack(fill="x", pady=(0, 8))
        self._label(header, "Solver", 30, True).pack(side="left")
        self.hide_btn = ctk.CTkButton(header, text="Hide · F8", width=88, height=30, fg_color=self.RAISED,
                                     text_color=self.MUTED, command=self.hide_window)
        self.hide_btn.pack(side="right")
        self._label(page, "Open a question set. We’ll take it from here.", 12, muted=True).pack(anchor="w", pady=(0, 16))
        self._build_homework_browser(page)
        self._build_bookwork_record(page)
        options = self._card(page, "Question set")
        option_row = ctk.CTkFrame(options, fg_color="transparent")
        option_row.pack(fill="x", padx=18, pady=(4,14))
        self.bookwork_checkbox = ctk.CTkCheckBox(option_row, text="Bookwork check",
            variable=self.bookwork_var, command=self._apply_preferences, checkbox_width=20, checkbox_height=20)
        self.bookwork_checkbox.pack(side="left", padx=(0,24))
        self.workings_checkbox = ctk.CTkCheckBox(option_row, text="Save workings",
            variable=self.workings_var, command=self._apply_preferences, checkbox_width=20, checkbox_height=20)
        self.workings_checkbox.pack(side="left")
        self.workings_autosave_label = self._label(options, 'Workings save automatically to Documents when enabled.', 11, muted=True)
        self.workings_autosave_label.configure(wraplength=490)
        self.workings_autosave_label.pack(anchor='w', padx=18, pady=(0,12))
        hero = self._card(page)
        self._orb = tk.Canvas(hero, width=90, height=90, highlightthickness=0, bd=0)
        self._orb.pack(side="left", padx=(16,10), pady=12)
        mode = ctk.CTkFrame(hero, fg_color="transparent")
        mode.pack(side="left", fill="both", expand=True, padx=(0,18), pady=16)
        self._set_progress_label = self._label(mode, "The whole set. Automatically.", 19, True)
        self._set_progress_label.pack(anchor="w")
        self._label(mode, "Question tabs are detected when you start.", 11, muted=True).pack(anchor="w", pady=(2,10))
        pace = ctk.CTkFrame(mode, fg_color="transparent")
        pace.pack(fill="x")
        self._label(pace,"Human pacing",11,muted=True).pack(side="left",padx=(0,14))
        self.human_switch = ctk.CTkSegmentedButton(pace, values=["Off", "On"], variable=self.human_var,
                                                height=28, width=120, command=self._set_human_mode)
        self.human_switch.pack(side="left")
        controls = ctk.CTkFrame(page, fg_color="transparent")
        controls.pack(fill="x", pady=(0, 14))
        controls.grid_columnconfigure(0, weight=3)
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(2, weight=1)
        self.start_btn = ctk.CTkButton(controls, text="Start solver  →", height=48, corner_radius=12,
                                      font=("Segoe UI", 14, "bold"), command=self._on_start)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 9))
        self.stop_btn = ctk.CTkButton(controls, text="Stop · Esc", height=48, corner_radius=12,
                                     fg_color=self.RAISED, hover_color="#8b3346", text_color=self.TEXT, command=self._on_stop)
        self.stop_btn.grid(row=0, column=2, sticky="ew")
        self.pause_btn = ctk.CTkButton(controls, text='Pause', height=48, corner_radius=12,
                                      fg_color=self.RAISED, text_color=self.TEXT,
                                      command=self._on_pause_resume, state='disabled')
        self.pause_btn.grid(row=0, column=1, sticky='ew', padx=(0,9))
        self.stop_btn.configure(state="disabled")
        self._accent_widgets.extend([self.start_btn])
        self.status_label = self._label(page, "Choose your homework browser above, then press Start.", 12, muted=True)
        self.status_label.configure(wraplength=520)
        self.status_label.pack(fill="x", pady=(0, 16))
        stats = ctk.CTkFrame(page, fg_color="transparent")
        stats.grid_columnconfigure((0, 1), weight=1)
        self._home_input = self._token_card(stats, "INPUT TOKENS", 0)
        self._home_output = self._token_card(stats, "OUTPUT TOKENS", 1)
        self._meter_note = self._label(page, "THIS SESSION  ·  Waiting for the first API response", 10, muted=True)

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

    def _build_homework_browser(self, parent):
        target = self._card(parent, "Homework browser")
        self.browser_mode_var = ctk.StringVar(value="Separate browser" if self.browser_var.get() else "Existing Chrome")
        self.browser_mode_switch = ctk.CTkSegmentedButton(target, values=["Existing Chrome", "Separate browser"],
                                                        variable=self.browser_mode_var, command=self._set_browser_mode, height=34)
        self.browser_mode_switch.pack(fill="x", padx=18, pady=(4, 6))
        self.browser_target_label = self._label(target, "", 11, muted=True)
        self.browser_target_label.configure(wraplength=480)
        self.browser_target_label.pack(fill="x", padx=18, pady=(0, 5))
        self.browser_open_btn = ctk.CTkButton(target, text="Open separate browser · sign in once", height=38, command=self._open_background_browser)
        self.browser_open_btn.pack(fill="x", padx=18, pady=8)
        self._accent_widgets.append(self.browser_open_btn)
        self.window_select_btn = ctk.CTkButton(target, text="Use existing Chrome window", fg_color=self.RAISED, text_color=self.TEXT,
                                             command=self._choose_existing_window, height=36)
        self.window_select_btn.pack(fill="x", padx=18, pady=7)
        row = ctk.CTkFrame(target, fg_color="transparent")
        self._browser_tools_row = row
        row.pack(fill="x", padx=18, pady=(0, 14))
        self.refresh_btn = ctk.CTkButton(row, text="Refresh windows", width=130, height=30, fg_color=self.RAISED, text_color=self.TEXT, command=self._quick_refresh_windows)
        self.refresh_btn.pack(side="left")
        self.roi_quick_btn = ctk.CTkButton(row, text="Select screen area", width=140, height=30, fg_color=self.RAISED, text_color=self.TEXT, command=self._select_region)
        self.roi_quick_btn.pack(side="left", padx=8)

    def _build_bookwork_record(self, parent):
        card = self._card(parent)
        self.homework_record_label = self._label(card, "", 11, muted=True)
        self.homework_record_label.configure(wraplength=480)
        self.homework_record_label.pack(fill="x", padx=18, pady=(12, 6))
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(0, 12))
        self.homework_new_btn = ctk.CTkButton(row, text="New set", width=110, fg_color=self.RAISED,
                                            text_color=self.TEXT, command=self._new_homework_record)
        self.homework_new_btn.pack(side="left")
        self.homework_open_btn = ctk.CTkButton(row, text="Open saved set", width=130, fg_color=self.RAISED,
                                             text_color=self.TEXT, command=self._open_homework_record)
        self.homework_open_btn.pack(side="left", padx=8)
        self.homework_review_btn = ctk.CTkButton(row, text="Review pending", width=125, fg_color=self.RAISED,
                                               text_color=self.TEXT, command=self._review_homework_record)
        self.homework_review_btn.pack(side="left")
        self._refresh_homework_record()

    def _new_homework_record(self):
        if self._is_running or self._opening_browser:
            return
        dialog = ctk.CTkInputDialog(title="New homework record", text="Name this assignment (use a different record for each homework):")
        name = dialog.get_input()
        if not name or self._is_running:
            return
        try:
            self.homework_store = HomeworkStore.create(name)
            self._homework_error = ""
            self._save_workspace()
        except (OSError, ValueError) as error:
            self._homework_error = str(error)
        self._refresh_homework_record()

    def _open_homework_record(self):
        if self._is_running or self._opening_browser:
            return
        from tkinter import filedialog
        directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".config"))) / "AutoAnswer" / "homework"
        filename = filedialog.askopenfilename(parent=self, title="Open homework record", initialdir=str(directory), filetypes=[("Homework record", "*.json")])
        if not filename or self._is_running:
            return
        try:
            store = HomeworkStore(filename)
            self.homework_store = store
            self._homework_error = ""
            self._save_workspace()
        except (OSError, ValueError, TypeError) as error:
            self._homework_error = f"Record was not opened: {error}"
        self._refresh_homework_record()

    def _review_homework_record(self):
        if self._is_running or not self.homework_store:
            return
        from tkinter import messagebox
        if messagebox.askyesno("Review uncertain submissions", "Inspect the browser first. Discard saved uncertain outcomes so you can try again? This does not mark any answer correct.", parent=self):
            try:
                self.homework_store.review_unresolved()
            except (OSError, ValueError) as error:
                self._homework_error = str(error)
            self._refresh_homework_record()

    def _refresh_homework_record(self):
        self._watch_workings_store()
        store = self.homework_store
        busy = self._is_running or self._opening_browser
        pending = store.unresolved_count if store else 0
        latest = {a['code']: a['status'] for a in store.data['attempts']} if store else {}
        count = sum(status == 'correct' for status in latest.values())
        state = (str(store.path) if store else None, count, pending, busy, self._homework_error)
        if state == getattr(self, "_homework_record_state", None):
            return
        self._homework_record_state = state
        text = f"{store.data['name']} · {count} verified answers · {pending} pending review" if store else "Create a question history or open a saved set."
        self.homework_record_label.configure(text=self._homework_error or text)
        self.homework_new_btn.configure(state="disabled" if busy else "normal")
        self.homework_open_btn.configure(state="disabled" if busy else "normal")
        self.homework_review_btn.configure(state="disabled" if busy or not pending else "normal")

    def _set_browser_mode(self, value):
        """Change input mode only while the solver and browser setup are idle."""
        if self._is_running or self._opening_browser:
            self.browser_mode_var.set("Separate browser" if self.browser_var.get() else "Existing Chrome")
            return
        self.browser_var.set(value == "Separate browser")
        if not self.browser_var.get():
            self.executor.browser_session = self.observer.browser_session = None
        self._apply_preferences()
        if self.browser_var.get():
            message = "Open your homework in the separate browser, then press Start."
        else:
            message = "Ready to solve in the selected window." if self.selected_region else "Choose your homework window above, then press Start."
        self.status_label.configure(text=message, text_color=self.MUTED)

    def _refresh_browser_controls(self):
        """Keep the primary browser selection and target visible on the Solver page."""
        separate = self.browser_var.get()
        busy = self._is_running or self._opening_browser
        state = (separate, busy, self.selected_window_title, repr(self.selected_region), bool(self.browser_session))
        if state == getattr(self, "_browser_controls_state", None):
            return
        self._browser_controls_state = state
        self.browser_open_btn.pack_forget()
        self.window_select_btn.pack_forget()
        self._browser_tools_row.pack_forget()
        if separate:
            self.browser_open_btn.pack(fill="x", padx=18, pady=(5, 14))
        else:
            self.window_select_btn.configure(text="Change homework window" if self.selected_window_title else "Choose homework window")
            self.window_select_btn.pack(fill="x", padx=18, pady=(5, 7))
            self._browser_tools_row.pack(fill="x", padx=18, pady=(0, 14))
        self.browser_mode_var.set("Separate browser" if separate else "Existing Chrome")
        self.browser_mode_switch.configure(state="disabled" if busy else "normal")
        self.browser_open_btn.configure(state="disabled" if busy else "normal")
        self.window_select_btn.configure(state="disabled" if busy else "normal")
        for button in (self.refresh_btn, self.roi_quick_btn):
            button.configure(state="disabled" if busy or separate else "normal")
        if separate:
            description = "Separate browser ready · sign in and open your homework there." if self.browser_session else "Open the separate browser and sign in to your homework."
        elif self.selected_window_title:
            description = "Selected: " + self.selected_window_title
        elif self.selected_region:
            region = self.selected_region
            description = f"Selected screen area: {region['width']} × {region['height']}"
        else:
            description = "Choose the Chrome window containing your homework."
        self.browser_target_label.configure(text=description)

    def _token_card(self, parent, title, column):
        frame = ctk.CTkFrame(parent, fg_color=self.PANEL, corner_radius=14)
        frame.grid(row=0, column=column, sticky="ew", padx=(0, 6) if column == 0 else (6, 0))
        self._label(frame, title, 10, True, True).pack(anchor="w", padx=18, pady=(13, 0))
        label = self._label(frame, "0", 28, True)
        label.pack(anchor="w", padx=18, pady=(0, 13))
        return label

    def _build_configuration(self):
        page = self._pages["Configuration"]
        self._heading(page, "Configuration", "Pacing, visibility, and your homework connection.")
        scroll = ctk.CTkScrollableFrame(page, fg_color="transparent", corner_radius=0)
        scroll.pack(fill="both", expand=True)
        run = self._card(scroll, "Question set")
        self._label(run,"Automatically solve every question in the visible set.",12).pack(anchor="w",padx=18,pady=(6,14))
        display = self._card(scroll, "Visibility")
        self.log_switch = ctk.CTkSwitch(display, text="Show live activity log on Solver", variable=self.log_visible_var, command=self._toggle_log)
        self.log_switch.pack(anchor="w", padx=18, pady=16)
        ctk.CTkButton(display, text="Open session logs", command=self._open_session_logs,
                      fg_color=self.RAISED, text_color=self.TEXT).pack(fill="x", padx=18, pady=(0,14))

        review = self._card(scroll, "Questions needing help")
        self._label(review, "1 wrong → high reasoning · 4 wrong → save & skip", 11, muted=True).pack(anchor="w", padx=18, pady=10)
        ctk.CTkButton(review, text="Open human review logs", command=self._open_review_logs,
                      fg_color=self.RAISED, text_color=self.TEXT).pack(fill="x", padx=18, pady=(0, 14))

        options = self._card(scroll, "Automation")
        self.option_switches = []
        for text, name in (("Finish sound", "sound"), ("Input delays", "delay"), ("Smooth pointer", "pointer"),
                           ("Numeric keyboard entry", "keyboard"), ("Fast submit · trust the last numeric entry", "submit")):
            switch = ctk.CTkSwitch(options, text=text, variable=getattr(self, name + "_var"), command=self._apply_preferences)
            switch.pack(anchor="w", padx=18, pady=8)
            self.option_switches.append(switch)
        self.wait_label = self._label(options, "Waiting Time: off", 11, muted=True)
        self.wait_label.pack(anchor="w", padx=18, pady=(8, 0))
        self.wait_slider = ctk.CTkSlider(options, from_=0, to=5, number_of_steps=50, command=self._set_waiting_time)
        self.wait_slider.set(self.waiting_time)
        self.wait_slider.pack(fill="x", padx=18, pady=(8, 20))
        provider = self._card(scroll, "Model connection")
        self.provider_var = ctk.StringVar(value="Remote API" if self.mode_var.get() == "remote" else "Local CLI")
        self.mode_switch = ctk.CTkSegmentedButton(provider, values=["Remote API", "Local CLI"], variable=self.provider_var, command=self._set_provider, height=34)
        self.mode_switch.pack(anchor="w", padx=18, pady=8)
        config = load_api_config()
        self._label(provider, "Server URL", 11, muted=True).pack(anchor="w", padx=18)
        self.base_url_entry = ctk.CTkEntry(provider, height=34)
        self.base_url_entry.insert(0, self._prefs.get("base_url", config["base_url"]))
        self.base_url_entry.pack(fill="x", padx=18, pady=(2, 10))
        self._label(provider, "Model · low effort, high after 1 wrong answer", 11, muted=True).pack(anchor="w", padx=18)
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

    def _open_session_logs(self):
        """Open local diagnostics for stopped runs and unexpected failures."""
        folder = self._runtime_log_path.parent
        if os.name == 'nt':
            os.startfile(str(folder))
        else:
            import subprocess, sys
            subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open', str(folder)])

    def _build_appearance(self):
        page = self._pages["Appearance"]
        self._heading(page, "Appearance", "Choose your colors, contrast, and motion.")
        card = self._card(page, "Color & contrast")
        self.appearance_choices = {}
        for label, variable, values in (("Theme", self.theme_var, ["Midnight", "Light", "System"]),
                                         ("Accent", self.accent_var, list(self.ACCENTS))):
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=14)
            self._label(row, label, 13).pack(anchor="w", pady=(0, 8))
            choices = {}
            option_grid = ctk.CTkFrame(row, fg_color="transparent")
            option_grid.pack(fill="x")
            option_grid.grid_columnconfigure((0, 1), weight=1)
            for index, value in enumerate(values):
                checked = ctk.BooleanVar(value=variable.get() == value)
                checkbox = ctk.CTkCheckBox(option_grid, text=value, variable=checked,
                                         command=lambda group=label, selected=value: self._choose_appearance(group, selected),
                                         height=34, checkbox_width=22, checkbox_height=22,
                                         text_color=self.TEXT, font=("Segoe UI", 13))
                checkbox.grid(row=index // 2, column=index % 2, sticky="w", pady=3, padx=(0, 15))
                choices[value] = (checkbox, checked)
            self.appearance_choices[label] = choices
        card = self._card(page, "Comfort")
        self.animation_switch = ctk.CTkSwitch(card, text="Animations · turn off for reduced motion", variable=self.animation_var, command=self._save_workspace)
        self.animation_switch.pack(anchor="w", padx=20, pady=18)
        self.pin_switch = ctk.CTkSwitch(card, text="Always on top · enabled", variable=self.pin_var, state="disabled")
        self.pin_switch.pack(anchor="w", padx=20, pady=(0, 18))
        self._label(page, "Your preferences are saved on this device.", 11, muted=True).pack(anchor="w", pady=10)

    def _choose_appearance(self, group, selected):
        """Keep exactly one checked choice per appearance group."""
        variable = self.theme_var if group == "Theme" else self.accent_var
        variable.set(selected)
        self._apply_appearance()

    def _watch_workings_store(self):
        store = self.homework_store
        enabled = self.workings_var.get()
        if store is self._workings_store and enabled == getattr(self, "_workings_enabled", None):
            return
        if self._workings_store:
            self._workings_store.on_change = None
        self._workings_store = store
        self._workings_enabled = enabled
        if store and enabled:
            path = store.path
            store.on_change = lambda snapshot, path=path: self._workings_exporter.schedule(snapshot, path, automatic=True)
            store.on_change(store.snapshot())
        message = ('Automatic save: ' + str(self._workings_exporter.automatic_path(store.snapshot()).parent)
                   if store and enabled else 'Create or open a set to save workings automatically in Documents.'
                   if enabled else 'Automatic workings saving is off. Existing files are kept.')
        if hasattr(self, 'workings_autosave_label'):
            self.workings_autosave_label.configure(text=message)
        if hasattr(self, "workings_status"):
            self.workings_status.configure(text=message)

    def _build_workings(self):
        page = self._pages["Workings"]
        self._heading(page, "Workings", "Minimal mathematical steps for your verified answers.")
        self.workings_coverage = self._label(page, "", 12)
        self.workings_coverage.pack(fill="x", pady=(0, 6))
        self._label(page, "Rendered mathematics · Markdown images and an offline browser copy", 11, muted=True).pack(anchor="w", pady=(0, 10))
        self.workings_source = ctk.CTkScrollableFrame(page, fg_color=self.PANEL, corner_radius=12)
        self.workings_source.pack(fill="both", expand=True)
        self._workings_images = []
        self.workings_review = ctk.CTkTextbox(page, height=90, fg_color=self.PANEL, text_color=self.MUTED, font=("Segoe UI", 11), corner_radius=12)
        self.workings_review.pack(fill="x", pady=(10, 10))
        self.workings_review.configure(state="disabled")
        buttons = ctk.CTkFrame(page, fg_color="transparent")
        buttons.pack(fill="x")
        self.workings_save_btn = ctk.CTkButton(buttons, text="Save Markdown", command=self._save_workings_markdown, height=36)
        self.workings_save_btn.pack(side="left")
        self._accent_widgets.append(self.workings_save_btn)
        self.workings_copy_btn = ctk.CTkButton(buttons, text="Copy workings", command=self._copy_workings, fg_color=self.RAISED, text_color=self.TEXT, width=110, height=36)
        self.workings_copy_btn.pack(side="left", padx=8)
        self.workings_folder_btn = ctk.CTkButton(buttons, text="Open folder", command=self._open_workings_folder, fg_color=self.RAISED, text_color=self.TEXT, width=110, height=36)
        self.workings_folder_btn.pack(side="left")
        self.workings_preview_btn = ctk.CTkButton(buttons, text="Open preview", command=self._open_workings_preview,
                                                fg_color=self.RAISED, text_color=self.TEXT, width=105, height=36)
        self.workings_preview_btn.pack(side="left", padx=(8, 0))
        self.workings_status = self._label(page, "Local checkpoints update automatically; files contain mathematical notation only.", 11, muted=True)
        self.workings_status.configure(wraplength=520)
        self.workings_status.pack(fill="x", pady=(10, 0))
        self._refresh_workings()

    def _refresh_workings(self):
        store = self.homework_store
        key = (id(store), store.revision if store else None, self._workings_saving)
        if key == getattr(self, "_workings_view_state", None):
            return
        self._workings_view_state = key
        snapshot = store.snapshot() if store else {'attempts': []}
        report = build_report(snapshot)
        self._workings_report = report
        preview_token = (id(store), store.revision if store else None)
        if report.ready and preview_token != getattr(self, '_workings_preview_token', None):
            self._workings_preview_token = preview_token
            self._workings_exporter.preview(snapshot, preview_token)
        name = snapshot.get('name', 'Select a homework record on Solver')
        self.workings_coverage.configure(text=f"{name}\n{len(report.ready)}/{report.total} captured questions ready · {len(report.review)} need review")
        self._render_workings_preview(report)
        notes = "\n".join(f"{code}: {reason}" for code, reason in report.review) or "No captured questions need review." if store else "Create or open a homework record on Solver. Older answers without saved workings cannot be reconstructed automatically."
        self.workings_review.configure(state="normal")
        self.workings_review.delete("1.0", "end")
        self.workings_review.insert("end", notes)
        self.workings_review.configure(state="disabled")
        self.workings_save_btn.configure(state="normal" if report.ready and not self._workings_saving else "disabled")
        self.workings_copy_btn.configure(state="normal" if report.ready else "disabled")
        self.workings_folder_btn.configure(state="normal" if store else "disabled")
        self.workings_preview_btn.configure(state="normal" if report.ready else "disabled")

    def _render_workings_preview(self, report):
        """Show equations as offline images instead of displaying TeX escape commands."""
        import re
        from io import BytesIO
        from PIL import Image
        for child in self.workings_source.winfo_children():
            child.destroy()
        self._workings_images = []
        assets = {name: source for source, name in report.assets}
        for block in re.finditer(r'^## ([^\n]+)$|\$\$\n(.*?)\n\$\$|!\[\]\(assets/([^\n]+)\)', report.markdown, re.M | re.S):
            if block[1]:
                self._label(self.workings_source, block[1], 18, bold=True).pack(anchor="w", padx=12, pady=(14, 5))
                continue
            try:
                if block[2]:
                    data = getattr(self, '_workings_preview_data', {}).get(block[2])
                    if data is None:
                        self._label(self.workings_source, "Rendering equation…", 11, muted=True).pack(anchor="w", padx=12, pady=8)
                        continue
                    image = Image.open(BytesIO(data))
                else:
                    image = Image.open(assets[block[3]])
                image.load()
                scale = min(1., 470 / image.width)
                size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                rendered = ctk.CTkImage(light_image=image, dark_image=image, size=size)
                self._workings_images.append(rendered)
                ctk.CTkLabel(self.workings_source, text="", image=rendered).pack(anchor="w", padx=10, pady=6)
            except Exception as error:
                self._label(self.workings_source, f"Preview unavailable: {error}", 11, muted=True).pack(anchor="w")
        if not report.ready:
            self._label(self.workings_source, "Verified workings will appear here after a correct answer.", 12, muted=True).pack(anchor="w", padx=12, pady=20)

    def _open_workings_preview(self):
        """Export the current verified snapshot and open its standalone HTML copy."""
        if not self.homework_store:
            return
        self._open_preview_when_ready = True
        self._workings_exporter.schedule(self.homework_store.snapshot(), self.homework_store.path,
                                        self._workings_output_path())

    def _save_workings_markdown(self):
        store = self.homework_store
        if not store or self._workings_saving:
            return
        from tkinter import filedialog
        destination = filedialog.asksaveasfilename(parent=self, title="Save mathematical workings", defaultextension=".md",
            initialfile="homework_workings.md", filetypes=[("Markdown", "*.md")])
        if not destination:
            return
        if Path(destination).suffix.lower() != ".md":
            self.workings_status.configure(text="Choose a Markdown filename ending in .md.")
            return
        if Path(destination).resolve() == store.path.resolve():
            self.workings_status.configure(text="Choose a different file; the homework record must remain intact.")
            return
        snapshot = store.snapshot()
        self._workings_saving = True
        self.workings_status.configure(text="Saving verified workings…")
        self._workings_exporter.schedule(snapshot, store.path, destination)
        self._refresh_workings()

    def _copy_workings(self):
        if self.homework_store:
            report = build_report(self.homework_store.snapshot())
            if report.ready:
                self.clipboard_clear()
                self.clipboard_append(report.markdown)
                self.workings_status.configure(text="Verified LaTeX workings copied.")

    def _workings_output_path(self):
        """Find automatic files first, retaining access to earlier local checkpoints."""
        snapshot = self.homework_store.snapshot()
        automatic = WorkingsExporter.automatic_path(snapshot)
        return automatic if self.workings_var.get() or automatic.exists() else WorkingsExporter.default_path(snapshot, self.homework_store.path)

    def _open_workings_folder(self):
        if not self.homework_store:
            return
        folder = self._workings_output_path().parent
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == "nt": os.startfile(str(folder))
            else:
                import subprocess, sys
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])
        except OSError as error:
            self.workings_status.configure(text=f"Could not open output folder: {error}")

    def _process_workings_events(self):
        try:
            while True:
                event = self._workings_queue.get_nowait()
                if event['kind'] == 'preview':
                    if event['token'] == getattr(self, '_workings_preview_token', None):
                        self._workings_preview_data = event['images']
                        self._render_workings_preview(self._workings_report)
                        if event['error']:
                            self.workings_status.configure(text="Preview could not render: " + event['error'])
                    continue
                if event['kind'] == 'save':
                    self._workings_saving = False
                if self.homework_store and event['id'] == self.homework_store.data['id']:
                    if event['error']:
                        self.workings_status.configure(text="Workings file was not saved: " + event['error'])
                        if event['kind'] == 'automatic':
                            self.workings_autosave_label.configure(text='Automatic save failed: ' + event['error'])
                    elif event.get("empty"):
                        self.workings_status.configure(text="No verified workings are ready yet. See the review list above.")
                    else:
                        self.workings_status.configure(text=("Saved automatically: " if event['kind'] == 'automatic' else "Saved: " if event['kind'] == 'save' else "Local checkpoint: ") + event['path'])
                        if event['kind'] == 'automatic':
                            self.workings_autosave_label.configure(text='Saved automatically: ' + event['path'])
                        if getattr(self, '_open_preview_when_ready', False) and event.get('preview_path'):
                            import webbrowser
                            self._open_preview_when_ready = False
                            webbrowser.open(Path(event['preview_path']).resolve().as_uri())
        except queue.Empty:
            pass

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
        for group, choices in self.appearance_choices.items():
            selected = self.theme_var.get() if group == "Theme" else self.accent_var.get()
            for value, (checkbox, checked) in choices.items():
                checked.set(value == selected)
                checkbox.configure(fg_color=(accent, shade), hover_color=(accent, shade),
                                   border_color=self.MUTED, checkmark_color=("#08141a", "#ffffff"), text_color=self.TEXT)
        self.mode_switch.configure(selected_color=(accent, shade), selected_hover_color=(accent, shade),
                                   unselected_color=self.RAISED, unselected_hover_color=self.RAISED, text_color=self.TEXT)
        self.browser_mode_switch.configure(selected_color=(accent, shade), selected_hover_color=(accent, shade),
                                           unselected_color=self.RAISED, unselected_hover_color=self.RAISED, text_color=self.TEXT)
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
        if hasattr(self, "bookwork_checkbox"):
            self.bookwork_checkbox.configure(state="disabled" if working else "normal")
        if hasattr(self, "workings_checkbox"):
            self.workings_checkbox.configure(state="disabled" if working else "normal")
        self.attributes("-alpha", 1.0)
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
        if self._is_hidden or not self.winfo_viewable() or self.tk.call("grab", "current"):
            return
        if not self.attributes("-topmost"):
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
        settings = {name: getattr(self, name + "_var").get() for name in ("sound", "delay", "pointer", "keyboard", "submit", "browser", "bookwork", "workings")}
        settings["homework_record"] = str(self.homework_store.path) if self.homework_store else None
        settings.update(theme=self.theme_var.get(), accent=self.accent_var.get(), animations=self.animation_var.get(),
                        live_log=self.log_visible_var.get(), pin=self.pin_var.get(), waiting_time=self.waiting_time,
                        provider=self.mode_var.get(),
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
        if self.controller and self._is_running:
            paused = self.controller.is_paused
            self.pause_btn.configure(text='Resume' if paused else 'Pause')
            if paused:
                self.status_label.configure(text='Paused: ' + self.controller.pause_reason + '. Correct the page, then Resume. Progress is saved.', text_color='orange')
        if now - getattr(self, "_last_front_check", 0) > 1:
            self._last_front_check = now
            self._keep_in_front()
        try:
            while True:
                self._record_usage(self._usage_queue.get_nowait())
        except queue.Empty:
            pass
        self._refresh_browser_controls()
        self._refresh_homework_record()
        self._process_workings_events()
        self._refresh_workings()
        self.reset_usage_btn.configure(state="disabled" if self._is_running else "normal")
        if self.controller and self._is_running and self.controller.question_inventory:
            inventory=self.controller.question_inventory
            correct=sum(item['correct'] for item in inventory)
            self._set_progress_label.configure(text=f"{correct} / {len(inventory)} questions complete")
        elif not self._is_running:
            self._set_progress_label.configure(text="The whole set. Automatically.")
        if self._active_page == "Solver":
            dark = ctk.get_appearance_mode() == "Dark"
            self._orb.configure(bg="#111c2b" if dark else "#ffffff")
            self._orb.delete("all")
            accent = self.ACCENTS.get(self.accent_var.get(), self.ACCENTS["Mint"])
            if self.animation_var.get() and dark and not self._is_running:
                accent = self._blend_color(accent,"#70b8ff",.25+.25*math.sin(self._anim_phase*.6))
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
