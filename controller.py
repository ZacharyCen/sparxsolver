"""
controller.py - Closed-Loop Control Orchestrator
Implements:
- Observe-Plan-Validate-Execute-Verify main state machine
- Dynamic question state tracking & maximum question limit
- Submission deduplication rule (prohibits blind repeated submits when result is pending)
- Anti-window-switch protection (frame diff check prior to physical action)
- Late result discarding on Esc cancellation
"""

import time
import os
import tempfile
import json
from typing import Dict, Any, Optional

import cv2
import numpy as np

from bookwork import BookworkReview
from retry_policy import QuestionRetries, RecoveryGuard, incorrect_feedback, question_tabs
from observer import Observer
from planner import BasePlanner
from executor import ActionExecutor
from browser_session import BrowserSessionError
from win_utils import get_window_bounds, activate_target_window


class AutoAnswerController:
    """
    Coordinates the visual observation, model planning, strict schema validation,
    physical execution, and post-verification loop.
    Supports dynamic target window tracking and auto-foregrounding.
    """

    def __init__(
        self,
        observer: Observer,
        planner: BasePlanner,
        executor: ActionExecutor,
        max_questions: int = 10,
        region: Optional[Dict[str, int]] = None,
        target_hwnd: Optional[int] = None,
        waiting_time: float = 0.3,
        fast_submit: bool = False,
        bookwork=None,
        require_bookwork: bool = False,
        submit_with_enter: bool = False
    ):
        self.observer = observer
        self.planner = planner
        self.executor = executor
        self.auto_complete = max_questions is None
        self.max_questions = max_questions if max_questions is not None else 0
        self.question_inventory = []
        self.review_codes = set()
        self._inventory_misses = 0
        self.waiting_time = max(0.0, min(5.0, waiting_time))
        self.fast_submit = fast_submit
        self.bookwork = bookwork
        self.require_bookwork = require_bookwork
        self.submit_with_enter = submit_with_enter
        self._last_answer_submission_at = 0.0
        self._bookwork_finish_passed = False
        self.target_hwnd = target_hwnd
        self.set_region(region or {"left": 0, "top": 0, "width": 800, "height": 600})

        self.finish_reason = "Ready"
        self.current_question_count = 0
        self.is_running = False
        self.is_paused = False
        self._last_submit_time: Optional[float] = None
        self._consecutive_submit_count = 0
        self.retries = QuestionRetries()
        self.skipped_question_count = 0

    def set_region(self, region: Dict[str, int]) -> None:
        """Sets the active ROI region and updates executor screen offset."""
        self.region = region
        self.executor.set_roi_offset(region["left"], region["top"])
        print(f"[Controller] Configured ROI global screen offset: ({region['left']}, {region['top']})")

    def start(self) -> None:
        """Starts the autonomous answering loop."""
        self.is_running = True
        self.is_paused = False
        self.executor.reset_cancellation()
        self.run_loop()

    def pause(self) -> None:
        """Pauses the answering loop."""
        self.is_paused = True

    def resume(self) -> None:
        """Resumes the answering loop."""
        self.is_paused = False

    def stop(self, reason: str = "Stopped by user") -> None:
        """Stops the answering loop and cancels pending operations."""
        self.finish_reason = reason
        self.is_running = False
        self.executor.cancel()

    def _handle_bookwork(self, frame):
        """Handle interruptions without touching ordinary retry/count state."""
        if not self.bookwork:
            return False
        activate = (lambda: activate_target_window(self.target_hwnd)) if self.target_hwnd else None
        return self.bookwork.handle(frame, lambda: self.observer.capture_region(self.region), activate)

    def finish_bookwork(self, frame=None):
        """Stay alive after the maths limit, navigate Summary, and verify the required check."""
        if not self.bookwork:
            raise BookworkReview('Bookwork was requested but no homework record is selected')
        print('[Finish] Maths finished; checking bookwork before ending the run.')
        deadline=time.monotonic()+15
        summary_clicked=False
        while self.is_running and not self.executor.is_cancelled() and time.monotonic()<deadline:
            if self.bookwork.last_completed_at > self._last_answer_submission_at:
                self._bookwork_finish_passed=True
                return True
            if frame is None:
                frame=self.observer.capture_region(self.region)
            if self._handle_bookwork(frame):
                frame=None
                continue
            # The footer's Summary/Continue is preferred. Header Summary is a
            # fallback after the requested maths count or a verified end page.
            target=self.observer.detect_correct_continue(frame)
            if target is None and not summary_clicked:
                items=self.observer.extract_auxiliary_reference(frame).get('ocr_results',[])
                choices=[item for item in items if item.get('text','').strip().lower() in ('summary','view summary')
                         and item.get('confidence',1)>=.85 and 'center' in item]
                if len(choices)==1:
                    target=choices[0]['center']
            if target and not summary_clicked:
                fresh=self.observer.capture_region(self.region)
                if self.observer.compute_frame_diff(frame,fresh)>.20:
                    frame=None
                    continue
                if self.executor.is_cancelled():return False
                if self.target_hwnd:activate_target_window(self.target_hwnd)
                self.executor.execute({'type':'click','purpose':'next','target':list(map(int,target))})
                summary_clicked=True
                print('[Finish] Opened the completion screen; waiting for bookwork.')
            self.executor.wait(150)
            frame=None
        if self.executor.is_cancelled() or not self.is_running:
            return False
        raise BookworkReview('Required bookwork check did not appear or could not be confirmed; the run is not marked complete')

    def sync_question_set(self, frame):
        """Keep the run inside the detected set, including earlier unfinished tabs."""
        inventory=self.observer.question_inventory(frame)
        if inventory:
            codes=[item['code'] for item in inventory]
            if self.question_inventory and codes!=[item['code'] for item in self.question_inventory]:
                raise BookworkReview('The question set changed during the run; stopped before answering another set')
            if not self.question_inventory:
                self.max_questions=sum(not item['correct'] for item in inventory)
                print('[Set] Detected '+', '.join(codes)+f'; {self.max_questions} unfinished questions.')
                if getattr(self.observer,'_independent_learning',False):
                    self.require_bookwork=False
                    print('[Set] Independent Learning detected; homework bookwork is not required.')
            self.question_inventory=inventory
            self._inventory_misses=0
        elif not self.question_inventory:
            self._inventory_misses+=1
            if self._inventory_misses>=3:
                raise BookworkReview('Could not read the complete question-tab row ending in Summary')
        return bool(self.question_inventory) and all(item['correct'] or item['code'] in self.review_codes for item in self.question_inventory)

    def next_unfinished_tab(self):
        return next((item for item in self.question_inventory if not item['correct'] and not item['selected'] and item['code'] not in self.review_codes),None)

    def run_loop(self) -> None:
        """
        Main Observe-Plan-Validate-Execute-Verify continuous loop.
        """
        previous_result: Optional[str] = None
        frame_idx = 0
        consecutive_errors = 0
        submitted = False
        correct_verified = False
        feedback_seen = False
        stalled_cycles = 0
        submission_ready = False
        keyboard_failed = False
        entry_failures = 0
        local_focus_attempts = 0
        ocr_refresh = True
        last_action_signature = None
        repeated_ineffective_actions = 0
        question_recovery_used = False
        recovery_guard = RecoveryGuard()

        while self.is_running and (self.auto_complete or self.current_question_count < self.max_questions):
            if self.is_paused or self.executor.is_cancelled():
                time.sleep(0.2)
                continue

            frame_idx += 1
            print(f"\n==================================================")
            if self.auto_complete:
                label=f"{self.question_inventory[0]['code']}–{self.question_inventory[-1]['code']}" if self.question_inventory else 'detecting question set'
                print(f"[Cycle #{frame_idx}] {label} · {self.current_question_count} newly correct")
            else:
                print(f"[Cycle #{frame_idx}] Question {self.current_question_count + 1}/{self.max_questions}")
            print(f"==================================================")

            frame_path = ""
            try:
                # Dynamically update bounds if tracking a target window
                if self.target_hwnd:
                    cur_bounds = get_window_bounds(self.target_hwnd)
                    if cur_bounds:
                        self.set_region(cur_bounds)

                # Expose the selected desktop window before reading pixels. A
                # terminal/chat window over the ROI is not the homework page.
                if self.target_hwnd:
                    activate_target_window(self.target_hwnd)
                    self.executor.wait(60)
                    if self.executor.is_cancelled():
                        break

                # 1. OBSERVE
                print(f"[Observe] Capturing ROI ({self.region['left']},{self.region['top']}) {self.region['width']}x{self.region['height']}...")
                frame_bgr = self.observer.capture_region(self.region)
                if getattr(self.executor, "browser_session", None) and isinstance(self.executor, ActionExecutor):
                    self.set_region({"left": 0, "top": 0, "width": frame_bgr.shape[1], "height": frame_bgr.shape[0]})

                if self._handle_bookwork(frame_bgr):
                    ocr_refresh = True
                    continue

                if self.auto_complete:
                    complete=self.sync_question_set(frame_bgr)
                    complete = complete or bool(self.question_inventory and self.current_question_count+len(self.review_codes)>=self.max_questions)
                    if complete and not self.retries.pending:
                        if self.require_bookwork:self.finish_bookwork(frame_bgr)
                        self.finish_reason=f'Finished all {len(self.question_inventory)} questions; {self.current_question_count} newly correct'
                        if self.review_codes:self.finish_reason+=f'; {len(self.review_codes)} need human review'
                        break
                    if not self.question_inventory:
                        self.executor.wait(120)
                        continue

                # Correct feedback is local UI navigation: bypass full OCR and the model.
                continue_target = self.observer.detect_correct_continue(frame_bgr)
                submit_target = self.observer.detect_enabled_submit(frame_bgr) if self.fast_submit and submission_ready else None
                local_entry = None
                local_choice = None
                local_choice_ready = False
                if (self.bookwork and isinstance(self.observer,Observer) and not self.retries.pending
                        and getattr(self.bookwork,'visual_parts',None) and len(self.bookwork.visual_parts)==1):
                    answer=self.bookwork.visual_parts[0].get('answer','')
                    choice=self.observer.letter_choice(frame_bgr,answer)
                    if choice:
                        if choice['complete']:
                            if self.fast_submit:submit_target=self.observer.detect_enabled_submit(frame_bgr)
                        else:
                            submit_target=None
                            local_choice={'type':'click','purpose':'answer','target':choice['target']}
                            local_choice_ready=choice['last_change']
                if self.bookwork and isinstance(self.observer, Observer) and self.bookwork.solution and not self.retries.pending:
                    if self.observer.answer_page_visible(frame_bgr):
                        if self.bookwork.verified_fields(frame_bgr):
                            submit_target = self.observer.detect_enabled_submit(frame_bgr) if self.fast_submit else None
                        else:
                            local_entry = self.bookwork.next_entry(frame_bgr, keypad=keyboard_failed or not getattr(self.executor, 'keyboard_entry_enabled', False))
                if self.retries.wrong >= 4:
                    continue_target = None
                    aux_data = self.observer.extract_auxiliary_reference(frame_bgr)
                    ocr_items = aux_data.get("ocr_results", [])
                    question_id, skip_target = question_tabs(frame_bgr, ocr_items)
                    if not self.retries.logged:
                        log_path = self.retries.save(question_id or f"Question after {self.current_question_count} correct / {self.skipped_question_count} skipped", frame_bgr)
                        print(f"[Needs Human intervention] Four incorrect answers. Review log: {log_path}")
                    if not question_id or not skip_target:
                        self.stop("Needs Human intervention; no verified next-question tab available. Review saved at " + str(self.retries.log_dir))
                        break
                    decision = {"page_state": "feedback", "action": {"type": "click", "purpose": "skip", "target": skip_target},
                                "expected_change": "A different question tab becomes selected"}
                elif continue_target:
                    if self.bookwork:
                        self.bookwork.grade(True, frame_bgr)
                    feedback_seen = correct_verified = True
                    if self.auto_complete:
                        remaining=self.next_unfinished_tab()
                        if remaining:continue_target=remaining['target']
                    aux_data = {}
                    ocr_items = []
                    decision = {
                        "page_state": "feedback",
                        "action": {"type": "click", "purpose": "next", "target": continue_target},
                        "expected_change": "Correct feedback closes and next question appears",
                    }
                    print("[Fast path] Correct + Continue verified locally; skipping full OCR and model request.")
                elif (self.auto_complete and not self.retries.pending and
                      any(item['selected'] and item['correct'] for item in self.question_inventory) and self.next_unfinished_tab()):
                    ocr_items=[]
                    decision={'page_state':'question','action':{'type':'click','purpose':'auto_tab','target':self.next_unfinished_tab()['target']},
                              'expected_change':'Open the first unfinished question in this set'}
                elif local_entry:
                    local_focus_attempts = local_focus_attempts + 1 if local_entry['type'] == 'click' else 0
                    if local_focus_attempts >= 4:
                        self.stop('Needs Human intervention: answer field could not be focused with a recognized keypad')
                        break
                    ocr_items = []
                    decision = {'page_state': 'question', 'action': local_entry,
                                'expected_change': 'The complete number appears in the answer field'}
                    print('[Fast path] Entering a saved numeric answer locally; skipping model request.')
                elif local_choice and not submit_target:
                    ocr_items=[]
                    decision={'page_state':'question','action':local_choice,'ready_to_submit':local_choice_ready,
                              'expected_change':'The saved answer choice is selected'}
                    print('[Fast path] Selecting the saved letter answer locally; skipping model request.')
                elif submit_target:
                    ocr_items = []
                    decision = {"page_state": "question", "action": {"type": "press_enter" if self.submit_with_enter else "click", "purpose": "submit", "target": submit_target},
                                "expected_change": "Grade the completed answer"}
                    print("[Fast path] Final entry confirmed and Submit enabled; skipping model request.")
                else:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_f:
                        frame_path = tmp_f.name
                    self.observer.compress_image_same_dimension(frame_bgr, quality=85, output_path=frame_path)
                    ocr_started = time.monotonic()
                    # Vision sees the current full question. Auto mode already has
                    # fresh tab/footer grounding; avoid another full-screen OCR pass.
                    needs_full_ocr = (not self.auto_complete or self.retries.pending or self.retries.feedback_latched)
                    aux_data = self.observer.extract_auxiliary_reference(frame_bgr) if needs_full_ocr and (ocr_refresh or self.retries.pending or self.retries.feedback_latched) else {}
                    if self.auto_complete and not aux_data:
                        header=[{'text':item['code'],'center':item['target'],'confidence':.99} for item in self.question_inventory]
                        footer=self.observer.footer_navigation_items(frame_bgr) if isinstance(self.observer,Observer) else []
                        aux_data={'ocr_results':header+footer}
                    print(f"[Timing] OCR: {time.monotonic() - ocr_started:.2f}s ({'refresh' if ocr_refresh else 'skipped; current image used'})")
                    aux_table = json.dumps([
                        {"text": item["text"], "center": item["center"]}
                        for item in aux_data.get("ocr_results", [])
                        if item.get("confidence", 1.0) >= 0.6 and "center" in item
                    ], ensure_ascii=False, separators=(",", ":")) if aux_data else ""
                    ocr_items = aux_data.get("ocr_results", [])
                    print(f"[Observer] Frame captured: {frame_bgr.shape[1]}x{frame_bgr.shape[0]} | OCR items: {len(ocr_items)}")
                    if self.bookwork:
                        self.bookwork.bind_question(frame_bgr, ocr_items)
                    wrong_visible = incorrect_feedback(ocr_items, frame_bgr.shape[0])
                    if wrong_visible and not self.retries.pending and not self.retries.wrong and hasattr(self.planner,'reasoning_effort'):
                        self.planner.reasoning_effort='high'
                        self.planner.retry_context='Resumed on incorrect feedback. Re-read and solve independently using high reasoning.'
                    newly_wrong = self.retries.observe(wrong_visible)
                    if newly_wrong:
                        if self.bookwork:
                            self.bookwork.grade(False)
                        self._last_submit_time = None
                        self._consecutive_submit_count = 0
                        submission_ready = False
                        submitted = False
                        question_recovery_used = False
                        recovery_guard = RecoveryGuard()
                        feedback_seen = correct_verified = False
                        rejected = getattr(self.planner, "question_memory", "")
                        if hasattr(self.planner, "question_memory"):
                            self.planner.accepted_solution = None
                            self.planner.question_memory = ""
                            self.planner.recent_steps = []
                            self.planner.reasoning_effort = "high" if self.retries.wrong >= 1 else "low"
                            self.planner.retry_context = (f"{self.retries.wrong} confirmed incorrect submissions on this question. "
                                f"Rejected solution: {rejected}. Re-read the original question and solve independently; "
                                "dismiss Try again if needed, then use Back/Question ONLY to recover question text for this retry. "
                                "Never advance from incorrect feedback. Check all parts and units before submitting.")
                        previous_result = f"Incorrect answer confirmed ({self.retries.wrong}/4). Re-solve this question."
                        print(f"[Retry] {self.retries.wrong}/4 incorrect answers; reasoning: {'high' if self.retries.wrong >= 1 else 'low'}.")
                        if self.retries.wrong >= 4:
                            ocr_refresh = True
                            continue
                    question_state = f"Question after {self.current_question_count} correct, {self.skipped_question_count} needing human review"
                    if self.bookwork and self.bookwork.code:
                        question_state += f"; verified question code: {self.bookwork.code}"
                    labels = {item.get('text', '').strip().lower() for item in ocr_items}
                    compact_labels = {''.join(label.split()) for label in labels}
                    recovery_control = next((item for item in ocr_items if item.get('text', '').strip().lower()
                        in ('back', 'back to question', 'question') and item.get('confidence', 1) >= .85 and 'center' in item), None)
                    retry_button = next((item for item in ocr_items if item.get('text', '').strip().lower().rstrip('!.') == 'try again'
                                         and item.get('center', [0,0])[1] > frame_bgr.shape[0]*.5 and item.get('confidence',1) >= .85), None)
                    if wrong_visible and retry_button:
                        decision = {'page_state': 'feedback', 'action': {'type': 'click', 'purpose': 'answer', 'target': retry_button['center']},
                                    'expected_change': 'Incorrect feedback closes so the answer can be corrected'}
                        print('[Retry] Dismissing verified incorrect feedback locally; skipping model request.')
                    elif (self.bookwork and not question_recovery_used and not self.bookwork.solution and not submitted
                            and recovery_control and 'submitanswer' in compact_labels
                            and ('enternumber' in compact_labels or {'7', '8', '9'}.issubset(labels))):
                        decision = {'page_state': 'question', 'action': {'type': 'click', 'purpose': 'answer',
                            'target': recovery_control['center']}, 'expected_change': 'Read the hidden question before solving'}
                        print('[Recovery] Answer keypad without saved solution; reading the question locally before any model request.')
                    else:
                        decision = self.planner.plan(
                            frame_path=frame_path,
                            question_state=question_state,
                            auxiliary_reference=aux_table,
                            previous_result=previous_result
                        )

                if self.executor.is_cancelled():
                    print("[Controller] Late result discarded due to cancellation.")
                    break

                page_state = decision.get("page_state", "question")
                action = decision.get("action", {})
                action_type = action.get("type")
                purpose = action.get("purpose")
                expected_change = decision.get("expected_change", "")

                print(f"[Plan] Decision received: page_state='{page_state}', type='{action_type}', purpose='{purpose}'")
                if "target" in action:
                    print(f"[Plan] Action target: {action['target']} | Expected: '{expected_change}'")

                # 3. VALIDATE & DEDUPLICATE
                if page_state == "completed" and self.bookwork and any(
                    item.get("text", "").strip().lower().rstrip("!.") == "correct"
                    and item.get("center", [0, 0])[1] > frame_bgr.shape[0] * .5 for item in ocr_items
                ):
                    self.bookwork.grade(True, frame_bgr)
                    if submitted:
                        self.current_question_count += 1
                        submitted = False
                        print(f'[Progress] Final correct answer this run: {self.current_question_count}/{self.max_questions}')
                if page_state == 'completed' and self.require_bookwork:
                    self.finish_bookwork(frame_bgr)
                    if self._bookwork_finish_passed:
                        self.finish_reason = f'Completed {self.current_question_count} new questions; bookwork check passed'
                    break
                if page_state in ("completed", "blocked"):
                    print(f"[Controller] Halting: page_state '{page_state}' reached.")
                    self.stop(f"Page state: {page_state}; {decision.get('answer_display', '')}")
                    break

                if action_type == "stop":
                    print(f"[Controller] Model requested stop: {action.get('reason')}")
                    self.stop(action.get("reason", "Model requested stop"))
                    break

                # Count completed answers only after submission and visible feedback.
                # A resumed session may already show a graded answer.
                if page_state == "feedback" and any(
                    item.get("text", "").strip().lower() in ("correct", "correct!", "well done", "well done!")
                    for item in ocr_items
                ):
                    if self.bookwork:
                        self.bookwork.grade(True, frame_bgr)
                    feedback_seen = correct_verified = True
                if submitted and page_state == "feedback":
                    feedback_seen = True
                if purpose == "next" and not correct_verified:
                    previous_result = "Next action rejected: no submitted answer with observed feedback. Solve and submit the current question; returning to question text is not completion."
                    stalled_cycles += 1
                    if stalled_cycles >= 3:
                        print("[Safety] Repeated unverified next actions. Stopping; inspect target window.")
                        self.stop("Repeated unverified navigation; no result feedback was confirmed")
                    continue

                for key in ("target", "start", "end"):
                    if key in action:
                        x, y = action[key]
                        if not (0 <= x < frame_bgr.shape[1] and 0 <= y < frame_bgr.shape[0]):
                            raise ValueError(f"Action {key} lies outside captured frame")

                if not ocr_items and isinstance(self.observer, Observer):
                    ocr_items = self.observer.footer_navigation_items(frame_bgr)

                # Do not let navigation masquerade as answer entry.
                target = action.get("target")
                forbidden_navigation = False
                recovering_question = False
                if target:
                    for item in ocr_items:
                        text = item.get("text", "").strip().lower()
                        if text in ("question", "back", "back to question", "previous", "< previous", "view question"):
                            points = item.get("box", [])
                            if not points and 'center' in item:
                                center=item['center']
                                if abs(center[0]-target[0])<100 and abs(center[1]-target[1])<40:
                                    forbidden_navigation=True
                            if points:
                                xs, ys = zip(*points)
                                if min(xs)-12 <= target[0] <= max(xs)+12 and min(ys)-12 <= target[1] <= max(ys)+12:
                                    forbidden_navigation = True
                if (forbidden_navigation and self.bookwork and not question_recovery_used
                        and not self.bookwork.solution and not self.retries.pending):
                    forbidden_navigation = False
                    recovering_question = True
                    print('[Recovery] One-time return to read the hidden question before solving.')
                if forbidden_navigation:
                    previous_result = "Navigation rejected: target is the Question/Previous button, not an answer input. Use the stored solution and click the answer slot or keypad."
                    stalled_cycles += 1
                    if stalled_cycles >= 3:
                        self.stop("Repeated navigation to Back/Question instead of answer controls")
                    continue

                recovery_stalled=recovery_guard.observe(action, ocr_items)
                if recovery_stalled and self.retries.wrong and not getattr(self,'_recovery_replanned',False):
                    self._recovery_replanned=True
                    recovery_guard=RecoveryGuard();question_recovery_used=False;stalled_cycles=0
                    self.planner.reasoning_effort='high';self.planner.accepted_solution=None
                    self.planner.question_memory='';self.planner.recent_steps=[]
                    previous_result='Recovery stalled. Use high reasoning to re-read the question once, then enter a corrected answer. Do not cycle Back and Answer.'
                    ocr_refresh=True
                    print('[Recovery] Replanning this question with high reasoning instead of ending the run.')
                    continue
                if recovery_stalled:
                    self.stop('Needs Human intervention: repeated waits or Back/Answer navigation without answer entry. Inspect the browser before retrying.')
                    print('[Safety] ' + self.finish_reason)
                    break

                # Submission Deduplication Guard
                if purpose == "submit":
                    if self.retries.feedback_latched:
                        previous_result = "Dismiss the incorrect feedback and edit/re-solve the answer before submitting again."
                        ocr_refresh = True
                        continue
                    now = time.time()
                    if self._last_submit_time and (now - self._last_submit_time < 3.0):
                        print("[Safety] Anti-deduplication: submission in flight, pausing.")
                        time.sleep(1.0)
                        continue
                    if self._consecutive_submit_count > 3:
                        print("[Safety] Excessive repeated submits without state change. Pausing for safety.")
                        self.stop("Repeated submissions without confirmed result feedback")
                        break
                # Anti-Window-Switch Safety Check
                verify_frame = self.observer.capture_region(self.region)
                diff = self.observer.compute_frame_diff(frame_bgr, verify_frame)
                if diff > 0.20:
                    print(f"[Safety Warning] Frame diff too high ({diff:.2f} > 0.20) prior to execution. Page changed or window switched. Re-observing...")
                    continue
                else:
                    print(f"[Safety] Pre-execution visual diff: {diff:.3f} (<= 0.20) [PASS]")

                if self._handle_bookwork(verify_frame):
                    ocr_refresh = True
                    continue
                if self.bookwork and isinstance(decision.get("solution"), dict):
                    if self.bookwork.accept_solution(decision["solution"]):
                        self.planner.accepted_solution = self.bookwork.solution
                if (purpose == 'submit' and self.bookwork and self.bookwork.solution
                        and isinstance(self.observer, Observer) and not self.bookwork.verified_fields(verify_frame)):
                    previous_result = 'Submission blocked: one or more visible numeric fields differ from the full solution. Correct every field, including decimal points.'
                    print('[Verify] ' + previous_result)
                    submission_ready = False
                    ocr_refresh = True
                    stalled_cycles += 1
                    if stalled_cycles >= 3:
                        self.stop('Needs Human intervention: numeric answers could not be verified before submission')
                    continue
                attempt_id = self.bookwork.prepare_submission(verify_frame) if self.bookwork and purpose == "submit" else None

                entry_index = None
                if action_type in ('type_text', 'keypad_number') and isinstance(self.observer, Observer):
                    before_fields = self.observer.numeric_fields(verify_frame)
                    entry_index = next((index for index,field in enumerate(before_fields)
                                        if abs(field['target'][0]-action['target'][0]) < 100 and abs(field['target'][1]-action['target'][1]) < 60), None)

                # 4. EXECUTE
                if self.target_hwnd:
                    activate_target_window(self.target_hwnd)
                print(f"[Execute] Executing action '{action_type}'...")
                if action_type == 'keypad_number':
                    for key in ['delete']*min(action['erase'],32) + list(action['text']):
                        if self.executor.is_cancelled():
                            break
                        fresh = self.observer.capture_region(self.region)
                        keys = self.observer.keypad_controls(fresh)
                        if not keys or key not in keys:
                            raise BookworkReview('Numeric keypad changed during entry; no stale key was clicked')
                        self.executor.execute({'type':'click','purpose':'answer','target':keys[key]})
                        self.executor.wait(45)
                elif action_type == "wait":
                    self.executor.wait(int(self.waiting_time * 1000))
                else:
                    self.executor.execute(action)
                if self.executor.is_cancelled():
                    break
                if recovering_question:
                    question_recovery_used = True
                    self.planner.question_memory = ""
                    self.planner.accepted_solution = None
                    self.planner.recent_steps = []
                    ocr_refresh = True
                if purpose == "submit":
                    if self.bookwork:
                        self.bookwork.dispatched(attempt_id)
                    self._last_submit_time = time.time()
                    self._last_answer_submission_at = self._last_submit_time
                    self._consecutive_submit_count += 1
                    self.retries.submitted(getattr(self.planner, "question_memory", ""), getattr(self.planner, "reasoning_effort", "low"))
                    submitted = True
                    submission_ready = False

                # 5. VERIFY
                if action_type != "wait":
                    self.executor.wait(int(self.waiting_time * 1000))
                    if (not self.waiting_time and action_type=='click' and purpose=='answer'
                            and action.get('target',[0,0])[1]>frame_bgr.shape[0]*.90):
                        # Wait for rendering locally instead of paying for an AI wait action.
                        deadline=time.monotonic()+.9
                        previous=self.observer.capture_region(self.region);stable=0
                        while time.monotonic()<deadline and not self.executor.is_cancelled():
                            self.executor.wait(60)
                            current=self.observer.capture_region(self.region)
                            stable=stable+1 if self.observer.compute_frame_diff(previous,current)<.002 else 0
                            previous=current
                            if stable>=3:break
                post_frame = self.observer.capture_region(self.region)
                if self._handle_bookwork(post_frame):
                    post_frame = self.observer.capture_region(self.region)
                if self.executor.is_cancelled():
                    break
                # Compare immediately before and after execution, excluding API latency.
                post_diff = self.observer.compute_frame_diff(verify_frame, post_frame)
                changed = post_diff > 0.0002
                if not changed and action_type not in ("wait", "stop"):
                    # A zero user delay still allows asynchronous browser updates to finish.
                    deadline = time.monotonic() + 0.8
                    while not changed and time.monotonic() < deadline:
                        if self.executor.is_cancelled():
                            break
                        self.executor.wait(50)
                        post_frame = self.observer.capture_region(self.region)
                        post_diff = self.observer.compute_frame_diff(verify_frame, post_frame)
                        changed = post_diff > 0.0002
                if self.executor.is_cancelled():
                    break
                if action_type=='press_enter' and isinstance(self.observer,Observer):
                    # Some custom pages ignore Enter. If Submit remains enabled
                    # after settling, click it once locally; never ask the model.
                    self.executor.wait(250)
                    post_frame=self.observer.capture_region(self.region)
                    enabled=self.observer.detect_enabled_submit(post_frame)
                    if enabled and self.observer.compute_frame_diff(verify_frame,post_frame)<.01:
                        self.executor.execute({'type':'click','purpose':'submit','target':enabled})
                        self.executor.wait(140)
                        post_frame=self.observer.capture_region(self.region)
                        print('[Submit] Enter was not handled; used the verified Submit control locally.')
                    post_diff=self.observer.compute_frame_diff(verify_frame,post_frame)
                    changed=post_diff>.0002
                if self.bookwork:
                    self.bookwork.observe_entry(action, post_frame)
                entry_verified = True
                if action_type in ("type_text", "keypad_number"):
                    entry_verified = self.observer.verify_numeric_entry(post_frame, action["target"], action["text"])
                    if not entry_verified:
                        # Allow the custom keypad to process the final key before readback.
                        self.executor.wait(120)
                        post_frame = self.observer.capture_region(self.region)
                        entry_verified = self.observer.verify_numeric_entry(post_frame, action['target'], action['text'])
                    if not entry_verified and entry_index is not None:
                        from bookwork import numeric_value
                        after_fields = self.observer.numeric_fields(post_frame)
                        if len(after_fields) == len(before_fields):
                            entry_verified = numeric_value(after_fields[entry_index]['value']) == numeric_value(action['text'])
                    entry_failures = 0 if entry_verified else entry_failures + 1
                    if entry_failures >= 3:
                        self.stop('Needs Human intervention: numeric entry failed readback three times; no answer was submitted')
                    if not entry_verified and action_type == "type_text":
                        keyboard_failed = True
                        if hasattr(self.planner, "keyboard_entry_enabled"):
                            self.planner.keyboard_entry_enabled = False
                        print("[Keyboard] Entry not confirmed; disabling keyboard shortcut for this run and using keypad fallback.")
                if action_type not in ("wait", "stop") and purpose not in ("submit", "next"):
                    submission_ready = bool(decision.get("ready_to_submit", False)) and entry_verified and changed
                if continue_target:
                    # Require the verified banner to disappear before counting progress.
                    deadline = time.monotonic() + 1.5
                    while self.observer.detect_correct_continue(post_frame) and time.monotonic() < deadline:
                        if self.executor.is_cancelled():
                            break
                        self.executor.wait(100)
                        post_frame = self.observer.capture_region(self.region)
                    if self.executor.is_cancelled():
                        break
                    changed = changed and not self.observer.detect_correct_continue(post_frame)
                    if not changed:
                        print("[Safety] Continue did not dismiss feedback; stopping without repeated clicks.")
                        self.stop("Continue click did not dismiss correct feedback; no repeat click attempted")
                # Continue may open a check during the readiness polling above.
                if continue_target and self._handle_bookwork(post_frame):
                    post_frame = self.observer.capture_region(self.region)
                    if self.executor.is_cancelled():
                        break
                    changed = changed and not self.observer.detect_correct_continue(post_frame)
                if purpose == "skip":
                    # A disappearing banner or animation does not prove navigation.
                    next_id = None
                    deadline = time.monotonic() + 2.0
                    while self.is_running and not self.executor.is_cancelled():
                        post_items = self.observer.extract_auxiliary_reference(post_frame).get("ocr_results", [])
                        next_id, _ = question_tabs(post_frame, post_items)
                        if next_id and next_id != question_id:
                            break
                        if time.monotonic() >= deadline:
                            break
                        self.executor.wait(100)
                        post_frame = self.observer.capture_region(self.region)
                    if not next_id or next_id == question_id:
                        self.stop("Needs Human intervention; next-question navigation could not be verified. Review log saved.")
                        break
                    self.skipped_question_count += 1
                    if self.auto_complete and question_id:self.review_codes.add(question_id)
                    changed = True
                    print(f"[Review] Skipped {question_id} to {next_id}; completed count unchanged.")
                if purpose in ("next", "skip", "auto_tab") and changed:
                    if purpose == "next" and submitted and correct_verified:
                        self.current_question_count += 1
                        print(f"[Progress] New correct answer this run: {self.current_question_count}/{self.max_questions}")
                    else:
                        print("[Progress] Skipped previously completed feedback; new-answer count unchanged.")
                    if self.bookwork:
                        self.bookwork.reset_question()
                    question_recovery_used = False
                    self.retries.reset()
                    self._recovery_replanned=False
                    submission_ready = False
                    submitted = feedback_seen = correct_verified = False
                    if hasattr(self.planner, "reset_question_memory"):
                        self.planner.reset_question_memory()
                    self._consecutive_submit_count = 0
                    self._last_submit_time = None
                    print(f"[Progress] Feedback observed and next click changed the page. Completed: {self.current_question_count}/{self.max_questions}")
                # Refresh grounding after layout changes or ineffective actions.
                # Small keypad edits remain visible in the image; stale OCR is never sent.
                ocr_refresh = (not changed or post_diff > 0.02 or purpose in ("submit", "next", "skip") or action_type == "scroll"
                               or (action_type == "click" and action.get("target",[0,0])[1] > frame_bgr.shape[0]*.90))
                signature = repr(sorted(action.items()))
                if action_type != "wait":
                    repeated_ineffective_actions = (
                        repeated_ineffective_actions + 1
                        if not changed and signature == last_action_signature else (0 if changed else 1)
                    )
                    last_action_signature = signature
                    stalled_cycles = 0 if changed else stalled_cycles + 1
                previous_result = f"Action '{action_type}' executed. Visual change score: {post_diff:.3f}. Expected: {expected_change}"
                previous_result += (". Visible change detected; check whether the expected UI is present." if changed else ". No change distinguishable from background activity. Do not assume success; re-locate the control.")
                if keyboard_failed:
                    previous_result += ". Keyboard entry was not confirmed. Use visible keypad clicks, not type_text or paste."
                if repeated_ineffective_actions >= 3 or stalled_cycles >= 8:
                    self.stop("No confirmed progress after repeated ineffective actions; last action: " + str(action))
                    print(f"[Safety] {self.finish_reason}")
                print(f"[Verify] Visual difference score = {post_diff:.3f}")
                consecutive_errors = 0

            except BookworkReview as e:
                self.stop("Needs Human intervention — " + str(e))
                print(f"[Bookwork] {self.finish_reason}")
                break
            except BrowserSessionError as e:
                self.stop(str(e))
                print(f"[Browser] {self.finish_reason}")
                break
            except Exception as e:
                consecutive_errors += 1
                import traceback
                print(f"[ERROR in Cycle #{frame_idx}] {e}")
                print(traceback.format_exc())
                if consecutive_errors >= 3:
                    print(f"[Safety Halt] 3 consecutive errors encountered in loop. Halting to prevent endless cycling: {e}")
                    self.stop(f"Three consecutive controller errors: {e}")
                    break
                time.sleep(1.0)

            finally:
                if frame_path and os.path.isfile(frame_path):
                    try:
                        os.remove(frame_path)
                    except OSError:
                        pass

        if not self.auto_complete and self.current_question_count >= self.max_questions and self.is_running and not self.executor.is_cancelled():
            if self.require_bookwork and not self._bookwork_finish_passed:
                try:
                    self.finish_bookwork()
                except BookworkReview as error:
                    self.stop('Needs Human intervention — ' + str(error))
            if self.is_running and not self.executor.is_cancelled():
                self.finish_reason = f"Completed {self.current_question_count}/{self.max_questions} new questions this run"
                if self.require_bookwork:
                    self.finish_reason += '; bookwork check passed'
                if self.skipped_question_count:
                    self.finish_reason += f"; {self.skipped_question_count} Needs Human intervention"
                print(f"[Controller] {self.finish_reason}")
        self.is_running = False
