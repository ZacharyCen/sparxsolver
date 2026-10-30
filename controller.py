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

from retry_policy import QuestionRetries, incorrect_feedback, question_tabs
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
        fast_submit: bool = False
    ):
        self.observer = observer
        self.planner = planner
        self.executor = executor
        self.max_questions = max_questions
        self.waiting_time = max(0.0, min(5.0, waiting_time))
        self.fast_submit = fast_submit
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
        ocr_refresh = True
        last_action_signature = None
        repeated_ineffective_actions = 0

        while self.is_running and self.current_question_count < self.max_questions:
            if self.is_paused or self.executor.is_cancelled():
                time.sleep(0.2)
                continue

            frame_idx += 1
            print(f"\n==================================================")
            print(f"[Cycle #{frame_idx}] Question {self.current_question_count + 1}/{self.max_questions}")
            print(f"==================================================")

            frame_path = ""
            try:
                # Dynamically update bounds if tracking a target window
                if self.target_hwnd:
                    cur_bounds = get_window_bounds(self.target_hwnd)
                    if cur_bounds:
                        self.set_region(cur_bounds)

                # 1. OBSERVE
                print(f"[Observe] Capturing ROI ({self.region['left']},{self.region['top']}) {self.region['width']}x{self.region['height']}...")
                frame_bgr = self.observer.capture_region(self.region)
                if getattr(self.executor, "browser_session", None) and isinstance(self.executor, ActionExecutor):
                    self.set_region({"left": 0, "top": 0, "width": frame_bgr.shape[1], "height": frame_bgr.shape[0]})

                # Correct feedback is local UI navigation: bypass full OCR and the model.
                continue_target = self.observer.detect_correct_continue(frame_bgr)
                submit_target = self.observer.detect_enabled_submit(frame_bgr) if self.fast_submit and submission_ready else None
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
                    feedback_seen = correct_verified = True
                    aux_data = {}
                    ocr_items = []
                    decision = {
                        "page_state": "feedback",
                        "action": {"type": "click", "purpose": "next", "target": continue_target},
                        "expected_change": "Correct feedback closes and next question appears",
                    }
                    print("[Fast path] Correct + Continue verified locally; skipping full OCR and model request.")
                elif submit_target:
                    ocr_items = []
                    decision = {"page_state": "question", "action": {"type": "click", "purpose": "submit", "target": submit_target},
                                "expected_change": "Grade the completed answer"}
                    print("[Fast path] Final entry confirmed and Submit enabled; skipping model request.")
                else:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_f:
                        frame_path = tmp_f.name
                    self.observer.compress_image_same_dimension(frame_bgr, quality=85, output_path=frame_path)
                    ocr_started = time.monotonic()
                    aux_data = self.observer.extract_auxiliary_reference(frame_bgr) if ocr_refresh or self.retries.pending or self.retries.feedback_latched else {}
                    print(f"[Timing] OCR: {time.monotonic() - ocr_started:.2f}s ({'refresh' if ocr_refresh else 'skipped; current image used'})")
                    aux_table = json.dumps([
                        {"text": item["text"], "center": item["center"]}
                        for item in aux_data.get("ocr_results", [])
                        if item.get("confidence", 1.0) >= 0.6 and "center" in item
                    ], ensure_ascii=False, separators=(",", ":")) if aux_data else ""
                    ocr_items = aux_data.get("ocr_results", [])
                    print(f"[Observer] Frame captured: {frame_bgr.shape[1]}x{frame_bgr.shape[0]} | OCR items: {len(ocr_items)}")
                    wrong_visible = incorrect_feedback(ocr_items, frame_bgr.shape[0])
                    if self.retries.observe(wrong_visible):
                        self._last_submit_time = None
                        self._consecutive_submit_count = 0
                        submission_ready = False
                        feedback_seen = correct_verified = False
                        rejected = getattr(self.planner, "question_memory", "")
                        if hasattr(self.planner, "question_memory"):
                            self.planner.question_memory = ""
                            self.planner.recent_steps = []
                            self.planner.reasoning_effort = "high" if self.retries.wrong >= 2 else "low"
                            self.planner.retry_context = (f"{self.retries.wrong} confirmed incorrect submissions on this question. "
                                f"Rejected solution: {rejected}. Re-read the original question and solve independently; "
                                "dismiss Try again if needed, then use Back/Question ONLY to recover question text for this retry. "
                                "Never advance from incorrect feedback. Check all parts and units before submitting.")
                        previous_result = f"Incorrect answer confirmed ({self.retries.wrong}/4). Re-solve this question."
                        print(f"[Retry] {self.retries.wrong}/4 incorrect answers; reasoning: {'high' if self.retries.wrong >= 2 else 'low'}.")
                        if self.retries.wrong >= 4:
                            ocr_refresh = True
                            continue
                    question_state = f"Question after {self.current_question_count} correct, {self.skipped_question_count} needing human review"
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

                # Do not let navigation masquerade as answer entry.
                target = action.get("target")
                forbidden_navigation = False
                if target and not submitted and not self.retries.wrong:
                    for item in ocr_items:
                        text = item.get("text", "").strip().lower()
                        if text in ("question", "back", "back to question", "previous", "< previous", "view question"):
                            points = item.get("box", [])
                            if points:
                                xs, ys = zip(*points)
                                if min(xs)-12 <= target[0] <= max(xs)+12 and min(ys)-12 <= target[1] <= max(ys)+12:
                                    forbidden_navigation = True
                if forbidden_navigation:
                    previous_result = "Navigation rejected: target is the Question/Previous button, not an answer input. Use the stored solution and click the answer slot or keypad."
                    stalled_cycles += 1
                    if stalled_cycles >= 3:
                        self.stop("Repeated navigation to Back/Question instead of answer controls")
                    continue

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

                # 4. EXECUTE
                if self.target_hwnd:
                    activate_target_window(self.target_hwnd)
                print(f"[Execute] Executing action '{action_type}'...")
                if action_type == "wait":
                    self.executor.wait(int(self.waiting_time * 1000))
                else:
                    self.executor.execute(action)
                if self.executor.is_cancelled():
                    break
                if purpose == "submit":
                    self._last_submit_time = time.time()
                    self._consecutive_submit_count += 1
                    self.retries.submitted(getattr(self.planner, "question_memory", ""), getattr(self.planner, "reasoning_effort", "low"))
                    submitted = True
                    submission_ready = False

                # 5. VERIFY
                if action_type != "wait":
                    self.executor.wait(int(self.waiting_time * 1000))
                post_frame = self.observer.capture_region(self.region)
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
                entry_verified = True
                if action_type == "type_text":
                    entry_verified = self.observer.verify_numeric_entry(post_frame, action["target"], action["text"])
                    if not entry_verified:
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
                    changed = True
                    print(f"[Review] Skipped {question_id} to {next_id}; completed count unchanged.")
                if purpose in ("next", "skip") and changed:
                    if purpose == "next" and submitted and correct_verified:
                        self.current_question_count += 1
                        print(f"[Progress] New correct answer this run: {self.current_question_count}/{self.max_questions}")
                    else:
                        print("[Progress] Skipped previously completed feedback; new-answer count unchanged.")
                    self.retries.reset()
                    submission_ready = False
                    submitted = feedback_seen = correct_verified = False
                    if hasattr(self.planner, "reset_question_memory"):
                        self.planner.reset_question_memory()
                    self._consecutive_submit_count = 0
                    self._last_submit_time = None
                    print(f"[Progress] Feedback observed and next click changed the page. Completed: {self.current_question_count}/{self.max_questions}")
                # Refresh grounding after layout changes or ineffective actions.
                # Small keypad edits remain visible in the image; stale OCR is never sent.
                ocr_refresh = not changed or post_diff > 0.02 or purpose in ("submit", "next", "skip") or action_type == "scroll"
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

        if self.current_question_count >= self.max_questions:
            self.finish_reason = f"Completed {self.current_question_count}/{self.max_questions} new questions this run"
            if self.skipped_question_count:
                self.finish_reason += f"; {self.skipped_question_count} Needs Human intervention"
            print(f"[Controller] {self.finish_reason}")
        self.is_running = False
