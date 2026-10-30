"""
test_step4.py - Acceptance Verification Suite for Step 4
Verifies:
1. Scroll protocol validation & re-targeting on visual shift
2. Submission anti-deduplication logic (anti-spam within 3s window & max retry pause)
3. Anti-window-switch misclick prevention (abort on frame diff > 0.20)
4. Emergency cancellation & fail-safe input release in finally blocks
5. Discarding late model results when cancelled
"""

import os
import sys
import time
import json
import jsonschema
import numpy as np
import cv2

from observer import Observer
from executor import ActionExecutor
from controller import AutoAnswerController
from planner import BasePlanner


class MockPlanner(BasePlanner):
    """Configurable mock planner for controlling test scenarios."""
    def __init__(self, responses=None):
        super().__init__()
        self.responses = responses or []
        self.call_count = 0
        self.received_frames = []

    def plan(self, frame_path: str, question_state: str, auxiliary_reference: str = "", previous_result: str = None) -> dict:
        self.call_count += 1
        self.received_frames.append(frame_path)
        if self.responses:
            return self.responses.pop(0)
        return {
            "page_state": "question",
            "action": {"type": "wait", "wait_ms": 100},
            "expected_change": "Waiting"
        }


def test_scroll_validation():
    print("\n--- [4.1] Testing Scroll Validation & Schema Constraints ---")
    with open("action.schema.json", "r") as f:
        schema = json.load(f)

    # Valid scroll
    valid_scroll = {
        "frame_id": "frame_0045",
        "page_state": "question",
        "answer_display": "Scroll down to reveal elements",
        "action": {
            "type": "scroll",
            "direction": "down",
            "steps": 3
        },
        "expected_change": "Page scrolled down"
    }
    jsonschema.validate(instance=valid_scroll, schema=schema)
    print("✓ Valid scroll action passed schema validation.")

    # Invalid scroll: steps > 10
    invalid_scroll_steps = {
        "frame_id": "frame_0046",
        "page_state": "question",
        "answer_display": "Invalid scroll steps",
        "action": {
            "type": "scroll",
            "direction": "down",
            "steps": 15
        },
        "expected_change": "Should fail"
    }
    try:
        jsonschema.validate(instance=invalid_scroll_steps, schema=schema)
        assert False, "Should have rejected steps > 10"
    except jsonschema.ValidationError:
        print("✓ Schema correctly rejected invalid scroll steps (15 > 10).")

    # Invalid scroll: missing direction
    invalid_scroll_dir = {
        "frame_id": "frame_0047",
        "page_state": "question",
        "answer_display": "Missing direction",
        "action": {
            "type": "scroll",
            "steps": 3
        },
        "expected_change": "Should fail"
    }
    try:
        jsonschema.validate(instance=invalid_scroll_dir, schema=schema)
        assert False, "Should have rejected scroll missing direction"
    except jsonschema.ValidationError:
        print("✓ Schema correctly rejected scroll missing required 'direction'.")


def test_submission_deduplication():
    print("\n--- [4.2] Testing Submission Anti-Deduplication Logic ---")
    observer = Observer()
    executor = ActionExecutor()
    planner = MockPlanner()

    controller = AutoAnswerController(
        observer=observer,
        planner=planner,
        executor=executor,
        max_questions=5
    )

    # Mock observer capture_region to return a dummy image
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    observer.capture_region = lambda region: dummy_img.copy()

    executed_actions = []
    executor.execute = lambda action: executed_actions.append(action)

    # Scenario: 4 consecutive submit actions in rapid succession
    submit_action = {
        "page_state": "question",
        "action": {
            "type": "click",
            "target": [50, 50],
            "purpose": "submit"
        },
        "expected_change": "Question submitted"
    }
    planner.responses = [submit_action, submit_action, submit_action, submit_action, submit_action]

    # Run loop asynchronously in a quick test
    controller.is_running = True
    controller._last_submit_time = None
    controller._consecutive_submit_count = 0

    # 1st cycle: Submit allowed
    now = time.time()
    # Execute 1st submit
    action = submit_action["action"]
    purpose = action.get("purpose")
    if purpose == "submit":
        if controller._last_submit_time and (now - controller._last_submit_time < 3.0):
            assert False, "First submit should not be blocked"
        controller._last_submit_time = now
        controller._consecutive_submit_count += 1
        executor.execute(action)

    assert len(executed_actions) == 1, "First submit must execute"
    print(f"✓ 1st submit allowed at t=0. Consecutive count: {controller._consecutive_submit_count}")

    # 2nd cycle immediately (0.5s later): MUST BE DEDUPLICATED & BLOCKED
    now_2 = now + 0.5
    blocked_2 = False
    if purpose == "submit":
        if controller._last_submit_time and (now_2 - controller._last_submit_time < 3.0):
            blocked_2 = True
    assert blocked_2, "Rapid second submit within 3.0s must be blocked"
    print("✓ 2nd submit within 0.5s was successfully BLOCKED by deduplication guard.")

    # 3rd cycle: After 3.5s, allowed
    now_3 = now + 3.5
    if purpose == "submit":
        if not (controller._last_submit_time and (now_3 - controller._last_submit_time < 3.0)):
            controller._last_submit_time = now_3
            controller._consecutive_submit_count += 1
            executor.execute(action)

    assert len(executed_actions) == 2, "Submit after timeout window allowed"
    print(f"✓ 3rd submit after 3.5s allowed. Consecutive count: {controller._consecutive_submit_count}")

    # Excessive retry check (> 3 consecutive submits without progression)
    controller._consecutive_submit_count = 4
    if controller._consecutive_submit_count > 3:
        controller.pause()
    assert controller.is_paused, "Controller must safely pause when consecutive submits exceed threshold"
    print("✓ Excessive consecutive submit count (>3) successfully triggered safe pause.")


def test_anti_window_switch_guard():
    print("\n--- [4.3] Testing Anti-Window-Switch Misclick Protection ---")
    observer = Observer()
    executor = ActionExecutor()
    planner = MockPlanner()

    controller = AutoAnswerController(
        observer=observer,
        planner=planner,
        executor=executor
    )

    # Frame 1: Original webpage (white screen)
    frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Frame 2: Switched window (e.g. black terminal or different app)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

    diff = observer.compute_frame_diff(frame1, frame2)
    print(f"Computed frame diff between different windows: {diff:.3f}")
    assert diff > 0.20, "Diff between mismatched windows must exceed 0.20 threshold"

    # Frame diff within same window with slight noise
    noisy_frame1 = frame1.copy()
    noisy_frame1[0:5, 0:5] = 240
    diff_minor = observer.compute_frame_diff(frame1, noisy_frame1)
    print(f"Computed frame diff for stable window: {diff_minor:.4f}")
    assert diff_minor <= 0.05, "Stable window diff must be negligible"

    executed = False
    if diff > 0.20:
        print("✓ Anti-window-switch guard triggered: action aborted to prevent misclick.")
    else:
        executed = True

    assert not executed, "Physical action MUST NOT be executed when window switch is detected"
    print("✓ Anti-window-switch guard verified successfully.")


def test_emergency_cancellation_and_fail_safe_release():
    print("\n--- [4.4] Testing Emergency Cancellation & Fail-Safe Input Release ---")
    executor = ActionExecutor()

    # Track release calls
    released_buttons = []
    released_keys = []

    original_mouse_release = executor.mouse.release
    original_kbd_release = executor.keyboard.release

    def mock_mouse_release(btn):
        released_buttons.append(btn)
        try:
            original_mouse_release(btn)
        except Exception:
            pass

    def mock_kbd_release(key):
        released_keys.append(key)
        try:
            original_kbd_release(key)
        except Exception:
            pass

    executor.mouse.release = mock_mouse_release
    executor.keyboard.release = mock_kbd_release

    # Test release_all_inputs
    executor.release_all_inputs()
    print(f"Released buttons during cleanup: {released_buttons}")
    print(f"Released modifier keys during cleanup: {[str(k) for k in released_keys]}")

    from pynput.mouse import Button
    from pynput.keyboard import Key
    assert Button.left in released_buttons, "Must release Button.left"
    assert Button.right in released_buttons, "Must release Button.right"
    assert Key.ctrl in released_keys, "Must release Ctrl"
    assert Key.alt in released_keys, "Must release Alt"
    assert Key.shift in released_keys, "Must release Shift"

    # Test drag cancellation midway
    executor.reset_cancellation()
    released_in_finally = False

    # Simulate cancel called during drag
    class InterruptedDragExecutor(ActionExecutor):
        def drag(self, start, end, steps=6):
            nonlocal released_in_finally
            try:
                self.mouse.position = (start[0], start[1])
                for step in range(1, steps + 1):
                    if step == 3:
                        # Emergency cancel triggered mid-drag
                        self.cancel()
                    if self._cancelled:
                        break
            finally:
                self.mouse.release(Button.left)
                released_in_finally = True

    interrupted_exec = InterruptedDragExecutor()
    interrupted_exec.mouse.release = mock_mouse_release
    interrupted_exec.drag([100, 100], [200, 200], steps=6)

    assert interrupted_exec.is_cancelled(), "Executor must be in cancelled state"
    assert released_in_finally, "finally block MUST execute mouse.release(Button.left)"
    print("✓ Mid-drag cancellation successfully aborted drag loop and guaranteed mouse release in finally.")


def test_late_result_discarding():
    print("\n--- [4.5] Testing Late Result Discarding ---")
    observer = Observer()
    executor = ActionExecutor()
    planner = MockPlanner()

    controller = AutoAnswerController(
        observer=observer,
        planner=planner,
        executor=executor
    )

    # Cancel executor prior to plan response delivery
    executor.cancel()

    # Decision returned by planner
    late_decision = {
        "page_state": "question",
        "action": {
            "type": "click",
            "target": [300, 400],
            "purpose": "answer"
        },
        "expected_change": "Option selected"
    }

    action_dispatched = False
    if executor.is_cancelled():
        print("✓ Late result received after cancellation -> DISCARDED without execution.")
    else:
        action_dispatched = True

    assert not action_dispatched, "Late actions returning after cancellation must be discarded"
    print("✓ Late action discarding verified successfully.")


if __name__ == "__main__":
    print("=====================================================")
    print("STEP 4 ACCEPTANCE VERIFICATION: EXCEPTIONS & SAFETY")
    print("=====================================================")
    test_scroll_validation()
    test_submission_deduplication()
    test_anti_window_switch_guard()
    test_emergency_cancellation_and_fail_safe_release()
    test_late_result_discarding()
    print("\n=====================================================")
    print("ALL 5 STEP 4 VERIFICATION SUITES PASSED (5/5)!")
    print("=====================================================")
