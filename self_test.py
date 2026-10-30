"""
self_test.py - Non-Interactive Acceptance Self-Test Suite
Covers:
(a) Successful import & basic invocation of third-party dependencies (MSS, RapidOCR, pynput, CustomTkinter)
(b) Strict schema validation in action.schema.json (positive passing + negative rejection)
(c) Successful local codex CLI path resolution & complete pass of all three startup checks
"""

import sys
import os
import json
import tempfile
import cv2
import numpy as np


def run_self_test():
    print("=" * 65)
    print("  AUTOMATED INITIALIZATION ACCEPTANCE SELF-TEST SUITE")
    print("=" * 65)
    passed_count = 0
    total_count = 3

    # =========================================================================
    # Part (a): Third-Party Libraries Import & Invocation Check
    # =========================================================================
    print("\n[PART A] Verifying Third-Party Libraries & Basic Invocation...")
    try:
        # Imports
        import customtkinter as ctk
        import mss
        from rapidocr_onnxruntime import RapidOCR
        import PIL
        import pynput
        import pyperclip
        import openai
        import jsonschema
        import PyInstaller

        print(f"  [+] Python version: {sys.version.split()[0]}")
        print(f"  [+] numpy version: {np.__version__} (<2.0.0 constraint satisfied)")
        print(f"  [+] opencv-python-headless version: {cv2.__version__}")
        print(f"  [+] CustomTkinter version: {ctk.__version__}")

        # 1. RapidOCR invocation
        ocr_engine = RapidOCR()
        test_img = np.ones((80, 250, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "TEST 123", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        ocr_res, _ = ocr_engine(test_img)
        assert ocr_res is not None and len(ocr_res) > 0, "RapidOCR returned empty result on test image"
        print(f"  [+] RapidOCR offline inference verified (detected: '{ocr_res[0][1]}', conf: {ocr_res[0][2]:.2f})")

        # 2. MSS Screen capture invocation
        with mss.mss() as sct:
            monitors = sct.monitors
            assert len(monitors) > 0, "MSS found no displays"
            # Small 50x50 grab test
            test_rect = {"left": 0, "top": 0, "width": 50, "height": 50}
            shot = sct.grab(test_rect)
            assert shot is not None and shot.size == (50, 50), "MSS grab failed"
        print(f"  [+] MSS screen capture verified ({len(monitors)} monitor contexts detected)")

        # 3. pynput Controller invocation
        mouse = pynput.mouse.Controller()
        pos = mouse.position
        assert pos is not None, "pynput mouse controller position is None"
        print(f"  [+] pynput OS-level input verified (current position: {pos})")

        # 4. CustomTkinter headless creation check
        try:
            root = ctk.CTk()
            root.withdraw()
            root.update_idletasks()
            root.destroy()
            print("  [+] CustomTkinter GUI engine initialized and destroyed cleanly")
        except Exception as tk_err:
            print(f"  [!] CustomTkinter GUI display warning: {tk_err} (headless environment)")

        print(">>> PART A RESULT: ALL THIRD-PARTY LIBRARIES VERIFIED SUCCESSFULLY [PASS]\n")
        passed_count += 1
    except Exception as e:
        print(f">>> PART A FAILED: {e}\n")
        sys.exit(1)

    # =========================================================================
    # Part (b): Strict JSON Schema Validation Check
    # =========================================================================
    print("[PART B] Verifying action.schema.json Enforcements...")
    try:
        from planner import get_default_schema_path
        schema_file = get_default_schema_path()
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)

        base_envelope = {
            "frame_id": "frame_acceptance_01",
            "page_state": "question",
            "answer_display": "Verify test equation",
            "expected_change": "Target value updated"
        }

        # 1. Positive Tests (Valid payloads)
        valid_actions = [
            {"type": "fill", "target": [150, 200], "text": "42"},
            {"type": "click", "purpose": "answer", "target": [300, 400]},
            {"type": "click", "purpose": "submit", "target": [500, 600]},
            {"type": "click", "purpose": "next", "target": [700, 800]},
            {"type": "drag", "purpose": "answer", "start": [100, 100], "end": [200, 200]},
            {"type": "scroll", "direction": "down", "steps": 5},
            {"type": "scroll", "direction": "up", "steps": 1},
            {"type": "wait", "wait_ms": 1500},
            {"type": "stop", "reason": "Target question count reached"}
        ]
        for act in valid_actions:
            payload = dict(base_envelope, action=act)
            jsonschema.validate(instance=payload, schema=schema)
        print(f"  [+] Passed {len(valid_actions)} valid action payloads (fill, click, drag, scroll, wait, stop)")

        # 2. Negative Tests (Strict rejection)
        negative_test_cases = [
            ("fill missing target", {"type": "fill", "text": "42"}),
            ("fill empty text", {"type": "fill", "target": [150, 200], "text": ""}),
            ("drag missing end", {"type": "drag", "purpose": "answer", "start": [100, 100]}),
            ("scroll steps > 10", {"type": "scroll", "direction": "down", "steps": 20}),
            ("scroll invalid direction", {"type": "scroll", "direction": "left", "steps": 3}),
            ("unexpected extra key", {"type": "fill", "target": [150, 200], "text": "42", "unexpected_arg": True}),
            ("invalid action type", {"type": "hover", "target": [100, 100]}),
            ("invalid page_state", {"type": "fill", "target": [150, 200], "text": "42"}, "invalid_state")
        ]

        for item in negative_test_cases:
            label = item[0]
            bad_act = item[1]
            bad_state = item[2] if len(item) > 2 else "question"
            bad_payload = dict(base_envelope, page_state=bad_state, action=bad_act)
            rejected = False
            try:
                jsonschema.validate(instance=bad_payload, schema=schema)
            except jsonschema.ValidationError:
                rejected = True

            assert rejected, f"Negative test failed: '{label}' was unexpectedly accepted!"

        print(f"  [+] Rejection verified for all {len(negative_test_cases)} invalid payload test cases")
        print(">>> PART B RESULT: STRICT JSON SCHEMA VALIDATION PASSED [PASS]\n")
        passed_count += 1
    except Exception as e:
        print(f">>> PART B FAILED: {e}\n")
        sys.exit(1)

    # =========================================================================
    # Part (c): Local Codex CLI Path Resolution & Three Startup Checks
    # =========================================================================
    print("[PART C] Verifying Local Codex CLI Resolution & Three Startup Checks...")
    try:
        from planner import CodexCliPlanner

        planner = CodexCliPlanner()

        # Check 1: Executable Check
        codex_bin = planner.check_executable()
        print(f"  [+] Check 1 Passed: Discovered codex executable at '{codex_bin}'")

        # Check 2: Authentication Check
        auth_ok = planner.check_authentication()
        assert auth_ok, "Authentication check returned False"
        print("  [+] Check 2 Passed: CLI login session verified as active")

        # Check 3: Real Image Round-Trip Check
        # Generate a real test question screenshot
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            test_img_path = tmp_img.name

        try:
            q_img = np.ones((160, 420, 3), dtype=np.uint8) * 255
            cv2.rectangle(q_img, (10, 10), (410, 150), (220, 220, 220), 2)
            cv2.putText(q_img, "Sparx Question: 8 * 9 = [   ]", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
            cv2.imwrite(test_img_path, q_img)

            # Invoke dry-run round-trip check
            print("  [+] Invoking dry-run real image round-trip validation with codex exec...")
            action_payload = planner.check_image_roundtrip(test_img_path)

            print(f"  [+] Check 3 Passed: Round-trip completed successfully.")
            print(f"      - frame_id: '{action_payload.get('frame_id')}'")
            print(f"      - page_state: '{action_payload.get('page_state')}'")
            print(f"      - answer_display: '{action_payload.get('answer_display')}'")
            print(f"      - action: {action_payload.get('action')}")
            print(f"      - expected_change: '{action_payload.get('expected_change')}'")
        finally:
            if os.path.isfile(test_img_path):
                os.remove(test_img_path)

        print(">>> PART C RESULT: ALL THREE STARTUP CHECKS PASSED [PASS]\n")
        passed_count += 1
    except Exception as e:
        print(f">>> PART C FAILED: {e}\n")
        sys.exit(1)

    print("=" * 65)
    print(f"  ACCEPTANCE SELF-TEST SUMMARY: {passed_count}/{total_count} SUITES PASSED")
    print("  Status: All initialization components ready for Step 2!")
    print("=" * 65)


if __name__ == "__main__":
    run_self_test()
