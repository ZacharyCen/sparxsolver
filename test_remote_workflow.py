"""Regression checks for question memory and answer-page navigation guards."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from controller import AutoAnswerController
from planner import RemoteApiPlanner
from observer import Observer


class RemoteWorkflowTests(unittest.TestCase):
    def test_correct_footer_uses_cached_templates_after_first_small_ocr(self):
        observer = Observer.__new__(Observer)
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        frame[450:560, 50:750] = (130, 190, 30)
        cv2.putText(frame, "Correct!", (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Continue", (550, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        observer._ocr = Mock(return_value=([
            ([[150, 20], [285, 20], [285, 55], [150, 55]], "Correct!", .99),
            ([[500, 20], [650, 20], [650, 55], [500, 55]], "Continue", .99),
        ], None))
        target = observer.detect_correct_continue(frame)
        self.assertIsNotNone(target)
        self.assertEqual(observer._ocr.call_args.args[0].shape, (110, 700, 3))
        self.assertEqual(observer.detect_correct_continue(frame.copy()), target)
        self.assertEqual(observer._ocr.call_count, 1)
        self.assertIsNone(observer.detect_correct_continue(np.zeros_like(frame)))
        self.assertEqual(observer._ocr.call_count, 1)

    def test_prior_correct_feedback_is_skipped_then_three_new_answers_count(self):
        observer, planner, executor = Mock(), Mock(), Mock()
        state = {"page": "correct"}
        observer.capture_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        observer.detect_correct_continue.side_effect = lambda frame: [50, 50] if state["page"] == "correct" else None
        observer.compute_frame_diff.side_effect = [0, .1] * 7
        observer.extract_auxiliary_reference.return_value = {"ocr_results": []}
        planner.plan.return_value = {"page_state": "question", "action": {
            "type": "click", "purpose": "submit", "target": [50, 50]}}
        executor.is_cancelled.return_value = False
        executor.execute.side_effect = lambda action: state.update(page="question" if action["purpose"] == "next" else "correct")
        control = AutoAnswerController(observer, planner, executor, max_questions=3, waiting_time=0)
        control.start()
        self.assertEqual(control.current_question_count, 3)
        self.assertEqual(planner.plan.call_count, 3)
        self.assertEqual(observer.extract_auxiliary_reference.call_count, 3)
        self.assertEqual(sum(call.args[0]["purpose"] == "next" for call in executor.execute.call_args_list), 4)
        self.assertIn("3/3 new questions", control.finish_reason)

    def test_starting_on_answer_keypad_recovers_question_locally_once(self):
        observer, planner, executor, bookwork = Mock(), Mock(), Mock(), Mock()
        bookwork.handle.return_value = False
        bookwork.solution = None
        bookwork.code = '2A'
        observer.capture_region.return_value = np.zeros((100,100,3),np.uint8)
        observer.detect_correct_continue.return_value = None
        observer.compute_frame_diff.side_effect = [0,.1]
        def item(text, center):
            x,y=center
            return {'text':text,'center':center,'confidence':.99,'box':[[x-5,y-5],[x+5,y-5],[x+5,y+5],[x-5,y+5]]}
        observer.extract_auxiliary_reference.side_effect = [
            {'ocr_results':[item('Back',[20,80]),item('Enter number',[50,40]),item('Submitanswer',[80,80])]},
            {'ocr_results':[]},
        ]
        executor.is_cancelled.return_value=False
        planner.plan.return_value={'page_state':'question','action':{'type':'stop','reason':'Question recovered'}}
        control=AutoAnswerController(observer,planner,executor,bookwork=bookwork,waiting_time=0)
        control.start()
        self.assertEqual(executor.execute.call_count,1)
        self.assertEqual(executor.execute.call_args.args[0],{'type':'click','purpose':'answer','target':[20,80]})
        self.assertEqual(planner.plan.call_count,1)
        self.assertEqual(control.current_question_count,0)
        self.assertEqual(control.finish_reason,'Question recovered')

    def test_failed_question_recovery_cannot_cycle_back_forever(self):
        observer, planner, executor, bookwork = Mock(), Mock(), Mock(), Mock()
        bookwork.handle.return_value=False;bookwork.solution=None;bookwork.code='2A'
        observer.capture_region.return_value=np.zeros((100,100,3),np.uint8)
        observer.detect_correct_continue.return_value=None
        observer.compute_frame_diff.side_effect=[0,.1]
        observer.extract_auxiliary_reference.return_value={'ocr_results':[
            {'text':'Back','center':[20,80],'confidence':.99,'box':[[15,75],[25,75],[25,85],[15,85]]},
            {'text':'Enter number','center':[50,40]}, {'text':'Submit answer','center':[80,80]}]}
        executor.is_cancelled.return_value=False
        planner.plan.return_value={'page_state':'question','action':{'type':'click','purpose':'answer','target':[20,80]}}
        control=AutoAnswerController(observer,planner,executor,bookwork=bookwork,waiting_time=0)
        control.start()
        self.assertEqual(executor.execute.call_count,1)
        self.assertEqual(planner.plan.call_count,3)
        self.assertIn('Repeated navigation',control.finish_reason)
        self.assertEqual(control.current_question_count,0)

    def test_small_multipart_progress_is_not_compared_with_old_animation(self):
        observer, planner, executor = Mock(), Mock(), Mock()
        observer.capture_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        observer.detect_correct_continue.return_value = None
        observer.extract_auxiliary_reference.return_value = {"ocr_results": []}
        observer.compute_frame_diff.side_effect = [.009, .006] * 4
        actions = [{"page_state": "question", "action": {
            "type": "click", "purpose": "answer", "target": [20 + n, 50]}} for n in range(4)]
        planner.plan.side_effect = actions + [{"action": {"type": "stop", "reason": "Test complete"}}]
        executor.is_cancelled.return_value = False
        control = AutoAnswerController(observer, planner, executor, waiting_time=0)
        control.start()
        self.assertEqual(executor.execute.call_count, 4)
        self.assertEqual(observer.extract_auxiliary_reference.call_count, 1)
        self.assertEqual(planner.plan.call_args_list[1].kwargs["auxiliary_reference"], "")
        self.assertEqual(control.finish_reason, "Test complete")
        self.assertEqual(control.current_question_count, 0)

    def test_unchanged_feedback_is_not_clicked_repeatedly(self):
        observer, planner, executor = Mock(), Mock(), Mock()
        observer.capture_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        observer.detect_correct_continue.return_value = [50, 50]
        observer.compute_frame_diff.side_effect = [0, .1]
        executor.is_cancelled.return_value = False
        control = AutoAnswerController(observer, planner, executor, max_questions=1, waiting_time=0)
        with patch("controller.time.monotonic", side_effect=[0, 2]):
            control.start()
        self.assertEqual(control.current_question_count, 0)
        executor.execute.assert_called_once()
        planner.plan.assert_not_called()

    def test_identical_frames_reuse_ocr_but_changed_frames_refresh(self):
        observer = Observer.__new__(Observer)
        observer._ocr = Mock(return_value=(None, None))
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        observer.extract_auxiliary_reference(image)
        observer.extract_auxiliary_reference(image.copy())
        self.assertEqual(observer._ocr.call_count, 1)
        image[0, 0] = 255
        observer.extract_auxiliary_reference(image)
        self.assertEqual(observer._ocr.call_count, 2)

    def test_solution_survives_answer_page_transition(self):
        planner = RemoteApiPlanner(api_key="test", base_url="http://localhost/v1")
        decision = {
            "frame_id": "test", "page_state": "question",
            "answer_display": "25^(-6) = 5^n, so n = -12",
            "action": {"type": "click", "purpose": "answer", "target": [50, 50]},
            "expected_change": "Answer page opens",
        }
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(decision)))])
        planner.client = client
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "frame.jpg"
            Image.new("RGB", (100, 100)).save(image)
            planner.plan(str(image), "Question 1", "")
            planner.plan(str(image), "Question 1", "")
        request = client.chat.completions.create.call_args.kwargs
        self.assertIn("n = -12", request["messages"][0]["content"][0]["text"])
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertLess(len(planner.recent_steps), 3)
        self.assertNotIn("description", request["response_format"]["json_schema"]["schema"])
        planner.reset_question_memory()
        self.assertEqual(planner.question_memory, "")
        self.assertEqual(planner.recent_steps, [])

    def test_card_crop_maps_model_and_ocr_coordinates(self):
        planner = RemoteApiPlanner(api_key="test", base_url="http://localhost/v1")
        decision = {"frame_id": "test", "page_state": "question", "answer_display": "n=10",
                    "expected_change": "Input filled", "action": {"type": "fill", "target": [50, 50], "text": "10"}}
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(decision)))])
        planner.client = client
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frame.png"
            frame = np.zeros((700, 1000, 3), dtype=np.uint8)
            frame[100:600, 200:800] = 255
            Image.fromarray(frame).save(path)
            result = planner.plan(str(path), "Question 1", json.dumps([
                {"text": "Answer", "center": [250, 150]}, {"text": "Browser", "center": [10, 10]}]))
        self.assertEqual(result["action"]["target"], [238, 138])
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"][0]["text"]
        self.assertIn('["Answer",62,62]', prompt)
        self.assertNotIn('Browser', prompt)
        self.assertIn('624x612', prompt)

    def test_verified_final_keyboard_entry_submits_without_extra_model_call(self):
        observer, planner, executor = Mock(), Mock(), Mock()
        observer.capture_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        observer.detect_correct_continue.side_effect = [None, None, [80, 80], None, None]
        observer.detect_enabled_submit.return_value = [70, 80]
        observer.verify_numeric_entry.return_value = True
        observer.extract_auxiliary_reference.return_value = {"ocr_results": []}
        observer.compute_frame_diff.side_effect = [0, .1] * 3
        planner.plan.return_value = {"page_state": "question", "ready_to_submit": True,
                                     "action": {"type": "type_text", "target": [50, 50], "text": "12"}}
        executor.is_cancelled.return_value = False
        c = AutoAnswerController(observer, planner, executor, max_questions=1, waiting_time=0, fast_submit=True, submit_with_enter=True)
        c.start()
        planner.plan.assert_called_once()
        observer.verify_numeric_entry.assert_called_once()
        self.assertEqual(c.current_question_count, 1)
        # Unknown controls use the grounded Submit button; Enter can open a dropdown.
        self.assertEqual([call.args[0]["type"] for call in executor.execute.call_args_list], ["type_text", "click", "click"])

    def test_intermediate_digit_never_triggers_fast_submit(self):
        observer, planner, executor = Mock(), Mock(), Mock()
        observer.capture_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        observer.detect_correct_continue.return_value = None
        observer.extract_auxiliary_reference.return_value = {"ocr_results": []}
        observer.compute_frame_diff.side_effect = [0, .1]
        planner.plan.side_effect = [
            {"page_state": "question", "ready_to_submit": False,
             "action": {"type": "click", "purpose": "answer", "target": [50, 50]}},
            {"action": {"type": "stop", "reason": "Test complete"}}]
        executor.is_cancelled.return_value = False
        c = AutoAnswerController(observer, planner, executor, waiting_time=0, fast_submit=True)
        c.start()
        observer.detect_enabled_submit.assert_not_called()
        self.assertEqual(c.current_question_count, 0)

    def test_submit_detection_rejects_disabled_button(self):
        observer = Observer.__new__(Observer)
        observer._ocr = Mock()
        frame = np.full((600, 800, 3), 255, dtype=np.uint8)
        frame[530:580, 600:780] = 220
        self.assertIsNone(observer.detect_enabled_submit(frame))
        observer._ocr.assert_not_called()
        frame[530:580, 600:780] = (210, 110, 20)
        cv2.putText(frame, "Submit", (605, 565), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        observer._ocr.return_value = [([[0, 0]]*4, "Submit answer", .99)], None
        target = observer.detect_enabled_submit(frame)
        self.assertIsNotNone(target)
        self.assertEqual(observer.detect_enabled_submit(frame), target)
        self.assertEqual(observer._ocr.call_count, 1)

    def test_finish_sound_is_not_repeated_by_duplicate_finish_callbacks(self):
        from main import AutoAnswerApp
        app = Mock()
        app._is_hidden = False
        app._notified_controller = None
        app.controller = SimpleNamespace(current_question_count=3, max_questions=3, finish_reason="Completed 3/3 new questions this run")
        app.sound_var.get.return_value = True
        AutoAnswerApp._on_finish(app)
        AutoAnswerApp._on_finish(app)
        app._play_finish_sound.assert_called_once_with(True)

    def test_failed_keyboard_verification_blocks_fast_submit(self):
        observer, planner, executor = Mock(), Mock(), Mock()
        observer.capture_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        observer.detect_correct_continue.return_value = None
        observer.verify_numeric_entry.return_value = False
        observer.extract_auxiliary_reference.return_value = {"ocr_results": []}
        observer.compute_frame_diff.side_effect = [0, .1]
        planner.plan.side_effect = [
            {"page_state": "question", "ready_to_submit": True,
             "action": {"type": "type_text", "target": [50, 50], "text": "12"}},
            {"action": {"type": "stop", "reason": "Test complete"}}]
        executor.is_cancelled.return_value = False
        c = AutoAnswerController(observer, planner, executor, waiting_time=0, fast_submit=True)
        c.start()
        observer.detect_enabled_submit.assert_not_called()
        self.assertFalse(planner.keyboard_entry_enabled)
        self.assertIn("Use visible keypad clicks", planner.plan.call_args.kwargs["previous_result"])

    def test_keyboard_exception_releases_key_without_spurious_mouse_events(self):
        from executor import ActionExecutor
        e = ActionExecutor.__new__(ActionExecutor)
        e._cancelled = False
        e.keyboard_entry_enabled = True
        e.click = Mock()
        e.mouse = Mock()
        e.keyboard = Mock()
        e.keyboard.press.side_effect = RuntimeError("Input error")
        with self.assertRaises(RuntimeError):
            e.type_text([10, 10], "12")
        e.keyboard.release.assert_called_once()
        e.mouse.release.assert_not_called()

    def test_browser_actions_and_cancel_never_use_desktop_input(self):
        from executor import ActionExecutor
        e = ActionExecutor.__new__(ActionExecutor)
        e.browser_session = Mock()
        e.browser_session.image_size = (1280, 800)
        e.mouse, e.keyboard = Mock(), Mock()
        e._cancelled = False
        e.keyboard_entry_enabled = True
        for action in [
            {"type": "click", "target": [10, 20]},
            {"type": "type_text", "target": [10, 20], "text": "-12.3"},
            {"type": "fill", "target": [10, 20], "text": "hello"},
            {"type": "drag", "start": [10, 20], "end": [30, 40]},
            {"type": "scroll", "direction": "down", "steps": 1},
        ]:
            e.execute(action)
        e.cancel()
        self.assertEqual(e.mouse.mock_calls, [])
        self.assertEqual(e.keyboard.mock_calls, [])
        e.browser_session.type_text.assert_called_once()
        self.assertTrue(e.is_cancelled())

    def test_browser_capture_avoids_desktop_and_checks_tab_scope(self):
        from browser_session import BrowserSession, BrowserSessionError
        observer = Observer.__new__(Observer)
        observer.browser_session = Mock()
        observer.browser_session.capture.return_value = np.zeros((20, 40, 3), dtype=np.uint8)
        self.assertEqual(observer.capture_region({}).shape, (20, 40, 3))
        session = BrowserSession(1)
        session.target_id = "homework"
        session._http = Mock(return_value=[{"id": "homework", "url": "https://example.org/"}])
        with self.assertRaises(BrowserSessionError):
            session.ensure_homework()
        self.assertTrue(session._allowed("https://maths.sparx-learning.com/student/"))
        self.assertFalse(session._allowed("https://sparx-learning.com.example.org/"))

    def test_browser_mouse_maps_screenshot_pixels_to_css(self):
        from browser_session import BrowserSession
        session = BrowserSession(1)
        session.scale = (2, 3)
        session.ensure_homework = Mock()
        session.command = Mock()
        session.click([200, 300])
        params = session.command.call_args_list[0].args[1]
        self.assertEqual((params["x"], params["y"]), (100, 100))
        self.assertIsNone(session.pressed_point)

    def test_start_does_not_require_a_manual_question_limit(self):
        from main import AutoAnswerApp
        app=Mock();app._is_running=False;app.browser_var.get.return_value=True
        app.browser_session=None;app._opening_browser=False
        AutoAnswerApp._on_start(app)
        app._open_background_browser.assert_called_once()
        app.limit_entry.get.assert_not_called()

    def test_replacement_key_is_loaded_and_legacy_key_is_ignored(self):
        import os
        from planner import load_api_config
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            root = Path(folder)
            (root / "API KEY.txt").write_text("API key: legacy-secret\nBase URL: http://legacy.invalid/v1")
            with patch("planner.__file__", str(root / "planner.py")), patch("planner.os.getcwd", return_value=folder):
                self.assertEqual(load_api_config()["api_key"], "")
                key = root / "OpenAI-KEY.txt"
                key.write_text("sk-replacement-test-key")
                config = load_api_config()
                self.assertEqual(config["api_key"], "sk-replacement-test-key")
                self.assertEqual(config["base_url"], "https://api.openai.com/v1")
                key.write_text('API_KEY=sk-labeled-test-key\nBASE_URL=https://example.org/v1')
                self.assertEqual(load_api_config()["api_key"], "sk-labeled-test-key")

    def test_usage_is_recorded_even_when_response_json_is_invalid(self):
        planner = RemoteApiPlanner(api_key="test", base_url="http://localhost/v1")
        planner.client = Mock()
        planner.usage_callback = Mock()
        planner.client.chat.completions.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 12}),
            choices=[SimpleNamespace(message=SimpleNamespace(content="invalid json"))])
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "frame.png"
            Image.new("RGB", (100, 100)).save(image)
            with self.assertRaises(json.JSONDecodeError):
                planner.plan(str(image), "test", "")
        self.assertEqual(planner.total_usage, {"input_tokens": 100, "output_tokens": 12, "requests": 1})
        record = planner.usage_callback.call_args.args[0]
        self.assertTrue(record["available"])
        self.assertEqual(record["input_tokens"], 100)

    def test_usage_meter_distinguishes_missing_usage_and_resets_pending_data(self):
        import queue
        from solver_ui import SolverUI
        ui = SolverUI()
        ui._token_totals = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "unreported": 0}
        ui._usage_rows = []
        ui._refresh_usage = Mock()
        ui._is_running = False
        ui._usage_queue = queue.SimpleQueue()
        ui._record_usage({"available": True, "input_tokens": 25, "output_tokens": 3})
        ui._record_usage({"available": True, "input_tokens": 25, "output_tokens": 3})
        ui._record_usage({"available": False, "input_tokens": 0, "output_tokens": 0})
        self.assertEqual(ui._token_totals, {"input_tokens": 50, "output_tokens": 6, "requests": 3, "unreported": 1})
        ui._usage_queue.put({"available": True, "input_tokens": 9, "output_tokens": 1})
        ui._reset_usage()
        self.assertEqual(ui._token_totals["requests"], 0)
        self.assertTrue(ui._usage_queue.empty())

    def test_solver_start_connects_live_usage_to_the_display_queue(self):
        import queue
        from main import AutoAnswerApp
        app = Mock()
        app._is_running = False
        app.limit_entry.get.return_value = "3"
        app.browser_var.get.return_value = False
        app.selected_hwnd = None
        app.winfo_rootx.return_value = 200
        app.winfo_rooty.return_value = 200
        app.winfo_width.return_value = 100
        app.winfo_height.return_value = 100
        app.selected_region = {"left": 0, "top": 0, "width": 100, "height": 100}
        app.mode_var.get.return_value = "remote"
        app.base_url_entry.get.return_value = "https://api.openai.com/v1"
        app.model_entry.get.return_value = "configured-model"
        app._usage_queue = queue.SimpleQueue()
        with patch("main.RemoteApiPlanner") as planner_type, patch("main.AutoAnswerController"), patch("main.threading.Thread"):
            AutoAnswerApp._on_start(app)
            planner_type.assert_called_once_with(base_url="https://api.openai.com/v1", model="configured-model")
            record = {"available": True, "input_tokens": 10, "output_tokens": 4}
            planner_type.return_value.usage_callback(record)
            self.assertEqual(app._usage_queue.get_nowait(), record)

    def test_pinned_foreground_panel_cannot_click_through_itself(self):
        from main import AutoAnswerApp
        app = Mock()
        app._is_running = False
        app.limit_entry.get.return_value = "3"
        app.browser_var.get.return_value = False
        app.selected_hwnd = None
        app.selected_region = {"left": 0, "top": 0, "width": 100, "height": 100}
        app.winfo_rootx.return_value = 50
        app.winfo_rooty.return_value = 50
        app.winfo_width.return_value = 100
        app.winfo_height.return_value = 100
        AutoAnswerApp._on_start(app)
        self.assertIn("outside the homework area", app.status_label.configure.call_args.kwargs["text"])
        app._set_working_visuals.assert_not_called()

    def test_browser_timeout_has_a_deadline_and_discards_failed_socket(self):
        from browser_session import BrowserSession, BrowserSessionError
        session = BrowserSession(1)
        socket = Mock()
        socket.recv.return_value = json.dumps({"method": "Page.loadEventFired"})
        session.socket = socket
        with patch("browser_session.time.monotonic", side_effect=[0, .1, 11]):
            with self.assertRaises(BrowserSessionError) as error:
                session.command("Page.captureScreenshot")
        self.assertIn("capturing its page", str(error.exception))
        self.assertIn("Use existing Chrome window", str(error.exception))
        self.assertIsNone(session.socket)
        socket.close.assert_called_once()

    def test_existing_window_choice_is_explicit_and_clears_background_input(self):
        from main import AutoAnswerApp
        app = Mock()
        app._is_running = False
        app._opening_browser = False
        AutoAnswerApp._choose_existing_window(app)
        app.browser_var.set.assert_called_once_with(False)
        self.assertIsNone(app.executor.browser_session)
        self.assertIsNone(app.observer.browser_session)
        app._open_window_selector.assert_called_once()

    def test_back_navigation_rejected_even_when_labeled_answer(self):
        observer, planner, executor = Mock(), Mock(), Mock()
        observer.detect_correct_continue.return_value = None
        observer.capture_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        observer.extract_auxiliary_reference.return_value = {
            "ocr_results": [{"text": "Back", "box": [[10, 40], [60, 40], [60, 60], [10, 60]]}]}
        planner.plan.return_value = {
            "page_state": "question", "action": {
                "type": "click", "purpose": "answer", "target": [30, 50]}}
        executor.is_cancelled.return_value = False
        controller = AutoAnswerController(observer, planner, executor, waiting_time=0)
        controller.start()
        executor.execute.assert_not_called()
        self.assertEqual(planner.plan.call_count, 3)
        self.assertEqual(controller.current_question_count, 0)


if __name__ == "__main__":
    unittest.main()
