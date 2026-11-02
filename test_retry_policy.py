"""Regression checks for graded failures, escalation and verified skipping."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from controller import AutoAnswerController
from planner import RemoteApiPlanner
from retry_policy import QuestionRetries, incorrect_feedback, question_tabs


class RetryTests(unittest.TestCase):
    def test_only_new_submitted_feedback_counts(self):
        retry = QuestionRetries()
        self.assertFalse(retry.observe(True))  # Pre-existing feedback.
        retry.observe(False)
        retry.submitted("x=3", "low")
        self.assertTrue(retry.observe(True))
        for _ in range(3):
            self.assertFalse(retry.observe(True))
        self.assertEqual(retry.wrong, 1)
        self.assertFalse(retry.observe(False))  # Loading/editing isn't failure.
        retry.submitted("x=4", "low")
        self.assertTrue(retry.observe(True))
        self.assertEqual(retry.wrong, 2)

    def test_question_prose_and_low_confidence_do_not_count(self):
        item = {"text": "Try again!", "confidence": .99, "center": [50, 20]}
        self.assertFalse(incorrect_feedback([item], 600))
        item["center"][1] = 550
        self.assertTrue(incorrect_feedback([item], 600))
        item["confidence"] = .5
        self.assertFalse(incorrect_feedback([item], 600))

    def run_fixture(self, navigates=True):
        observer, executor = Mock(), Mock()
        planner = SimpleNamespace(reasoning_effort="low", question_memory="rejected", recent_steps=[], retry_context="")
        state = {"page": "answer", "question": 0}
        efforts = []
        model_dismissals = []
        frame = np.full((600, 800, 3), 255, dtype=np.uint8)
        items = [{"text": "1A", "center": [200, 70]}, {"text": "1B", "center": [300, 70]}]
        def capture(region):
            result = frame.copy()
            x = 200 if state["question"] == 0 else 300
            result[50:90, x-35:x+35] = (100, 55, 40)
            return result
        def ocr(image):
            output = list(items)
            if state["page"] == "wrong":
                output.append({"text": "Incorrect!", "center": [400, 550], "confidence": .99})
                output.append({"text": "Try again", "center": [500, 550], "confidence": .99})
            return {"ocr_results": output}
        def plan(**kwargs):
            if state["question"]:
                efforts.append(planner.reasoning_effort)
                return {"action": {"type": "stop", "reason": "Next question reached"}}
            if state["page"] == "wrong":
                model_dismissals.append(True)
                return {"action": {"type": "click", "purpose": "answer", "target": [400, 500]}}
            efforts.append(planner.reasoning_effort)
            return {"action": {"type": "click", "purpose": "submit", "target": [700, 550]}}
        def execute(action):
            if action["purpose"] == "skip":
                if navigates:
                    state.update(question=1, page="answer")
            else:
                state["page"] = "wrong" if action["purpose"] == "submit" else "answer"
        def reset():
            planner.reasoning_effort = "low"
            planner.question_memory = planner.retry_context = ""
        planner.plan = plan
        planner.reset_question_memory = reset
        observer.capture_region.side_effect = capture
        observer.extract_auxiliary_reference.side_effect = ocr
        observer.detect_correct_continue.return_value = None
        observer.compute_frame_diff.side_effect = lambda a, b: .1  # Actual tab identity still checked.
        executor.is_cancelled.return_value = False
        executor.execute.side_effect = execute
        control = AutoAnswerController(observer, planner, executor, waiting_time=0)
        with tempfile.TemporaryDirectory() as folder:
            control.retries = QuestionRetries(folder)
            control.start()
            rows = [json.loads(row) for row in (Path(folder) / "questions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["wrong_answers"], 4)
            self.assertEqual(rows[0]["status"], "Needs Human intervention")
            self.assertTrue(Path(rows[0]["screenshot"]).exists())
        self.assertFalse(model_dismissals, "Verified Try again should bypass the model")
        self.assertEqual(control.current_question_count, 0)
        self.assertEqual(efforts[:4], ["low", "high", "high", "high"])
        return control, efforts

    def test_four_failures_escalate_log_skip_and_reset(self):
        control, efforts = self.run_fixture()
        self.assertEqual(control.skipped_question_count, 1)
        self.assertEqual(efforts[-1], "low")
        self.assertEqual(control.retries.wrong, 0)

    def test_animation_without_new_selected_tab_does_not_confirm_skip(self):
        control, _ = self.run_fixture(navigates=False)
        self.assertEqual(control.skipped_question_count, 0)
        self.assertIn("navigation could not be verified", control.finish_reason)

    def test_escalated_request_sends_high_effort(self):
        planner = RemoteApiPlanner(api_key="test", base_url="http://localhost/v1")
        planner.reasoning_effort = "high"
        planner.retry_context = "Two rejected submissions; re-solve."
        decision = {"frame_id": "test", "page_state": "question", "answer_display": "x=4",
                    "action": {"type": "click", "purpose": "answer", "target": [50, 50]},
                    "expected_change": "Answer opens"}
        planner.client = Mock()
        planner.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(decision)))])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frame.png"
            import cv2
            cv2.imwrite(str(path), np.full((100, 100, 3), 255, dtype=np.uint8))
            planner.plan(str(path), "Question 1", "")
        request = planner.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertIn("Two rejected", request["messages"][0]["content"][0]["text"])

    def test_remote_memory_reset_restores_low_effort(self):
        planner = RemoteApiPlanner(api_key="test", base_url="http://localhost/v1")
        planner.reasoning_effort = "high"
        planner.retry_context = "Incorrect answer"
        planner.reset_question_memory()
        self.assertEqual(planner.reasoning_effort, "low")
        self.assertEqual(planner.retry_context, "")


if __name__ == "__main__":
    unittest.main()
