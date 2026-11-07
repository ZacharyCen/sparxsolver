"""Animated dropdown loops and manual repair must preserve the live run."""
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from controller import AutoAnswerController
from executor import ActionExecutor, ExecutionPaused
from retry_policy import InteractionGuard


class InteractionTests(unittest.TestCase):
    def test_small_coordinate_jitter_does_not_hide_a_click_loop(self):
        guard = InteractionGuard()
        offsets = [0,9,-7,12,-11,4,3,-8]
        results = [guard.observe({'type':'click','target':[100+(i%2)*200+offset,300]})
                   for i,offset in enumerate(offsets)]
        self.assertIsNotNone(results[-1])

    def test_alternating_dropdown_coordinates_are_bounded_without_ocr(self):
        guard = InteractionGuard()
        results = [guard.observe({'type':'click','target':[100+i%2*200,300]}) for i in range(8)]
        self.assertFalse(any(results[:-1]))
        self.assertIn('four times', results[-1])

    def test_distinct_numeric_entries_are_progress_not_a_repeat(self):
        guard = InteractionGuard()
        for digit in range(10):
            self.assertIsNone(guard.observe({'type':'type_text','target':[100,300],'text':str(digit)}))

    def fixture(self):
        obs, planner, executor = Mock(), Mock(), Mock()
        executor.is_cancelled.return_value = False
        obs.capture_region.return_value = np.full((600,800,3),255,np.uint8)
        obs.detect_correct_continue.return_value = None
        obs.compute_frame_diff.return_value = .1  # Every dropdown animates.
        obs.extract_auxiliary_reference.return_value = {'ocr_results':[]}
        return AutoAnswerController(obs,planner,executor,max_questions=10,waiting_time=0), planner, executor

    def test_animated_loop_replans_twice_then_pauses_instead_of_stopping(self):
        control, planner, executor = self.fixture()
        count = 0
        def plan(**kwargs):
            nonlocal count
            count += 1
            return {'action':{'type':'click','purpose':'answer','target':[100+count%2*200,300]}}
        planner.plan.side_effect = plan
        paused = threading.Event()
        real_pause = control.pause
        def pause(reason):
            real_pause(reason)
            paused.set()
        control.pause = pause
        control.current_question_count = 1
        worker = threading.Thread(target=control.start)
        worker.start()
        try:
            self.assertTrue(paused.wait(4))
            self.assertTrue(control.is_running)
            self.assertTrue(control.is_paused)
            self.assertEqual(control.current_question_count,1)
            self.assertEqual(control._entry_replans['current'],2)
            self.assertLessEqual(executor.execute.call_count,21)
            executor.cancel.assert_not_called()
        finally:
            control.stop()
            worker.join(3)
        self.assertFalse(worker.is_alive())

    def test_pause_resume_invalidates_in_flight_decision_and_keeps_pending_grade(self):
        control, planner, executor = self.fixture()
        control.retries.pending = True
        count = 0
        def plan(**kwargs):
            nonlocal count
            count += 1
            if count == 1:
                control.pause()
                control.resume()
            else:
                control.stop()
            return {'action':{'type':'click','purpose':'answer','target':[100,300]}}
        planner.plan.side_effect = plan
        control.start()
        executor.execute.assert_not_called()
        self.assertTrue(control.retries.pending)

    def test_paused_executor_does_not_dispatch_or_cancel_the_run(self):
        ex = ActionExecutor.__new__(ActionExecutor)
        ex._cancelled = False
        ex.click = Mock()
        ex.pause()
        with self.assertRaises(ExecutionPaused):
            ex.execute({'type':'click','target':[100,300]})
        ex.click.assert_not_called()
        self.assertFalse(ex.is_cancelled())
        ex.resume()
        ex.execute({'type':'click','target':[100,300]})
        ex.click.assert_called_once()

    def test_watchdog_records_hang_once_per_unresponsive_interval(self):
        from runtime_log import InterfaceWatchdog
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            watch = InterfaceWatchdog(Mock(),Path(folder)/'session.log',timeout=1)
            with patch('runtime_log.time.monotonic',return_value=watch.last_beat+2), patch('runtime_log.faulthandler.dump_traceback') as dump:
                watch.check();watch.check()
                self.assertEqual(dump.call_count,1)
                watch.beat()
                watch.check()
                self.assertEqual(dump.call_count,1)


if __name__ == '__main__':
    unittest.main()
