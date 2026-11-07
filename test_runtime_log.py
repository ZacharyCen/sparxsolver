"""Worker logging must not call Tk; failure diagnostics must survive the run."""
import queue
import tempfile
import threading
import unittest
from unittest.mock import Mock


class RuntimeTests(unittest.TestCase):
    def test_worker_log_queues_without_calling_tk(self):
        from main import AutoAnswerApp
        app=Mock();app._ui_log_queue=queue.SimpleQueue()
        worker=threading.Thread(target=AutoAnswerApp.log,args=(app,'Answer readback pending'))
        worker.start();worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        app.after.assert_not_called()
        self.assertIn('Answer readback pending',app._ui_log_queue.get_nowait()[1])

    def test_controller_failure_reports_reason_and_enqueues_finish_without_closing(self):
        from main import AutoAnswerApp
        app=Mock();app._ui_log_queue=queue.SimpleQueue()
        app.controller.start.side_effect=RuntimeError('readback failed')
        AutoAnswerApp._run_controller_thread(app)
        app.controller.stop.assert_called_once_with('Controller error: readback failed')
        self.assertEqual(app._ui_log_queue.get_nowait(),('finish',None))
        app.destroy.assert_not_called();app.after.assert_not_called()

    def test_diagnostics_are_local_and_redact_api_keys(self):
        import sys
        from runtime_log import start_session_log
        original,worker_original=sys.excepthook,threading.excepthook
        try:
            with tempfile.TemporaryDirectory() as folder:
                logger,path=start_session_log(folder,native_traces=False)
                logger.info('Key: sk-private-example-key')
                for handler in logger.handlers:
                    handler.flush();handler.close()
                logger.handlers.clear()
                self.assertIn('[API KEY REDACTED]',path.read_text())
                self.assertNotIn('sk-private',path.read_text())
        finally:
            sys.excepthook=original;threading.excepthook=worker_original


if __name__=='__main__':unittest.main()
