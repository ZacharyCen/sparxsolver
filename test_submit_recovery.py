"""A completed entry must not be abandoned on an unsupported model blocker."""
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from controller import AutoAnswerController
from observer import Observer


class SubmitRecoveryTests(unittest.TestCase):
    def fixture(self):
        observer = Observer.__new__(Observer)
        observer.read_question_code = Mock(return_value='2A')
        observer.detect_enabled_submit = Mock(return_value=[700,550])
        frame = np.full((600,800,3),255,np.uint8)
        book = Mock()
        book.code = '2A'
        book.pending = None
        book.solution = {'question_code':'2A','parts':[{'part':'','answer':'3'}]}
        book.verified_fields.return_value = [{'value':'3'}]
        control = AutoAnswerController(observer,Mock(),Mock(),bookwork=book)
        return control,observer,book,frame

    def test_complete_readback_provides_a_real_submit_click(self):
        control,observer,book,frame = self.fixture()
        self.assertEqual(control.verified_submit_action(frame),
                         {'type':'click','purpose':'submit','target':[700,550]})

    def test_pending_grades_and_unmatched_fields_never_resubmit(self):
        control,observer,book,frame = self.fixture()
        control.retries.pending = True
        self.assertIsNone(control.verified_submit_action(frame))
        control.retries.pending = False
        book.verified_fields.return_value = None
        self.assertIsNone(control.verified_submit_action(frame))
        book.verified_fields.return_value = [{'value':'3'}]
        observer.read_question_code.return_value = '2B'
        self.assertIsNone(control.verified_submit_action(frame))

    def test_blocked_three_is_recovered_through_normal_submission_checks(self):
        control,observer,book,frame = self.fixture()
        observer.capture_region = Mock(return_value=frame)
        observer.compute_frame_diff = Mock(return_value=0)
        observer.detect_correct_continue = Mock(return_value=None)
        observer.answer_page_visible = Mock(return_value=True)
        observer.numeric_choice_cards = Mock(return_value=None)
        observer.numeric_choice_state = Mock(return_value=None)
        observer.input_context = Mock(return_value={'kind':'numeric_fields'})
        observer.extract_auxiliary_reference = Mock(return_value={'ocr_results':[]})
        observer.footer_navigation_items = Mock(return_value=[])
        observer.compress_image_same_dimension = Mock()
        book.handle.return_value = False
        book.visual_parts = None
        book.next_entry.return_value = None
        control.executor.is_cancelled.side_effect = lambda: not control.is_running
        control.executor.execute.side_effect = lambda action: control.stop('Test submit dispatched')
        control.planner.plan.return_value = {'page_state':'blocked','answer_display':'3',
                                             'action':{'type':'stop','reason':'3'}}
        control.start()
        book.prepare_submission.assert_called_once()
        control.executor.execute.assert_called_once_with({'type':'click','purpose':'submit','target':[700,550]})
        self.assertEqual(control.finish_reason,'Test submit dispatched')

    def test_short_submit_label_is_grounded_but_disabled_button_is_rejected(self):
        observer = Observer.__new__(Observer)
        observer._ocr = Mock(return_value=([([], 'Submit', .99)],None))
        frame = np.full((600,800,3),255,np.uint8)
        cv2.rectangle(frame,(620,500),(780,565),(235,130,45),-1)
        self.assertIsNotNone(observer.detect_enabled_submit(frame))
        frame[500:566,620:781] = 210
        self.assertIsNone(observer.detect_enabled_submit(frame))


if __name__ == '__main__':
    unittest.main()
