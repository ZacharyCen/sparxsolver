"""Bookwork matching, per-homework persistence and isolated input regression tests."""
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from bookwork import BookworkDetector, BookworkSession, BookworkReview, BookworkView, numeric_value
from homework_store import HomeworkStore
from controller import AutoAnswerController


def fixture(selected=None):
    frame = np.full((800, 1000, 3), 245, dtype=np.uint8)
    items = [{'text': 'Bookwork check', 'center': [500, 150], 'confidence': .99},
             {'text': 'Select the answer for 1B', 'center': [500, 210], 'confidence': .99}]
    for index, text in enumerate(('5', '8', '7', '6', '4', '1')):
        x, y = 180 + index % 3 * 270, 300 + index // 3 * 130
        cv2.rectangle(frame, (x, y), (x+170, y+90), (180, 230, 200) if selected == index else (255, 255, 255), -1)
        cv2.rectangle(frame, (x, y), (x+170, y+90), (70, 80, 90), 2)
        cv2.putText(frame, text, (x+65, y+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
        items.append({'text': text, 'center': [x+85, y+45], 'confidence': .99})
    return frame, items


class StoreTests(unittest.TestCase):
    def test_exact_numbers(self):
        for text in ('1/2', '0.5', '+0.50', '.5'):
            self.assertEqual(numeric_value(text), Fraction(1, 2))
        self.assertEqual(numeric_value('−3/2'), Fraction(-3, 2))
        for text in ('1/0', '5 cm', '__import__("os")', 'NaN', '2**300', '1e20', '1' * 70):
            self.assertIsNone(numeric_value(text))

    def test_verified_records_are_isolated_and_survive_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = HomeworkStore.create('A', folder), HomeworkStore.create('B', folder)
            key = a.prepare('1B', '5')
            self.assertIsNone(a.verified_answer('1B'))
            with self.assertRaises(ValueError): a.set_status(key, 'correct')
            a.set_status(key, 'pending')
            a.set_status(key, 'correct')
            self.assertEqual(HomeworkStore(a.path).verified_answer('1B'), '5')
            self.assertIsNone(b.verified_answer('1B'))

    def test_crash_intent_is_not_an_answer_or_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            store = HomeworkStore.create('A', folder)
            store.prepare('1B', '5')
            reopened = HomeworkStore(store.path)
            self.assertEqual(reopened.unresolved_count, 1)
            self.assertIsNone(reopened.verified_answer('1B'))
            with self.assertRaises(ValueError): reopened.prepare('1B', '5')
            reopened.review_unresolved()
            self.assertEqual(reopened.unresolved_count, 0)
            self.assertIsNone(reopened.verified_answer('1B'))

    def test_storage_failure_rolls_back_promotion(self):
        with tempfile.TemporaryDirectory() as folder:
            store = HomeworkStore.create('A', folder)
            key = store.prepare('1B', '5')
            store.set_status(key, 'pending')
            with patch.object(store, '_save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError): store.set_status(key, 'correct')
            self.assertIsNone(store.verified_answer('1B'))
            self.assertEqual(store.unresolved_count, 1)

    def test_new_uncertain_attempt_does_not_reuse_an_older_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            store = HomeworkStore.create('A', folder)
            first = store.prepare('1B','5')
            store.set_status(first,'pending')
            store.set_status(first,'correct')
            store.prepare('1B','7')
            self.assertIsNone(store.verified_answer('1B'))
            store.review_unresolved()
            self.assertIsNone(store.verified_answer('1B'))

    def test_invalid_record_is_not_silently_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.json'
            path.write_text('{"version":99}')
            with self.assertRaises(ValueError): HomeworkStore(path)


class BookworkTests(unittest.TestCase):
    def test_detector_grounds_all_six_cards_and_code(self):
        frame, items = fixture()
        result = BookworkDetector.from_items(frame, items)
        self.assertEqual(result.code, '1B')
        self.assertEqual([text for text, _ in result.options], ['5', '8', '7', '6', '4', '1'])

    def test_unclear_title_is_suspected_not_normal_question(self):
        frame, items = fixture()
        view = BookworkDetector.from_items(frame, items[1:])
        self.assertIsNone(view.code)
        self.assertEqual(view.options, [])

    def setup_session(self, folder):
        observer, executor = Mock(), Mock()
        executor.is_cancelled.return_value = False
        store = HomeworkStore.create('A', folder)
        key = store.prepare('1B', '5')
        store.set_status(key, 'pending')
        store.set_status(key, 'correct')
        session = BookworkSession(observer, executor, store)
        return session, executor

    def test_option_confirmation_and_acceptance_do_not_touch_pending_question(self):
        with tempfile.TemporaryDirectory() as folder:
            session, executor = self.setup_session(folder)
            session.pending = ('ordinary-pending-id', '1C')
            frame, _ = fixture()
            chosen, _ = fixture(selected=0)
            view = BookworkView('1B', [('8', [700, 400]), ('5', [300, 400])], control=[700, 650])
            accepted = BookworkView('1B', [], 'accepted', [700, 650])
            session.detector.inspect = Mock(side_effect=[view, view, view, view, accepted, accepted, None])
            executor.execute.side_effect = lambda action: None
            capture = Mock(side_effect=[frame, chosen, chosen, chosen, chosen, chosen])
            self.assertTrue(session.handle(frame, capture))
            actions = [call.args[0] for call in executor.execute.call_args_list]
            self.assertEqual(actions[0]['target'], [300, 400])
            self.assertTrue(all(action['purpose'] == 'bookwork' for action in actions))
            self.assertEqual(session.pending, ('ordinary-pending-id', '1C'))

    def test_ambiguous_choices_do_not_click(self):
        with tempfile.TemporaryDirectory() as folder:
            session, executor = self.setup_session(folder)
            session.store.data['attempts'][-1]['answer'] = '0.5'
            session.detector.inspect = Mock(return_value=BookworkView('1B', [('1/2', [100, 300]), ('0.5', [300, 300])]))
            with self.assertRaises(BookworkReview): session.handle(np.zeros((800, 1000, 3), np.uint8), Mock())
            executor.execute.assert_not_called()

    def test_unknown_answer_does_not_guess(self):
        with tempfile.TemporaryDirectory() as folder:
            session, executor = self.setup_session(folder)
            session.detector.inspect = Mock(return_value=BookworkView('2A', [('5', [100, 300]), ('8', [300, 300])]))
            with self.assertRaises(BookworkReview): session.handle(np.zeros((800, 1000, 3), np.uint8), Mock())
            executor.execute.assert_not_called()

    def test_reordered_choices_before_input_do_not_click_stale_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            session, executor = self.setup_session(folder)
            one = BookworkView('1B', [('5', [100, 300]), ('8', [300, 300])])
            two = BookworkView('1B', [('8', [100, 300]), ('5', [300, 300])])
            session.detector.inspect = Mock(side_effect=[one, two])
            with self.assertRaises(BookworkReview): session.handle(np.zeros((800, 1000, 3), np.uint8), lambda: np.zeros((800, 1000, 3), np.uint8))
            executor.execute.assert_not_called()

    def test_disappearance_without_positive_feedback_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            session, executor = self.setup_session(folder)
            view = BookworkView('1B', [('5', [100, 300]), ('8', [300, 300])])
            session.detector.inspect = Mock(side_effect=[view, view, None])
            frame = np.zeros((800, 1000, 3), np.uint8)
            with self.assertRaises(BookworkReview): session.handle(frame, lambda: frame)
            self.assertEqual(executor.execute.call_count, 1)

    def test_solution_requires_bound_code_and_accepts_numeric_multipart(self):
        session = BookworkSession(Mock(), Mock())
        self.assertFalse(session.accept_solution({'question_code': None, 'parts': [{'part': '', 'answer': '5'}]}))
        session.code = '1B'
        self.assertTrue(session.accept_solution({'question_code': '1B', 'parts': [{'part': 'a', 'answer': '5'}, {'part': 'b', 'answer': '8'}]}))
        self.assertTrue(session.accept_solution({'question_code': '1B', 'parts': [{'part': '', 'answer': '5'}]}))

    def test_readback_uses_white_answer_field_not_a_blue_keypad(self):
        observer = Mock()
        observer._ocr.return_value = [([[5,5],[25,5],[25,30],[5,30]], '5', .99)], None
        session = BookworkSession(observer, Mock())
        frame = np.full((600,800,3), 240, np.uint8)
        cv2.rectangle(frame,(100,200),(300,270),(255,255,255),-1)
        cv2.rectangle(frame,(100,200),(300,270),(30,70,140),2)
        cv2.rectangle(frame,(500,200),(700,270),(220,190,150),-1)
        cv2.rectangle(frame,(500,200),(700,270),(30,70,140),2)
        self.assertTrue(session.verify_entry(frame,[200,235],'5'))
        self.assertFalse(session.verify_entry(frame,[600,235],'5'))

    def test_unreadable_card_does_not_allow_a_false_unique_match(self):
        with tempfile.TemporaryDirectory() as folder:
            session, executor = self.setup_session(folder)
            session.detector.inspect = Mock(return_value=BookworkView('1B',[('5',[100,300]),('8',[300,300])],complete=False))
            with self.assertRaises(BookworkReview): session.handle(np.zeros((800,1000,3),np.uint8),Mock())
            executor.execute.assert_not_called()

    def test_cancelled_check_does_not_click_or_modify_ordinary_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            session, executor = self.setup_session(folder)
            session.pending = ('ordinary', '1C')
            session.detector.inspect = Mock(return_value=BookworkView('1B',[('5',[100,300]),('8',[300,300])]))
            executor.is_cancelled.return_value = True
            self.assertTrue(session.handle(np.zeros((800,1000,3),np.uint8),Mock()))
            executor.execute.assert_not_called()
            self.assertEqual(session.pending,('ordinary','1C'))

    def test_restart_does_not_replay_interrupted_bookwork_input(self):
        with tempfile.TemporaryDirectory() as folder:
            session, executor = self.setup_session(folder)
            session.store.begin_check('1B')
            session.store = HomeworkStore(session.store.path)
            self.assertEqual(session.store.unresolved_count, 1)
            session.detector.inspect = Mock(return_value=BookworkView('1B',[('5',[100,300]),('8',[300,300])]))
            with self.assertRaises(BookworkReview): session.handle(np.zeros((800,1000,3),np.uint8),Mock())
            executor.execute.assert_not_called()
            session.store.review_unresolved()
            self.assertEqual(session.store.unresolved_count, 0)
            self.assertEqual(session.store.verified_answer('1B'),'5')

    def test_controller_intercepts_check_without_ordinary_model_or_retry(self):
        observer, planner, executor, session = Mock(), Mock(), Mock(), Mock()
        executor.is_cancelled.return_value = False
        observer.capture_region.return_value = np.zeros((100, 100, 3), np.uint8)
        session.handle.side_effect = BookworkReview('No saved answer')
        controller = AutoAnswerController(observer, planner, executor, bookwork=session)
        controller.start()
        planner.plan.assert_not_called()
        executor.execute.assert_not_called()
        self.assertEqual(controller.current_question_count, 0)
        self.assertEqual(controller.retries.wrong, 0)
        self.assertIn('No saved answer', controller.finish_reason)


if __name__ == '__main__': unittest.main()
