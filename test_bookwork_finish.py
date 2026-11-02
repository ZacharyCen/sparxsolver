"""Finishing-phase and whole-card bookwork regression tests."""
import itertools
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from bookwork import BookworkDetector, BookworkView, matching_answer, reliable_options
from controller import AutoAnswerController


class FinishTests(unittest.TestCase):
    def fixture(self, required=True, show_check=True):
        obs,planner,ex,book=Mock(),Mock(),Mock(),Mock()
        frame=np.full((800,1000,3),255,np.uint8)
        obs.capture_region.return_value=frame
        obs.compute_frame_diff.return_value=0
        obs.detect_correct_continue.return_value=None
        obs.extract_auxiliary_reference.return_value={'ocr_results':[{'text':'Summary','center':[800,100],'confidence':.99}]}
        ex.is_cancelled.return_value=False
        book.last_completed_at=0.;book.completed_checks=0
        state={'stage':0}
        def execute(action):state['stage']=1
        def wait(ms):
            if state['stage']:state['stage']+=1
        def handle(*args):
            if show_check and state['stage']>=3:
                book.last_completed_at=100.;book.completed_checks+=1
                return True
            return False
        ex.execute.side_effect=execute;ex.wait.side_effect=wait;book.handle.side_effect=handle
        control=AutoAnswerController(obs,planner,ex,max_questions=0,waiting_time=0,bookwork=book,require_bookwork=required)
        return control,obs,planner,ex,book

    def test_limit_does_not_stop_before_delayed_bookwork(self):
        control,obs,planner,ex,book=self.fixture()
        control.start()
        self.assertEqual(book.completed_checks,1)
        self.assertIn('bookwork check passed',control.finish_reason)
        self.assertEqual(ex.execute.call_count,1)
        planner.plan.assert_not_called()

    def test_checkbox_off_does_not_open_summary(self):
        control,obs,planner,ex,book=self.fixture(required=False)
        control.start()
        ex.execute.assert_not_called();book.handle.assert_not_called()

    def test_missing_required_check_is_not_reported_as_success(self):
        control,*_=self.fixture(show_check=False)
        with patch('controller.time.monotonic',side_effect=itertools.count()):control.start()
        self.assertIn('Required bookwork check did not appear',control.finish_reason)
        self.assertTrue(control.finish_reason.startswith('Needs Human intervention'))

    def test_cancel_during_finishing_is_preserved(self):
        control,obs,planner,ex,book=self.fixture(show_check=False)
        ex.wait.side_effect=lambda ms:control.stop()
        control.start()
        self.assertEqual(control.finish_reason,'Stopped by user')
        self.assertFalse(control._bookwork_finish_passed)

    def test_five_new_questions_are_followed_by_delayed_check(self):
        import time
        control,obs,planner,ex,book=self.fixture()
        control.max_questions=5
        state={'page':'question','submitted':0,'captures':0}
        frame=np.full((800,1000,3),255,np.uint8)
        def capture(region):
            if state['page']=='waiting':
                state['captures']+=1
                if state['captures']>=4:state['page']='bookwork'
            return frame.copy()
        obs.capture_region.side_effect=capture
        obs.compute_frame_diff.return_value=.1
        obs.detect_correct_continue.side_effect=lambda frame:[800,740] if state['page']=='correct' else None
        obs.extract_auxiliary_reference.return_value={'ocr_results':[]}
        planner.plan.return_value={'page_state':'question','action':{'type':'click','purpose':'submit','target':[800,740]}}
        def execute(action):
            if action['purpose']=='submit':
                state['submitted']+=1;state['page']='correct'
            elif action['purpose']=='next':
                state['page']='waiting' if state['submitted']==5 else 'question'
        def handle(*args):
            if state['page']=='bookwork':
                book.completed_checks=1;book.last_completed_at=time.time()+1
                state['page']='summary';return True
            return False
        ex.execute.side_effect=execute;ex.wait.side_effect=None;book.handle.side_effect=handle
        control.start()
        self.assertEqual(control.current_question_count,5)
        self.assertTrue(control._bookwork_finish_passed)
        self.assertEqual(planner.plan.call_count,5)
        self.assertIn('bookwork check passed',control.finish_reason)

    def test_model_completed_page_still_checks_bookwork(self):
        control,obs,planner,ex,book=self.fixture()
        control.max_questions=5
        planner.plan.return_value={'page_state':'completed','action':{'type':'stop','reason':'All correct'}}
        control.start()
        self.assertEqual(book.completed_checks,1)
        self.assertIn('bookwork check passed',control.finish_reason)
        self.assertEqual(planner.plan.call_count,1)


class CardTests(unittest.TestCase):
    def test_image_cards_keep_complete_answers_together(self):
        frame=np.full((1000,1400,3),255,np.uint8)
        values=[('-0.14','2.73','-12.86','-22.73'),('-1','5','-5','1'),('11','56','-2','-9'),('1','5','4.6','-2.2'),('-1.33','1.67','-2.33','-2.33')]
        for index in range(5):
            x=100+index%3*300;y=250+index//3*260
            cv2.rectangle(frame,(x,y),(x+250,y+200),(255,244,238),-1)
            cv2.rectangle(frame,(x,y),(x+250,y+200),(200,190,180),2)
            for slot in range(4):
                left=x+35+slot%2*130;top=y+35+slot//2*90
                cv2.rectangle(frame,(left,top),(left+45,top+55),(255,255,255),-1)
        reader=Mock(side_effect=[[value] for card in values for value in card])
        items=[{'text':'Bookwork check','center':[600,100],'confidence':.99}, {'text':'Bookwork5C','center':[600,180],'confidence':.99}]
        view=BookworkDetector.from_items(frame,items,numeric_reader=reader)
        self.assertEqual(view.code,'5C');self.assertTrue(view.complete)
        matches=[target for answer,target in view.options if matching_answer(answer,values[3])]
        self.assertEqual(matches,[[225,610]])

    def test_passed_screen_is_accepted_without_a_repeated_question_code(self):
        frame=np.full((800,1000,3),255,np.uint8)
        items=[{'text':'Bookwork check','center':[500,120],'confidence':.99},
               {'text':'Passed','center':[500,450],'confidence':.99},
               {'text':'Continue','center':[500,720],'confidence':.99}]
        view=BookworkDetector.from_items(frame,items)
        self.assertEqual(view.status,'accepted')
        self.assertIsNone(view.code)
        self.assertEqual(view.control,[500,720])

    def test_summary_footer_is_local_and_cached(self):
        from observer import Observer
        obs=Observer.__new__(Observer)
        frame=np.full((800,1000,3),255,np.uint8);frame[675:795,100:980]=(245,230,215)
        obs._ocr=Mock(return_value=([([[50,15],[130,15],[130,45],[50,45]],'Correct!',.99),
                                    ([[600,15],[720,15],[720,45],[600,45]],'Summary',.99)],None))
        target=obs.footer_continue(frame)
        self.assertEqual(target,[760,702])
        self.assertEqual(obs.footer_continue(frame.copy()),target)
        self.assertEqual(obs._ocr.call_count,1)

    def test_windows_cleanup_does_not_release_idle_right_button(self):
        from executor import ActionExecutor
        from types import SimpleNamespace
        executor=ActionExecutor.__new__(ActionExecutor);executor.browser_session=None
        executor.mouse=Mock();executor.keyboard=Mock()
        user32=SimpleNamespace(GetAsyncKeyState=Mock(return_value=0))
        with patch('executor.sys.platform','win32'),patch('ctypes.windll',SimpleNamespace(user32=user32),create=True):
            executor.release_all_inputs()
            executor.mouse.release.assert_not_called()
            user32.GetAsyncKeyState.return_value=0x8000
            executor.release_all_inputs()
            self.assertEqual(executor.mouse.release.call_count,2)

    def test_root_pair_order_is_irrelevant_but_coordinates_are_not(self):
        expected=('1','5','4.6','-2.2')
        self.assertTrue(matching_answer(('4.6','-2.2','1','5'),expected))
        self.assertFalse(matching_answer(('5','1','4.6','-2.2'),expected))

    def test_unreadable_distractor_must_be_provably_different(self):
        expected=('1','5','4.6','-2.2')
        view=BookworkView('5C',[(('11','56','-2',None),[100,200]),(expected,[400,200])],complete=False)
        self.assertTrue(reliable_options(view,expected))
        view.options[0]=(('1','5','4.6',None),[100,200])
        self.assertFalse(reliable_options(view,expected))

if __name__=='__main__':unittest.main()
