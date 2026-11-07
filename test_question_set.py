"""Complete-set discovery and finishing without a manual question count."""
import unittest
import threading
from unittest.mock import Mock, patch
import numpy as np
from controller import AutoAnswerController
from question_set import read_inventory


class SetTests(unittest.TestCase):
    def test_review_question_is_not_counted_as_completed_or_sent_to_bookwork(self):
        obs,planner,executor,book=Mock(),Mock(),Mock(),Mock()
        obs.capture_region.return_value=np.zeros((800,1000,3),np.uint8)
        obs.question_inventory.return_value=[
            {'code':'3A','target':[100,100],'selected':False,'correct':True},
            {'code':'3B','target':[200,100],'selected':True,'correct':False}]
        obs._independent_learning=False
        obs.extract_auxiliary_reference.return_value={'ocr_results':[]}
        obs.detect_correct_continue.return_value=None
        obs.compute_frame_diff.return_value=0
        book.handle.return_value=False;book.pending=None;book.solution=None;book.code='3B'
        executor.is_cancelled.return_value=False
        control=AutoAnswerController(obs,planner,executor,max_questions=None,bookwork=book,require_bookwork=True)
        control.review_codes={'3B'};control.current_question_count=1
        control._entry_replans['3B']=2
        control.finish_bookwork=Mock()
        paused=threading.Event();pause=control.pause
        def record_pause(reason):pause(reason);paused.set()
        control.pause=record_pause
        worker=threading.Thread(target=control.start);worker.start()
        try:
            self.assertTrue(paused.wait(3))
            self.assertTrue(control.is_running)
            self.assertIn('3B',control.pause_reason)
            control.finish_bookwork.assert_not_called()
            planner.plan.assert_not_called()
            def retry(**kwargs):
                control.stop('Test reached retry')
                return {'action':{'type':'wait'}}
            planner.plan.side_effect=retry
            control.resume();worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertNotIn('3B',control.review_codes)
            planner.plan.assert_called_once()
        finally:
            control.stop();worker.join(3)

    def test_review_flags_do_not_make_an_unfinished_inventory_complete(self):
        obs=Mock();obs._independent_learning=False
        obs.question_inventory.return_value=[{'code':'3B','correct':False,'selected':True,'target':[100,100]}]
        control=AutoAnswerController(obs,Mock(),Mock(),max_questions=None)
        control.review_codes={'3B'}
        self.assertFalse(control.sync_question_set(None))

    def test_correct_inventory_clears_resolved_review_flags(self):
        obs=Mock();obs._independent_learning=False
        obs.question_inventory.return_value=[{'code':'3B','correct':True,'selected':True,'target':[100,100]}]
        control=AutoAnswerController(obs,Mock(),Mock(),max_questions=None)
        control.review_codes={'3B'}
        self.assertTrue(control.sync_question_set(None))
        self.assertFalse(control.review_codes)

    def test_missing_final_i_cannot_silently_shorten_the_set(self):
        frame=np.full((800,1400,3),255,np.uint8)
        items=[{'text':'1'+letter,'center':[200+i*100,150],'confidence':.99}
               for i,letter in enumerate('ABCDEFGH')]
        items.append({'text':'Summary','center':[1125,150],'confidence':.99})
        self.assertIsNone(read_inventory(frame,items))
        items.append({'text':'1I','center':[1000,150],'confidence':.99})
        self.assertEqual(len(read_inventory(frame,items)),9)

    def test_inventory_recovers_isolated_i_from_visible_slot(self):
        from observer import Observer
        frame=np.full((800,1400,3),255,np.uint8)
        items=[]
        for i,letter in enumerate('ABCDEFGHI'):
            x=150+i*100
            frame[130:180,x-45:x+45]=(248,236,220)
            if letter!='I':items.append({'text':'1'+letter,'center':[x,150],'confidence':.99})
        items.append({'text':'Summary','center':[1070,150],'confidence':.99})
        observer=Observer.__new__(Observer)
        observer._ocr=Mock(return_value=([([[0,0],[10,0],[10,10],[0,10]],'11',.95)],None))
        result=observer._read_inventory_slots(frame,items)
        self.assertEqual([item['code'] for item in result],['1'+c for c in 'ABCDEFGHI'])
        observer._ocr.assert_called_once()

    def test_failed_inventory_is_not_reused_as_a_permanent_cached_failure(self):
        from observer import Observer
        observer=Observer.__new__(Observer)
        observer._ocr=Mock(return_value=([],None))
        frame=np.full((800,1400,3),255,np.uint8)
        self.assertIsNone(observer.question_inventory(frame))
        self.assertIsNone(observer.question_inventory(frame))
        self.assertEqual(observer._ocr.call_count,2)

    def test_complete_contiguous_row_and_existing_correct_tabs(self):
        frame=np.full((800,1200,3),255,np.uint8);items=[]
        for i,letter in enumerate('ABCDEFG'):
            x=100+i*100
            frame[80:120,x-35:x+35]=(100,55,40) if i==0 else (40,180,60) if i==1 else (250,240,230)
            items.append({'text':'1'+letter,'center':[x,100],'confidence':.99})
        items.append({'text':'Summary','center':[900,100],'confidence':.99})
        result=read_inventory(frame,items)
        self.assertEqual([item['code'] for item in result],['1'+l for l in 'ABCDEFG'])
        self.assertTrue(result[0]['selected']);self.assertTrue(result[1]['correct'])
        self.assertIsNone(read_inventory(frame,items[:-1]))
        self.assertIsNone(read_inventory(frame,[i for i in items if i['text']!='1C']))

    def run_set(self,already_correct=0):
        observer,planner,executor=Mock(),Mock(),Mock()
        frame=np.full((800,1000,3),255,np.uint8)
        state={'correct':already_correct,'page':'question','current':already_correct}
        def inventory(image):
            if state['page']=='summary':return None
            return [{'code':'2'+letter,'target':[100+i*100,100],
                     'correct':i<state['correct'],'selected':i==state['current']} for i,letter in enumerate('ABCD')]
        observer.question_inventory.side_effect=inventory
        observer.capture_region.return_value=frame
        observer.compute_frame_diff.return_value=.1
        observer.detect_correct_continue.side_effect=lambda image:[800,730] if state['page']=='correct' else None
        observer.extract_auxiliary_reference.return_value={'ocr_results':[]}
        observer._independent_learning=True
        planner.plan.return_value={'page_state':'question','action':{'type':'click','purpose':'submit','target':[800,730]}}
        executor.is_cancelled.return_value=False
        def execute(action):
            if action['purpose']=='submit':state['correct']+=1;state['page']='correct'
            elif action['purpose']=='next':
                state['current']+=1;state['page']='summary' if state['current']==4 else 'question'
        executor.execute.side_effect=execute
        control=AutoAnswerController(observer,planner,executor,max_questions=None,waiting_time=0)
        control.start()
        self.assertTrue(control.finish_reason.startswith('Finished all 4 questions'))
        self.assertEqual(control.current_question_count,4-already_correct)
        self.assertEqual(planner.plan.call_count,4-already_correct)
        return control

    def test_all_four_questions_finish_without_count(self):self.run_set()
    def test_existing_correct_questions_are_not_reanswered(self):self.run_set(1)

    def test_enter_submission_does_not_use_the_mouse(self):
        from executor import ActionExecutor
        from pynput.keyboard import Key
        executor=ActionExecutor.__new__(ActionExecutor)
        executor._cancelled=False;executor.browser_session=None
        executor.keyboard=Mock();executor.mouse=Mock()
        executor.execute({'type':'press_enter','purpose':'submit'})
        executor.keyboard.press.assert_called_once_with(Key.enter)
        executor.keyboard.release.assert_called_once_with(Key.enter)
        executor.mouse.click.assert_not_called()

    def test_unreadable_set_never_guesses_a_default_limit(self):
        observer,planner,executor=Mock(),Mock(),Mock()
        observer.capture_region.return_value=np.full((800,1000,3),255,np.uint8)
        observer.question_inventory.return_value=None
        executor.is_cancelled.return_value=False
        control=AutoAnswerController(observer,planner,executor,max_questions=None,waiting_time=0)
        control.is_running=True
        control._inventory_missing_since=0
        with patch('controller.time.monotonic',return_value=9):
            self.assertFalse(control.sync_question_set(observer.capture_region.return_value))
        self.assertTrue(control.is_running)
        self.assertTrue(control.is_paused)
        self.assertIn('Could not read the complete',control.pause_reason)
        planner.plan.assert_not_called()

    def test_three_rapid_inventory_misses_do_not_end_the_run(self):
        observer,planner,executor=Mock(),Mock(),Mock()
        observer.question_inventory.return_value=None
        control=AutoAnswerController(observer,planner,executor,max_questions=None)
        control.is_running=True
        with patch('controller.time.monotonic',side_effect=[0,0,.4,.8]):
            for _ in range(3):self.assertFalse(control.sync_question_set(None))
        self.assertFalse(control.is_paused)
        self.assertTrue(control.is_running)
        executor.cancel.assert_not_called()

    def test_recovered_inventory_clears_the_startup_failure_timer(self):
        observer,planner,executor=Mock(),Mock(),Mock()
        row=[{'code':'3A','selected':True,'correct':False,'target':[100,100]}]
        observer.question_inventory.side_effect=[None,row]
        observer._independent_learning=False
        control=AutoAnswerController(observer,planner,executor,max_questions=None)
        self.assertFalse(control.sync_question_set(None))
        self.assertFalse(control.sync_question_set(None))
        self.assertIsNone(control._inventory_missing_since)
        self.assertEqual(control._inventory_misses,0)
        self.assertEqual(control.question_inventory,row)

if __name__=='__main__':unittest.main()
