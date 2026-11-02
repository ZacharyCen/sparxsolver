"""Complete-set discovery and finishing without a manual question count."""
import unittest
from unittest.mock import Mock
import numpy as np
from controller import AutoAnswerController
from question_set import read_inventory


class SetTests(unittest.TestCase):
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
        control.start()
        self.assertIn('Could not read the complete',control.finish_reason)
        planner.plan.assert_not_called()

if __name__=='__main__':unittest.main()
