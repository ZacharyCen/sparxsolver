"""Saved multi-select choices are reconciled locally and never submitted partially."""
import unittest
from unittest.mock import Mock
import numpy as np
from observer import Observer
from controller import AutoAnswerController


class ChoiceTests(unittest.TestCase):
    def test_remove_extra_then_add_missing_then_confirm_complete(self):
        obs=Observer.__new__(Observer)
        choices={'A':{'target':[100,300],'selected':True},'B':{'target':[200,300],'selected':False},'C':{'target':[300,300],'selected':True}}
        obs.choice_buttons=Mock(return_value=choices)
        result=obs.letter_choice(None,'A and B')
        self.assertEqual(result['target'],[300,300]);self.assertFalse(result['last_change'])
        choices['C']['selected']=False
        result=obs.letter_choice(None,'A,B')
        self.assertEqual(result['target'],[200,300]);self.assertTrue(result['last_change'])
        choices['B']['selected']=True
        self.assertEqual(obs.letter_choice(None,'A,B'),{'complete':True})
        self.assertIsNone(obs.letter_choice(None,'A,A'))
        self.assertIsNone(obs.letter_choice(None,'A or B'))

    def test_all_choices_and_submission_need_no_additional_model_calls(self):
        obs=Mock(spec=Observer);planner=Mock();executor=Mock();book=Mock()
        book.solution=None;book.visual_parts=[{'answer':'A,B'}];book.handle.return_value=False
        book.prepare_submission.return_value=None
        frame=np.full((600,800,3),255,np.uint8)
        state={'A':False,'B':False,'graded':False}
        def choice(frame,answer):
            if not state['A']:return {'complete':False,'target':[200,300],'last_change':False}
            if not state['B']:return {'complete':False,'target':[400,300],'last_change':True}
            return {'complete':True}
        obs.letter_choice.side_effect=choice
        obs.capture_region.return_value=frame
        obs.detect_correct_continue.side_effect=lambda frame:[700,550] if state['graded'] else None
        obs.detect_enabled_submit.side_effect=lambda frame:[700,550] if state['A'] and not state['graded'] else None
        obs.compute_frame_diff.return_value=.1
        obs.footer_navigation_items.return_value=[]
        executor.is_cancelled.return_value=False
        def execute(action):
            if action['type']=='press_enter':
                self.assertTrue(state['A'] and state['B']);state['graded']=True
            elif action['target']==[200,300]:state['A']=True
            elif action['target']==[400,300]:state['B']=True
            elif action.get('purpose')=='next':state['graded']=False
        executor.execute.side_effect=execute
        control=AutoAnswerController(obs,planner,executor,max_questions=1,waiting_time=0,fast_submit=True,submit_with_enter=True,bookwork=book)
        control.start()
        planner.plan.assert_not_called()
        obs.extract_auxiliary_reference.assert_not_called()
        self.assertEqual(control.current_question_count,1)
        self.assertEqual([call.args[0]['type'] for call in executor.execute.call_args_list],['click','click','press_enter','click'])

if __name__=='__main__':unittest.main()
