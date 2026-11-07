"""Fast entry skips redundant observations while preserving grading and identity."""
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from bookwork import BookworkSession, BookworkReview
from controller import AutoAnswerController
from homework_store import HomeworkStore
from observer import Observer


class FastSubmissionTests(unittest.TestCase):
    def fixture(self, folder):
        obs=Mock(spec=Observer); ex=Mock(); planner=Mock()
        obs.read_question_code.return_value='2A'
        obs.detect_enabled_submit.return_value=[700,550]
        ex.is_cancelled.return_value=False
        store=HomeworkStore.create('Fast entry',folder)
        book=BookworkSession(obs,ex,store);book.code='2A'
        book.accept_solution({'question_code':'2A','parts':[
            {'part':'a','answer':'5'}, {'part':'b','answer':'8'}]})
        book.verified_fields=Mock(side_effect=AssertionError('No repeated full-field OCR'))
        control=AutoAnswerController(obs,planner,ex,bookwork=book,fast_submit=True)
        control.is_running=True
        return control,obs,ex,book,store

    def test_immediate_submit_saves_pending_intent_then_requires_real_grading(self):
        with tempfile.TemporaryDirectory() as folder:
            c,obs,ex,book,store=self.fixture(folder)
            self.assertTrue(c.submit_completed_entry(None,{'type':'type_text','text':'8'},True,True,0))
            ex.execute.assert_called_once_with({'type':'click','purpose':'submit','target':[700,550]})
            self.assertIsNone(store.verified_check_values('2A'))
            self.assertEqual(store.data['attempts'][-1]['status'],'pending')
            book.grade(True,np.zeros((600,800,3),np.uint8))
            self.assertEqual(store.verified_check_values('2A'),('5','8'))
            book.verified_fields.assert_not_called()

    def test_disabled_incomplete_unverified_and_pending_entries_do_not_submit(self):
        with tempfile.TemporaryDirectory() as folder:
            c,obs,ex,book,store=self.fixture(folder)
            action={'type':'type_text','text':'8'}
            self.assertFalse(c.submit_completed_entry(None,action,False,True,0))
            self.assertFalse(c.submit_completed_entry(None,action,True,False,0))
            c.fast_submit=False
            self.assertFalse(c.submit_completed_entry(None,action,True,True,0))
            c.fast_submit=True;c.retries.pending=True
            self.assertFalse(c.submit_completed_entry(None,action,True,True,0))
            c.retries.pending=False;obs.detect_enabled_submit.return_value=None
            self.assertFalse(c.submit_completed_entry(None,action,True,True,0))
            ex.execute.assert_not_called()
            self.assertEqual(store.data['attempts'],[])

    def test_changed_question_is_never_submitted(self):
        with tempfile.TemporaryDirectory() as folder:
            c,obs,ex,book,store=self.fixture(folder)
            obs.read_question_code.return_value='2B'
            with self.assertRaises(BookworkReview):
                c.submit_completed_entry(None,{'type':'type_text','text':'8'},True,True,0)
            ex.execute.assert_not_called()

    def test_last_numeric_entry_goes_directly_to_submit_after_one_capture(self):
        obs,planner,ex=Mock(),Mock(),Mock()
        state={'typed':False,'submitted':False,'graded':False,'entry_captures':0}
        frame=np.zeros((600,800,3),np.uint8)
        def capture(region):
            if state['typed'] and not state['submitted']:
                state['entry_captures']+=1
                self.assertEqual(state['entry_captures'],1)
            return frame
        def execute(action):
            if action['type']=='type_text':state['typed']=True
            elif action.get('purpose')=='submit':state['submitted']=state['graded']=True
            elif action.get('purpose')=='next':state['graded']=False
        obs.capture_region.side_effect=capture
        obs.detect_correct_continue.side_effect=lambda f:[700,550] if state['graded'] else None
        obs.detect_enabled_submit.return_value=[700,550]
        obs.verify_numeric_entry.return_value=True
        obs.extract_auxiliary_reference.return_value={'ocr_results':[]}
        obs.compute_frame_diff.return_value=.1
        planner.plan.return_value={'ready_to_submit':True,'action':{'type':'type_text','purpose':'answer','target':[100,300],'text':'8'}}
        ex.is_cancelled.return_value=False;ex.execute.side_effect=execute
        c=AutoAnswerController(obs,planner,ex,max_questions=1,waiting_time=0,fast_submit=True)
        c.start()
        self.assertEqual(state['entry_captures'],1)
        planner.plan.assert_called_once()
        self.assertEqual(c.current_question_count,1)

    def test_local_entry_marks_only_the_last_unfinished_slot_ready(self):
        obs=Mock();book=BookworkSession(obs,Mock());book.code='2A'
        book.accept_solution({'question_code':'2A','parts':[{'part':'a','answer':'5'},{'part':'b','answer':'8'}]})
        fields=[{'target':[100,300],'value':''},{'target':[300,300],'value':''}]
        obs.numeric_fields.return_value=fields
        book.next_entry(None)
        self.assertFalse(book.last_entry_completes)
        fields[0]['value']='5'
        book.next_entry(None)
        self.assertTrue(book.last_entry_completes)

    def test_unchanged_readback_is_reused_but_pixels_and_solution_invalidate_it(self):
        obs=Observer.__new__(Observer)
        obs.numeric_choice_state=Mock(return_value=None)
        obs.answer_fields=Mock(return_value=[{'value':'5'}])
        book=BookworkSession(obs,Mock());book.code='2A'
        book.accept_solution({'question_code':'2A','parts':[{'part':'','answer':'5'}]})
        frame=np.zeros((600,800,3),np.uint8)
        self.assertTrue(book.verified_fields(frame))
        self.assertTrue(book.verified_fields(frame.copy()))
        obs.answer_fields.assert_called_once()
        frame[0,0]=255
        self.assertTrue(book.verified_fields(frame))
        self.assertEqual(obs.answer_fields.call_count,2)
        book.solution['parts'][0]['answer']='6'
        self.assertIsNone(book.verified_fields(frame))
        self.assertEqual(obs.answer_fields.call_count,3)


if __name__=='__main__':unittest.main()
