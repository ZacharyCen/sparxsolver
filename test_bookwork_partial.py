"""Complete bookwork selection and submission despite transient OCR/rendering changes."""
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from bookwork import BookworkDetector, BookworkReview, BookworkSession, BookworkView
from homework_store import HomeworkStore


class PartialBookworkTests(unittest.TestCase):
    def run_check(self, folder, *, selected=False, loading=False, stale_control=False):
        store=HomeworkStore.create('Bookwork',folder)
        key=store.prepare('1B','5');store.set_status(key,'pending');store.set_status(key,'correct')
        executor=Mock();executor.is_cancelled.return_value=False
        session=BookworkSession(Mock(),executor,store)
        session.pending=('ordinary-answer','1C')
        state={'page':int(selected),'captures':0,'reads':0,'stale':stale_control}
        if selected:
            store.begin_check('1B');store.check_phase('selected')
        def capture():
            state['captures']+=1
            frame=np.full((800,1000,3),245,np.uint8)
            if state['page']==1:
                # Hover/selection animations change color without changing answers.
                frame[260:340,150:250]=30+state['captures']%3*8
            return frame
        def inspect(frame,force=False):
            state['reads']+=1
            if loading and state['reads']==1:
                return BookworkView(None,[])
            if state['page']==3:return None
            if state['page']==2:return BookworkView('1B',[],'accepted',[700,650])
            shift=1 if force else 0
            options=[('5',[200+shift,300]),('9' if force else '8',[500,300+shift])]
            control=[700+shift,650] if state['page']==1 else None
            if force and state['page']==1 and state['stale']:
                state['stale']=False;control=None
            return BookworkView('1B',options,control=control,
                                regions={(200+shift,300):(150,260,100,80)},
                                selected=[200+shift,300] if state['page']==1 else None)
        def execute(action):
            target=action['target']
            expected=[200,300] if state['page']==0 else [700,650]
            self.assertLessEqual(max(abs(a-b) for a,b in zip(target,expected)),1)
            self.assertEqual(action['purpose'],'bookwork')
            state['page']+=1
        executor.execute.side_effect=execute
        session.detector.inspect=Mock(side_effect=inspect)
        self.assertTrue(session.handle(capture(),capture))
        self.assertEqual(state['page'],3)
        self.assertEqual(executor.execute.call_count,2 if selected else 3)
        self.assertIsNone(store.data['bookwork_pending'])
        self.assertEqual(session.completed_checks,1)
        self.assertGreater(session.last_completed_at,0)
        self.assertEqual(session.pending,('ordinary-answer','1C'))

    def test_selection_color_and_ocr_jitter_do_not_prevent_submit(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_check(folder,stale_control=True)

    def test_late_question_code_is_read_before_any_input(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_check(folder,loading=True)

    def test_resume_visibly_selected_saved_answer_submits_without_reclicking(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_check(folder,selected=True)

    def test_resume_never_submits_a_different_selected_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            store=HomeworkStore.create('Bookwork',folder)
            key=store.prepare('1B','5');store.set_status(key,'pending');store.set_status(key,'correct')
            store.begin_check('1B');store.check_phase('selected')
            executor=Mock();executor.is_cancelled.return_value=False
            session=BookworkSession(Mock(),executor,store)
            session.detector.inspect=Mock(return_value=BookworkView(
                '1B',[('5',[200,300]),('8',[500,300])],control=[700,650],selected=[500,300]))
            with self.assertRaises(BookworkReview):
                session.handle(np.zeros((800,1000,3),np.uint8),Mock())
            executor.execute.assert_not_called()
            self.assertEqual(store.data['bookwork_pending']['phase'],'selected')

    def test_explicit_spaced_code_wins_over_answer_text_and_button_arrows(self):
        frame=np.full((800,1000,3),255,np.uint8)
        items=[{'text':text,'center':center,'confidence':.99} for text,center in (
            ('Bookwork check',[500,100]),('Bookwork 6 H',[500,220]),
            ('2A',[200,300]),('Submit >',[700,700]))]
        view=BookworkDetector.from_items(frame,items)
        self.assertEqual(view.code,'6H')
        self.assertEqual(view.control,[700,700])

    def test_code_badge_has_a_separate_read_when_full_page_ocr_omits_it(self):
        frame=np.full((800,1000,3),255,np.uint8)
        cv2.rectangle(frame,(450,240),(550,272),(230,110,20),-1)
        observer=Mock()
        observer.extract_auxiliary_reference.return_value={'ocr_results':[
            {'text':'Bookwork check','center':[500,100],'confidence':.99}]}
        badge=[([[0,0],[100,0],[100,30],[0,30]],'Bookwork 6 H',.99)]
        observer._ocr.side_effect=[(badge,None),(badge,None)]
        detector=BookworkDetector(observer)
        self.assertEqual(detector.inspect(frame).code,'6H')


if __name__=='__main__':
    unittest.main()
