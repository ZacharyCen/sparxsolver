"""Regression checks for the observed decimal-entry and retry-loop failures."""
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from bookwork import BookworkSession, BookworkReview
from homework_store import HomeworkStore
from retry_policy import RecoveryGuard
from workings_export import WorkingsExporter, build_report, validate_workings


def multipart():
    return {'question_code': '5C', 'parts': [
        {'part': 'x1', 'answer': '1', 'workings_latex': r'5x^2-28x+23=0\implies (x-1)(5x-23)=0\implies x=1'},
        {'part': 'y1', 'answer': '5', 'workings_latex': 'y=7-2(1)=5'},
        {'part': 'x2', 'answer': '4.6', 'workings_latex': r'5x-23=0\implies x=23/5=4.6'},
        {'part': 'y2', 'answer': '-2.2', 'workings_latex': 'y=7-2(4.6)=-2.2'}]}


class RecoveryTests(unittest.TestCase):
    def test_animation_cannot_keep_waits_running_forever(self):
        guard = RecoveryGuard()
        self.assertEqual([guard.observe({'type': 'wait'}, []) for _ in range(4)], [False, False, False, True])

    def test_back_answer_oscillation_is_bounded(self):
        guard = RecoveryGuard()
        items = [{'text': 'Back', 'center': [100,500]}, {'text': 'Answer', 'center': [700,500]}]
        results = [guard.observe({'type':'click','target':[100 if i%2 else 700,500]},items) for i in range(6)]
        self.assertTrue(results[-1])
        self.assertFalse(any(results[:-1]))

    def test_minus_22_is_blocked_and_local_repair_preserves_decimal(self):
        observer = Mock()
        session = BookworkSession(observer, Mock()); session.code = '5C'
        self.assertTrue(session.accept_solution(multipart()))
        fields = [{'target': [100+i*100,300], 'value':value} for i,value in enumerate(('1','5','4.6','-22'))]
        observer.numeric_fields.return_value = fields
        self.assertIsNone(session.verified_fields(None))
        self.assertEqual(session.next_entry(None)['text'], '-2.2')
        fields[-1]['value'] = '-2.2'
        self.assertEqual(session.verified_fields(None), fields)

    def test_multipart_submission_grade_creates_markdown_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            store = HomeworkStore.create('Live regression', folder)
            observer = Mock(); session = BookworkSession(observer,Mock(),store); session.code = '5C'
            frame = np.full((600,800,3),255,np.uint8); frame[50:90,165:235]=(100,55,40)
            items = [{'text':'5C','center':[200,70],'confidence':.99}]
            observer.extract_auxiliary_reference.return_value={'ocr_results':items}
            observer._ocr.return_value=([([[180,60],[220,60],[220,80],[180,80]],'5C',.99)],None)
            observer.numeric_fields.return_value=[{'target':[100+i*100,300],'value':p['answer']} for i,p in enumerate(multipart()['parts'])]
            events=[]; exporter=WorkingsExporter(events.append)
            store.on_change=lambda snapshot:exporter.schedule(snapshot,store.path)
            session.accept_solution(multipart())
            identifier=session.prepare_submission(frame); session.dispatched(identifier)
            self.assertFalse(build_report(store.snapshot()).ready)
            session.grade(True,frame)
            exporter.close(wait=True)
            report=build_report(store.snapshot())
            self.assertEqual(report.ready,['5C'])
            path=exporter.default_path(store.snapshot(),store.path)
            self.assertIn('7-2(4.6)=-2.2',path.read_text())
            self.assertIsNone(store.verified_answer('5C'))  # Multipart is not a guessed scalar bookwork answer.
            self.assertTrue(all(not event['error'] for event in events))

    def test_keypad_recognition_includes_disabled_decimal_and_delete(self):
        from observer import Observer
        obs=Observer.__new__(Observer)
        obs._ocr=Mock(side_effect=[([([],str(n),.99)],None) for n in (7,8,9)])
        frame=np.full((1000,1400,3),255,np.uint8)
        # Pale page surrounds the keys: RETR_EXTERNAL would miss enclosed controls.
        cv2.rectangle(frame,(10,200),(1390,990),(255,240,230),3)
        for row,count in enumerate((4,4,3,2)):
            for col in range(count):
                x=700+col*120;y=400+row*70;w=100
                if row==3:
                    x=700 if col==0 else 940;w=220 if col==0 else 100
                disabled=(row,col) in ((0,3),(3,1))
                cv2.rectangle(frame,(x,y),(x+w,y+50),(255,250,247) if disabled else (248,236,223),-1)
                cv2.putText(frame,str(row*4+col),(x+25,y+32),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,120,30),1)
        keys=obs.keypad_controls(frame)
        self.assertEqual(set(keys),set('0123456789.-')|{'delete'})
        self.assertEqual(keys['.'],[990,635])

    def test_narrow_digit_one_is_read_without_confusing_an_empty_caret(self):
        from observer import Observer
        obs=Observer.__new__(Observer);obs._ocr=Mock(return_value=([],None))
        digit=np.full((80,200,3),255,np.uint8);digit[20:55,90:100]=0
        self.assertEqual(obs.read_numeric_labels(digit),['1'])
        caret=np.full((80,200,3),255,np.uint8);caret[20:55,90:92]=0
        self.assertEqual(obs.read_numeric_labels(caret),[])

    def test_rounded_quadratic_workings_are_supported(self):
        latex=r'x=\frac{-13+\sqrt{162}}{2}\approx -0.14'
        self.assertEqual(validate_workings(latex,'-0.14'),latex)

if __name__ == '__main__': unittest.main()
