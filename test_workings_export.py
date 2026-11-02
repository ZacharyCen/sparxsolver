"""Word-free numeric workings, snapshot provenance and failure-safe local exports."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from homework_store import HomeworkStore
from workings_export import WorkingsError, WorkingsExporter, atomic_write, build_report, validate_workings


def solution(code='1B', answer='5', latex=r'x+2=7\implies x=7-2=5'):
    return {'question_code': code, 'parts': [{'part': '', 'answer': answer, 'workings_latex': latex}]}


def correct(store, value=None):
    value = value or solution()
    code = value['question_code']
    store.note_question(code)
    store.save_proposal(code, value['parts'])
    identifier = store.prepare(code, value['parts'][0]['answer'], solution=value)
    store.set_status(identifier, 'pending')
    store.set_status(identifier, 'correct')
    return identifier


class WorkingsTests(unittest.TestCase):
    def test_disabled_workings_keep_verified_bookwork_values(self):
        from bookwork import BookworkSession
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            store=HomeworkStore.create('No workings',directory)
            session=BookworkSession(Mock(),Mock(),store,collect_workings=False);session.code='1B'
            original=solution()
            self.assertTrue(session.accept_solution(original))
            self.assertIsNone(session.solution['parts'][0]['workings_latex'])
            self.assertIsNotNone(original['parts'][0]['workings_latex'])
            key=store.prepare('1B','5',solution=session.solution)
            store.set_status(key,'pending');store.set_status(key,'correct')
            self.assertEqual(store.verified_values('1B'),('5',))
            self.assertFalse(build_report(store.snapshot()).ready)

    def test_disabled_workings_detach_automatic_exports(self):
        from solver_ui import SolverUI
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            store=HomeworkStore.create('Off',directory)
            ui=Mock();ui.homework_store=store;ui._workings_store=None
            ui.workings_var.get.return_value=False
            SolverUI._watch_workings_store(ui)
            self.assertIsNone(store.on_change)
            ui._workings_exporter.schedule.assert_not_called()
            ui.workings_var.get.return_value=True
            SolverUI._watch_workings_store(ui)
            ui._workings_exporter.schedule.assert_called_once()
            ui.workings_var.get.return_value=False
            SolverUI._watch_workings_store(ui)
            self.assertIsNone(store.on_change)

    def test_verified_geometry_diagrams_export_with_portable_assets(self):
        import numpy as np
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            store=HomeworkStore.create('Geometry',directory)
            image=np.full((200,200,3),255,np.uint8)
            cv2.line(image,(40,100),(160,100),(20,20,240),10)
            identifier=store.prepare_visual('2A',image)
            self.assertFalse(build_report(store.snapshot()).ready)
            store.set_status(identifier,'pending');store.set_status(identifier,'correct')
            report=build_report(store.snapshot())
            self.assertEqual(report.ready,['2A'])
            events=[];exporter=WorkingsExporter(events.append)
            output=Path(directory)/'copy'/'Geometry workings.md'
            exporter.schedule(store.snapshot(),store.path,output).result(timeout=5)
            exporter.close(wait=True)
            self.assertFalse(events[-1]['error'])
            self.assertIn('Geometry workings_assets/',output.read_text())
            files=list((output.parent/'Geometry workings_assets').glob('*.png'))
            self.assertEqual(len(files),1)
            self.assertTrue(np.array_equal(cv2.imread(str(files[0])),image))
            self.assertEqual(build_report(HomeworkStore(store.path).snapshot()).ready,['2A'])

    def test_numeric_fraction_decimal_power_and_variable_workings(self):
        cases = [(r'x+2=7\implies x=7-2=5', '5'),
                 (r'\frac{7}{10}-\frac25=\frac3{10}', '.3'),
                 (r'\frac{7}{10}-\frac{2}{5}=\frac{3}{10}', '.3'),
                 (r'2^r=\frac{1}{32}=2^{-5}\\r=-5', '-5'),
                 ('0.3-0.1=0.2', '.2'), ('2(3+4)=14', '14')]
        # Unbraced TeX shorthand is intentionally unsupported in the first grammar.
        for latex, answer in [cases[0], *cases[2:]]:
            self.assertEqual(validate_workings(latex, answer), latex)
        with self.assertRaises(WorkingsError): validate_workings(*cases[1])

    def test_spaced_equalities_remain_separate_mathematical_steps(self):
        value = r'2^r=\frac{1}{32}=2^{-5}\qquad r=-5'
        self.assertEqual(validate_workings(value, '-5'), value)

    def test_words_html_commands_incorrect_math_and_missing_steps_are_rejected(self):
        values = ['Subtract two to get 5', r'\text{The answer is }5', r'\href{https://example.com}{5}',
                  '<script>alert(1)</script>', '5', 'x=5', '7-2=6', '7-2=5', r'\frac{1}{0}=5',
                  r'\frac{1}{2=5', '2^999=5', '__import__("os")=5', '5=5']
        for value in values:
            with self.assertRaises(WorkingsError, msg=value):
                validate_workings(value, '6' if value == '7-2=5' else '5')

    def test_only_latest_verified_snapshots_export_and_numeric_order_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('A', directory)
            correct(store, solution('10A'))
            correct(store, solution('2A'))
            store.note_question('1A')
            store.save_proposal('1C', solution('1C')['parts'])
            pending = store.prepare('1C', '5', solution=solution('1C'))
            store.set_status(pending, 'pending')
            report = build_report(store.snapshot())
            self.assertEqual(report.ready, ['2A', '10A'])
            self.assertEqual(report.total, 4)
            self.assertEqual([code for code,_ in report.review], ['1A', '1C'])
            self.assertNotIn('1C', report.markdown)
            self.assertNotIn('A\n', report.markdown.replace('## 2A\n', '').replace('## 10A\n', ''))
            self.assertIn('$$\nx+2=7', report.markdown)

    def test_workings_quality_does_not_erase_a_verified_bookwork_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('A', directory)
            correct(store, solution(latex='The answer is five'))
            self.assertEqual(store.verified_answer('1B'), '5')
            report = build_report(store.snapshot())
            self.assertFalse(report.ready)
            self.assertEqual(len(report.review), 1)

    def test_submission_snapshot_cannot_be_overwritten_by_later_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('A', directory)
            correct(store)
            original = store.snapshot()['attempts'][0]['parts'][0]['workings_latex']
            store.save_proposal('1B', solution(latex='9-4=5')['parts'])
            report = build_report(store.snapshot())
            self.assertFalse(report.ready)
            self.assertIn('newer solution', report.review[0][1])
            self.assertEqual(store.snapshot()['attempts'][0]['parts'][0]['workings_latex'], original)

    def test_rejected_revision_is_replaced_instead_of_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('A', directory)
            bad = solution(answer='6', latex='7-1=6')
            store.save_proposal('1B', bad['parts'])
            identifier = store.prepare('1B', '6', solution=bad)
            store.set_status(identifier, 'pending'); store.set_status(identifier, 'incorrect')
            correct(store)
            report = build_report(store.snapshot())
            self.assertEqual(report.markdown.count('## 1B'), 1)
            self.assertNotIn('7-1=6', report.markdown)

    def test_navigation_null_workings_does_not_erase_solved_calculation(self):
        from bookwork import BookworkSession
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            store=HomeworkStore.create('A',directory)
            session=BookworkSession(Mock(),Mock(),store);session.code='1B'
            session.accept_solution(solution())
            session.accept_solution(solution(latex=None))
            self.assertEqual(session.solution['parts'][0]['workings_latex'],r'x+2=7\implies x=7-2=5')
            key=store.prepare('1B','5',solution=session.solution)
            store.set_status(key,'pending');store.set_status(key,'correct')
            self.assertEqual(build_report(store.snapshot()).ready,['1B'])
            self.assertEqual(store.data['proposals']['1B']['revision'],1)

    def test_changed_or_rejected_answer_does_not_inherit_old_workings(self):
        from bookwork import BookworkSession
        from unittest.mock import Mock
        session=BookworkSession(Mock(),Mock());session.code='1B'
        session.accept_solution(solution())
        session.accept_solution(solution(answer='6',latex=None))
        self.assertIsNone(session.solution['parts'][0]['workings_latex'])
        session.accept_solution(solution())
        session.grade(False)
        session.accept_solution(solution(latex=None))
        self.assertIsNone(session.solution['parts'][0]['workings_latex'])

    def test_empty_initial_checkpoint_does_not_create_a_blank_file(self):
        with tempfile.TemporaryDirectory() as directory:
            store=HomeworkStore.create('A',directory)
            events=[];exporter=WorkingsExporter(events.append)
            exporter.schedule(store.snapshot(),store.path).result(timeout=5)
            exporter.close(wait=True)
            self.assertFalse(exporter.default_path(store.snapshot(),store.path).exists())
            self.assertTrue(events[0]['empty'])
            self.assertIsNone(events[0]['error'])

    def test_legacy_correct_answers_have_visible_missing_workings_status(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('Legacy', directory)
            identifier = store.prepare('1B','5')
            store.set_status(identifier,'pending'); store.set_status(identifier,'correct')
            reloaded = HomeworkStore(store.path)
            report = build_report(reloaded.snapshot())
            self.assertEqual(report.total, 1)
            self.assertFalse(report.ready)
            self.assertTrue(report.review)

    def test_auto_checkpoints_are_local_and_export_failures_preserve_good_files(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('A', directory)
            events = []
            exporter = WorkingsExporter(events.append)
            store.on_change = lambda snap: exporter.schedule(snap, store.path)
            correct(store)
            exporter.close(wait=True)
            output = exporter.default_path(store.snapshot(), store.path)
            self.assertEqual(output.read_text(), build_report(store.snapshot()).markdown)
            self.assertTrue(events)
            self.assertTrue(all(event['error'] is None for event in events))
            store.on_change = None
            before = output.read_text()
            with patch('workings_export.Path.replace', side_effect=OSError('disk full')):
                with self.assertRaises(OSError): atomic_write(output, 'replacement')
            self.assertEqual(output.read_text(), before)
            self.assertFalse(list(output.parent.glob('.workings-*.tmp')))
            self.assertEqual(HomeworkStore(store.path).verified_answer('1B'), '5')

    def test_no_ready_save_does_not_overwrite_an_existing_document(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('A', directory)
            output = Path(directory)/'workings.md'; output.write_text('previous good file')
            events=[]; exporter=WorkingsExporter(events.append)
            exporter.schedule(store.snapshot(),store.path,output).result(timeout=5)
            exporter.close(wait=True)
            self.assertTrue(events[0]['error'])
            self.assertEqual(output.read_text(),'previous good file')

    def test_snapshot_is_detached_and_proposal_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            store=HomeworkStore.create('A',directory); correct(store)
            snapshot=store.snapshot(); snapshot['attempts'][0]['parts'][0]['answer']='999'
            self.assertEqual(store.verified_answer('1B'),'5')
            self.assertEqual(build_report(HomeworkStore(store.path).snapshot()).ready,['1B'])


if __name__ == '__main__': unittest.main()
