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
    def test_homework_integer_powers_export_and_incorrect_equalities_still_fail(self):
        workings = r'2^{15}=2\times 16384=32768'
        self.assertEqual(validate_workings(workings,'32768'),workings)
        with self.assertRaises(WorkingsError):
            validate_workings(r'2^{15}=32767','32767')
        with self.assertRaises(WorkingsError):
            validate_workings(r'2^{100000}=32768','32768')

    def test_checked_workings_save_to_documents_after_each_correct_answer(self):
        from solver_ui import SolverUI
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('Homework / 1', directory)
            events = []
            exporter = WorkingsExporter(events.append)
            ui = Mock()
            ui.homework_store = store
            ui._workings_store = None
            ui.workings_var.get.return_value = True
            ui._workings_exporter = exporter
            with patch.object(WorkingsExporter, 'documents_folder', return_value=Path(directory)/'Documents'):
                SolverUI._watch_workings_store(ui)
                output = exporter.automatic_path(store.snapshot())
                self.assertIn('Sparx Solver', str(output))
                self.assertEqual(output.name,'workings.md')
                correct(store)
                correct(store,solution('1C','6','x=3+3=6'))
                exporter.close(wait=True)
            self.assertTrue(output.is_file())
            self.assertIn('1B',output.read_text())
            self.assertIn('1C',output.read_text())
            self.assertTrue(output.with_suffix('.html').is_file())
            self.assertTrue(list(output.with_name('workings_assets').glob('*.png')))
            self.assertTrue(all(event['kind']=='automatic' and not event['error'] for event in events))

    def test_automatic_paths_do_not_mix_assignments_with_the_same_name(self):
        with tempfile.TemporaryDirectory() as directory:
            a = HomeworkStore.create('Same title', directory)
            b = HomeworkStore.create('Same title', directory)
            with patch.object(WorkingsExporter, 'documents_folder', return_value=Path(directory)):
                self.assertNotEqual(WorkingsExporter.automatic_path(a.snapshot()), WorkingsExporter.automatic_path(b.snapshot()))

    def test_ordering_export_requires_readback_snapshot_and_correct_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('Ordering', directory)
            parts = [{'part':str(i), 'answer':answer, 'workings_latex':None}
                     for i,answer in enumerate(('-7.8','-7.2','-4.4','-4','4.3','7.6'))]
            store.save_proposal('2C', parts)
            identifier = store.prepare('2C', '-7.8', {'question_code':'2C','parts':parts}, ordered=True)
            self.assertFalse(build_report(store.snapshot()).ready)
            store.set_status(identifier, 'pending')
            store.set_status(identifier, 'correct')
            report = build_report(HomeworkStore(store.path).snapshot())
            self.assertEqual(report.ready, ['2C'])
            self.assertIn(r'-7.8 \leq -7.2 \leq -4.4', report.markdown)
            parts[0],parts[1] = parts[1],parts[0]
            with self.assertRaises(ValueError):
                store.prepare('2D', '-7.2', {'question_code':'2D','parts':parts}, ordered=True)

    def test_preview_rendering_runs_on_worker_and_returns_png_bytes(self):
        import threading
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('Preview', directory)
            correct(store)
            events = []
            current = threading.get_ident()
            exporter = WorkingsExporter(lambda event: events.append((threading.get_ident(),event)))
            exporter.preview(store.snapshot(), ('record',1)).result(timeout=15)
            exporter.close(wait=True)
            worker,event = events[-1]
            self.assertNotEqual(current,worker)
            self.assertEqual(event['kind'],'preview')
            self.assertIsNone(event['error'])
            self.assertTrue(all(image.startswith(b'\x89PNG') for image in event['images'].values()))

    def test_rendered_export_is_portable_offline_and_contains_no_raw_tex(self):
        import base64
        import re
        from io import BytesIO
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            store = HomeworkStore.create('Rendered fractions', directory)
            correct(store, solution(answer='.3', latex=r'\frac{7}{10}-\frac{2}{5}=\frac{3}{10}'))
            events = []
            exporter = WorkingsExporter(events.append)
            output = Path(directory) / 'My workings.md'
            exporter.schedule(store.snapshot(), store.path, output).result(timeout=15)
            exporter.close(wait=True)
            self.assertIsNone(events[-1]['error'])
            markdown = output.read_text()
            self.assertNotIn(r'\frac', markdown)
            self.assertNotIn('$$', markdown)
            target = re.search(r'!\[\]\(<(.+)>\)', markdown)[1]
            with Image.open(output.parent / target) as image:
                size = image.size
                self.assertGreater(image.width, 80)
                self.assertGreater(image.height, 40)
            document = output.with_suffix('.html').read_text()
            self.assertNotIn('<script', document)
            self.assertNotIn(r'\frac', document)
            embedded = base64.b64decode(re.search(r'base64,([^\"]+)', document)[1])
            self.assertEqual(Image.open(BytesIO(embedded)).size, size)

    def test_rendering_handles_supported_multiline_and_tex_commands(self):
        from workings_render import equation_png
        from PIL import Image
        from io import BytesIO
        for source in (r'x+2=7\implies x=7-2=5', r'\dfrac{1}{2}+\tfrac{1}{2}=1',
                       r'2^r=\frac{1}{32}=2^{-5}\\r=-5', r'\sqrt{4}=2'):
            with self.subTest(source=source):
                image = Image.open(BytesIO(equation_png(source)))
                self.assertEqual(image.format, 'PNG')
                self.assertGreater(image.width, 40)

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
            self.assertIn('![](<', output.read_text())
            self.assertNotIn('$$', output.read_text())
            self.assertTrue(output.with_suffix('.html').is_file())
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
