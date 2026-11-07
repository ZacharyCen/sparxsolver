"""Versioned, local-only verified numeric answers for one homework."""
import json
import copy
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class HomeworkStore:
    """Keep submission intent separate from confirmed results; never replay intent."""
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.on_change = None
        self.revision = 0
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if (not isinstance(self.data, dict) or self.data.get("version") != 1
                or not isinstance(self.data.get("id"), str) or not isinstance(self.data.get("name"), str)
                or not isinstance(self.data.get("attempts"), list)):
            raise ValueError("Unsupported or damaged homework record")
        try:
            uuid.UUID(self.data['id'])
        except (ValueError, AttributeError) as error:
            raise ValueError('Invalid homework record identifier') from error
        check = self.data.get('bookwork_pending')
        if check is not None and (not isinstance(check, dict) or not re.fullmatch(r'\d{1,3}[A-Z]', str(check.get('code', ''))) or check.get('phase') not in ('prepared', 'selected', 'confirmed', 'accepted')):
            raise ValueError('Damaged saved bookwork interruption')
        for attempt in self.data['attempts']:
            if (not isinstance(attempt, dict) or not re.fullmatch(r'\d{1,3}[A-Z]', str(attempt.get('code', '')))
                    or not isinstance(attempt.get('answer'), str)
                    or attempt.get('status') not in ('prepared', 'pending', 'correct', 'incorrect', 'review')
                    or not isinstance(attempt.get('id'), str)):
                raise ValueError("Damaged homework submission record")

        observed = self.data.setdefault('observed_codes', [])
        if not isinstance(observed, list) or any(not isinstance(code, str) or not re.fullmatch(r'\d{1,3}[A-Z]', code) for code in observed):
            raise ValueError('Damaged observed question list')
        proposals = self.data.setdefault('proposals', {})
        if not isinstance(proposals, dict):
            raise ValueError('Damaged saved solution proposals')
        for code, proposal in proposals.items():
            if (not re.fullmatch(r'\d{1,3}[A-Z]', code) or not isinstance(proposal, dict)
                    or type(proposal.get('revision')) is not int or proposal['revision'] < 1
                    or not self._valid_parts(proposal.get('parts'))):
                raise ValueError('Damaged saved solution proposal')
        for attempt in self.data['attempts']:
            if attempt.get('kind')=='diagram':
                asset=Path(attempt.get('workings_image',''))
                if asset.parent.resolve()!=(self.path.parent/'assets').resolve() or not re.fullmatch(r'[0-9a-f]{32}\.png',asset.name):
                    raise ValueError('Invalid visual workings asset')
            if 'solution_revision' in attempt and (type(attempt['solution_revision']) is not int or attempt['solution_revision'] < 0):
                raise ValueError('Damaged solution revision')
            if 'parts' in attempt and not self._valid_parts(attempt['parts']):
                raise ValueError('Damaged submitted workings snapshot')

    def note_question(self, code):
        """Include observed but unsupported questions in export coverage."""
        if not re.fullmatch(r'\d{1,3}[A-Z]', code):
            raise ValueError('Invalid observed question code')
        with self._lock:
            codes = self.data.setdefault('observed_codes', [])
            if code not in codes:
                codes.append(code)
                try: self._save()
                except Exception:
                    codes.pop()
                    raise

    @staticmethod
    def _valid_parts(parts):
        return (isinstance(parts, list) and 0 < len(parts) <= 10 and all(
            isinstance(part, dict) and isinstance(part.get('part'), str) and len(part['part']) <= 32
            and isinstance(part.get('answer'), str) and len(part['answer']) <= 512
            and (part.get('workings_latex') is None or isinstance(part['workings_latex'], str) and len(part['workings_latex']) <= 2048)
            for part in parts))

    def snapshot(self):
        """Return a consistent copy for export workers and UI coverage."""
        with self._lock:
            return copy.deepcopy(self.data)

    def save_proposal(self, code, parts):
        """Retain workings without declaring either the answer or derivation correct."""
        if not re.fullmatch(r'\d{1,3}[A-Z]', code) or not self._valid_parts(parts):
            raise ValueError('Invalid solution proposal')
        with self._lock:
            previous = self.data['proposals'].get(code)
            if previous and previous['parts'] == parts:
                return previous['revision']
            revision = previous['revision'] + 1 if previous else 1
            self.data['proposals'][code] = {'revision': revision, 'parts': copy.deepcopy(parts)}
            try:
                self._save()
            except Exception:
                if previous is None: self.data['proposals'].pop(code, None)
                else: self.data['proposals'][code] = previous
                raise
            return revision

    @classmethod
    def create(cls, name, directory=None):
        """Create an isolated assignment record outside the source checkout."""
        directory = Path(directory or Path(os.environ.get("LOCALAPPDATA", str(Path.home() / '.config'))) / 'AutoAnswer' / 'homework')
        directory.mkdir(parents=True, exist_ok=True)
        identifier = str(uuid.uuid4())
        path = directory / (identifier + '.json')
        data = {'version': 1, 'id': identifier, 'name': name.strip()[:120] or 'Homework', 'attempts': []}
        with path.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        return cls(path)

    def _save(self):
        temporary = self.path.with_suffix('.tmp')
        try:
            with temporary.open('w', encoding='utf-8') as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
            self.revision += 1
            if self.on_change:
                try:
                    self.on_change(self.snapshot())
                except Exception as error:
                    print(f'[Workings] Checkpoint could not be queued: {error}')
        finally:
            temporary.unlink(missing_ok=True)

    def prepare(self, code, answer, solution=None, ordered=False):
        """Persist intent before physical input; unresolved attempts are never answers."""
        if not re.fullmatch(r'\d{1,3}[A-Z]', code):
            raise ValueError('Invalid question code')
        from bookwork import numeric_value
        if numeric_value(answer) is None:
            raise ValueError('Only plain numeric answers are supported')
        with self._lock:
            if any(a['code'] == code and a['status'] in ('prepared', 'pending') for a in self.data['attempts']):
                raise ValueError(f'{code} has an unresolved submission; review it before another attempt')
            attempt = {'id': str(uuid.uuid4()), 'code': code, 'answer': answer, 'status': 'prepared',
                       'time': datetime.now(timezone.utc).isoformat()}
            if solution is not None:
                parts = solution.get('parts')
                if (solution.get('question_code') != code or not self._valid_parts(parts)
                        or any(numeric_value(part['answer']) is None for part in parts) or numeric_value(parts[0]['answer']) != numeric_value(answer)):
                    raise ValueError('Solution snapshot does not match the submitted numeric value')
                attempt['parts'] = copy.deepcopy(parts)
                attempt['solution_revision'] = self.data['proposals'].get(code, {}).get('revision', 0)
                if ordered:
                    values = [numeric_value(part['answer']) for part in parts]
                    if len(values) < 3 or not (values == sorted(values) or values == sorted(values, reverse=True)):
                        raise ValueError('Ordering answers must form a complete monotonic sequence')
                    attempt['kind'] = 'ordering'
            self.data['attempts'].append(attempt)
            try:
                self._save()
            except Exception:
                self.data['attempts'].pop()
                raise
            return attempt['id']

    def prepare_expression(self, code, solution):
        """Persist a complete expression readback; grading is still required separately."""
        from answer_values import expression_value
        parts=solution.get('parts')
        if (not re.fullmatch(r'\d{1,3}[A-Z]',code) or solution.get('question_code')!=code
                or not self._valid_parts(parts) or any(expression_value(part['answer']) is None for part in parts)):
            raise ValueError('Unsupported or incomplete expression submission')
        with self._lock:
            if any(a['code']==code and a['status'] in ('prepared','pending') for a in self.data['attempts']):
                raise ValueError('An earlier submission still needs grading')
            identifier=str(uuid.uuid4())
            attempt={'id':identifier,'code':code,'answer':parts[0]['answer'],'parts':copy.deepcopy(parts),
                     'kind':'expression','status':'prepared','time':datetime.now(timezone.utc).isoformat(),
                     'solution_revision':self.data['proposals'].get(code,{}).get('revision',0)}
            self.data['attempts'].append(attempt)
            try:self._save()
            except Exception:
                self.data['attempts'].pop();raise
            return identifier

    def prepare_visual(self, code, image, parts=None):
        """Save an observed diagram before submission; export still requires correct grading."""
        import cv2
        if not re.fullmatch(r'\d{1,3}[A-Z]',code):raise ValueError('Invalid visual question code')
        with self._lock:
            if any(a['code']==code and a['status'] in ('prepared','pending') for a in self.data['attempts']):
                raise ValueError('An earlier submission still needs grading')
            directory=self.path.parent/'assets';directory.mkdir(parents=True,exist_ok=True)
            asset=directory/(uuid.uuid4().hex+'.png')
            ok,encoded=cv2.imencode('.png',image)
            if not ok:raise OSError('Could not encode the answer diagram')
            temporary=asset.with_suffix('.tmp');temporary.write_bytes(encoded.tobytes());temporary.replace(asset)
            identifier=str(uuid.uuid4())
            parts=copy.deepcopy(parts) if self._valid_parts(parts) else [{'part':'','answer':'diagram','workings_latex':None}]
            attempt={'id':identifier,'code':code,'answer':parts[0]['answer'],'parts':parts,'status':'prepared',
                     'kind':'diagram','workings_image':str(asset),'solution_revision':self.data['proposals'].get(code,{}).get('revision',0),
                     'time':datetime.now(timezone.utc).isoformat()}
            self.data['attempts'].append(attempt)
            try:self._save()
            except Exception:
                self.data['attempts'].pop();asset.unlink(missing_ok=True);raise
            return identifier

    def set_status(self, identifier, status):
        """Only a dispatched, pending attempt can be promoted to correct."""
        with self._lock:
            attempt = next(a for a in self.data['attempts'] if a['id'] == identifier)
            allowed = {'prepared': {'pending', 'review'}, 'pending': {'correct', 'incorrect', 'review'}}
            if status not in allowed.get(attempt['status'], set()):
                raise ValueError('Invalid submission state transition')
            old = attempt['status']
            attempt['status'] = status
            try:
                self._save()
            except Exception:
                attempt['status'] = old
                raise

    def verified_answer(self, code):
        with self._lock:
            for attempt in reversed(self.data['attempts']):
                if attempt['code'] == code:
                    return attempt['answer'] if attempt['status'] == 'correct' and len(attempt.get('parts', [{}])) == 1 else None
        return None

    def verified_values(self, code):
        """Return the complete numeric submission only after confirmed grading."""
        from bookwork import numeric_value
        with self._lock:
            for attempt in reversed(self.data['attempts']):
                if attempt['code'] == code:
                    if attempt['status'] != 'correct':
                        return None
                    values = tuple(part['answer'] for part in attempt.get('parts', []))
                    if not values:
                        values = (attempt['answer'],)
                    return values if all(numeric_value(value) is not None for value in values) else None
        return None

    def verified_check_values(self, code):
        """Read exact rational/surd answers only from the latest graded submission."""
        from answer_values import expression_value
        with self._lock:
            for attempt in reversed(self.data['attempts']):
                if attempt['code'] != code:
                    continue
                if attempt['status'] != 'correct' or attempt.get('kind') == 'diagram':
                    return None
                values = tuple(part['answer'] for part in attempt.get('parts', [])) or (attempt['answer'],)
                return values if all(expression_value(value) is not None for value in values) else None
        return None

    def begin_check(self, code):
        """Record an interruption before input; a restart must not replay it."""
        with self._lock:
            if self.data.get('bookwork_pending'):
                raise ValueError('A previous bookwork action needs review before retrying')
            self.data['bookwork_pending'] = {'code': code, 'phase': 'prepared'}
            try:
                self._save()
            except Exception:
                self.data['bookwork_pending'] = None
                raise

    def check_phase(self, phase):
        with self._lock:
            old = self.data['bookwork_pending']['phase']
            self.data['bookwork_pending']['phase'] = phase
            try:
                self._save()
            except Exception:
                self.data['bookwork_pending']['phase'] = old
                raise

    def finish_check(self):
        with self._lock:
            old = self.data.get('bookwork_pending')
            self.data['bookwork_pending'] = None
            try:
                self._save()
            except Exception:
                self.data['bookwork_pending'] = old
                raise

    def review_unresolved(self):
        """User acknowledgement discards uncertain outcomes without declaring them correct."""
        with self._lock:
            self.data['bookwork_pending'] = None
            for attempt in self.data['attempts']:
                if attempt['status'] in ('prepared', 'pending'):
                    attempt['status'] = 'review'
            self._save()

    @property
    def unresolved_count(self):
        return sum(a['status'] in ('prepared', 'pending') for a in self.data['attempts']) + bool(self.data.get('bookwork_pending'))
