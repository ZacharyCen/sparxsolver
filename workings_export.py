"""Restricted word-free LaTeX checks and atomic, local Markdown exports."""
import ast
import hashlib
import math
import json
import uuid
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from bookwork import numeric_value
from answer_values import expression_value


class WorkingsError(ValueError):
    """Workings fall outside the supported numeric export grammar."""


def _translate(source, depth=0):
    if depth > 16:
        raise WorkingsError('Nested fractions are too complex')
    output, index = [], 0
    while index < len(source):
        if source[index] != '\\':
            output.append(source[index]); index += 1; continue
        match = re.match(r'\\([A-Za-z]+|[,;!])', source[index:])
        if not match:
            raise WorkingsError('Unsupported LaTeX escape')
        command = match[1]
        index += len(match[0])
        if command in ('frac', 'dfrac', 'tfrac', 'sqrt'):
            groups = []
            for _ in range(1 if command == 'sqrt' else 2):
                while index < len(source) and source[index].isspace(): index += 1
                if index >= len(source) or source[index] != '{':
                    raise WorkingsError('Fractions need two braced arguments')
                start, level = index+1, 1
                index += 1
                while index < len(source) and level:
                    if source[index] == '{': level += 1
                    elif source[index] == '}': level -= 1
                    index += 1
                if level:
                    raise WorkingsError('Unbalanced braces')
                groups.append(_translate(source[start:index-1], depth+1))
            output.append(f'sqrt({groups[0]})' if command == 'sqrt' else f'(({groups[0]})/({groups[1]}))')
        elif command == 'approx': output.append('~')
        elif command in ('cdot', 'times'): output.append('*')
        elif command == 'div': output.append('/')
        elif command in ('left', 'right'): pass
        elif command in ('quad', 'qquad'): output.append('|')
        elif command in (',', ';', '!'): output.append(' ')
        elif command in ('implies', 'Rightarrow', 'rightarrow'): output.append('|')
        else:
            raise WorkingsError(f'Unsupported LaTeX command: {command}')
    return ''.join(output)


def _expression(source):
    source = source.strip().replace('^', '**').replace('{', '(').replace('}', ')')
    source = re.sub(r'(\d|\))\s*([A-Za-z(])', r'\1*\2', source)
    source = re.sub(r'(?<![A-Za-z])([A-Za-z])\s*(\d|\()', r'\1*\2', source)
    try:
        root = ast.parse(source, mode='eval')
    except (SyntaxError, RecursionError) as error:
        raise WorkingsError('Malformed mathematical expression') from error
    if len(list(ast.walk(root))) > 128:
        raise WorkingsError('Expression is too complex')
    operations = 0
    def visit(node):
        nonlocal operations
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Name) and re.fullmatch('[A-Za-z]', node.id): return None
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            text = ast.get_source_segment(source, node)
            if not text or len(text) > 64 or not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)', text):
                raise WorkingsError('Unsupported number')
            return Fraction(text)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand)
            return -value if value is not None and isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'sqrt' and len(node.args) == 1 and not node.keywords:
            operations += 1
            argument = visit(node.args[0])
            if argument is not None and argument < 0:
                raise WorkingsError('Square root of a negative number')
            return None  # Syntax checking only; not a symbolic proof checker.
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            operations += 1
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and right is not None and (right.denominator != 1 or abs(right) > 64):
                raise WorkingsError('Only bounded integer powers are supported')
            if isinstance(node.op, ast.Div) and right == 0:
                raise WorkingsError('Division by zero')
            if left is None or right is None: return None
            try:
                if isinstance(node.op, ast.Add): value = left + right
                elif isinstance(node.op, ast.Sub): value = left - right
                elif isinstance(node.op, ast.Mult): value = left * right
                elif isinstance(node.op, ast.Div): value = left / right
                else: value = left ** int(right)
            except (ValueError, ZeroDivisionError, OverflowError) as error:
                raise WorkingsError('Undefined numeric expression') from error
            if value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
                raise WorkingsError('Numeric expression is too large')
            return value
        raise WorkingsError('Use mathematical expressions without words or executable content')
    return visit(root), operations


def validate_workings(latex, answer):
    """Check restricted syntax and exact rational/surd results, not symbolic proofs."""
    if not isinstance(latex, str) or not latex.strip():
        raise WorkingsError('No workings were saved for this answer')
    if len(latex) > 2048 or re.search(r'[^A-Za-z0-9\\\s{}().+*/^=,;!|−–-]', latex):
        raise WorkingsError('Use only supported word-free mathematical notation')
    expected=expression_value(answer)
    if expected is None:
        raise WorkingsError('Answer expression consistency is unsupported')
    source = latex.strip().replace('−', '-').replace('–', '-')
    source = source.replace('\\\\', '|')
    translated = _translate(source)
    if translated.count('{') != translated.count('}'):
        raise WorkingsError('Unbalanced braces')
    relations, operations, final = 0, 0, None
    for step in re.split(r'[|\n]+', translated):
        if not step.strip(): continue
        approximate = '~' in step
        sides = re.split('[=~]', step)
        if len(sides) < 2 or any(not side.strip() for side in sides):
            raise WorkingsError('Each step must contain a complete equality')
        relations += 1
        previous = None
        previous_exact = None
        for side in sides:
            result, count = _expression(side)
            operations += count
            exact=expression_value(side)
            if previous_exact is not None and exact is not None and previous_exact!=exact:
                if not approximate:
                    raise WorkingsError('A mathematical equality in the working is incorrect')
                digits=len(side.strip().split('.')[-1]) if '.' in side.strip() else 0
                before=sum(float(coefficient)*math.sqrt(root) for root,coefficient in previous_exact)
                after=sum(float(coefficient)*math.sqrt(root) for root,coefficient in exact)
                if abs(before-after)>.5*10**(-digits)+1e-12:
                    raise WorkingsError('An approximation in the working is incorrect')
            previous_exact=exact
            if previous is not None and result is not None and previous != result:
                digits = len(side.strip().split('.')[-1]) if '.' in side.strip() else 0
                if not approximate or abs(previous-result) > Fraction(1, 2*10**digits):
                    raise WorkingsError('A numeric equality in the working is incorrect')
            previous, final = result, result
    if not relations or not operations:
        raise WorkingsError('Include the essential calculation, not only the final answer')
    if expression_value(sides[-1]) != expected:
        raise WorkingsError('The final expression does not match the verified answer')
    return latex.strip()


@dataclass
class WorkingsReport:
    markdown: str
    ready: list
    review: list
    total: int
    assets: list = field(default_factory=list)


def build_report(snapshot):
    """Render the latest verified snapshots; proposals and rejected attempts never export."""
    proposals = snapshot.get('proposals', {})
    latest = {item['code']: item for item in snapshot.get('attempts', [])}
    codes = set(proposals) | set(latest) | set(snapshot.get('observed_codes', []))
    def order(code):
        match = re.fullmatch(r'(\d+)([A-Z])', code)
        return (int(match[1]), match[2]) if match else (10**9, code)
    sections, ready, review, assets = [], [], [], []
    for code in sorted(codes, key=order):
        attempt = latest.get(code)
        if not attempt or attempt.get('status') != 'correct':
            review.append((code, 'Answer has not been verified correct')); continue
        proposal = proposals.get(code, {})
        if proposal.get('revision', 0) > attempt.get('solution_revision', 0):
            review.append((code, 'A newer solution has not been graded')); continue
        if attempt.get('kind')=='diagram':
            image=Path(attempt.get('workings_image',''))
            if not image.is_file() or image.suffix!='.png' or image.read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n':
                review.append((code,'Verified diagram file is missing'));continue
            assets.append((str(image),image.name))
            sections.append(f'## {code}\n\n![](assets/{image.name})\n')
            ready.append(code)
            continue
        parts = attempt.get('parts')
        if not isinstance(parts, list) or not parts:
            review.append((code, 'No saved workings snapshot')); continue
        if attempt.get('kind') == 'ordering':
            values = [numeric_value(part.get('answer')) for part in parts]
            if len(values) < 3 or any(value is None for value in values) or not (values == sorted(values) or values == sorted(values, reverse=True)):
                review.append((code, 'Saved ordering is not a supported numeric sequence')); continue
            symbol = r' \leq ' if values == sorted(values) else r' \geq '
            source = symbol.join(str(part['answer']) for part in parts)
            sections.append(f'## {code}\n\n$$\n{source}\n$$\n')
            ready.append(code)
            continue
        try:
            if expression_value(parts[0].get('answer')) != expression_value(attempt['answer']):
                raise WorkingsError('Saved solution and submitted answer differ')
            sources = [validate_workings(part.get('workings_latex'), part['answer']) for part in parts]
        except (WorkingsError, TypeError, KeyError) as error:
            review.append((code, str(error))); continue
        if not re.fullmatch(r'\d{1,3}[A-Z]', code):
            review.append((code, 'Unsupported question code')); continue
        sections.append(f'## {code}\n\n' + '\n'.join('$$\n'+source+'\n$$\n' for source in sources))
        ready.append(code)
    return WorkingsReport('\n'.join(sections), ready, review, len(codes), assets)


def atomic_write(path, content):
    """Replace an entire file atomically; never leave a partial previous export."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent, prefix='.workings-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary: temporary.unlink(missing_ok=True)


class WorkingsExporter:
    """Serialize checkpoints and Save As jobs away from the controller/Tk threads."""
    def __init__(self, callback):
        self.callback = callback
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='workings-export')
        self._lock = threading.Lock()
        self._fingerprints = {}

    @staticmethod
    def default_path(snapshot, record_path):
        return Path(record_path).parent / 'exports' / (str(uuid.UUID(snapshot['id'])) + '_workings.md')

    @staticmethod
    def documents_folder():
        """Use the user's Documents folder, including redirected Windows folders."""
        documents = Path.home() / 'Documents'
        if os.name == 'nt':
            import ctypes
            buffer = ctypes.create_unicode_buffer(32768)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buffer) == 0 and buffer.value:
                documents = Path(buffer.value)
        return documents

    @staticmethod
    def automatic_path(snapshot):
        """Give each assignment its own stable, portable export folder."""
        documents = WorkingsExporter.documents_folder()
        name = re.sub(r'[^\w -]', '_', snapshot.get('name', 'Homework')).strip(' .')[:40] or 'Homework'
        folder = name + '_' + str(uuid.UUID(snapshot['id']))
        return documents / 'Sparx Solver' / 'Workings' / folder / 'workings.md'

    def schedule(self, snapshot, record_path, destination=None, automatic=False):
        kind = 'automatic' if automatic else 'save' if destination is not None else 'checkpoint'
        path = self.automatic_path(snapshot) if automatic else Path(destination) if destination is not None else self.default_path(snapshot, record_path)
        signature = hashlib.blake2b(json.dumps({key: snapshot.get(key) for key in ('id','attempts','proposals','observed_codes')}, sort_keys=True).encode(), digest_size=16).digest()
        with self._lock:
            if kind != 'save' and self._fingerprints.get(str(path)) == signature:
                return None
            if kind != 'save': self._fingerprints[str(path)] = signature
            return self._worker.submit(self._run, snapshot, path, kind, signature)

    def preview(self, snapshot, token):
        """Render away from Tk; the UI must never wait on the math renderer lock."""
        def render():
            from workings_render import equation_png
            images, error = {}, None
            try:
                report = build_report(snapshot)
                for source in re.findall(r'\$\$\n(.*?)\n\$\$', report.markdown, re.S):
                    images[source] = equation_png(source)
            except Exception as failure:
                error = str(failure)
            self.callback({'id': snapshot.get('id'), 'kind': 'preview', 'token': token,
                           'images': images, 'error': error})
        return self._worker.submit(render)

    def _run(self, snapshot, path, kind, signature):
        try:
            report = build_report(snapshot)
            if kind == 'save' and not report.ready:
                raise WorkingsError('No verified workings are ready to save')
            from workings_render import rendered_markdown, html_preview
            directory = path.parent / (path.stem + '_assets')
            content, image_data = rendered_markdown(report.markdown, directory.name)
            for source,name in report.assets:
                image_data[name] = Path(source).read_bytes()
                content=content.replace(f'](assets/{name})',f'](<{directory.name}/{name}>)')
            for name, data in image_data.items():
                directory.mkdir(parents=True, exist_ok=True)
                destination = directory / name
                with tempfile.NamedTemporaryFile(dir=directory,delete=False,suffix='.tmp') as stream:
                    stream.write(data);temporary=Path(stream.name)
                temporary.replace(destination)
            empty = not report.ready
            # Do not advertise a new zero-byte document before any answer is ready.
            # If a checkpoint exists, clear its now-stale contents atomically.
            if not empty or path.exists():
                atomic_write(path.with_suffix('.html'), html_preview(content, image_data))
                atomic_write(path, content)
            event = {'id': snapshot['id'], 'kind': kind, 'path': str(path), 'preview_path': str(path.with_suffix('.html')), 'error': None, 'empty': empty}
        except Exception as error:
            with self._lock:
                if self._fingerprints.get(str(path)) == signature:
                    self._fingerprints.pop(str(path), None)
            event = {'id': snapshot['id'], 'kind': kind, 'path': str(path), 'error': str(error)}
        self.callback(event)

    def close(self, wait=False):
        self._worker.shutdown(wait=wait)
