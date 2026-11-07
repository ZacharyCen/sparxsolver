"""Conservative, local numeric bookwork checks; no model requests for matching."""
import re
import copy
import hashlib
import time
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np

from retry_policy import question_tabs
from answer_values import expression_value, entry_matches


class BookworkReview(RuntimeError):
    """The current check or saved attempt needs the user to inspect it."""


def numeric_value(text):
    """Parse bounded exact numbers without eval, units, or floating point."""
    value = str(text).strip().replace('−', '-').replace('–', '-')
    if len(value) > 64 or not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?', value):
        exact=expression_value(value)
        if exact==():return Fraction(0)
        if exact is not None and len(exact)==1 and exact[0][0]==1:
            return exact[0][1]
        return None
    try:
        parts = value.replace(' ', '').split('/')
        return Fraction(parts[0]) / Fraction(parts[1]) if len(parts) == 2 else Fraction(parts[0])
    except (ValueError, ZeroDivisionError):
        return None


@dataclass
class BookworkView:
    code: str | None
    options: list
    status: str = 'select'
    control: list | None = None
    complete: bool = True
    regions: dict | None = None
    selected: list | None = None

    @property
    def identity(self):
        return self.code, tuple((text, tuple(target)) for text, target in self.options)


class BookworkDetector:
    """Cache a cropped title gate; run full OCR only for a suspected check."""
    def __init__(self, observer):
        self.observer = observer
        self._gate_image = None
        self._gate_present = False
        self._card_cache = {}

    def inspect(self, frame, force=False):
        height, width = frame.shape[:2]
        x, y = int(width * .12), int(height * .08)
        crop = frame[y:int(height * .32), x:int(width * .90)]
        signature = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (400, 220))
        # Static header/question regions do not need OCR on every keypad edit.
        if force or self._gate_image is None or np.mean(cv2.absdiff(signature, self._gate_image)) > .25:
            raw, _ = self.observer._ocr(crop)
            text = ' '.join(str(item[1]).lower() for item in raw or [] if item[2] >= .75)
            self._gate_present = bool(re.search(r'book\s*work', text))
            self._gate_image = signature
        if not self._gate_present:
            return None
        items = self.observer.extract_auxiliary_reference(frame).get('ocr_results', [])
        view = self.from_items(frame, items, self.observer._ocr, self._card_cache, self.read_answer_labels)
        if view.code is None and view.status == 'select':
            # The white-on-blue code badge is sometimes omitted by full-page
            # OCR. Read that badge independently without guessing from answers.
            badge = frame[int(height*.16):int(height*.40),int(width*.25):int(width*.75)]
            hsv = cv2.cvtColor(badge,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,(90,100,90),(130,255,255))
            codes = set()
            for contour in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
                bx,by,bw,bh=cv2.boundingRect(contour)
                if not (width*.025 < bw < width*.22 and 20 < bh < height*.08):
                    continue
                crop=badge[max(0,by-3):by+bh+3,max(0,bx-3):bx+bw+3]
                raw,_=self.observer._ocr(cv2.resize(crop,None,fx=3,fy=3))
                text=' '.join(str(label) for _,label,confidence in raw or [] if confidence>=.85)
                match=re.fullmatch(r'\s*book\s*work\s*(\d{1,3})\s*([A-Z])\s*',text,re.I)
                if match:codes.add(match[1]+match[2].upper())
            if len(codes)==1:view.code=codes.pop()
        return view

    def read_answer_labels(self, crop):
        """Keep stacked fractions and square roots intact inside bookwork cards."""
        fraction = self.observer.read_stacked_fraction(crop)
        if fraction is not None:
            return [fraction]
        ink = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 160)
        boxes = [cv2.boundingRect(c) for c in cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]
        height, width = crop.shape[:2]
        # OCR often omits the radical and reads its raised radicand first.
        # Require the actual connected hook/roof and read its two sides apart.
        radicals = [(x,y,w,h) for x,y,w,h in boxes
                    if y < height*.35 and h > height*.4 and w > height*.6
                    and w > h*1.05]
        if len(radicals) == 1:
            x,y,w,h = radicals[0]
            digits = [(a,b,c,d) for a,b,c,d in boxes if a > x+w*.3 and a+c <= x+w+1 and b > y]
            signs = [(a,b,c,d) for a,b,c,d in boxes if a+c < x and a > width*.15
                     and (c > d*3 or abs(c-d) < max(3,d*.25))]
            if digits:
                a = min(b[0] for b in digits)
                b = min(b[1] for b in digits)
                root = self.observer.read_numeric_labels(crop[b:y+h+1, a:x+w+1])
                if len(root)>1 and all(re.fullmatch(r'\d+',label) for label in root):
                    # A confirmed radical's integer may be split into separate
                    # digit detections. Reassemble spatially, never OCR order.
                    field=crop[b:y+h+1,a:x+w+1]
                    padded=cv2.copyMakeBorder(cv2.resize(field,None,fx=2,fy=2),16,16,16,16,
                                             cv2.BORDER_CONSTANT,value=(255,255,255))
                    raw,_=self.observer._ocr(padded,use_cls=False)
                    tokens=[]
                    for points,text,confidence in raw or []:
                        if confidence>=.8:
                            points=np.asarray(points)
                            tokens.append((str(text).strip(),points.min(axis=0),points.max(axis=0)))
                    unique=[]
                    for index,token in enumerate(tokens):
                        text,left,right=token
                        tolerance=max(4,(right[1]-left[1])*.2)
                        covered=any(other!=index and ((len(word)>len(text) and text in word) or (word==text and other<index))
                                    and (start[0]<=left[0]+tolerance and end[0]>=right[0]-tolerance
                                         or len(word)>len(text) and min(end[0],right[0])-max(start[0],left[0])
                                             >min(end[0]-start[0],right[0]-left[0])*.55)
                                    and min(end[1],right[1])-max(start[1],left[1])>min(end[1]-start[1],right[1]-left[1])*.5
                                    for other,(word,start,end) in enumerate(tokens))
                        if not covered:unique.append(token)
                    unique.sort(key=lambda item:item[1][0])
                    separate=all(after[1][0]>=before[2][0]-max(2,min(before[2][0]-before[1][0],
                                                                                   after[2][0]-after[1][0])*.2)
                                 for before,after in zip(unique,unique[1:]))
                    if (unique and all(re.fullmatch(r'\d+',text) for text,_,_ in unique)
                            and separate
                            and max((left[1]+right[1])/2 for _,left,right in unique)
                                -min((left[1]+right[1])/2 for _,left,right in unique)<field.shape[0]*.8):
                        root=[''.join(text for text,_,_ in sorted(unique,key=lambda item:item[1][0]))]
                if len(root) == 1 and re.fullmatch(r'\d+', root[0]):
                    prefix = [(a,b,c,d) for a,b,c,d in boxes if a+c < x]
                    radical = 'sqrt(' + root[0] + ')'
                    if not prefix:
                        return [radical]
                    if len(prefix) == 1 and prefix[0][2] > prefix[0][3]*3:
                        return ['-' + radical]
                    if len(signs) == 1:
                        sx,sy,sw,sh = signs[0]
                        left = self.read_answer_labels(crop[:, :sx-2])
                        if len(left) == 1 and numeric_value(left[0]) is not None:
                            return [left[0] + ('-' if sw > sh*3 else '+') + radical]
            return []  # Never reinterpret an unreadable radical as ordinary digits.
        labels = self.observer.read_numeric_labels(crop)
        if len(labels) > 1:
            # A detached minus is sometimes returned as a Chinese horizontal stroke.
            ordered = sorted(boxes)
            if len(ordered) >= 2 and ordered[0][2] > ordered[0][3]*3:
                digits = [label for label in labels if re.fullmatch(r'\d+(?:\.\d+)?', label)]
                if len(digits) == 1 and all(label in digits or label in ('一','-','−') for label in labels):
                    return ['-' + digits[0]]
            return []  # Token order alone is not mathematical layout.
        return labels

    @staticmethod
    def from_items(frame, items, ocr=None, card_cache=None, numeric_reader=None):
        """Ground options in visible bordered cards; do not click arbitrary numbers."""
        height, width = frame.shape[:2]
        items = [item for item in items if item.get('confidence', 1) >= .85 and 'center' in item]
        titles = [item for item in items if re.search(r'book\s*work', item.get('text', ''), re.I)]
        if not titles:
            return BookworkView(None, [])  # Suspected check stays out of ordinary solving.
        title_y = min(item['center'][1] for item in titles)
        codes, explicit_codes = set(), set()
        for item in items:
            if title_y - 15 <= item['center'][1] < title_y + height * .3:
                explicit_codes.update(number+letter for number,letter in re.findall(
                    r'BOOK\s*WORK\s*(\d{1,3})\s*([A-Z])(?![A-Z0-9])',item['text'].upper()))
                codes.update(re.findall(r'(?<!\d)\d{1,3}[A-Z](?![A-Z0-9])', item['text'].upper()))
        if explicit_codes:codes=explicit_codes
        code = next(iter(codes)) if len(codes) == 1 else None
        words = {item['text'].strip().lower().rstrip('!. >›»→'): item['center'] for item in items if item['center'][1] >= title_y}
        status = 'rejected' if any(word in words for word in ('incorrect', 'try again', 'not quite', 'failed')) else 'select'
        if any(word in words for word in ('correct', 'passed', 'bookwork check passed', 'check passed', 'bookwork check complete', 'well done', 'correct answer')):
            status = 'accepted'
        control = None
        labels = ('continue', 'continue homework', 'next') if status == 'accepted' else ('confirm', 'check answer', 'submit answer', 'submit')
        for label in labels:
            if label in words:
                control = list(map(int, words[label]))
                break
        if numeric_reader and status == 'select':
            vectors = BookworkDetector.vector_options(frame, title_y, numeric_reader)
            if vectors is not None:
                options, complete, regions, selected = vectors
                return BookworkView(code, options, status, control, complete, regions, selected)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(cv2.Canny(gray, 40, 130), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            left, top, w, h = cv2.boundingRect(contour)
            if 35 <= w <= width * .55 and 25 <= h <= height * .24 and top > title_y:
                if len(cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)) == 4:
                    boxes.append((left, top, w, h))
        options = []
        seen = set()
        for item in items:
            if expression_value(item['text']) is None:
                continue
            px, py = map(int, item['center'])
            if py <= title_y or not width * .08 < px < width * .92:
                continue
            if any(l < px < l+w and t < py < t+h for l, t, w, h in boxes) and (px, py) not in seen:
                options.append((item['text'], [px, py]))
                seen.add((px, py))
        complete = True
        if ocr and status == 'select':
            unique = []
            for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
                l, t, w, h = box
                if w*h < 3500 or any(abs(l+w/2-(a+c/2)) < 10 and abs(t+h/2-(b+d/2)) < 10 for a,b,c,d in unique):
                    continue
                unique.append(box)
            groups = [[other for other in unique if .75 < other[2]/box[2] < 1.3 and .75 < other[3]/box[3] < 1.3] for box in unique]
            cards = max(groups, key=len) if groups else []
            if len(cards) < 2:
                complete = False
            else:
                options = []
                for l,t,w,h in cards:
                    crop = frame[t+5:t+h-5, l+5:l+w-5]
                    crop = cv2.copyMakeBorder(cv2.resize(crop, None, fx=2, fy=2), 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255,255,255))
                    key = hashlib.blake2b(crop.tobytes(), digest_size=16).digest()
                    labels = card_cache.get(key) if card_cache is not None else None
                    if labels is None:
                        raw, _ = ocr(crop)
                        labels = [str(text).strip() for _,text,confidence in raw or [] if confidence >= .85]
                        if card_cache is not None:
                            if len(card_cache) >= 24:
                                card_cache.clear()
                            card_cache[key] = labels
                    if len(labels) != 1 or expression_value(labels[0]) is None:
                        complete = False
                    else:
                        options.append((labels[0], [int(l+w/2), int(t+h/2)]))
        return BookworkView(code, sorted(options, key=lambda item: (item[1][1], item[1][0])), status, control, complete)


    @staticmethod
    def vector_options(frame, title_y, reader):
        """Read numeric fields inside whole answer-image cards, keeping each tuple together."""
        h,w=frame.shape[:2]
        edges=cv2.Canny(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),40,130)
        edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        boxes=[]
        contours=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True):
            x,y,bw,bh=cv2.boundingRect(contour)
            if not (w*.10 < bw < w*.36 and h*.10 < bh < h*.36 and y>title_y):
                continue
            if len(cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)) != 4:
                continue
            if any(abs(x+bw/2-(a+c/2))<15 and abs(y+bh/2-(b+d/2))<15 for a,b,c,d in boxes):
                continue
            boxes.append((x,y,bw,bh))
        groups=[[other for other in boxes if .8 < other[2]/box[2] < 1.25 and .8 < other[3]/box[3] < 1.25] for box in boxes]
        cards=max(groups,key=len) if groups else []
        if not 2 <= len(cards) <= 12:
            return None
        options=[];regions={};complete=True;has_vector=False;selected=None
        for x,y,bw,bh in sorted(cards,key=lambda box:(round(box[1]/25),box[0])):
            card=frame[y+3:y+bh-3,x+3:x+bw-3]
            dark=np.mean(np.max(card,axis=2)<170)>.4
            threshold=175 if dark else 245
            mask=cv2.inRange(card,(threshold,threshold,threshold),(255,255,255))
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            fields=[]
            for contour in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
                l,t,fw,fh=cv2.boundingRect(contour)
                if 18 < fw < bw*.80 and 20 < fh < bh*.80 and l>1 and t>1:
                    fields.append((l,t,fw,fh))
            if not fields:
                return None  # Ordinary scalar cards use the existing scalar path.
            values=[]
            for l,t,fw,fh in sorted(fields,key=lambda box:(round((box[1]+box[3]/2)/20),box[0])):
                field=card[t+2:t+fh-2,l+2:l+fw-2]
                labels=reader(255-field if dark else field)
                if len(labels)==1 and expression_value(labels[0]) is not None:
                    values.append(labels[0])
                elif len(labels)==1 and re.fullmatch(r'[A-Za-z]',labels[0]):
                    values.append(labels[0].upper())  # Recognized letter distractors cannot match numeric answers.
                elif len(labels)==1 and labels[0].strip().lower() == 'have':
                    continue  # A proof acknowledgement is not a fifth numeric answer.
                else:
                    values.append(None)
                    complete=False
            if len(values)>1:has_vector=True
            if not values:complete=False
            target=[x+bw//2,y+bh//2]
            options.append((tuple(values),target));regions[tuple(target)]=(x,y,bw,bh)
            if dark:
                if selected is not None:complete=False
                selected=target
        return (options,complete,regions,selected) if has_vector else None


def matching_answer(option, expected):
    """Match exact numbers; simultaneous-equation root pairs may be displayed in either order."""
    values=option if isinstance(option,(tuple,list)) else (option,)
    if len(values)!=len(expected):return False
    actual=tuple(expression_value(value) for value in values)
    wanted=tuple(expression_value(value) for value in expected)
    if None in actual or None in wanted:return False
    if len(actual)==4:
        return sorted((actual[:2],actual[2:])) == sorted((wanted[:2],wanted[2:]))
    return actual==wanted


def reliable_options(view, expected):
    """An unreadable distractor is harmless only when its other values rule it out."""
    if view.complete:return True
    for values,_ in view.options:
        if not isinstance(values,tuple):return False
        if None not in values:continue
        if len(values)!=len(expected):return False
        actual=tuple(expression_value(value) if value is not None else None for value in values)
        wanted=tuple(expression_value(value) for value in expected)
        orders=(wanted,wanted[2:]+wanted[:2]) if len(wanted)==4 else (wanted,)
        if any(all(a is None or a==b for a,b in zip(actual,order)) for order in orders):
            return False
    return True


def stable_bookwork_controls(previous, current, expected):
    """Allow small OCR/hover shifts while keeping the same unique answer card."""
    if current is None or previous.code != current.code or previous.status != current.status:
        return False
    def near(a,b):
        return a is None and b is None or a is not None and b is not None and max(abs(x-y) for x,y in zip(a,b))<=4
    if not near(previous.control,current.control):
        return False
    if current.status != 'select':
        return previous.identity == current.identity
    if not reliable_options(current,expected) or len(previous.options)!=len(current.options):
        return False
    old_matches=[target for values,target in previous.options if matching_answer(values,expected)]
    new_matches=[target for values,target in current.options if matching_answer(values,expected)]
    if len(old_matches)!=1 or len(new_matches)!=1 or not near(old_matches[0],new_matches[0]):
        return False
    # Tolerate a differently read distractor, never reordered or moved cards.
    old_targets=[target for _,target in previous.options]
    new_targets=[target for _,target in current.options]
    return (all(sum(near(a,b) for b in new_targets)==1 for a in old_targets)
            and all(sum(near(a,b) for a in old_targets)==1 for b in new_targets))


class BookworkSession:
    """Own bookwork input independently from ordinary question retries and counting."""
    def __init__(self, observer, executor, store=None, collect_workings=True):
        self.observer, self.executor, self.store = observer, executor, store
        self.collect_workings = collect_workings
        self.detector = BookworkDetector(observer)
        self.code = None
        self.solution = None
        self.entry = None
        self.visual_parts = None
        self.pending = None
        self._header = None
        self.completed_checks = 0
        self.last_completed_at = 0.0

    def bind_question(self, frame, items):
        from observer import Observer
        if isinstance(self.observer,Observer):
            code=self.observer.read_question_code(frame)
        else:
            code, _ = question_tabs(frame, [item for item in items if item.get('confidence', 1) >= .80])
        if code and code != self.code:
            if self.pending:
                raise BookworkReview('Question changed while its saved submission outcome is unresolved')
            self.code, self.solution, self.entry = code, None, None
            if self.store:
                try: self.store.note_question(code)
                except (OSError, ValueError) as error:
                    raise BookworkReview(f'Could not save question identity: {error}') from error
        return self.code

    def accept_solution(self, solution):
        if not self.code or not isinstance(solution, dict) or solution.get('question_code') != self.code:
            return False
        parts = copy.deepcopy(solution.get('parts', []))
        self.visual_parts = copy.deepcopy(parts) if self.store and self.store._valid_parts(parts) else None
        if not self.collect_workings and isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict):
                    part['workings_latex'] = None
        # Navigation responses may repeat the answer with null workings. Keep the
        # calculation accepted during this attempt, never across a changed answer.
        if (self.collect_workings and self.solution and isinstance(parts,list)
                and len(parts)==len(self.solution['parts'])):
            for incoming,previous in zip(parts,self.solution['parts']):
                if (isinstance(incoming,dict) and not incoming.get('workings_latex')
                        and expression_value(incoming.get('answer')) is not None
                        and expression_value(incoming['answer'])==expression_value(previous['answer'])):
                    incoming['workings_latex']=previous.get('workings_latex')
        if self.store and self.store._valid_parts(parts):
            try:
                self.store.save_proposal(self.code, parts)
            except (OSError, ValueError) as error:
                raise BookworkReview(f'Could not save question workings: {error}') from error
        if not isinstance(parts, list) or not parts or any(not isinstance(part, dict) for part in parts):
            self.solution = self.entry = None
            return False
        for part in parts:
            answer = str(part.get('answer', '')).strip()
            assignment = re.fullmatch(r'[A-Za-z]\s*=\s*(.+)', answer)
            if assignment:
                answer = assignment[1].strip()
            if expression_value(answer) is None:
                self.solution = self.entry = None
                return False
            part['answer'] = answer
        self.solution = {'question_code': self.code, 'parts': parts}
        return True

    def verified_fields(self, frame):
        """Reuse one readback for identical pixels and the same solution."""
        from observer import Observer
        if self.solution and isinstance(self.observer,Observer) and isinstance(frame,np.ndarray):
            key=(self.code,tuple(part['answer'] for part in self.solution['parts']),frame.shape,
                 hashlib.blake2b(frame.tobytes(),digest_size=16).digest())
            cache=getattr(self,'_field_readbacks',{})
            if key in cache:
                return cache[key]
            result=self._read_verified_fields(frame)
            if len(cache)>=4:cache.clear()
            cache[key]=result
            self._field_readbacks=cache
            return result
        return self._read_verified_fields(frame)

    def _read_verified_fields(self, frame):
        """Match every complete numeric or expression field to the saved solution."""
        if not self.solution:
            return None
        parts = self.solution['parts']
        from observer import Observer
        if isinstance(self.observer,Observer):
            choices=self.observer.numeric_choice_state(frame,[part['answer'] for part in parts])
            if choices is not None:
                return choices['cards'] if choices['complete'] else None
        fields = (self.observer.answer_fields(frame,[part['answer'] for part in parts])
                  if isinstance(self.observer,Observer) else self.observer.numeric_fields(frame))
        if len(fields) != len(parts):
            return None
        if all(entry_matches(field['value'],part['answer']) for field,part in zip(fields,parts)):
            return fields
        return None

    def next_entry(self, frame, keypad=False):
        """Locate one unfinished slot locally; leave unfamiliar layouts to the model."""
        self.last_entry_completes = False
        if not self.solution:
            return None
        if any(not re.fullmatch(r'-?\d+(?:\.\d+)?',part['answer']) for part in self.solution['parts']):
            return None  # Fractions and expressions need their actual mathematical controls.
        fields = self.observer.numeric_fields(frame)
        if len(fields) != len(self.solution['parts']):
            return None
        if any(field.get('kind') == 'ordered_card' for field in fields):
            return None  # Reordering needs a drag, never keyboard entry into a card.
        self.last_entry_completes = sum(not entry_matches(field['value'],part['answer'])
                                        for field,part in zip(fields,self.solution['parts'])) == 1
        for field,part in zip(fields,self.solution['parts']):
            if not entry_matches(field['value'],part['answer']):
                if keypad:
                    keys = self.observer.keypad_controls(frame)
                    if not field.get('focused'):
                        return {'type':'click','purpose':'answer','target':field['target']}
                    if not keys:
                        return None
                    if not re.fullmatch(r'-?\d+(?:\.\d+)?', part['answer']):
                        return None
                    return {'type':'keypad_number','purpose':'answer','target':field['target'],
                            'text':part['answer'],'erase':len(field['value']) if numeric_value(field['value']) is not None else 0}
                return {'type': 'type_text', 'purpose': 'answer', 'target': field['target'], 'text': part['answer']}
        return None

    def verify_entry(self, frame, target, expected):
        """Read the enclosing white answer field, excluding nearby keypad labels."""
        x, y = target
        height, width = frame.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return False
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 40, 130)
        edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            left, top, w, h = cv2.boundingRect(contour)
            if (80 <= w <= width * .5 and 30 <= h <= height * .22
                    and left+4 < x < left+w-4 and top+4 < y < top+h-4
                    and len(cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)) == 4):
                inner = frame[top+5:top+h-5, left+5:left+w-5]
                if inner.size and np.mean(np.min(inner, axis=2) > 225) > .75:
                    boxes.append((left, top, w, h))
        if not boxes:
            return False
        left, top, w, h = max(boxes, key=lambda box: box[2] * box[3])
        crop = frame[top+5:top+h-5, left+5:left+w-5]
        from observer import Observer
        if isinstance(self.observer, Observer):
            values = self.observer.read_numeric_labels(crop)
        else:
            raw, _ = self.observer._ocr(crop)
            values = [str(text).strip() for _, text, confidence in raw or [] if confidence >= .85]
        return len(values) == 1 and entry_matches(values[0],expected)

    def observe_entry(self, action, frame):
        if action.get('type') in ('type_text', 'fill') and self.solution and len(self.solution['parts']) == 1:
            answer = self.solution['parts'][0]['answer']
            if numeric_value(action['text']) == numeric_value(answer) and self.verify_entry(frame, action['target'], action['text']):
                self.entry = (list(action['target']), action['text'])
                return
        if action.get('purpose') != 'submit' and action.get('type') != 'wait':
            self.entry = None

    def visual_choice_readback(self, frame):
        """Keep image evidence for expression cards OCR cannot verify; never invent values."""
        from observer import Observer
        if not self.solution or not isinstance(self.observer, Observer):
            return None
        cards = self.observer.numeric_choice_cards(frame)
        if not cards or not 3 <= len(cards) <= 16:
            return None
        selected = [card for card in cards if card['selected']]
        if len(selected) != len(self.solution['parts']) or not all(card.get('bounds') for card in cards):
            return None
        left = min(card['bounds'][0] for card in cards)
        top = min(card['bounds'][1] for card in cards)
        right = max(card['bounds'][0]+card['bounds'][2] for card in cards)
        bottom = max(card['bounds'][1]+card['bounds'][3] for card in cards)
        return frame[top:bottom,left:right].copy()

    def prepare_submission(self, frame, visual_choices=None, trust_entry=False):
        if not self.store:
            return None
        if self.pending:
            raise BookworkReview('A saved submission is still awaiting grading; it will not be submitted again')
        if visual_choices is not None:
            if self.observer.read_question_code(frame) != self.code:
                raise BookworkReview('Visual choice submission question code could not be verified')
            return self.store.prepare_visual(self.code, visual_choices, self.visual_parts or self.solution['parts'])
        if not self.solution:
            from observer import Observer
            if self.collect_workings and self.code and isinstance(self.observer,Observer):
                diagram=self.observer.selected_visual_answer(frame)
                if diagram is not None:
                    code=self.observer.read_question_code(frame)
                    if code!=self.code:raise BookworkReview('Diagram submission code could not be verified')
                    return self.store.prepare_visual(code,diagram,self.visual_parts)
            return None
        # Multi-field submissions must be read in full, including the final decimal.
        if trust_entry and any(not re.fullmatch(r'-?\d+(?:\.\d+)?',part['answer']) for part in self.solution['parts']):
            raise BookworkReview('Fast entry intent supports plain numeric fields only')
        fields = None if trust_entry else self.verified_fields(frame)
        if not trust_entry and fields is None and (len(self.solution['parts']) != 1 or not self.entry):
            raise BookworkReview('Entered answers do not match the saved solution; submission was blocked')
        from observer import Observer
        if isinstance(self.observer,Observer):
            code=self.observer.read_question_code(frame)
        else:
            items = self.observer.extract_auxiliary_reference(frame).get('ocr_results', [])
            code, _ = question_tabs(frame, [item for item in items if item.get('confidence', 1) >= .80])
        if code != self.code:
            raise BookworkReview('Submission question code could not be verified')
        answer = self.solution['parts'][0]['answer']
        if not trust_entry and fields is None and not self.verify_entry(frame, self.entry[0], answer):
            raise BookworkReview('Entered answer could not be verified')
        try:
            if any(numeric_value(part['answer']) is None for part in self.solution['parts']):
                return self.store.prepare_expression(code,self.solution)
            return self.store.prepare(code, answer, solution=self.solution,
                                      ordered=bool(fields and all(field.get('kind') == 'ordered_card' for field in fields)))
        except (OSError, ValueError) as error:
            raise BookworkReview(f'Homework record could not save submission intent: {error}') from error

    def dispatched(self, identifier):
        if identifier:
            try:
                self.store.set_status(identifier, 'pending')
            except (OSError, ValueError) as error:
                raise BookworkReview(f'Could not save dispatched submission: {error}') from error
            self.pending = (identifier, self.code)

    def grade(self, correct, frame=None):
        if self.pending and correct:
            if frame is None:
                raise BookworkReview('Correct feedback needs a fresh question identity before saving')
            from observer import Observer
            if isinstance(self.observer,Observer):
                current_code=self.observer.read_question_code(frame)
            else:
                raw, _ = self.observer._ocr(frame[:int(frame.shape[0] * .35)])
                items = []
                for points, text, confidence in raw or []:
                    if confidence >= .80:
                        items.append({'text': text, 'center': np.mean(points, axis=0).tolist()})
                current_code, _ = question_tabs(frame, [item for item in items if item.get('confidence', 1) >= .80])
            if current_code != self.pending[1]:
                raise BookworkReview('Correct feedback could not be bound to the submitted question code')
        if self.pending:
            identifier, code = self.pending
            try:
                self.store.set_status(identifier, 'correct' if correct else 'incorrect')
            except (OSError, ValueError) as error:
                raise BookworkReview(f'Could not save grading; feedback was preserved: {error}') from error
            print(f'[Bookwork memory] {code}: {"verified correct" if correct else "rejected"}.')
            self.pending = None
        if not correct:
            self.solution = self.entry = self.visual_parts = None

    def reset_question(self):
        self.code = self.solution = self.entry = self.visual_parts = None

    def _check_store(self, method, *args):
        try:
            return getattr(self.store, method)(*args)
        except (OSError, ValueError) as error:
            raise BookworkReview(f'Could not save bookwork interruption state: {error}') from error

    def handle(self, frame, capture, activate=None):
        view = self.detector.inspect(frame)
        if view is None:
            return False
        for _ in range(2):
            if view.code or view.status != 'select':
                break
            if self.executor.is_cancelled():
                return True
            self.executor.wait(150)
            frame=capture()
            view=self.detector.inspect(frame,force=True) or BookworkView(None,[])
        pending = self.store.data.get('bookwork_pending') if self.store else None
        resume_accepted = bool(view.status == 'accepted' and pending and pending['phase'] in ('confirmed','accepted'))
        check_code = view.code or (pending['code'] if resume_accepted else None)
        if not self.store or not check_code:
            raise BookworkReview('Bookwork check needs a selected homework record and a readable question code')
        expected = self.store.verified_check_values(check_code)
        if not expected:
            raise BookworkReview(f'No verified answer saved for Bookwork {check_code}. This question may have been completed before recording started; enter its answer manually in the browser. No choice was guessed.')
        answer = ', '.join(expected)
        resume_selected = bool(pending and pending['code']==check_code and pending['phase']=='selected'
                               and view.selected and reliable_options(view,expected)
                               and sum(matching_answer(values,expected) for values,_ in view.options)==1
                               and any(target==view.selected and matching_answer(values,expected) for values,target in view.options))
        if pending and not (resume_accepted or resume_selected):
            raise BookworkReview("A previous bookwork action is unresolved. Inspect the browser, then use Review pending before retrying")
        original_code = check_code
        selected = confirmed = accepted = continued = False
        selected_target = None
        selected_appearance = None
        selection_slice = None
        intent_saved = resume_accepted or resume_selected
        last_frame = frame
        deadline = time.monotonic() + 40
        if view.selected and any(target==view.selected and matching_answer(values,expected) for values,target in view.options):
            selected=True
            selected_target=view.selected
            sx,sy,sw,sh=view.regions[tuple(selected_target)]
            selection_slice=(slice(sy,sy+sh),slice(sx,sx+sw))
            selected_appearance=frame[selection_slice].copy()
            print('[Bookwork] Saved answer is already visibly selected; verifying before confirmation.')
        while time.monotonic() <= deadline:
            if self.executor.is_cancelled():
                return True
            if view is None:
                if accepted:
                    self._check_store('finish_check')
                    self.completed_checks += 1
                    self.last_completed_at = time.time()
                    print(f'[Bookwork] {original_code}: accepted; ordinary solving resumes.')
                    return True
                raise BookworkReview('Bookwork screen disappeared without confirmed acceptance; inspect the browser')
            if time.monotonic() > deadline:
                break
            if view.code != original_code and view.status != 'accepted':
                raise BookworkReview('Bookwork question code changed; no stale choice will be clicked')
            if view.status == 'rejected':
                raise BookworkReview(f'Bookwork selection for {original_code} was rejected; saved answer needs review')
            target = None
            if view.status == 'accepted':
                accepted = True
                if view.control and not continued:
                    target, continued = view.control, True
            elif not selected:
                if not reliable_options(view,expected) or not 2 <= len(view.options) <= 12:
                    raise BookworkReview('Bookwork answer cards were not reliably recognized')
                matches = [position for text, position in view.options if matching_answer(text, expected)]
                if len(matches) != 1:
                    raise BookworkReview(f'No unique matching bookwork choice for {original_code}: {answer}')
                target, selected = matches[0], True
                selected_target = target
                region=(view.regions or {}).get(tuple(target))
                if region:
                    sx,sy,sw,sh=region
                    selection_slice=(slice(sy,sy+sh),slice(sx,sx+sw))
                else:
                    sx,sy=target
                    selection_slice=(slice(max(0,sy-30),sy+31),slice(max(0,sx-45),sx+46))
            elif view.control and not confirmed:
                # Selection must visibly alter the chosen card before confirmation.
                confirmed, target = True, view.control
            if target:
                if activate:
                    activate()
                fresh = capture()
                fresh_view = self.detector.inspect(fresh, force=True)
                for _ in range(2):
                    if stable_bookwork_controls(view,fresh_view,expected):
                        break
                    if self.executor.is_cancelled():
                        return True
                    self.executor.wait(100)
                    fresh=capture()
                    fresh_view=self.detector.inspect(fresh,force=True)
                if not stable_bookwork_controls(view,fresh_view,expected):
                    raise BookworkReview('Bookwork controls changed before input; re-observe before retrying')
                if accepted or confirmed:
                    target=fresh_view.control
                else:
                    target=next(position for values,position in fresh_view.options if matching_answer(values,expected))
                x, y = target
                if not 0 <= x < fresh.shape[1] or not 0 <= y < fresh.shape[0]:
                    raise BookworkReview('Bookwork target is outside the captured region')
                if self.executor.is_cancelled():
                    return True
                if confirmed and not accepted and selected_appearance is not None:
                    sx, sy = selected_target
                    patch = fresh[selection_slice]
                    visibly_selected=(fresh_view.selected is not None and
                                      max(abs(a-b) for a,b in zip(fresh_view.selected,selected_target))<=4)
                    if not visibly_selected and (patch.shape != selected_appearance.shape or np.mean(cv2.absdiff(patch, selected_appearance)) > 1):
                        raise BookworkReview('The selected bookwork card changed before confirmation')
                if not intent_saved:
                    self._check_store('begin_check', original_code)
                    intent_saved = True
                print(f'[Bookwork] {original_code}: {"continue" if accepted else "confirm" if confirmed else "select " + answer}.')
                self.executor.execute({'type': 'click', 'purpose': 'bookwork', 'target': target})
                self._check_store('check_phase', 'accepted' if accepted else 'confirmed' if confirmed else 'selected')
                if self.executor.is_cancelled():
                    return True
            self.executor.wait(150)
            current = capture()
            if selected and not confirmed and selection_slice is not None:
                # CSS/browser rendering can finish after the first capture. Poll
                # pixels briefly before spending OCR on a stale frame.
                ready_until=time.monotonic()+1.0
                while (current.shape==last_frame.shape and
                       np.mean(cv2.absdiff(current[selection_slice],last_frame[selection_slice]))<.5 and
                       time.monotonic()<ready_until and not self.executor.is_cancelled()):
                    self.executor.wait(60)
                    current=capture()
            next_view = self.detector.inspect(current)
            if selected and not confirmed and next_view and next_view.status == 'select' and next_view.control:
                # No visible selection change: never blindly confirm or repeat the option click.
                x, y = selected_target
                patch = selection_slice
                if current.shape != last_frame.shape or np.mean(cv2.absdiff(current[patch], last_frame[patch])) < .5:
                    raise BookworkReview('Bookwork choice did not visibly select; no blind confirmation attempted')
                selected_appearance = current[patch].copy()
            last_frame, view = current, next_view
        if accepted and view is None:
            self._check_store('finish_check')
            self.completed_checks += 1
            self.last_completed_at = time.time()
            print(f'[Bookwork] {original_code}: accepted; ordinary solving resumes.')
            return True
        raise BookworkReview('Bookwork result or Continue was not verified within the bounded wait')
