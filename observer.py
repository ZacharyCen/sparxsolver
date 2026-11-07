"""
observer.py - Visual Perception and Auxiliary Analysis Layer
Implements:
- Windows Per-Monitor DPI Aware v2 initialization
- Fast MSS screen region capture
- Same-dimension image quality compression (strictly preserving dimensions)
- RapidOCR offline text detection + OpenCV candidate contour reference extraction
- Frame-to-frame diff verification for anti-window-switch protection
"""

import os
import sys
import ctypes
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple

import cv2
import mss
import numpy as np
from rapidocr_onnxruntime import RapidOCR


def init_dpi_awareness() -> bool:
    """
    Enables Windows Per-Monitor DPI Aware v2 prior to UI creation.
    Calls SetProcessDpiAwarenessContext(-4) on Windows.
    Safely no-ops on macOS and Linux.
    """
    if sys.platform == "win32":
        try:
            # -4 corresponds to DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            dpi_context = ctypes.c_void_p(-4)
            result = ctypes.windll.user32.SetProcessDpiAwarenessContext(dpi_context)
            return bool(result)
        except Exception:
            try:
                # Fallback to SetProcessDpiAwareness(2) for legacy Win8.1/Win10
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
                return True
            except Exception:
                return False
    return False


class Observer:
    """
    Observer responsible for capturing the designated screen ROI,
    performing quality-only compression without resizing, and generating
    OCR + geometric auxiliary reference data.
    """

    def __init__(self, rapid_ocr_instance: Optional[RapidOCR] = None):
        """
        Initialize the Observer with an optional reusable RapidOCR instance.
        """
        self.browser_session = None
        # Small UI crops suffer from ONNX creating a large thread pool per model.
        # Bound CPU parallelism so OCR does not compete with capture and the UI.
        self._ocr = rapid_ocr_instance or RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        self._sct = mss.mss()

    def capture_region(self, region: Dict[str, int]) -> np.ndarray:
        """
        Capture the specified region from the screen using MSS.

        :param region: Dict with keys 'left', 'top', 'width', 'height'
        :return: BGR numpy image array
        """
        if getattr(self, "browser_session", None):
            return self.browser_session.capture()
        monitor = {
            "top": int(region["top"]),
            "left": int(region["left"]),
            "width": int(region["width"]),
            "height": int(region["height"]),
        }
        raw_shot = self._sct.grab(monitor)
        # MSS returns BGRA; convert to BGR for OpenCV / model processing
        img_np = np.array(raw_shot, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        return img_bgr

    def compress_image_same_dimension(
        self,
        image: np.ndarray,
        quality: int = 85,
        output_path: Optional[str] = None
    ) -> Tuple[bytes, Tuple[int, int]]:
        """
        Compresses image encoding quality without changing image width or height.
        Strictly prohibits any downscaling or dimension resizing.

        :param image: Source BGR image numpy array
        :param quality: JPEG quality parameter (1-100), default 85
        :param output_path: Optional path to save compressed image
        :return: (compressed_bytes, (width, height))
        """
        orig_h, orig_w = image.shape[:2]
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, quality))]
        success, encoded_buf = cv2.imencode(".jpg", image, encode_params)
        if not success:
            raise RuntimeError("Failed to encode image to JPEG format.")

        compressed_bytes = encoded_buf.tobytes()

        # Empirical self-check: verify decoded dimensions exactly equal original dimensions
        decoded = cv2.imdecode(np.frombuffer(compressed_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        dec_h, dec_w = decoded.shape[:2]
        if (dec_w, dec_h) != (orig_w, orig_h):
            raise ValueError(
                f"Dimension mismatch detected during compression! "
                f"Original: ({orig_w}x{orig_h}), Decoded: ({dec_w}x{dec_h})"
            )

        if output_path:
            parent_dir = os.path.dirname(output_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(compressed_bytes)

        return compressed_bytes, (orig_w, orig_h)

    def detect_correct_continue(self, image: np.ndarray) -> Optional[list]:
        """Find graded feedback in the footer, caching verified label templates.

        Green banners use cached templates. Other layouts use a small cached footer
        read, including the final Summary button, rather than a full model request.
        """
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 100, 90), (95, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < width * 0.5 or h < 35 or y < height * 0.5:
                continue
            banner = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(banner, cv2.COLOR_BGR2GRAY)
            templates = getattr(self, "_feedback_templates", None)
            if templates:
                matches = []
                for patch in templates:
                    if patch.shape[0] > h or patch.shape[1] > w:
                        break
                    _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED))
                    if score < 0.96:
                        break
                    matches.append(location)
                if len(matches) == 2 and matches[0][0] < matches[1][0]:
                    patch = templates[1]
                    return [x + matches[1][0] + patch.shape[1] // 2,
                            y + matches[1][1] + patch.shape[0] // 2]
            # One small OCR pass verifies both labels and locates the button.
            raw, _ = self._ocr(banner)
            labels = {}
            for points, text, confidence in raw or []:
                label = str(text).strip().lower().rstrip("!")
                if label in ("correct", "continue") and confidence >= 0.85:
                    pts = np.array(points, dtype=np.int32)
                    left, top = np.min(pts, axis=0)
                    right, bottom = np.max(pts, axis=0)
                    labels[label] = (max(0, left-2), max(0, top-2), min(w, right+3), min(h, bottom+3))
            if "correct" in labels and "continue" in labels:
                boxes = [labels["correct"], labels["continue"]]
                self._feedback_templates = [gray[t:b, l:r].copy() for l, t, r, b in boxes]
                left, top, right, bottom = labels["continue"]
                return [int(x + (left+right)//2), int(y + (top+bottom)//2)]
        return self.footer_continue(image)

    def footer_continue(self, image):
        """Recognize Correct + Continue/Summary in the footer, never a navigation tab."""
        h,w=image.shape[:2];x,y=int(w*.10),int(h*.84)
        crop=image[y:,x:int(w*.98)]
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        colored=cv2.inRange(hsv,(30,12,80),(130,255,255))
        if np.mean(colored>0)<.01:return None
        signature=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(400,80))
        cached=getattr(self,'_footer_signature',None)
        if cached is not None and np.mean(cv2.absdiff(cached,signature))<.4:
            return self._footer_target
        raw,_=self._ocr(crop)
        labels={}
        for points,text,confidence in raw or []:
            text=str(text).strip().lower().rstrip('!.')
            if confidence>=.85:
                labels[text]=np.mean(points,axis=0)
        correct=any(label in labels for label in ('correct','well done','correct answer'))
        control=next((labels[label] for label in ('continue','summary','view summary') if label in labels),None)
        target=[int(x+control[0]),int(y+control[1])] if correct and control is not None else None
        self._footer_items=[{'text':label,'center':[int(x+point[0]),int(y+point[1])],'confidence':.99}
                            for label,point in labels.items()]
        self._footer_signature,self._footer_target=signature,target
        return target

    def footer_navigation_items(self, image):
        """Reuse current footer grounding for navigation guards between full OCR refreshes."""
        self.footer_continue(image)
        return getattr(self,'_footer_items',[])

    def detect_enabled_submit(self, image: np.ndarray) -> Optional[list]:
        """Locate a blue Submit answer control, never a disabled gray button."""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (95, 110, 100), (125, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x < width * .55 or y < height * .75 or w < 80 or h < 25:
                continue
            button = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(button, cv2.COLOR_BGR2GRAY)
            template = getattr(self, "_submit_template", None)
            if template is not None and template.shape == gray.shape:
                if cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)[0, 0] > .96:
                    return [x+w//2, y+h//2]
            raw, _ = self._ocr(button)
            text = " ".join(str(item[1]).strip().lower() for item in raw or [] if item[2] >= .85)
            if text in ('submit answer', 'submit', 'check answer'):
                self._submit_template = gray.copy()
                return [x+w//2, y+h//2]
        return None

    def choice_buttons(self, image):
        """Read a complete contiguous letter-button group and its current selection locally."""
        if not self.answer_page_visible(image):return None
        h,w=image.shape[:2];x,y=int(w*.15),int(h*.25)
        crop=image[y:int(h*.88),x:int(w*.9)]
        key=hashlib.blake2b(crop.tobytes(),digest_size=16).digest()
        if getattr(self,'_choices_key',None)==key:return self._choices
        self._choices_key,self._choices=key,None
        choices={};seen=[]
        edges=cv2.morphologyEx(cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),40,130),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
        glyph_cache=getattr(self,'_choice_glyphs',{})
        for contour in sorted(contours,key=cv2.contourArea,reverse=True):
            l,t,bw,bh=cv2.boundingRect(contour);cx,cy=l+bw//2,t+bh//2
            if not (w*.015<bw<w*.12 and h*.02<bh<h*.1 and x<cx<w*.9 and y<cy<h*.88):continue
            if len(cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True))!=4:continue
            if any(abs(cx-a)<10 and abs(cy-b)<10 for a,b in seen):continue
            seen.append((cx,cy))
            button=image[t+8:t+bh-8,l+8:l+bw-8]
            hsv=cv2.cvtColor(button,cv2.COLOR_BGR2HSV)
            selected=np.mean(cv2.inRange(hsv,(95,90,130),(125,255,255))>0)>.3
            ink=cv2.inRange(hsv,(0,0,215),(179,40,255)) if selected else cv2.inRange(cv2.cvtColor(button,cv2.COLOR_BGR2GRAY),0,155)
            points=cv2.findNonZero(ink)
            if points is None:continue
            gx,gy,gw,gh=cv2.boundingRect(points)
            glyph=255-ink[gy:gy+gh,gx:gx+gw]
            glyph=cv2.copyMakeBorder(glyph,10,10,10,10,cv2.BORDER_CONSTANT,value=255)
            signature=hashlib.blake2b(glyph.tobytes(),digest_size=16).digest()
            if signature not in glyph_cache:
                raw,_=self._ocr(glyph,use_det=False,use_cls=False)
                glyph_cache[signature]=[(str(item[0]).strip(),item[1]) for item in raw or []]
            labels=glyph_cache[signature]
            if len(labels)!=1 or labels[0][1]<.85 or not re.fullmatch('[A-Z]',labels[0][0]):continue
            label=labels[0][0]
            if label in choices:return None
            choices[label]={'target':[cx,cy],'selected':bool(selected)}
        if len(glyph_cache)>64:glyph_cache.clear()
        self._choice_glyphs=glyph_cache
        letters=sorted(choices)
        if len(letters)<2 or letters!=[chr(65+i) for i in range(len(letters))]:return None
        self._choices_key,self._choices=key,choices
        return choices

    def letter_choice(self, image, expected):
        """Reconcile every saved choice, removing extras before selecting missing answers."""
        text=str(expected).strip()
        if not re.fullmatch(r'[A-Z](?:\s*(?:,|and|&)\s*[A-Z])*',text):return None
        wanted=re.findall('[A-Z]',text)
        if len(set(wanted))!=len(wanted):return None
        choices=self.choice_buttons(image)
        if not choices or any(letter not in choices for letter in wanted):return None
        extra=[letter for letter,choice in choices.items() if choice['selected'] and letter not in wanted]
        missing=[letter for letter in wanted if not choices[letter]['selected']]
        changes=extra+missing
        if not changes:return {'complete':True}
        return {'complete':False,'target':choices[changes[0]]['target'],'last_change':len(changes)==1}

    def selected_visual_answer(self, image):
        """Extract the square diagram inside a selected blue answer card, excluding controls."""
        h,w=image.shape[:2]
        mask=cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),(95,90,130),(125,255,255))
        for contour in sorted(cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea,reverse=True):
            x,y,bw,bh=cv2.boundingRect(contour)
            if not (w*.10<bw<w*.45 and h*.12<bh<h*.6 and y>h*.22):continue
            card=image[y:y+bh,x:x+bw]
            dark=cv2.inRange(cv2.cvtColor(card,cv2.COLOR_BGR2GRAY),0,70)
            for edge in sorted(cv2.findContours(dark,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea,reverse=True):
                l,t,fw,fh=cv2.boundingRect(edge)
                if fw>bw*.4 and fh>bh*.4 and .85<fw/fh<1.15:
                    return card[max(0,t-3):min(bh,t+fh+3),max(0,l-3):min(bw,l+fw+3)].copy()
            white=cv2.inRange(card,(245,245,245),(255,255,255))
            for panel in sorted(cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea,reverse=True):
                l,t,fw,fh=cv2.boundingRect(panel)
                if fw>bw*.6 and fh>bh*.4 and fw<bw*.98 and fh<bh*.95:
                    return card[t:t+fh,l:l+fw].copy()
        # Letter-only geometry pages can have several selected correct answers.
        choices=self.choice_buttons(image)
        letters=[letter for letter,choice in (choices or {}).items() if choice['selected']]
        if letters:
            text=', '.join(sorted(letters))
            answer=np.full((120,max(140,65*len(text)),3),255,np.uint8)
            cv2.putText(answer,text,(20,85),cv2.FONT_HERSHEY_SIMPLEX,2,(25,25,25),3,cv2.LINE_AA)
            return answer
        return None

    def task_completion(self, image):
        """Recognize Sparx's completed-task summary, rather than question prose."""
        items = self.extract_auxiliary_reference(image).get('ocr_results', [])
        labels = [(re.sub(r'\s+', '', item.get('text','').lower()), item.get('center',[0,0]))
                  for item in items if item.get('confidence',1) >= .85]
        tasks = [(text,center) for text,center in labels if re.fullmatch(r'task\d+',text)]
        navigation = [center for text,center in labels if text in ('backtotasklist','nexttask')]
        for text,center in labels:
            match = re.fullmatch(r'(\d+)outof(\d+)correct', text)
            if not match or not 0 < int(match[1]) == int(match[2]) <= 1000:
                continue
            if any(top[1] < center[1] for _,top in tasks) and any(bottom[1] > center[1] for bottom in navigation):
                return {'correct':int(match[1]), 'total':int(match[2])}
        return None

    def question_inventory(self, image):
        """Cache the tab row; input edits never need another full-screen OCR pass."""
        from question_set import read_inventory
        h,w=image.shape[:2];bottom=int(h*.32)
        crop=image[:bottom]
        signature=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(500,100))
        previous=getattr(self,'_inventory_signature',None)
        if (previous is not None and getattr(self,'_inventory_value',None) is not None
                and previous.shape==signature.shape and np.mean(cv2.absdiff(previous,signature))<.15):
            return self._inventory_value
        raw,_=self._ocr(crop,use_cls=False)
        items=[{'text':str(text),'confidence':confidence,'center':np.mean(points,axis=0).tolist()} for points,text,confidence in raw or []]
        self._independent_learning=any('independent' in item['text'].lower() for item in items)
        result=read_inventory(image,items)
        if result is None:
            result = self._read_inventory_slots(image, items)
        self._inventory_signature,self._inventory_value=signature,result
        return result

    def _read_inventory_slots(self, image, items):
        """Recover missed tab labels from their individual visible coloured slots."""
        from question_set import read_inventory
        h,w = image.shape[:2]
        summaries = [item for item in items if item.get('text','').strip().lower()=='summary'
                     and item.get('confidence',1)>=.75 and 0<item['center'][1]<h*.35]
        if len(summaries)!=1:
            return None
        summary = summaries[0];sx,sy=map(int,summary['center'])
        top,bottom=max(0,sy-20),min(h,sy+25)
        strip=image[top:bottom]
        mask=cv2.inRange(cv2.cvtColor(strip,cv2.COLOR_BGR2HSV),(35,8,35),(140,255,255))
        columns=np.mean(mask>0,axis=0)>.50
        edges=np.diff(np.r_[False,columns,False].astype(np.int8))
        slots=[(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))
               if 35 < b-a < w*.12 and b<sx]
        if not slots or len(slots)>26:
            return None
        recovered=[summary]
        prefix=None
        for index,(left,right) in enumerate(slots):
            known=[item for item in items if left<item.get('center',[0,0])[0]<right
                   and abs(item['center'][1]-sy)<25 and item.get('confidence',1)>=.75
                   and re.fullmatch(r'\d{1,3}[A-Z](?:\s*[✓✔×✕Xx])?',item.get('text','').strip().upper())]
            if len(known)==1:
                label=known[0]['text'].strip().upper()
                code=re.match(r'\d{1,3}[A-Z]',label)[0]
                if index==0 and code.endswith('A'):prefix=code[:-1]
                if prefix is not None and code==prefix+chr(ord('A')+index):
                    recovered.append(known[0]);continue
            crop=image[max(0,sy-24):min(h,sy+28),left:right]
            enlarged=cv2.copyMakeBorder(cv2.resize(crop,None,fx=3,fy=3),18,18,18,18,cv2.BORDER_CONSTANT,value=(255,255,255))
            labels,_=self._ocr(enlarged,use_cls=False)
            candidates=[]
            for attempt in range(2):
                for _,text,confidence in labels or []:
                    label=str(text).strip().upper().replace(' ','')
                    if index==8 and prefix=='1' and label[:1] in ('I','L','|'):
                        label='1'+label[1:]
                    ambiguous=re.fullmatch(r'(\d{1,3})[1L|]',label)
                    if ambiguous and index==8:label=ambiguous[1]+'I'
                    code=re.match(r'\d{1,3}[A-Z]',label)
                    if (confidence>=.75 and re.fullmatch(r'\d{1,3}[A-Z](?:[✓✔])?',label)
                            and (prefix is None or code[0]==prefix+chr(ord('A')+index))):
                        candidates.append(label)
                if candidates:break
                gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                mode=cv2.THRESH_BINARY if np.median(gray)>170 else cv2.THRESH_BINARY_INV
                glyph=cv2.threshold(gray,180,255,mode)[1]
                glyph=cv2.copyMakeBorder(cv2.resize(glyph,None,fx=3,fy=3),18,18,18,18,cv2.BORDER_CONSTANT,value=255)
                labels,_=self._ocr(glyph,use_cls=False)
            if len(set(candidates))!=1:
                return None
            recovered.append({'text':candidates[0],'center':[(left+right)//2,sy],'confidence':.99})
        return read_inventory(image,recovered)

    def read_question_code(self, image):
        """Read only the navy selected tab when full-screen OCR misses its code."""
        inventory=self.question_inventory(image)
        selected=[item['code'] for item in inventory or [] if item['selected']]
        if len(selected)==1:return selected[0]
        h,w = image.shape[:2]
        top = image[:int(h*.35)]
        mask=cv2.inRange(cv2.cvtColor(top,cv2.COLOR_BGR2HSV),(100,65,35),(140,255,155))
        candidates=[]
        for contour in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            x,y,bw,bh=cv2.boundingRect(contour)
            if 35 < bw < w*.14 and 25 < bh < h*.10 and h*.08 < y < h*.22 and w*.1 < x < w*.85:
                crop=top[y:y+bh,x:x+bw]
                if np.mean(mask[y:y+bh,x:x+bw]>0) < .5:
                    continue
                key=hashlib.blake2b(crop.tobytes(),digest_size=16).digest()
                cache=getattr(self,'_tab_codes',{})
                if key not in cache:
                    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                    glyph=cv2.threshold(gray,180,255,cv2.THRESH_BINARY_INV)[1]
                    glyph=cv2.copyMakeBorder(cv2.resize(glyph,None,fx=3,fy=3),20,20,20,20,cv2.BORDER_CONSTANT,value=255)
                    raw,_=self._ocr(glyph)
                    codes=[]
                    for _,text,confidence in raw or []:
                        match=re.fullmatch(r'(\d{1,3}[A-Z])(?:\s*[✓✔×✕Xx])?',str(text).strip())
                        if match and confidence >= .75:
                            codes.append(match[1])
                    cache[key]=codes[0] if len(set(codes))==1 else None
                    self._tab_codes=cache
                if cache[key]:candidates.append(cache[key])
        if len(set(candidates))==1:
            return candidates[0]
        inventory=self.question_inventory(image)
        selected=[item['code'] for item in inventory or [] if item['selected']]
        return selected[0] if len(selected)==1 else None

    def answer_page_visible(self, image):
        """Recognize the submission footer even when empty fields disable it."""
        h,w = image.shape[:2]
        crop = image[int(h*.90):, int(w*.70):]
        key = hashlib.blake2b(crop.tobytes(),digest_size=16).digest()
        if getattr(self,'_answer_footer_key',None) == key:
            return self._answer_footer_present
        raw,_ = self._ocr(crop)
        text = ' '.join(str(item[1]).lower().replace(' ','') for item in raw or [] if item[2] >= .8)
        present = any(label in text for label in ('submitanswer','submit','checkanswer'))
        self._answer_footer_key,self._answer_footer_present = key,present
        return present

    def numeric_fields(self, image):
        """Read bordered white answer slots in visual reading order, excluding keys."""
        from bookwork import numeric_value
        height, width = image.shape[:2]
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 40, 130)
        edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes, tinted_boxes = [], set()
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, w, h = cv2.boundingRect(contour)
            if not (width*.045 <= w <= width*.24 and height*.023 <= h <= height*.10
                    and height*.22 < y < height*.86 and w > h*1.4):
                continue
            if len(cv2.approxPolyDP(contour, .025*cv2.arcLength(contour, True), True)) != 4:
                continue
            inner = image[y+6:y+h-6, x+6:x+w-6]
            if not inner.size:
                continue
            tinted = np.mean(cv2.inRange(cv2.cvtColor(inner,cv2.COLOR_BGR2HSV),(95,8,200),(125,110,255)) > 0) > .78
            if np.mean(np.min(inner, axis=2) > 245) < .78 and not tinted:
                continue
            if any(abs(x+w/2-(a+c/2)) < 12 and abs(y+h/2-(b+d/2)) < 12 for a,b,c,d in boxes):
                continue
            boxes.append((x,y,w,h))
            if tinted:
                tinted_boxes.add((x,y))
        white_boxes = [box for box in boxes if box[:2] not in tinted_boxes]
        boxes = [box for box in boxes if box[:2] not in tinted_boxes or
                 sum(abs(box[1]+box[3]/2-(other[1]+other[3]/2))<12
                     and abs(box[2]-other[2])<box[2]*.25 for other in white_boxes)>=2]
        # A selected ordering card shares a row with white cards. Numeric keypad
        # buttons are blue too, but must never be counted as answer fields.
        tinted_boxes.intersection_update((box[0],box[1]) for box in boxes)
        fields = []
        for x,y,w,h in sorted(boxes, key=lambda b: (round((b[1]+b[3]/2)/30), b[0])):
            crop = image[y+5:y+h-5, x+5:x+w-5]
            labels = self.read_numeric_labels(crop)
            text = labels[0] if len(labels) == 1 else ''
            if not labels or numeric_value(text) is not None or text.lower().replace(' ', '') in ('enternumber', 'enteranswer'):
                fields.append({'target': [x+w//2,y+h//2], 'value': text, 'focused': bool(np.any(cv2.inRange(cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2HSV),(90,65,180),(125,255,255))))})
        if (tinted_boxes and 3 <= len(fields) <= 10
                and max(field['target'][1] for field in fields)-min(field['target'][1] for field in fields) < 12
                and max(field['target'][0] for field in fields)-min(field['target'][0] for field in fields) > width*.4):
            for field in fields:
                field['kind'] = 'ordered_card'
        return fields

    def keypad_controls(self, image):
        """Ground Sparx's four-row numeric keypad in visible key rectangles."""
        h,w = image.shape[:2]
        mask = cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),(95,2,200),(125,100,255))
        boxes=[]
        for contour in cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]:
            x,y,bw,bh=cv2.boundingRect(contour)
            if w*.035 < bw < w*.18 and h*.022 < bh < h*.08 and y > h*.35 and x > w*.15:
                boxes.append((x,y,bw,bh))
        rows=[]
        for box in sorted(boxes,key=lambda b:b[1]):
            if not rows or abs(box[1]-rows[-1][0][1]) > 12:
                rows.append([])
            rows[-1].append(box)
        for start in range(max(0,len(rows)-3)):
            grid=[sorted(row) for row in rows[start:start+4]]
            if [len(row) for row in grid] != [4,4,3,2]:
                continue
            # Recognize the top numeric row before assigning the fixed keypad layout.
            recognized=[]
            for x,y,bw,bh in grid[0][:3]:
                crop=image[y:y+bh,x:x+bw]
                key=hashlib.blake2b(crop.tobytes(),digest_size=16).digest()
                cache=getattr(self,'_key_labels',{})
                if key not in cache:
                    raw,_=self._ocr(crop)
                    cache[key]=''.join(str(t).strip() for _,t,c in raw or [] if c >= .8)
                    self._key_labels=cache
                recognized.append(cache[key])
            if recognized != ['7','8','9']:
                continue
            labels=[['7','8','9','delete'],['4','5','6','-'],['1','2','3'],['0','.']]
            return {label:[x+bw//2,y+bh//2] for row,names in zip(grid,labels) for (x,y,bw,bh),label in zip(row,names)}
        return None

    def read_stacked_fraction(self, crop: np.ndarray) -> Optional[str]:
        """Read numerator/denominator around a visible bar, without OCRing it as minus."""
        h,w=crop.shape[:2]
        ink=cv2.inRange(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),0,160)
        horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(7,int(h*.05))),np.uint8))
        for contour in sorted(cv2.findContours(horizontal,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea,reverse=True):
            x,y,bw,bh=cv2.boundingRect(contour)
            if not h*.25<y<h*.75 or bw<8 or bh>h*.08:continue
            margin=max(4,bw//2)
            left,right=max(0,x-margin),min(w,x+bw+margin)
            above=ink[:max(0,y-3),left:right]
            below=ink[min(h,y+bh+3):,left:right]
            top_points=cv2.findNonZero(above);bottom_points=cv2.findNonZero(below)
            if top_points is None or bottom_points is None:continue
            tx,ty,tw,th=cv2.boundingRect(top_points)
            bx,by,bottom_width,bottom_height=cv2.boundingRect(bottom_points)
            if th<h*.12 or bottom_height<h*.12 or not .60<=th/bottom_height<=1.67:continue
            center=x+bw/2
            if (abs(left+tx+tw/2-center)>max(5,bw*.55)
                    or abs(left+bx+bottom_width/2-center)>max(5,bw*.55)):continue
            outside=cv2.findNonZero(ink[:,:max(0,x-margin)])
            negative=False
            if outside is not None:
                ox,oy,ow,oh=cv2.boundingRect(outside)
                if oh>max(5,h*.07) or abs(oy+oh/2-(y+bh/2))>max(5,h*.07):continue
                negative=ow>=5
            numerator=self.read_numeric_labels(crop[:max(0,y-3),left:right])
            denominator=self.read_numeric_labels(crop[min(h,y+bh+3):,left:right])
            if (len(numerator)==len(denominator)==1 and re.fullmatch(r'\d+',numerator[0])
                    and re.fullmatch(r'\d+',denominator[0]) and int(denominator[0])!=0):
                return ('-' if negative else '')+numerator[0]+'/'+denominator[0]
        return None

    def expression_fields(self, image):
        """Read complete fraction/surd selections, rather than their coefficients."""
        from answer_values import expression_value
        h,w=image.shape[:2]
        edges=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),40,130)
        edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        boxes=[]
        for contour in sorted(cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea,reverse=True):
            x,y,bw,bh=cv2.boundingRect(contour)
            if not (w*.025<=bw<=w*.30 and h*.023<=bh<=h*.16 and h*.22<y<h*.86 and bw>bh*.40):continue
            if len(cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True))!=4:continue
            inner=image[y+6:y+bh-6,x+6:x+bw-6]
            if not inner.size or np.mean(np.min(inner,axis=2)>245)<.72:continue
            if any(abs(x+bw/2-(a+c/2))<12 and abs(y+bh/2-(b+d/2))<12 for a,b,c,d in boxes):continue
            boxes.append((x,y,bw,bh))
        fields=[]
        cache=getattr(self,'_expression_cache',{})
        for x,y,bw,bh in sorted(boxes,key=lambda b:(round((b[1]+b[3]/2)/30),b[0])):
            crop=image[y+5:y+bh-5,x+5:x+bw-5]
            key=hashlib.blake2b(crop.tobytes(),digest_size=16).digest()
            if key not in cache:
                from bookwork import BookworkDetector
                labels = BookworkDetector(self).read_answer_labels(crop)
                value = labels[0] if len(labels) == 1 else ''
                cache[key] = value if expression_value(value) is not None else ''
            fields.append({'target':[x+bw//2,y+bh//2],'value':cache[key],
                           'bounds':[x,y,bw,bh],'kind':'expression'})
        if len(cache)>128:cache={}
        self._expression_cache=cache
        return fields

    def answer_fields(self, image, expected):
        """Prefer scalar readback; use full mathematical fields for surds/fractions."""
        from answer_values import expression_value
        numeric=self.numeric_fields(image)
        if len(numeric)==len(expected) and all(expression_value(field['value'])==expression_value(answer)
                                               and expression_value(answer) is not None for field,answer in zip(numeric,expected)):
            return numeric
        expressions=self.expression_fields(image)
        return expressions if len(expressions)==len(expected) else numeric

    def numeric_choice_cards(self, image: np.ndarray) -> Optional[list]:
        """Ground unlabeled mathematical choice cards, including blue selections."""
        from answer_values import expression_value
        h,w=image.shape[:2]
        if self.keypad_controls(image):return None
        edges=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),40,130)
        edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        boxes=[]
        for contour in sorted(cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea,reverse=True):
            x,y,bw,bh=cv2.boundingRect(contour)
            if not (w*.04<bw<w*.28 and h*.04<bh<h*.16 and h*.30<y<h*.84 and bw>bh*.65):continue
            if len(cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True))!=4:continue
            if any(abs(x+bw/2-(a+c/2))<12 and abs(y+bh/2-(b+d/2))<12 for a,b,c,d in boxes):continue
            crop=image[y+6:y+bh-6,x+6:x+bw-6]
            blue=np.mean(cv2.inRange(cv2.cvtColor(crop,cv2.COLOR_BGR2HSV),(95,80,150),(125,255,255))>0)>.65
            if not blue and np.mean(np.min(crop,axis=2)>245)<.70:continue
            boxes.append((x,y,bw,bh))
        if not 3<=len(boxes)<=16:return None
        rows={round((y+bh/2)/40) for x,y,bw,bh in boxes}
        if len(rows)>3:return None
        cache=getattr(self,'_math_choice_cache',{})
        cards=[]
        for x,y,bw,bh in sorted(boxes,key=lambda b:(round((b[1]+b[3]/2)/40),b[0])):
            crop=image[y+6:y+bh-6,x+6:x+bw-6]
            selected=np.mean(cv2.inRange(cv2.cvtColor(crop,cv2.COLOR_BGR2HSV),(95,80,150),(125,255,255))>0)>.65
            key=hashlib.blake2b(crop.tobytes(),digest_size=16).digest()
            if key not in cache:
                normalized=cv2.cvtColor(255-cv2.inRange(crop,(200,200,200),(255,255,255)),cv2.COLOR_GRAY2BGR) if selected else crop
                value=self.read_stacked_fraction(normalized)
                if value is None:
                    padded=cv2.copyMakeBorder(cv2.resize(normalized,None,fx=3,fy=3),16,16,16,16,cv2.BORDER_CONSTANT,value=(255,255,255))
                    raw,_=self._ocr(padded,use_cls=False)
                    value=''.join(str(text).strip() for points,text,c in raw or [] if c>=.75)
                    value=re.sub(r'V(?=\d)','√',value)
                    ink=cv2.inRange(cv2.cvtColor(normalized,cv2.COLOR_BGR2GRAY),0,160)
                    components=sorted((cv2.boundingRect(c) for c in cv2.findContours(ink,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]),key=lambda b:b[0])
                    if len(components)>1:
                        cx,cy,cw,ch=components[0]
                        if cw>=7 and cw>ch*3 and not value.startswith(('-', '−')):
                            value='-'+value
                cache[key]=value if expression_value(value) is not None else ''
            cards.append({'target':[x+bw//2,y+bh//2],'value':cache[key], 'selected':selected,'kind':'math_choice',
                          'bounds':[x,y,bw,bh]})
        self._math_choice_cache=cache if len(cache)<128 else {}
        return cards if sum(bool(card['value']) for card in cards)>=3 else None

    def numeric_choice_state(self, image: np.ndarray, answers: list) -> Optional[dict]:
        """Compare actual selected values and identify one required selection change."""
        from answer_values import expression_value
        cards=self.numeric_choice_cards(image)
        if not cards:return None
        expected=[expression_value(answer) for answer in answers]
        if any(value is None for value in expected):return {'complete':False,'cards':cards,'target':None}
        selected=[card for card in cards if card['selected']]
        if any(expression_value(card['value']) is None for card in selected):
            return {'complete':False,'cards':selected,'target':None}
        extra=next((card for card in selected if expression_value(card['value']) not in expected),None)
        if extra:return {'complete':False,'cards':selected,'target':extra['target']}
        matched=[]
        for value in expected:
            choices=[card for card in cards if expression_value(card['value'])==value]
            if len(choices)!=1:return {'complete':False,'cards':selected,'target':None}
            matched.append(choices[0])
        missing=next((card for card in matched if not card['selected']),None)
        return {'complete':missing is None and len(selected)==len(expected),
                'cards':matched,'target':missing['target'] if missing else None}

    def input_context(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify visible answer controls before choosing an input shortcut."""
        from bookwork import numeric_value
        h,w=image.shape[:2]
        crop=image[int(h*.22):]
        key=hashlib.blake2b(crop.tobytes(),digest_size=16).digest()
        if getattr(self,'_input_context_key',None)==key:
            return self._input_context_value
        if not self.answer_page_visible(image):
            result={'kind':'question','instruction':'Open Answer before entering values'}
        else:
            fields=self.numeric_fields(image)
            expressions=self.expression_fields(image)
            keys=self.keypad_controls(image)
            values=[field['value'] for field in expressions if field['value']]
            symbolic=any(numeric_value(value) is None for value in values)
            fractional=any('/' in value for value in values)
            ordering=bool(fields and all(field.get('kind')=='ordered_card' for field in fields))
            choices=self.numeric_choice_cards(image)
            if choices:
                kind,instruction='math_choices','Select the actual mathematical values; these cards have no letter labels. Never invent A/B/C names or type into cards'
            elif ordering:
                kind,instruction='ordering','Drag answer cards; never type into them'
            elif symbolic and not keys:
                kind,instruction='expression_selector','Open each slot and click the complete matching expression; no typing or paste'
            elif symbolic or fractional:
                kind,instruction='math_controls','Use the fraction/surd controls and enter complete expressions; no numeric shortcut'
            elif keys:
                kind,instruction='numeric_keypad','Plain decimals may use numeric keyboard entry; fractions require the actual fraction controls'
            elif fields:
                kind,instruction='numeric_fields','Type only into a visible numeric field; inspect dropdown choices if a click opens them'
            else:
                kind,instruction='unknown','Inspect the actual answer slot with a click before selecting an entry method'
            result={'kind':kind,'instruction':instruction,'visible_values':values[:6],
                    'numeric_field_count':len(fields),'math_field_count':len(expressions)}
        self._input_context_key,self._input_context_value=key,result
        return result

    def keypad_delete_enabled(self, image: np.ndarray, keys: dict) -> bool:
        """Sparx draws an active delete key in red and a disabled empty key in grey."""
        if 'delete' not in keys:return False
        x,y=map(int,keys['delete']);h,w=image.shape[:2]
        crop=image[max(0,y-24):min(h,y+25),max(0,x-40):min(w,x+41)]
        if not crop.size:return False
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        red=cv2.inRange(hsv,(0,80,100),(12,255,255)) | cv2.inRange(hsv,(168,80,100),(180,255,255))
        return bool(np.count_nonzero(red)>=4)

    def read_numeric_labels(self, crop):
        """OCR a confirmed numeric field; normalize the narrow 1/I glyph only here."""
        key = hashlib.blake2b(crop.tobytes(), digest_size=16).digest()
        cache = getattr(self, '_numeric_cache', {})
        if key in cache:
            return cache[key]
        padded = cv2.copyMakeBorder(cv2.resize(crop,None,fx=2,fy=2),16,16,16,16,cv2.BORDER_CONSTANT,value=(255,255,255))
        raw, _ = self._ocr(padded, use_cls=False)
        labels = [str(text).strip() for _,text,confidence in raw or [] if confidence >= .80]
        if not raw or (len(raw) == 1 and str(raw[0][1]).strip() in ('I','l','L','|') and raw[0][2] >= .50):
            ink = cv2.inRange(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),0,150)
            points = cv2.findNonZero(ink)
            if points is not None:
                _,_,w,h = cv2.boundingRect(points)
                if h > 12 and .14 < w/h < .4:
                    labels = ['1']
        if len(cache) > 64:
            cache.clear()
        cache[key] = labels
        self._numeric_cache = cache
        return labels

    def verify_numeric_entry(self, image: np.ndarray, target: list, expected: str) -> bool:
        """Read the enclosing field, avoiding labels beside the input."""
        from bookwork import BookworkSession
        return BookworkSession(self, None).verify_entry(image, target, expected)

    def extract_auxiliary_reference(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Performs RapidOCR inference and OpenCV contour detection on the image.
        Outputs a structured reference table containing text boxes, centers,
        confidence scores, and detected candidate input/clickable regions.

        :param image: BGR numpy image array
        :return: Dict with 'ocr_results', 'candidate_boxes', and 'formatted_table'
        """
        cached_frame = getattr(self, "_aux_frame", None)
        if cached_frame is not None and np.array_equal(cached_frame, image):
            return self._aux_reference
        img_h, img_w = image.shape[:2]
        ocr_results: List[Dict[str, Any]] = []

        # 1. Run RapidOCR
        try:
            raw_ocr, _ = self._ocr(image)
            if raw_ocr:
                for item in raw_ocr:
                    # item format: [box_points, text, confidence]
                    # box_points is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    pts = np.array(item[0], dtype=np.int32)
                    text = str(item[1]).strip()
                    conf = float(item[2])

                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))

                    ocr_results.append({
                        "text": text,
                        "box": pts.tolist(),
                        "center": [cx, cy],
                        "confidence": round(conf, 3),
                    })
        except Exception as e:
            ocr_results.append({
                "text": f"[OCR Error: {e}]",
                "box": [[0, 0], [0, 0], [0, 0], [0, 0]],
                "center": [0, 0],
                "confidence": 0.0,
            })

        # 2. Run OpenCV contour extraction for candidate input boxes / buttons
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes: List[Dict[str, Any]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter noise and entire-screen borders
            if 20 <= w < (img_w * 0.95) and 15 <= h < (img_h * 0.95):
                aspect_ratio = float(w) / max(1, h)
                if 0.2 <= aspect_ratio <= 15.0:
                    candidate_boxes.append({
                        "bbox": [x, y, w, h],
                        "center": [x + w // 2, y + h // 2]
                    })

        # Keep top most prominent candidates to prevent prompt clutter
        candidate_boxes = sorted(
            candidate_boxes,
            key=lambda b: b["bbox"][2] * b["bbox"][3],
            reverse=True
        )[:20]

        # 3. Build human/model readable Markdown reference table
        lines = [
            f"Image Dimensions: {img_w}x{img_h}",
            "### Detected OCR Text Elements:",
            "| ID | Text | Center [X, Y] | Bounding Box | Confidence |",
            "|---|---|---|---|---|"
        ]
        if ocr_results:
            for idx, item in enumerate(ocr_results):
                pts_str = str(item["box"])
                lines.append(
                    f"| T{idx+1} | {item['text']} | {item['center']} | {pts_str} | {item['confidence']} |"
                )
        else:
            lines.append("| - | No text detected | - | - | - |")

        if candidate_boxes:
            lines.append("\n### Candidate Interactive Box Regions (OpenCV):")
            lines.append("| ID | Region [X, Y, W, H] | Center [X, Y] |")
            lines.append("|---|---|---|")
            for idx, c in enumerate(candidate_boxes[:10]):
                lines.append(f"| B{idx+1} | {c['bbox']} | {c['center']} |")

        formatted_table = "\n".join(lines)

        result = {
            "image_size": [img_w, img_h],
            "ocr_results": ocr_results,
            "candidate_boxes": candidate_boxes,
            "formatted_table": formatted_table
        }
        self._aux_frame = image.copy()
        self._aux_reference = result
        return result

    @staticmethod
    def compute_frame_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """
        Computes the visual difference ratio between two same-sized frames.
        Used to prevent misclicks caused by window switching or unexpected popups.

        :return: Difference score in range [0.0, 1.0]
        """
        if frame_a.shape != frame_b.shape:
            return 1.0

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_a, gray_b)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        non_zero = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        return float(non_zero) / float(max(1, total_pixels))
