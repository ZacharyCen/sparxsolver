"""Read and track one complete visible Sparx question set."""
import re
import cv2
import numpy as np


def read_inventory(frame, items):
    """Require a contiguous A–last row ending in Summary before trusting its size."""
    h,w=frame.shape[:2];tabs=[];summaries=[]
    for item in items:
        if item.get('confidence',1)<.75:continue
        x,y=map(int,item.get('center',[0,0]));text=item.get('text','').strip().upper()
        if not (0<y<h*.35 and 0<x<w):continue
        if text=='SUMMARY':summaries.append((x,y));continue
        match=re.fullmatch(r'(\d{1,3})([A-Z])(?:\s*[✓✔×✕Xx])?',text)
        if not match:continue
        patch=frame[max(0,y-12):y+13,max(0,x-22):x+23]
        hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
        navy=np.mean(cv2.inRange(hsv,(100,65,35),(140,255,155))>0)>.3
        green=np.mean(cv2.inRange(hsv,(35,65,60),(95,255,255))>0)>.25
        tabs.append({'code':match[1]+match[2],'target':[x,y],'selected':bool(navy),
                     'correct':bool(green or re.search('[✓✔]',text))})
    tabs.sort(key=lambda t:t['target'][0])
    if not tabs or len({t['code'][:-1] for t in tabs})!=1:return None
    letters=[t['code'][-1] for t in tabs]
    if letters!=[chr(ord('A')+i) for i in range(len(tabs))]:return None
    if not any(x>tabs[-1]['target'][0] and abs(y-tabs[-1]['target'][1])<35 for x,y in summaries):return None
    return tabs
