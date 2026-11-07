"""Offline equation images shared by the desktop preview and portable exports."""
import base64
from functools import lru_cache
import hashlib
import html
from io import BytesIO
import re
import threading

_render_lock = threading.Lock()


@lru_cache(maxsize=256)
def equation_png(source: str) -> bytes:
    """Render the supported, already validated LaTeX grammar without a TeX install."""
    import matplotlib
    matplotlib.use('Agg')  # Export workers must never initialize a Tk GUI backend.
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    from PIL import Image

    # Mathtext accepts individual expressions, rather than TeX display environments.
    lines = re.split(r'\\\\|\n', source.strip())
    images = []
    with _render_lock:
        for line in lines:
            if not line.strip():
                continue
            line = re.sub(r'\\(?:dfrac|tfrac)\b', r'\\frac', line)
            line = re.sub(r'\\implies\b', r'\\Rightarrow', line)
            data = BytesIO()
            math_to_image('$' + line + '$', data, prop=FontProperties(size=19, math_fontfamily='stix'),
                          dpi=160, format='png', color='#142235')
            image = Image.open(BytesIO(data.getvalue())).convert('RGBA')
            images.append(image)
    if not images:
        raise ValueError('No equation to render')
    canvas = Image.new('RGB', (max(item.width for item in images) + 40,
                              sum(item.height for item in images) + 20 * (len(images) + 1)), 'white')
    y = 20
    for image in images:
        canvas.paste(image, (20, y), image)
        y += image.height + 20
    output = BytesIO()
    canvas.save(output, format='PNG')
    return output.getvalue()


def rendered_markdown(markdown: str, directory: str) -> tuple[str, dict]:
    """Replace display math with portable image links, preserving verified diagrams."""
    images = {}
    def replace(match):
        source = match[1].strip()
        name = 'equation_' + hashlib.sha256(source.encode()).hexdigest()[:20] + '.png'
        images[name] = equation_png(source)
        return '![](<'+ directory + '/' + name + '>)'
    return re.sub(r'\$\$\n(.*?)\n\$\$', replace, markdown, flags=re.S), images


def html_preview(markdown: str, assets: dict) -> str:
    """Make a standalone browser/print document with embedded images and no network."""
    blocks = []
    for line in markdown.splitlines():
        if line.startswith('## '):
            blocks.append('<h2>' + html.escape(line[3:]) + '</h2>')
        elif line.startswith('![]('):
            target = line[4:-1].strip('<>')
            name = target.rsplit('/', 1)[-1]
            if name not in assets:
                raise ValueError('Missing workings image: ' + name)
            encoded = base64.b64encode(assets[name]).decode('ascii')
            blocks.append('<img alt="Mathematical workings" src="data:image/png;base64,' + encoded + '">')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Homework workings</title><style>'
            'body{max-width:900px;margin:40px auto;padding:0 24px;color:#142235;background:white;'
            'font-family:Georgia,serif}h2{margin-top:32px;font-size:22px}img{display:block;'
            'max-width:100%;height:auto;margin:16px 0;break-inside:avoid}'
            '@media print{body{margin:0}h2{break-after:avoid}}</style>'
            + ''.join(blocks) + '</html>')
