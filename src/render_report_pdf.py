#!/usr/bin/env python3
"""Convert the checked Japanese Markdown report into a printable CJK PDF."""
from pathlib import Path
import html,markdown,re
from weasyprint import HTML, CSS
root=Path(__file__).resolve().parent.parent
text=(root/'REPORT_99_PERCENT.md').read_text()
# Markdown source keeps LaTeX; print PDF uses plain Unicode equations for fidelity.
# Explicit replacements, once each.
start=text.find('\\[\n')
if start>=0:
 end=text.find('\\]',start)
 if end>=0:
  text=text[:start]+'\n\n**局所比較コスト：** (I_p − R_(p+Δ))² + 2.5 (Gx_p − Gx_(p+Δ))² + 2.5 (Gy_p − Gy_(p+Δ))² + 0.004(Δx²+Δy²)\n\n'+text[end+2:]
start=text.find('\\[\n')
if start>=0:
 end=text.find('\\]',start)
 if end>=0:
  text=text[:start]+'\n\n**合成距離：** d = 0.12 × d_pixel + 0.88 × d_local\n\n'+text[end+2:]
html_body=markdown.markdown(text,extensions=['tables','fenced_code','sane_lists'])
style=r'''
@page { size: A4; margin: 19mm 17mm 19mm 17mm; @bottom-left { content: "MNIST 10-class classification | Experiment report"; color: #667085; font: 8pt "Noto Sans CJK JP"; } @bottom-right { content: counter(page) " / " counter(pages); color:#667085; font: 8pt "Noto Sans CJK JP"; } }
html { font-family: "Noto Sans CJK JP","Noto Sans", sans-serif; color:#202a36; font-size:9.4pt; line-height:1.66; }
body { font-family: "Noto Sans CJK JP", sans-serif; }
h1 { font-size:19pt; line-height:1.4; color:#132843; margin:0 0 12pt; padding-bottom:10pt; border-bottom: 2pt solid #324f72; }
h2 { font-size:12.5pt; color:#17365a; margin:19pt 0 8pt; border-bottom: .4pt solid #d1dae5; padding-bottom:5pt; break-after:avoid; }
h3 { font-size:10.3pt; color:#17365a; margin:11pt 0 5pt; break-after:avoid; }
p { margin:0 0 8pt; orphans:3; widows:3; }
a { color:#174d82; text-decoration:none; }
strong { color:#102f54; }
table { border-collapse:collapse; width:100%; margin:9pt 0 14pt; font-size:7.6pt; page-break-inside:auto; }
thead { display:table-header-group; }
tr { break-inside:avoid; }
th, td { border-bottom:.5pt solid #d2dae4; padding:4pt 6pt; vertical-align:top; }
th { color:#223f60; background:#edf2f8; font-weight:bold; }
code { font-family:"DejaVu Sans Mono","Noto Sans Mono CJK JP", monospace; font-size:8pt; }
pre { border:1px solid #d7e1ee; background:#f5f8fc; padding:8pt 10pt; white-space:pre-wrap; font-size:8pt; line-height:1.45; break-inside:avoid; }
li { margin-bottom:2pt; }
'''
doc=f'<!doctype html><html lang="ja"><head><meta charset="utf-8"></head><body>{html_body}</body></html>'
out=root/'REPORT_99_PERCENT.pdf'
HTML(string=doc,base_url=str(root)).write_pdf(str(out),stylesheets=[CSS(string=style)])
print('Rendered',out,'bytes',out.stat().st_size)
