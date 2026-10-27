#!/usr/bin/env python3
"""Verify compiled document counts, text, copied scientific inputs and hashes."""
from pathlib import Path
import hashlib
import json
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    rows=[]
    for rel,pages in [('01_CLIENT_PROPOSAL_CN.pdf',9),
                      ('07_MENTOR_BRIEF_CN.pdf',2),
                      ('02_PAPER_SEED/main.pdf',10)]:
        p=ROOT/rel
        reader=PdfReader(p)
        text='\n'.join(page.extract_text() or '' for page in reader.pages)
        assert len(reader.pages)==pages,(rel,len(reader.pages))
        assert '\ufffd' not in text and '\x00' not in text,rel
        assert '196,608' in text,rel
        if rel.endswith('main.pdf'):
            assert 'References' in text and '30/29/30' in text,rel
        rows.append({'path':rel,'pages':pages,'bytes':p.stat().st_size,
                     'sha256':sha(p),'text_characters':len(text)})
    paper=ROOT/'02_PAPER_SEED'
    log=(paper/'main.log').read_text(errors='replace')
    for bad in ['Overfull','undefined','LaTeX Warning:','! LaTeX Error']:
        assert bad not in log,bad
    copies=[]
    for source in sorted((ROOT/'03_FIGURES').glob('*.pdf')):
        dest=paper/'figures'/source.name
        assert dest.is_file() and sha(source)==sha(dest),source
        copies.append({'source':str(source.relative_to(ROOT)),
                       'paper_copy':str(dest.relative_to(ROOT)),
                       'sha256':sha(source)})
    for source,target in [('Tab1_environment.tex','environment.tex'),
                          ('Tab2_historical_results.tex','historical_results.tex'),
                          ('Tab3_new_pilot.tex','new_pilot.tex'),
                          ('Tab4_ablation_budget.tex','ablation_budget.tex')]:
        src=ROOT/'04_TABLES'/source;dst=paper/'tables'/target
        assert sha(src)==sha(dst),target
        copies.append({'source':str(src.relative_to(ROOT)),
                       'paper_copy':str(dst.relative_to(ROOT)),
                       'sha256':sha(src)})
    report={'status':'PASS','scope':'Compiled output, text and exact generated-input copies; visual review recorded separately',
            'documents':rows,'source_copies':copies,
            'latex_warnings':0,'latex_overfull_boxes':0,
            'required_sections':sorted(p.name for p in (paper/'sections').glob('*.tex')),
            'compiler':'system pdfTeX 3.141592653-2.6-1.40.22 / BibTeX; TeX Live 2022/dev/Debian',
            'renderer':'/usr/bin/pdftoppm (system Poppler)',
            'note':'Bundled Poppler requires newer GLIBC; system Poppler used without installation. No system maintenance performed.'}
    (ROOT/'DOCUMENT_QA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','documents':rows,'copied_sources':len(copies)},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
