"""Render four Kawaii report-only figures from a versioned source manifest.

For legacy fixture tests only: --allow-historical (NEVER a corrected report).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eeg_software.workflows.kawaii_session2.report_presentation import render_report_figures


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--allow-historical',action='store_true',help='Test historical table rendering only; NOT scientifically corrected')
    a=p.parse_args()
    result=render_report_figures(a.input_manifest,a.output_dir,allow_historical=a.allow_historical)
    print('Rendered',len(result['figures']),'files, status=',result['scientific_data_status'])
    print('Collaborator release approval: NOT GRANTED BY THIS RUN')
    return 0

if __name__=='__main__': raise SystemExit(main())
