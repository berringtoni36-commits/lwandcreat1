"""Inspect figure paths and text in the historical v09 compiled PDF.

Usage: python archive/v09_pdf_probe/probe_pdf_figures.py --probe
This diagnostic script does not generate revised figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PDF = ROOT / "稿件" / "v09历史稿" / "Overleaf_润色稿_v09_编译版.pdf"
PAGES = {3: 13, 4: 14, 5: 15, 6: 16, 7: 17}  # figure number -> 1-based page


def spans(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                t = s["text"].strip()
                if t:
                    out.append({"text": t, "rect": pymupdf.Rect(s["bbox"]), "size": s["size"]})
    return out


def to_num(t):
    t = t.replace("−", "-").replace("–", "-")
    try:
        return float(t)
    except ValueError:
        return None


def figure_region(page, fig_no):
    cap = [s for s in spans(page) if s["text"].startswith(f"Figure {fig_no}:")]
    if not cap:
        raise RuntimeError(f"caption of Figure {fig_no} not found")
    return pymupdf.Rect(0, 0, page.rect.width, cap[0]["rect"].y0)


def polyline(item_list):
    pts = []
    for it in item_list:
        if it[0] == "l":
            p1, p2 = it[1], it[2]
            if not pts or (abs(pts[-1][0] - p1.x) > 1e-6 or abs(pts[-1][1] - p1.y) > 1e-6):
                pts.append((p1.x, p1.y))
            pts.append((p2.x, p2.y))
    return np.array(pts)


def probe():
    doc = pymupdf.open(PDF)
    for fig, pno in PAGES.items():
        page = doc[pno - 1]
        region = figure_region(page, fig)
        print(f"=== Figure {fig} (page {pno}) region {region}")
        for s in spans(page):
            if s["rect"].intersects(region):
                print("   text", repr(s["text"]), [round(v, 1) for v in s["rect"]], round(s["size"], 1))
        for dr in page.get_drawings():
            if not dr["rect"].intersects(region):
                continue
            n = len(dr["items"])
            kinds = sorted({it[0] for it in dr["items"]})
            col = tuple(round(v, 3) for v in dr["color"]) if dr.get("color") else None
            fill = tuple(round(v, 3) for v in dr["fill"]) if dr.get("fill") else None
            if n >= 3 or col not in (None,):
                print("   path n=%d kinds=%s col=%s fill=%s w=%s dash=%s rect=%s" % (
                    n, kinds, col, fill, round(dr.get("width") or 0, 2), dr.get("dashes"),
                    [round(v, 1) for v in dr["rect"]]))


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
