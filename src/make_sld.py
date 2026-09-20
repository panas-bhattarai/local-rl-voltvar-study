"""Single-line diagram of the CIGRE European MV benchmark, drawn with TikZ.

The topology, line lengths and types, load sizes and switch states are read
from `data/cigre_mv_european_tb575.json`, so the drawing cannot drift from the
network the notebooks build. Only the positions of the symbols on the page are
set by hand, in the layout block below.

Writes `figures/cigre_mv_sld.pdf`, and `figures/cigre_mv_sld.png` as well when
PyMuPDF is installed. Needs a LaTeX installation with TikZ on the path. It is
not part of the study and no notebook depends on it.

    python src/make_sld.py
"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir))
OUT = os.path.join(ROOT, "figures")
with open(os.path.join(ROOT, "data", "cigre_mv_european_tb575.json")) as f:
    d = json.load(f)

PV_KW = {3: 690, 4: 690, 5: 680, 6: 680, 7: 740, 8: 740, 9: 740, 10: 740, 11: 740}
TR1_TAP = 4.375

# Loads are labelled the way the brochure tabulates them (Table 6.15): the
# residential and the commercial/industrial part of each node, in kVA.
load_kva = {}
for l in d["loads"]:
    parts = []
    for key, tag in (("residential", "R"), ("commercial_industrial", "CI")):
        if l.get(key) and l[key].get("S_kVA"):
            parts.append(rf"{tag} {l[key]['S_kVA']:g}")
    if parts:
        # the label, and the arm length it needs so that it clears the drop
        text = r"\enspace ".join(parts) + r"\,kVA"
        chars = len("  ".join(parts)) + 4
        load_kva[l["bus"]] = (text, max(1.62, 0.80 + 0.112 * chars / 2))
seg = {}
for l in d["lines"]:
    seg[(l["from_bus"], l["to_bus"])] = l
    seg[(l["to_bus"], l["from_bus"])] = l
sw = {s["id"]: s for s in d["switches"]}

# --- page layout -----------------------------------------------------------
HV_Y = 3.2
Y1, Y2, Y3, SPLIT_Y = 0.8, -1.2, -3.2, -5.2
R1, R2, R3, R4 = -7.0, -9.0, -11.0, -13.0
XL, XH, XR, XF2 = -0.9, 2.0, 5.3, 9.6
X7, Y7 = 2.8, -7.6

# bus -> (x, y, side the load and PV hang on)
POS = {1: (XH, Y1, "l"), 2: (XH, Y2, None), 3: (XH, Y3, "l"),
       4: (XL, R1, "l"), 5: (XL, R2, "l"), 6: (XL, R3, "l"),
       8: (XR, R1, "r"), 9: (XR, R2, "r"), 10: (XR, R3, "r"), 11: (XR, R4, "r"),
       7: (X7, Y7, "l"),
       12: (XF2, Y1, "r"), 13: (XF2, -1.6, "r"), 14: (XF2, -4.6, "r")}
BAR = 0.75
# buses 4 and 8 also receive a tie line from above, so their bars are longer
BARW = {4: 1.00, 8: 1.00}


def bw(b):
    return BARW.get(b, BAR)


def att(b):
    """Where a tie line meets a bus bar: inside the tip, towards the centre."""
    return 0.80 if b in BARW else 0.50

T = []
add = T.append


def bus_bar(b, x, y, side):
    """The bus number goes on the side away from the load and PV drops."""
    s = -1 if side == "r" else 1
    add(rf"\draw[bus] ({x - bw(b)},{y}) -- ({x + bw(b)},{y});")
    if b != 7:
        add(rf"\node[bnum] at ({x + s * (bw(b) + 0.40)},{y}) {{{b}}};")


def inverter(x, y, face, h=0.30):
    """Converter symbol; `face` is the edge the connection arrives on, and the
    AC half is drawn against it so the grid never meets the DC side."""
    add(rf"\draw[pvbox] ({x - h},{y - h}) rectangle ({x + h},{y + h});")
    if face == "l":
        add(rf"\draw[pvbox] ({x - h},{y + h}) -- ({x + h},{y - h});")
        add(rf"\node[glyph] at ({x - h * 0.45},{y - h * 0.48}) {{$\sim$}};")
        add(rf"\node[glyph] at ({x + h * 0.45},{y + h * 0.42}) {{$=$}};")
    else:
        add(rf"\draw[pvbox] ({x - h},{y - h}) -- ({x + h},{y + h});")
        add(rf"\node[glyph] at ({x - h * 0.45},{y + h * 0.42}) {{$=$}};")
        add(rf"\node[glyph] at ({x + h * 0.45},{y - h * 0.48}) {{$\sim$}};")


def taps(b, x, y, side, load_drop=0.45):
    """Load and PV both drop off the bus bar and then turn out to the side."""
    s = -1 if side == "l" else 1
    if b in load_kva:
        text, arm = load_kva[b]
        tx, ay = x + s * 0.52, y - load_drop
        add(rf"\draw[tap] ({tx},{y}) -- ({tx},{ay});")
        add(rf"\draw[load,-{{Triangle[length=2.6mm,width=2.0mm]}}] "
            rf"({tx},{ay}) -- ({x + s * arm},{ay});")
        add(rf"\node[rating,anchor=south] at ({x + s * arm},{ay + 0.08}) "
            rf"{{{text}}};")
    if b in PV_KW:
        tx, py, bx = x + s * 0.25, y - 1.05, x + s * 1.40
        add(rf"\draw[tap] ({tx},{y}) -- ({tx},{py}) -- ({bx - s * 0.30},{py});")
        inverter(bx, py, "l" if s > 0 else "r")
        add(rf"\node[rating,anchor=north,text=pv] at ({bx},{py - 0.36}) "
            rf"{{PV {PV_KW[b]}\,kW}};")


def line(a, b, path, lbl_at, anchor):
    s = seg[(a, b)]
    sty = "cable" if s["installation"] == "underground" else "ohl"
    add(rf"\draw[{sty}] {path};")
    add(rf"\node[seg,anchor={anchor}] at {lbl_at} {{{s['length_km']:.2f}\,km}};")


def tie(sid, path, xsw, ysw, above=False):
    """Tie line drawn to the bus it really reaches, with an open switch on it."""
    s = sw[sid]
    kind = "overhead" if seg[tuple(s["between_buses"])]["installation"] == "overhead" \
        else "cable"
    add(rf"\draw[tie] {path};")
    add(rf"\fill[white] ({xsw - 0.46},{ysw - 0.14}) rectangle ({xsw + 0.46},{ysw + 0.52});")
    add(rf"\fill[tiec] ({xsw - 0.42},{ysw}) circle (0.06);")
    add(rf"\fill[tiec] ({xsw + 0.42},{ysw}) circle (0.06);")
    add(rf"\draw[tiesw] ({xsw - 0.42},{ysw}) -- ({xsw + 0.30},{ysw + 0.42});")
    a, bb = s["between_buses"]
    anchor, dy = ("south", 0.60) if above else ("north", -0.28)
    add(rf"\node[seg,anchor={anchor}] at ({xsw},{ysw + dy}) "
        rf"{{{sid} open, bus {a}--bus {bb}, {s['line_length_km']:.2f}\,km {kind}}};")


# --- HV source and transformers -------------------------------------------
add(rf"\draw[bus] (1.3,{HV_Y}) -- (9.7,{HV_Y});")
add(rf"\node[bnum] at (1.00,{HV_Y}) {{0}};")
add(rf"\node[seg,anchor=west] at (9.95,{HV_Y}) {{110\,kV}};")
add(rf"\node[seg,anchor=south east] at ({XH - 0.20},{Y1 + 0.12}) {{20\,kV}};")
add(rf"\draw[cable] (5.50,{HV_Y}) -- (5.50,{HV_Y + 0.80});")
add(rf"\draw[cable] (5.50,{HV_Y + 1.15}) circle (0.35);")
add(rf"\node[glyph] at (5.50,{HV_Y + 1.15}) {{$\sim$}};")
add(rf"\node[val,anchor=west,align=left] at (6.05,{HV_Y + 1.15}) "
    r"{110\,kV subtransmission\\ $S_{\mathrm{sc}}$ = 5000\,MVA, $X/R$ = 10};")

for name, x, tap in (("TR1", XH, TR1_TAP), ("TR2", XF2, 3.125)):
    yc = (HV_Y + Y1) / 2
    add(rf"\draw[cable] ({x},{HV_Y}) -- ({x},{yc + 0.60});")
    add(rf"\draw[cable] ({x},{yc - 0.60}) -- ({x},{Y1});")
    add(rf"\draw[cable] ({x},{yc + 0.27}) circle (0.33);")
    add(rf"\draw[cable] ({x},{yc - 0.27}) circle (0.33);")
    add(rf"\node[val,anchor=west,align=left,text=tapc] at ({x + 0.55},{yc + 0.30}) "
        rf"{{{name}\\ 110/20\,kV, 25\,MVA\\ $u_k$ = 12\,\%\\ tap {tap:+.3f}\,\%}};")

# --- feeder 1 --------------------------------------------------------------
line(1, 2, f"({XH},{Y1}) -- ({XH},{Y2})", (XH, (Y1 + Y2) / 2), "center")
line(2, 3, f"({XH},{Y2}) -- ({XH},{Y3})", (XH, (Y2 + Y3) / 2), "center")
add(rf"\draw[cable] ({XH},{Y3}) -- ({XH},{SPLIT_Y});")
line(3, 4, f"({XH},{SPLIT_Y}) -- ({XL},{SPLIT_Y}) -- ({XL},{R1})",
     (XL, (SPLIT_Y + R1) / 2), "center")
line(3, 8, f"({XH},{SPLIT_Y}) -- ({XR},{SPLIT_Y}) -- ({XR},{R1})",
     (XR, (SPLIT_Y + R1) / 2), "center")
line(4, 5, f"({XL},{R1}) -- ({XL},{R2})", (XL, (R1 + R2) / 2 - 0.45), "center")
line(5, 6, f"({XL},{R2}) -- ({XL},{R3})", (XL, (R2 + R3) / 2 - 0.45), "center")
line(8, 9, f"({XR},{R1}) -- ({XR},{R2})", (XR, (R1 + R2) / 2 - 0.45), "center")
line(9, 10, f"({XR},{R2}) -- ({XR},{R3})", (XR, (R2 + R3) / 2 - 0.45), "center")
line(10, 11, f"({XR},{R3}) -- ({XR},{R4})", (XR, (R3 + R4) / 2 - 0.45), "center")

# bus 7 hangs off bus 8: down from the bar, then across to the side of bus 7
line(7, 8, f"({XR - att(8)},{R1}) -- ({XR - att(8)},{R1 + 0.55}) -- "
           f"({X7 + 0.35},{R1 + 0.55}) -- ({X7 + 0.35},{Y7})",
     ((XR - att(8) + X7 + 0.35) / 2, R1 + 0.60), "south")
add(rf"\node[bnum] at ({X7 - 0.15},{Y7 + 0.45}) {{7}};")

# --- feeder 2 --------------------------------------------------------------
line(12, 13, f"({XF2},{Y1}) -- ({XF2},{-1.6})", (XF2, (Y1 - 1.6) / 2), "center")
line(13, 14, f"({XF2},{-1.6}) -- ({XF2},{-4.6})", (XF2, -3.1), "center")

for b, (x, y, side) in POS.items():
    bus_bar(b, x, y, side)
    if side:
        taps(b, x, y, side, load_drop=0.45)

# --- normally-open ties, drawn all the way to the bus they reach -----------
tie("S1", f"({XF2 - att(14)},{-4.6}) -- ({XF2 - att(14)},{-5.9}) -- "
          f"({XR + att(8)},{-5.9}) -- ({XR + att(8)},{R1})", 7.60, -5.9, above=True)
tie("S2", f"({XL + att(6)},{R3}) -- ({XL + att(6)},{-12.9}) -- "
          f"({X7 + 0.15},{-12.9}) -- ({X7 + 0.15},{Y7})", 1.40, -12.9)
tie("S3", f"({XR - att(11)},{R4}) -- ({XR - att(11)},{-14.2}) -- "
          f"(-3.80,{-14.2}) -- (-3.80,{R1 + 0.60}) -- "
          f"({XL - att(4)},{R1 + 0.60}) -- ({XL - att(4)},{R1})", 3.30, -14.2)

# --- titles and legend -----------------------------------------------------
add(r"\node[anchor=north west,align=left] at (-4.0,6.3) {\large\bfseries "
    r"CIGRE European MV distribution benchmark\\[2pt]"
    r"\normalsize\mdseries 20\,kV, 50\,Hz, radial base case. "
    r"Nine PV inverters on feeder 1, 6.44\,MW in total.};")

LX, LY = 8.20, -6.4
add(rf"\node[anchor=north west,align=left,draw=black!25,line width=0.5pt,"
    rf"rounded corners=2pt,inner sep=7pt,fill=black!2] at ({LX},{LY}) {{%")
add(r"\footnotesize\begin{tabular}{@{}l@{\ \ }l@{}}")
add(r"\tikz{\draw[cable] (0,0) -- (0.62,0);} & 20\,kV cable, NA2XS2Y 120\,mm$^2$\\[3pt]")
add(r"\tikz{\draw[ohl] (0,0) -- (0.62,0);} & 20\,kV overhead, A1 63\,mm$^2$\\[3pt]")
add(r"\tikz{\draw[tie] (0,0) -- (0.62,0);} & tie line, switch open\\[3pt]")
add(r"\tikz{\draw[load,-{Triangle[length=2.2mm,width=1.7mm]}] (0,0) -- (0.62,0);} "
    r"& load, peak apparent power\\[3pt]")
add(r"\tikz{\draw[pvbox] (0.16,-0.21) rectangle (0.58,0.21);"
    r"\draw[pvbox] (0.16,0.21) -- (0.58,-0.21);} & PV inverter, $P_{\mathrm{mpp}}$\\")
add(r"\end{tabular}};")
add(rf"\node[anchor=north west,align=left,text width=5.0cm] at ({LX},{LY - 3.15}) "
    r"{\footnotesize R and CI are the residential and the commercial/industrial part "
    r"of a load, split as in Table 6.15 of the brochure. Inverter rating is "
    r"$1.1\,P_{\mathrm{mpp}}$, so each unit keeps reactive capability at full sun. "
    r"Bus 1 and bus 12 carry the other feeders on the same transformer and are not "
    r"part of the modelled feeders.};")
add(rf"\node[anchor=north west,align=left,text width=5.0cm,text=black!55] "
    rf"at ({LX},{LY - 6.55}) "
    r"{\scriptsize Network, loads and line data: CIGRE Technical Brochure 575, "
    r"Section 6.2. PV ratings: Wagle et al., \emph{Front.\ Energy Res.} "
    r"10:1054870 (2023), Table 1. The 1.5\,MW wind unit of the brochure at bus 7 "
    r"is replaced by PV here.};")

PRE = r"""\documentclass[border=8pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc}
\definecolor{pv}{RGB}{13,79,138}
\definecolor{loadc}{RGB}{176,72,24}
\definecolor{tiec}{RGB}{125,125,125}
\definecolor{tapc}{RGB}{90,90,90}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,
  cable/.style={line width=1.45pt,black,line cap=round},
  ohl/.style={line width=0.65pt,black},
  bus/.style={line width=3.0pt,black,line cap=round},
  tap/.style={line width=0.65pt,black},
  tie/.style={line width=0.9pt,tiec,dash pattern=on 3pt off 2.4pt},
  tiesw/.style={line width=1.0pt,tiec,line cap=round},
  load/.style={line width=0.9pt,loadc},
  pvbox/.style={line width=0.9pt,pv},
  bnum/.style={circle,draw=black,line width=0.6pt,fill=white,inner sep=0pt,
               minimum size=4.6mm,font=\footnotesize\bfseries},
  val/.style={font=\footnotesize,inner sep=1pt},
  rating/.style={font=\scriptsize,inner sep=1pt},
  seg/.style={font=\scriptsize,text=black!55,inner sep=1pt,fill=white},
  glyph/.style={font=\scriptsize,inner sep=0pt}]
"""
tex = PRE + "\n".join(T) + "\n\\end{tikzpicture}\n\\end{document}\n"
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "cigre_mv_sld.tex"), "w", encoding="utf-8") as f:
    f.write(tex)

r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    "cigre_mv_sld.tex"], cwd=OUT, capture_output=True, text=True)
if r.returncode:
    out = r.stdout.splitlines()
    first = next((i for i, ln in enumerate(out) if ln.startswith("!")), None)
    print("\n".join(out[first:first + 12] if first is not None else out[-25:]))
    sys.exit(1)
for ext in (".aux", ".log"):
    os.remove(os.path.join(OUT, "cigre_mv_sld" + ext))
print("wrote figures/cigre_mv_sld.pdf")

try:
    import pymupdf
except ImportError:
    print("PyMuPDF is not installed, skipping the PNG")
else:
    page = pymupdf.open(os.path.join(OUT, "cigre_mv_sld.pdf"))[0]
    page.get_pixmap(dpi=200).save(os.path.join(OUT, "cigre_mv_sld.png"))
    print("wrote figures/cigre_mv_sld.png")
