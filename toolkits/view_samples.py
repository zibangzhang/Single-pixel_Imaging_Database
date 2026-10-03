#!/usr/bin/env python3
"""Interactive viewer for Sample.dat (raw little-endian float64) files.

Usage:
    python3 view_samples.py [data_dir_or_file]

Controls:
    position slider (below the axes)  drag horizontally to scroll through the data
    width slider                      number of points shown in the window
    left / right                      scroll by 1/4 window   (shift: a full window)
    up / down  (or PageUp/PageDown)   previous / next file
    Home / End                        jump to start / end
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
files = [root] if root.is_file() else sorted(root.glob("*/Sample.dat"))
if not files:
    sys.exit(f"no Sample.dat found under {root}")

state = {"i": 0, "data": None}


def load(i):
    state["i"] = i
    state["data"] = np.fromfile(files[i], dtype="<f8")


load(0)
n = state["data"].size
init_w = 1000

fig = plt.figure(figsize=(14, 7))
ax = fig.add_axes([0.10, 0.25, 0.86, 0.68])
(line,) = ax.plot([], [], lw=0.8)
ax.set_xlabel("Index of sample")
ax.set_ylabel("Voltage (V)")
ax.grid(alpha=0.3)

s_pos = Slider(fig.add_axes([0.10, 0.12, 0.80, 0.04]), "position", 0, 1, valinit=0, valfmt="%d")
s_wid = Slider(fig.add_axes([0.10, 0.05, 0.80, 0.04]), "width", 50, 20000, valinit=init_w, valstep=50, valfmt="%d")


def width():
    return int(min(s_wid.val, state["data"].size))


def redraw(_=None):
    d = state["data"]
    w = width()
    start = int(s_pos.val)
    start = max(0, min(start, d.size - w))
    seg = d[start:start + w]
    line.set_data(np.arange(start + 1, start + seg.size + 1), seg)
    ax.set_xlim(start + 1, start + w)
    lo, hi = seg.min(), seg.max()
    pad = (hi - lo) * 0.05 or 1
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"[{state['i'] + 1}/{len(files)}] {files[state['i']].parent.name}   "
                 f"samples {start + 1}–{start + w} of {d.size}")
    fig.canvas.draw_idle()


def set_file(i):
    i %= len(files)
    load(i)
    s_pos.valmax = state["data"].size
    s_pos.ax.set_xlim(0, s_pos.valmax)
    s_pos.set_val(0)


def on_key(e):
    step = width() * (1 if "shift" in (e.key or "") else 0.25)
    k = (e.key or "").replace("shift+", "")
    if k == "right":
        s_pos.set_val(min(s_pos.val + step, state["data"].size - width()))
    elif k == "left":
        s_pos.set_val(max(s_pos.val - step, 0))
    elif k in ("down", "pagedown"):
        set_file(state["i"] + 1)
    elif k in ("up", "pageup"):
        set_file(state["i"] - 1)
    elif k == "home":
        s_pos.set_val(0)
    elif k == "end":
        s_pos.set_val(state["data"].size)


s_pos.valmax = n
s_pos.ax.set_xlim(0, n)
s_pos.on_changed(redraw)
s_wid.on_changed(redraw)
fig.canvas.mpl_connect("key_press_event", on_key)
redraw()
plt.show()
