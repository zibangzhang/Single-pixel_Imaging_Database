#!/usr/bin/env python3
"""Interactive viewer for Sample.dat (raw little-endian float64) files.

Tkinter application:
    - Each dataset is shown on its own "Directory:" row, with a color swatch,
      a "Close" button to remove it, and a "+" button to add a new row.
    - At least one "Directory:" row is always shown, even before any data is
      loaded; the row's "Open" button loads a folder into that row.
    - All loaded datasets are plotted on the same axes with different colors,
      so curves can be compared directly.
    - The plot is embedded via matplotlib's TkAgg backend.

Controls (in the plot):
    position slider (below the axes)  drag horizontally to scroll through the data
    mouse wheel                       zoom the horizontal (x) axis around the cursor
    left / right                      scroll by 1/4 window   (shift: a full window)
    up / down  (or PageUp/PageDown)   previous / next file of the active dataset
    Home / End                        jump to start / end
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import tkinter as tk
from tkinter import filedialog, colorchooser, ttk

import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Slider


# Palette used to assign a distinct color to each new dataset.
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


class Dataset:
    """One loaded directory: its file list, data and color."""

    def __init__(self, root, files, color):
        self.root = root            # Path of the directory the user opened
        self.files = files          # list[Path]
        self.color = color          # hex string
        self.data = np.fromfile(files[0], dtype="<f8")
        self.index = 0              # which file in self.files is loaded

    @property
    def current_file(self):
        return self.files[self.index]

    def load(self, i):
        self.index = i
        self.data = np.fromfile(self.files[i], dtype="<f8")


class RawSampleViewer:
    def __init__(self, master, initial_root=None):
        self.master = master
        master.title("Raw Sample Viewer")
        master.minsize(700, 400)

        self.datasets = []          # list[Dataset]
        self.active = 0             # index into self.datasets
        self.active_var = tk.IntVar(master, value=0)   # mirrors self.active for the radio buttons
        self._color_idx = 0

        # Number of extra empty Directory rows created with the "+" button.
        self._pending_rows = 0

        # Current number of samples shown in the window.
        self.win_width = 1000
        self.min_width = 10

        # ---- Directory rows container (at least one row always shown) ----
        self.rows_frame = ttk.Frame(master)
        self.rows_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        # ---- Matplotlib figure embedded in Tk ----
        self.fig = Figure(figsize=(10, 5.5))
        self.ax = self.fig.add_axes([0.10, 0.18, 0.86, 0.75])
        self.ax.set_xlabel("Index of sample")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.grid(alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        # ---- Position slider (only slider) ----
        self.s_pos = Slider(
            self.fig.add_axes([0.10, 0.06, 0.80, 0.04]),
            "Position", 0, 1, valinit=0, valfmt="%d",
        )
        self.s_pos.on_changed(self.redraw)

        # ---- Build the initial (empty) UI ----
        self._rebuild_rows()

        # ---- Load initial directory if provided ----
        if initial_root is not None:
            self.add_directory(Path(initial_root))
        else:
            self.redraw()

    # ------------------------------------------------------------------
    def _next_color(self):
        c = COLORS[self._color_idx % len(COLORS)]
        self._color_idx += 1
        return c

    # ------------------------------------------------------------------
    def current(self):
        if not self.datasets:
            return None
        return self.datasets[self.active]

    # ------------------------------------------------------------------
    def visible_width(self):
        ds = self.current()
        if ds is None:
            return 0
        return int(max(self.min_width, min(self.win_width, ds.data.size)))

    # ------------------------------------------------------------------
    def _sync_slider_range(self):
        """Match the position slider's range to the active dataset."""
        ds = self.current()
        n = ds.data.size if ds is not None else 1
        self.s_pos.valmax = n
        self.s_pos.ax.set_xlim(0, n)

    # ------------------------------------------------------------------
    def _set_pos(self, val):
        """Move the slider and redraw exactly once.

        Slider.set_val fires on_changed (-> redraw), so the slider's callback
        is muted here and redraw() is called explicitly.
        """
        self.s_pos.eventson = False
        self.s_pos.set_val(val)
        self.s_pos.eventson = True
        self.redraw()

    # ------------------------------------------------------------------
    def _reset_window(self):
        """Back to the start with the default zoom (used when the data set changes)."""
        ds = self.current()
        self._sync_slider_range()
        if ds is not None:
            self.win_width = min(1000, max(self.min_width, ds.data.size))
        self._set_pos(0)

    # ------------------------------------------------------------------
    def _pick_folder(self, row_index=None):
        """Ask the user for a folder and load it.

        If row_index points at an existing dataset, that dataset is replaced.
        Otherwise a new dataset is appended. An empty ("+") row that the user
        filled in is consumed (its pending slot is removed).
        """
        folder = filedialog.askdirectory(title="Select the folder containing Sample.dat")
        if folder:
            self.add_directory(Path(folder), row_index)

    # ------------------------------------------------------------------
    def add_directory(self, root: Path, row_index=None):
        """Load `root` into the row at `row_index`, or as a new dataset."""
        files = [root] if root.is_file() else sorted(root.rglob("Sample.dat"))
        if not files:
            self.ax.set_title(f"{root}  (no Sample.dat found)")
            self.canvas.draw_idle()
            return

        ds = Dataset(root, files, self._next_color())
        if row_index is not None and 0 <= row_index < len(self.datasets):
            self.datasets[row_index] = ds
            self.active = row_index
        else:
            self.datasets.append(ds)
            self.active = len(self.datasets) - 1
            if self._pending_rows > 0:
                self._pending_rows -= 1

        self._rebuild_rows()
        self._reset_window()

    # ------------------------------------------------------------------
    def remove_dataset(self, idx, redraw=True):
        if not (0 <= idx < len(self.datasets)):
            return
        del self.datasets[idx]
        if idx < self.active:
            self.active -= 1           # keep pointing at the same dataset
        self.active = max(0, min(self.active, len(self.datasets) - 1))
        self._rebuild_rows()
        if redraw:
            # Keep the current position/zoom; just re-fit the slider range.
            self._sync_slider_range()
            self._set_pos(self.s_pos.val)

    # ------------------------------------------------------------------
    def _on_select_active(self):
        """A row's radio button was clicked: make that dataset the active one."""
        self.active = self.active_var.get()
        self._sync_slider_range()
        self._set_pos(self.s_pos.val)
        # Hand keyboard focus back to the plot so left/right/up/down keep working.
        self.canvas.get_tk_widget().focus_set()

    # ------------------------------------------------------------------
    def on_pick_color(self, idx):
        if not (0 <= idx < len(self.datasets)):
            return
        ds = self.datasets[idx]
        current = mcolors.to_hex(ds.color)
        rgb, chosen = colorchooser.askcolor(color=current, title="Choose line color")
        if chosen:
            ds.color = chosen
            self._rebuild_rows()
            self.redraw()

    # ------------------------------------------------------------------
    def _rebuild_rows(self):
        """Recreate all Directory rows.

        Layout: one row per loaded dataset, followed by `_pending_rows`
        empty rows created with the "+" button. If there are no datasets
        and no pending rows, a single empty row is shown so that a
        'Directory:' line is always visible.
        """
        for w in self.rows_frame.winfo_children():
            w.destroy()
        self.active_var.set(self.active)

        for i, ds in enumerate(self.datasets):
            self._build_row(row_index=i, ds=ds)

        for _ in range(self._pending_rows):
            self._build_row(row_index=None, ds=None)

        # Always keep at least one empty row visible.
        if not self.datasets and self._pending_rows == 0:
            self._build_row(row_index=None, ds=None)

        # Force layout so the labels get their proper width right away.
        self.rows_frame.update_idletasks()

    # ------------------------------------------------------------------
    def _build_row(self, row_index, ds):
        """Build a single Directory row.

        row_index is the dataset index this row controls, or None for an
        empty row that will create a new dataset when 'Open' is clicked.

        Buttons are packed first (from the right) so the path label, packed
        last with expand=True, can occupy all remaining horizontal space.
        The path is set directly via `text=` (not textvariable) so it is
        displayed reliably regardless of widget lifetime.
        """
        row = ttk.Frame(self.rows_frame)
        row.pack(side=tk.TOP, fill=tk.X, pady=1)

        # --- Buttons first, packed from the right ---

        # "+" — add another empty Directory row.
        tk.Button(
            row, text="+", width=2, cursor="hand2",
            command=self._on_add_row,
        ).pack(side=tk.RIGHT, padx=(4, 0))

        # "Close" / color swatch — only for rows that already hold data.
        if ds is not None:
            tk.Button(
                row, text="Close", width=6, cursor="hand2",
                command=lambda idx=row_index: self.remove_dataset(idx),
            ).pack(side=tk.RIGHT, padx=(4, 0))

            color = mcolors.to_hex(ds.color)
            # A Label rather than a Button: macOS's native Button ignores bg.
            swatch = tk.Label(
                row, width=4, relief="raised", bd=2, cursor="hand2", bg=color,
            )
            swatch.bind("<Button-1>", lambda _e, idx=row_index: self.on_pick_color(idx))
            swatch.pack(side=tk.RIGHT, padx=(4, 0))

        # "Open" — load a folder into this row (or create a new dataset).
        tk.Button(
            row, text="Open", width=6, cursor="hand2",
            command=lambda idx=row_index: self._pick_folder(idx),
        ).pack(side=tk.RIGHT, padx=(4, 0))

        # --- Labels last, so they fill the remaining space ---
        # The radio button picks the "active" dataset: the one that the
        # slider range and the up/down file keys apply to.
        if ds is not None:
            tk.Radiobutton(
                row, variable=self.active_var, value=row_index,
                command=self._on_select_active, cursor="hand2",
            ).pack(side=tk.LEFT)
        tk.Label(row, text="Directory:").pack(side=tk.LEFT)

        path_text = str(ds.root) if ds is not None else "(none)"
        # Use a plain tk.Label with text= (a strong reference is unnecessary
        # and text= is more robust than textvariable for static content).
        path_lbl = tk.Label(
            row, text=path_text, anchor="w", relief="sunken", bd=1,
            bg="white", fg="black",
        )
        path_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))

    # ------------------------------------------------------------------
    def _on_add_row(self):
        """Add a new empty Directory row below the current ones."""
        self._pending_rows += 1
        self._rebuild_rows()

    # ------------------------------------------------------------------
    def redraw(self, _=None):
        # Clear the axes and redraw every dataset.
        self.ax.cla()
        self.ax.set_xlabel("Index of sample")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.grid(alpha=0.3)

        if not self.datasets:
            self.canvas.draw_idle()
            return

        w = self.visible_width()
        if w <= 0:
            self.canvas.draw_idle()
            return

        ds_active = self.current()
        start = int(self.s_pos.val)
        start = max(0, min(start, ds_active.data.size - w))

        lo_all, hi_all = np.inf, -np.inf
        for ds in self.datasets:
            seg = ds.data[start:start + w]
            if seg.size == 0:
                continue
            x = np.arange(start + 1, start + seg.size + 1)
            label = ds.current_file.parent.name or str(ds.current_file.parent)
            self.ax.plot(x, seg, lw=0.8, color=ds.color, label=label)
            lo_all = min(lo_all, seg.min())
            hi_all = max(hi_all, seg.max())

        if np.isfinite(lo_all) and np.isfinite(hi_all):
            pad = (hi_all - lo_all) * 0.05 or 1
            self.ax.set_ylim(lo_all - pad, hi_all + pad)
            self.ax.set_xlim(start + 1, start + w)

        self.ax.legend(loc="upper right", fontsize=8)
        self.ax.set_title(
            f"{ds_active.current_file.parent.name}  "
            f"[file {ds_active.index + 1}/{len(ds_active.files)}]   "
            f"samples {start + 1}–{start + w} of {ds_active.data.size}"
        )
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def on_scroll(self, e):
        ds = self.current()
        if ds is None or ds.data.size == 0:
            return
        if e.inaxes != self.ax:
            return

        old_w = self.visible_width()
        if old_w <= 0:
            return

        factor = 0.8 if e.step > 0 else 1.25
        new_w = int(round(old_w * factor))
        new_w = max(self.min_width, min(new_w, ds.data.size))
        if new_w == old_w:
            return

        old_start = int(self.s_pos.val)
        if e.xdata is not None:
            anchor = int(e.xdata)
        else:
            anchor = old_start + old_w // 2
        frac = (anchor - old_start) / max(1, old_w)

        self.win_width = new_w
        new_start = int(anchor - frac * new_w)
        new_start = max(0, min(new_start, ds.data.size - new_w))
        self._set_pos(new_start)

    # ------------------------------------------------------------------
    def set_file(self, i):
        ds = self.current()
        if ds is None:
            return
        i %= len(ds.files)
        ds.load(i)
        # Keep the current position and zoom so neighbouring files can be compared.
        self._sync_slider_range()
        self._set_pos(self.s_pos.val)

    # ------------------------------------------------------------------
    def on_key(self, e):
        ds = self.current()
        if ds is None or ds.data.size == 0:
            return
        step = self.visible_width() * (1 if "shift" in (e.key or "") else 0.25)
        k = (e.key or "").replace("shift+", "")
        if k == "right":
            self.s_pos.set_val(min(self.s_pos.val + step, ds.data.size - self.visible_width()))
        elif k == "left":
            self.s_pos.set_val(max(self.s_pos.val - step, 0))
        elif k in ("down", "pagedown"):
            self.set_file(ds.index + 1)
        elif k in ("up", "pageup"):
            self.set_file(ds.index - 1)
        elif k == "home":
            self.s_pos.set_val(0)
        elif k == "end":
            self.s_pos.set_val(ds.data.size)


def main():
    initial = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    root = tk.Tk()
    app = RawSampleViewer(root, initial_root=initial)
    root.mainloop()


if __name__ == "__main__":
    main()