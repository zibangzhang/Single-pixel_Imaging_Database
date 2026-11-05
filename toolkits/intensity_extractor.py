#!/usr/bin/env python3
"""Turn a Sample.dat sample array into a single-pixel intensity array.

A Sample.dat file is one long sample array: little-endian float64 voltages,
stored pattern by pattern. Pattern j occupies samples
``[j * n, (j + 1) * n)``. Averaging a user-chosen segment of those n samples
produces one intensity. m patterns therefore produce an intensity array of
length m.

The segment is 1-based and inclusive. Samples ``start`` through ``end`` are
the same window as ``data[start - 1:end]``. The output keeps the original
pattern order.

Tkinter application:
    Open file / Open folder     load one Sample.dat, or every Sample.dat under a folder
    Samples per pattern         n, the length of each pattern block
    Segment start / end         which samples inside every pattern are averaged
    Pattern                     which pattern the sample view starts at
    Blocks                      how many consecutive patterns the sample view shows
    mouse wheel                 zoom the intensity axis around the cursor
    left / right                scroll the intensity view (shift: a full window)
    up / down                   previous / next Sample.dat
    Home / End                  jump to the start / end of the intensity array
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# The repository root (this file lives in <root>/toolkits), where the Open dialogs start.
REPO_ROOT = Path(__file__).resolve().parent.parent


def infer_n_sample(path: Path) -> int | None:
    """Read a ``<n>samples`` token from a file or folder name, if present."""
    matches = re.findall(r"(\d+)samples", str(path), flags=re.IGNORECASE)
    if not matches:
        return None
    value = int(matches[-1])      # the innermost name wins over parent folders
    return value if value >= 1 else None


def pattern_matrix(samples: np.ndarray, n_sample: int) -> np.ndarray:
    """Reshape a sample array into ``(n_sample, n_pattern)``, one column per pattern.

    Trailing samples that do not fill a whole pattern are left out.
    """
    if n_sample < 1:
        raise ValueError("samples per pattern must be at least 1")
    n_pattern = int(samples.size // n_sample)
    if n_pattern < 1:
        raise ValueError(
            f"{samples.size} samples cannot fill one pattern of length {n_sample}"
        )
    usable = n_pattern * n_sample
    return np.asarray(samples[:usable], dtype=np.float64).reshape((n_sample, n_pattern), order="F")


def intensity_from_samples(samples: np.ndarray, n_sample: int, start: int, end: int) -> np.ndarray:
    """Average samples ``start``..``end`` (1-based, inclusive) of every pattern.

    Returns the intensity array, one value per pattern.
    """
    matrix = pattern_matrix(samples, n_sample)
    if not 1 <= start <= end <= n_sample:
        raise ValueError(f"segment [{start}, {end}] is outside 1..{n_sample}")
    return matrix[start - 1:end, :].mean(axis=0)


class IntensityExtractor:
    def __init__(self, master, initial_path=None):
        self.master = master
        master.title("Intensity Extractor")
        master.minsize(860, 640)

        self.samples = None          # 1D sample array of the open file
        self.files = []              # Sample.dat paths when a folder was opened
        self.file_index = 0
        self.n_sample = 50
        self.start = 1               # 1-based inclusive segment
        self.end = 50
        self.pattern = 1             # 1-based pattern where the sample view starts
        self._followed_pattern = None
        self.blocks = 4              # consecutive patterns drawn in the sample view
        self.intensity = None

        self.win_width = 2000
        self.min_width = 10
        self._updating = True

        self._build_controls()
        self._build_figure()
        self._updating = False
        self._apply_state_to_widgets()
        self.redraw()

        if initial_path is not None:
            self.load_path(Path(initial_path))

    # ------------------------------------------------------------------
    def _build_controls(self):
        bar = ttk.Frame(self.master)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 2))

        ttk.Button(bar, text="Open file", command=self._open_file).pack(side=tk.LEFT)
        ttk.Button(bar, text="Open folder", command=self._open_folder).pack(side=tk.LEFT, padx=(6, 0))
        self.prev_btn = ttk.Button(bar, text="◀", width=3, command=lambda: self.step_file(-1))
        self.prev_btn.pack(side=tk.LEFT, padx=(12, 0))
        self.next_btn = ttk.Button(bar, text="▶", width=3, command=lambda: self.step_file(1))
        self.next_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.path_label = ttk.Label(bar, text="No file loaded", anchor="w")
        self.path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        params = ttk.Frame(self.master)
        params.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)

        ttk.Label(params, text="Samples per pattern").pack(side=tk.LEFT)
        self.n_var = tk.StringVar(value=str(self.n_sample))
        self.n_spin = ttk.Spinbox(
            params, from_=1, to=1_000_000, width=7, textvariable=self.n_var,
            command=self.apply_from_widgets,
        )
        self.n_spin.pack(side=tk.LEFT, padx=(4, 16))
        self._bind_apply(self.n_spin)

        ttk.Label(params, text="Blocks").pack(side=tk.LEFT)
        self.blocks_var = tk.StringVar(value=str(self.blocks))
        self.blocks_spin = ttk.Spinbox(
            params, from_=1, to=24, width=4, textvariable=self.blocks_var,
            command=self.apply_from_widgets,
        )
        self.blocks_spin.pack(side=tk.LEFT, padx=(4, 16))
        self._bind_apply(self.blocks_spin)

        self.count_label = ttk.Label(params, text="Patterns: —")
        self.count_label.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(params, text="Save intensity", command=self.save_intensity).pack(side=tk.RIGHT)
        self.save_all_btn = ttk.Button(params, text="Save all files", command=self.save_all)
        self.save_all_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self.format_var = tk.StringVar(value=".dat")
        ttk.Combobox(
            params, textvariable=self.format_var, values=(".dat", ".npy"),
            width=5, state="readonly",
        ).pack(side=tk.RIGHT, padx=(0, 4))
        ttk.Label(params, text="Batch format").pack(side=tk.RIGHT)

        segment = ttk.LabelFrame(self.master, text="Segment averaged inside every pattern (1-based, inclusive)")
        segment.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        self.start_scale, self.start_var = self._labeled_scale(segment, "Start", self.start)
        self.end_scale, self.end_var = self._labeled_scale(segment, "End", self.end)
        self.pattern_scale, self.pattern_var = self._labeled_scale(segment, "Pattern", self.pattern)

        self.info_label = ttk.Label(self.master, text="Load a Sample.dat file to build the intensity array.")
        self.info_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 4))

    def _labeled_scale(self, parent, caption, initial):
        row = ttk.Frame(parent)
        row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text=caption, width=10).pack(side=tk.LEFT)
        var = tk.StringVar(value=str(initial))
        spin = ttk.Spinbox(row, from_=1, to=1, width=8, textvariable=var, command=self.apply_from_widgets)
        spin.pack(side=tk.LEFT)
        self._bind_apply(spin)
        scale = ttk.Scale(row, from_=1, to=1, orient=tk.HORIZONTAL)
        scale.config(command=lambda value, linked=var: self._on_linked_scale(linked, value))
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        scale.set(initial)
        return scale, var

    def _on_linked_scale(self, var, value):
        if self._updating:
            return
        var.set(str(int(round(float(value)))))
        self.apply_from_widgets()

    def _bind_apply(self, widget):
        widget.bind("<Return>", lambda _e: self.apply_from_widgets())
        widget.bind("<FocusOut>", lambda _e: self.apply_from_widgets())

    # ------------------------------------------------------------------
    def _build_figure(self):
        self.fig = Figure(figsize=(10, 6.4))
        self.ax_sample = self.fig.add_axes([0.08, 0.58, 0.88, 0.36])
        self.ax_intensity = self.fig.add_axes([0.08, 0.10, 0.88, 0.36])
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.get_tk_widget().bind("<Button-1>", lambda _e: self.canvas.get_tk_widget().focus_set())

        self._touchpad_acc = 0.0
        try:
            self.canvas.get_tk_widget().bind("<TouchpadScroll>", self._on_touchpad_scroll)
        except tk.TclError:
            pass

        nav = ttk.Frame(self.master)
        nav.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(nav, text="Intensity position").pack(side=tk.LEFT)
        self.pos_var = tk.DoubleVar(value=0)
        self._pos_maximum = None
        self.pos_scale = ttk.Scale(
            nav, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.pos_var,
            command=self._on_position,
        )
        self.pos_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.pos_label = ttk.Label(nav, text="", width=42, anchor="e")
        self.pos_label.pack(side=tk.LEFT)

    def _on_position(self, value):
        """Redraw from the scale's new value.

        The command can run before ``pos_var`` itself updates, so the title and
        the plotted window have to follow this argument. Rebuilding the scale
        range on every drag also snaps the thumb back on macOS.
        """
        if self._updating:
            return
        self.redraw(position=float(value))

    def _style_axes(self):
        self.ax_sample.set_ylabel("Voltage (V)")
        self.ax_sample.grid(alpha=0.3)
        self.ax_intensity.set_xlabel("Pattern index")
        self.ax_intensity.set_ylabel("Intensity (V)")
        self.ax_intensity.grid(alpha=0.3)

    # ------------------------------------------------------------------
    def _open_file(self):
        chosen = filedialog.askopenfilename(
            title="Select Sample.dat",
            initialdir=REPO_ROOT,
            filetypes=[("Sample data", "*.dat"), ("All files", "*.*")],
        )
        if chosen:
            self.load_path(Path(chosen), single_file=True)

    def _open_folder(self):
        chosen = filedialog.askdirectory(
            title="Select a folder containing Sample.dat", initialdir=REPO_ROOT,
        )
        if chosen:
            self.load_path(Path(chosen), single_file=False)

    def load_path(self, path: Path, single_file=None):
        path = path.expanduser()
        if single_file is None:
            single_file = path.is_file()
        if single_file:
            files = [path]
        else:
            files = sorted(path.rglob("Sample.dat"))
            if not files:
                messagebox.showerror("No data", f"No Sample.dat found under {path}")
                return
        self.files = files
        self.file_index = 0
        guessed = infer_n_sample(path)
        if guessed is not None:
            self.n_sample = guessed
            self.start = 1
            self.end = guessed
        self._load_current_file()

    def step_file(self, delta):
        if not self.files:
            return
        nxt = self.file_index + delta
        if not 0 <= nxt < len(self.files):
            return
        self.file_index = nxt
        self._load_current_file()

    def _load_current_file(self):
        path = self.files[self.file_index]
        try:
            samples = np.fromfile(path, dtype="<f8")
        except OSError as exc:
            messagebox.showerror("Read error", f"Could not read {path}:\n{exc}")
            return
        if samples.size == 0:
            messagebox.showerror("Empty file", f"{path} contains no samples.")
            return
        self.samples = samples
        self.n_sample = min(self.n_sample, int(samples.size))
        self.end = min(self.end, self.n_sample)
        self.start = min(self.start, self.end)
        self.win_width = 2000
        shown = path.name if len(self.files) == 1 else f"{path.parent.name}/{path.name}"
        self.path_label.config(text=f"{shown}    [{self.file_index + 1}/{len(self.files)}]")
        self._apply_state_to_widgets()
        self.redraw()
        self.canvas.get_tk_widget().focus_set()

    # ------------------------------------------------------------------
    def _parse_int(self, raw, fallback):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return fallback

    def apply_from_widgets(self):
        if self._updating:
            return
        n_cap = int(self.samples.size) if self.samples is not None and self.samples.size else 1_000_000
        self.n_sample = min(n_cap, max(1, self._parse_int(self.n_var.get(), self.n_sample)))
        self.blocks = min(24, max(1, self._parse_int(self.blocks_var.get(), self.blocks)))
        start = min(self.n_sample, max(1, self._parse_int(self.start_var.get(), self.start)))
        end = min(self.n_sample, max(1, self._parse_int(self.end_var.get(), self.end)))
        self.pattern = max(1, self._parse_int(self.pattern_var.get(), self.pattern))
        if start > end:
            # Keep the edge the user just moved, and pull the other edge with it.
            if start != self.start:
                end = start
            else:
                start = end
        self.start = start
        self.end = end
        self._apply_state_to_widgets()
        self.redraw()

    def _apply_state_to_widgets(self):
        """Push the current integers into the spins and scales."""
        self._updating = True
        try:
            n_cap = int(self.samples.size) if self.samples is not None and self.samples.size else 1
            n_patterns = int(self.samples.size // self.n_sample) if self.samples is not None else 1
            n_patterns = max(1, n_patterns)
            self.pattern = min(self.pattern, n_patterns)
            self.n_var.set(str(self.n_sample))
            self.blocks_var.set(str(self.blocks))
            self.start_var.set(str(self.start))
            self.end_var.set(str(self.end))
            self.pattern_var.set(str(self.pattern))
            self.n_spin.config(to=max(1, n_cap))
            self._retarget_scale(self.start_scale, 1, self.n_sample, self.start)
            self._retarget_scale(self.end_scale, 1, self.n_sample, self.end)
            self._retarget_scale(self.pattern_scale, 1, n_patterns, self.pattern)
            self.prev_btn.config(state=tk.NORMAL if self.file_index > 0 else tk.DISABLED)
            at_end = not self.files or self.file_index >= len(self.files) - 1
            self.next_btn.config(state=tk.DISABLED if at_end else tk.NORMAL)
        finally:
            self._updating = False

    def _retarget_scale(self, scale, lo, hi, value):
        scale.config(from_=lo, to=max(lo, hi))
        scale.set(value)

    # ------------------------------------------------------------------
    def _visible_width(self):
        n = 0 if self.intensity is None else int(self.intensity.size)
        if n <= 0:
            return 0
        # Zoom stops at min_width, but a short recording is shown in full.
        floor = min(self.min_width, n)
        return int(min(n, max(floor, self.win_width)))

    def redraw(self, position=None):
        if self._updating or not hasattr(self, "canvas"):
            return
        self._requested_position = position
        self.ax_sample.cla()
        self.ax_intensity.cla()
        self._style_axes()

        if self.samples is None:
            self.count_label.config(text="Patterns: —")
            self.canvas.draw_idle()
            return

        try:
            matrix = pattern_matrix(self.samples, self.n_sample)
        except ValueError as exc:
            self.info_label.config(text=str(exc))
            self.canvas.draw_idle()
            return

        n_pattern = matrix.shape[1]
        unused = int(self.samples.size - n_pattern * self.n_sample)
        self.intensity = matrix[self.start - 1:self.end, :].mean(axis=0)
        length = self.end - self.start + 1
        unused_text = f", {unused} unused trailing samples" if unused else ""
        self.count_label.config(text=f"Patterns: {n_pattern}{unused_text}")
        self.info_label.config(
            text=(
                f"Intensity array length {n_pattern}: each value is the mean of "
                f"samples {self.start}–{self.end} ({length} values) inside that pattern."
            )
        )

        self._draw_samples(matrix)
        self._draw_intensity()
        self.canvas.draw_idle()

    def _draw_samples(self, matrix):
        n_sample, n_pattern = matrix.shape
        first = min(self.pattern, n_pattern) - 1
        count = min(self.blocks, n_pattern - first)
        block = matrix[:, first:first + count]
        flat = block.ravel(order="F")
        x = np.arange(1, flat.size + 1)
        self.ax_sample.plot(x, flat, color="#1f77b4", lw=1.0)

        for i in range(count):
            left = i * n_sample + self.start - 0.5
            right = i * n_sample + self.end + 0.5
            self.ax_sample.axvspan(left, right, color="#ff7f0e", alpha=0.28, lw=0)
            segment = block[self.start - 1:self.end, i]
            mean = float(segment.mean())
            self.ax_sample.plot(
                [i * n_sample + self.start, i * n_sample + self.end],
                [mean, mean],
                color="#d62728",
                lw=1.6,
            )
            if i:
                self.ax_sample.axvline(i * n_sample + 0.5, color="0.5", ls="--", lw=0.7)

        last_pattern = first + count
        self.ax_sample.set_xlim(0.5, flat.size + 0.5)
        self.ax_sample.set_xlabel("Sample index within the shown patterns")
        self.ax_sample.set_title(
            f"Patterns {first + 1}–{last_pattern}    "
            f"orange band = averaged segment {self.start}–{self.end}    "
            f"red = that pattern's intensity"
        )

    def _draw_intensity(self):
        intensity = self.intensity
        width = self._visible_width()
        if width <= 0:
            return
        marker = min(self.pattern, intensity.size)
        # Pattern chooses which block the upper plot shows. The intensity values
        # stay the same; the window scrolls so that pattern's value is on screen.
        follow_pattern = (
            self._requested_position is None and self._followed_pattern != self.pattern
        )
        if follow_pattern:
            requested = marker - 1 - width // 2
        elif self._requested_position is not None:
            requested = self._requested_position
        else:
            requested = self.pos_var.get()
        start = int(round(float(requested)))
        start = max(0, min(start, intensity.size - width))
        stop = start + width
        self._followed_pattern = self.pattern
        x = np.arange(start + 1, stop + 1)
        y = intensity[start:stop]
        self.ax_intensity.plot(x, y, color="#2ca02c", lw=0.8)
        if start < marker <= stop:
            self.ax_intensity.axvline(marker, color="#d62728", lw=1.0)
        pad = (float(y.max()) - float(y.min())) * 0.05 or 1.0
        self.ax_intensity.set_xlim(start + 0.5, stop + 0.5)
        self.ax_intensity.set_ylim(float(y.min()) - pad, float(y.max()) + pad)
        self.ax_intensity.set_title(
            f"Intensity array    samples {self.start}–{self.end} averaged    "
            f"patterns {start + 1}–{stop} of {intensity.size}"
        )
        self._sync_position_scale(start, width, intensity.size)
        self.pos_label.config(
            text=f"samples {self.start}–{self.end}    {start + 1}–{stop} / {intensity.size}"
        )

    def _sync_position_scale(self, start, width, n_pattern):
        maximum = max(0, n_pattern - width)
        self._updating = True
        try:
            if self._pos_maximum != maximum:
                self._pos_maximum = maximum
                self.pos_scale.configure(from_=0, to=maximum)
            if abs(float(self.pos_var.get()) - start) > 0.5:
                self.pos_var.set(start)
        finally:
            self._updating = False

    # ------------------------------------------------------------------
    def save_intensity(self):
        if self.intensity is None:
            messagebox.showinfo("Nothing to save", "Load a Sample.dat file first.")
            return
        chosen = filedialog.asksaveasfilename(
            title="Save intensity array",
            defaultextension=".dat",
            filetypes=[
                ("Little-endian float64", "*.dat"),
                ("NumPy array", "*.npy"),
                ("All files", "*.*"),
            ],
        )
        if not chosen:
            return
        destination = Path(chosen)
        self._write_intensity(destination, self.intensity)
        self.info_label.config(text=f"Saved {self.intensity.size} intensities to {destination}")

    @staticmethod
    def _write_intensity(destination: Path, intensity: np.ndarray):
        if destination.suffix.lower() == ".npy":
            np.save(destination, intensity)
        else:
            np.asarray(intensity, dtype="<f8").tofile(destination)

    def save_all(self):
        """Apply the current n / start / end to every loaded file and save one array each."""
        if not self.files:
            messagebox.showinfo("Nothing to save", "Load a Sample.dat file or folder first.")
            return
        chosen = filedialog.askdirectory(title="Select an output folder")
        if not chosen:
            return
        out_dir = Path(chosen)
        suffix = self.format_var.get() or ".dat"
        names = [p.parent.name or p.stem for p in self.files]
        unique = len(set(names)) == len(names)
        saved = 0
        for i, path in enumerate(self.files):
            try:
                samples = np.fromfile(path, dtype="<f8")
                intensity = intensity_from_samples(samples, self.n_sample, self.start, self.end)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Save error", f"{path}:\n{exc}")
                return
            stem = names[i] if unique else f"{i + 1:03d}_{names[i]}"
            self._write_intensity(out_dir / f"{stem}{suffix}", intensity)
            saved += 1
        self.info_label.config(
            text=f"Saved {saved} intensity arrays (samples {self.start}–{self.end}) to {out_dir}"
        )

    # ------------------------------------------------------------------
    def _on_touchpad_scroll(self, tk_event):
        dy = self.master.tk.call("tk::PreciseScrollDeltas", tk_event.delta)[1]
        self._touchpad_acc += float(dy)
        steps = int(self._touchpad_acc / 12)
        if steps == 0:
            return
        self._touchpad_acc -= steps * 12
        x, y = self.canvas._event_mpl_coords(tk_event)
        for _ in range(abs(steps)):
            ev = MouseEvent("scroll_event", self.canvas, x, y, step=-1 if steps > 0 else 1)
            self.canvas.callbacks.process("scroll_event", ev)

    def on_scroll(self, event):
        if self.intensity is None or event.inaxes != self.ax_intensity:
            return
        old_w = self._visible_width()
        if old_w <= 0:
            return
        factor = 0.8 if event.step > 0 else 1.25
        new_w = int(round(old_w * factor))
        floor = min(self.min_width, int(self.intensity.size))
        new_w = max(floor, min(new_w, int(self.intensity.size)))
        if new_w == old_w:
            return
        old_start = int(round(self.pos_var.get()))
        anchor = int(event.xdata) if event.xdata is not None else old_start + old_w // 2
        frac = (anchor - (old_start + 1)) / max(1, old_w)
        self.win_width = new_w
        new_start = int(round(anchor - frac * new_w)) - 1
        new_start = max(0, min(new_start, self.intensity.size - new_w))
        self.pos_var.set(new_start)

    def on_key(self, event):
        if self.intensity is None or self.intensity.size == 0:
            if (event.key or "") in ("up", "down", "pageup", "pagedown"):
                self.step_file(-1 if (event.key or "") in ("up", "pageup") else 1)
            return
        step = self._visible_width() * (1 if "shift" in (event.key or "") else 0.25)
        key = (event.key or "").replace("shift+", "")
        if key == "right":
            self.pos_var.set(min(self.pos_var.get() + step, self.intensity.size - self._visible_width()))
        elif key == "left":
            self.pos_var.set(max(self.pos_var.get() - step, 0))
        elif key in ("down", "pagedown"):
            self.step_file(1)
        elif key in ("up", "pageup"):
            self.step_file(-1)
        elif key == "home":
            self.pos_var.set(0)
        elif key == "end":
            self.pos_var.set(max(0, self.intensity.size - self._visible_width()))


def main():
    initial = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    root = tk.Tk()
    IntensityExtractor(root, initial_path=initial)
    root.mainloop()


if __name__ == "__main__":
    main()
