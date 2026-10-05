# Single-pixel Imaging Database

This project aims to share real and raw data obtained from single-pixel imaging experiments, along with the tools for reconstructing the final image from the data.

## Repository layout

```
data/        raw datasets (see "Data format")
toolkits/    tools for inspecting and processing the data
pyproject.toml, uv.lock   Python dependencies, managed with uv
```

## Data format

Each recording is stored as a `Sample.dat` file inside its own timestamped folder:

```
<dataset>/
├── 20260919_102009/Sample.dat
├── 20260919_102013/Sample.dat
└── ...
```

`Sample.dat` is a headerless binary file of little-endian 64-bit floats (`float64`), one value per sample, in volts. It can be read directly with NumPy:

```python
import numpy as np

samples = np.fromfile("Sample.dat", dtype="<f8")
```

## Setup

The tools are managed with [uv](https://docs.astral.sh/uv/) and need Python 3.12 or newer. uv downloads a suitable Python (with Tk, which the viewer needs) on first use.

```bash
brew install uv        # or see the uv documentation for other platforms
uv sync                # create .venv and install the dependencies
```

## Tools

### Raw sample inspector

An interactive viewer for `Sample.dat` files.

```bash
uv run toolkits/raw_sample_inspector.py [path/to/dataset]
```

The path is optional; use the **Open** button in the window to load a folder. Every `Sample.dat` found below the chosen folder is loaded.

- **Open / + / Close**: load a folder into a row, add a row, or remove one. Each row is plotted on the same axes so datasets can be compared.
- **Color swatch**: change a dataset's line color.
- **Radio button**: choose the active dataset, which the up/down keys act on.
- **Mouse wheel / trackpad scroll**: zoom the horizontal axis around the cursor.
- **Position slider**: scroll through the samples.
- **Left / Right**: scroll by a quarter of the window (hold Shift for a full window).
- **Up / Down** (or Page Up / Page Down): previous / next file of the active dataset.
- **Home / End**: jump to the start / end.

Sample indices on the horizontal axis start at 1.

## License

See [LICENSE](LICENSE).
