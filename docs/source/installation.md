# Installation

dcatoolkit requires Python 3.12 or later and is available on [PyPI](https://pypi.org/project/dcatoolkit/):

```bash
pip install dcatoolkit
```

To also install matplotlib, which the plotting example uses:

```bash
pip install "dcatoolkit[plot]"
```

On Python 3.10 or 3.11, `pip` automatically installs dcatoolkit 0.3.x, the last release series supporting those versions.

Upgrading from 0.2.x? See [Upgrading from 0.2.x](changelog.md#upgrading-from-02x) in the changelog.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/RaheelSyedAhmed/dcatoolkit.git
cd dcatoolkit
uv sync  # Installs dcatoolkit and the dev group (pytest, ruff), which uv includes by default.
```

Add `--group docs` to also install the documentation tools, and `--extra plot` for matplotlib. Use `uv sync --no-dev` to install only dcatoolkit and its dependencies.

Run the test suite (requires internet to fetch structures from RCSB), lint, and build these docs:

```bash
uv run pytest
uv run ruff check
uv run --group docs sphinx-build -W -b html docs/source docs/build/html
```
