# Video Content Gap Analyzer

FiftyOne plugin that identifies coverage gaps in video datasets using Twelve Labs Marengo embeddings and Pegasus descriptions.

Built at the Video Understanding AI Hackathon at Northeastern.

## Prerequisites

- Python 3.10+
- [Homebrew](https://brew.sh/) (macOS)
- A [Twelve Labs](https://twelvelabs.io/) API key

## Installation

### 1. Install system dependencies

```bash
brew install ffmpeg
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On macOS x86_64, `llvmlite` (a transitive dependency of `umap-learn`) may fail to build from source. If you hit a cmake/LLVM error, install the pre-built wheels first:
>
> ```bash
> pip install --only-binary :all: llvmlite numba
> pip install -r requirements.txt
> ```

### 4. Configure your Twelve Labs API key

Create a `.env` file in the project root (this file is already in `.gitignore`):

```bash
echo 'TWELVELABS_API_KEY=your_key_here' > .env
```

Or export it directly in your shell:

```bash
export TWELVELABS_API_KEY="your_key_here"
```

### 5. Verify the setup

```bash
python -c "import fiftyone as fo; print('fiftyone', fo.__version__)"
python -c "from twelvelabs import TwelveLabs; print('twelvelabs OK')"
python -c "from sklearn.cluster import KMeans; from umap import UMAP; print('clustering OK')"
ffmpeg -version | head -1
```

## Usage

Register the plugin with FiftyOne, then use the `analyze_coverage` and `show_gap_report` operators from the FiftyOne App.
