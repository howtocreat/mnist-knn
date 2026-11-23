# MNIST handwritten digit recognition

A Python implementation of k-nearest neighbors (KNN) for recognizing handwritten digits. It includes a single-image predictor and the original experiment comparing the custom classifier with scikit-learn on MNIST.

## Setup

Use Python 3.13. From the repository root in PowerShell, create a virtual environment and install the dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Recognize one image

```powershell
.\.venv\Scripts\python.exe predict.py "examples/digit-7.png"
```

The example prints `Predicted digit: 7`. Replace the path with your own PNG or JPEG. Use a tightly cropped, high-contrast image of **one handwritten digit**. The program detects light or dark ink, crops the foreground, and centers it on a 28 × 28 canvas before classification. The classifier always chooses one of 0–9 for an image that passes preprocessing, so its output on unrelated images or strings of digits is unreliable.

## Reproduce the experiment

```powershell
.\.venv\Scripts\python.exe RN.py --quick
.\.venv\Scripts\python.exe RN.py --k 1 3 5 7 9 30000 59999
```

The first command compares both implementations on the smaller coursework subset. The second uses the full bundled data and can take much longer. Add `--plot` to display the custom classifier's confusion matrices.

The [coursework report](docs/report.pdf) recorded a peak custom KNN accuracy of **97.17% at k = 3**. [Original plots](docs/plots/) show the sample images and confusion matrices from that experiment. Timings depend on the machine.

## Data and attribution

The bundled arrays preserve the pixel values and labels of the MNIST PNG files in the original coursework archive, including its smaller benchmark subset. The main split contains 60,000 training images and 9,999 test images. See [data provenance and licensing](data/README.md).
