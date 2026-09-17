# OpenCV k-NN Shape and Color Classifier

A computer vision tool built with OpenCV and k-Nearest Neighbors (k-NN) to segment and classify 2D geometric shapes based on HSV color thresholds and normalized 10x10 feature vectors.

## Features

* **Interactive Dataset Generator**: Automatically extracts object contours, resizes them into $10 \times 10$ binary feature matrices, and captures user keypresses for shape labeling.
* **HSV Color Segmentation**: Isolates target objects across predefined HSV ranges (Yellow, Green, Red, Blue, Purple).
* **k-NN Shape Recognition**: Classifies contours into pre-trained categories (Circle, Pentagon, Triangle, Rectangle, Square, Lightning, Trapeze, Rhombus).
* **Annotated Visual Overlay**: Renders bounding boxes and labels (Color + Shape) on top of input images in real time.

## Project Structure

* `knn.py` — Interactive script for contour extraction, ROI normalization ($10 \times 10$), and manual dataset labeling.
* `lab5.py` — Core detection engine that trains the OpenCV k-NN model, processes HSV color channels, and classifies detected shapes.
* `generalsamples.data` — Flattened $100$-element feature matrix generated during training.
* `generalresponses.data` — Label array corresponding to trained shape classes.

## Requirements

* Python 3.x
* OpenCV (`opencv-python`)
* NumPy
* imutils

```bash
pip install opencv-python numpy imutils
```

## Usage

1. Training & Dataset Generation
Run knn.py to process the template image, view extracted region proposals, and manually assign numeric labels using keys 0-9:

```bash
python knn.py
```

2. Running Shape & Color Recognition
Run lab5.py to train the k-NN model on the saved datasets and view the color/shape detection results:

```bash
python lab5.py
```
