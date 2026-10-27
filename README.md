# CLIP Zero-Shot Image Classification

A Python project demonstrating **zero-shot image classification** using OpenAI's CLIP model. The project compares images with natural-language descriptions to estimate which descriptions best match each image, without training a separate classifier on a labeled dataset.

## Overview

This project uses CLIP (Contrastive Language–Image Pre-training) with the `ViT-B/32` architecture to evaluate the similarity between images and a predefined collection of text descriptions.

Instead of training a traditional image classifier with fixed output classes, the model uses text descriptions supplied at inference time. This makes it possible to explore image classification using natural-language prompts.

For each image, the program calculates similarity scores between its visual representation and the text representations of the candidate descriptions, then prints the results in descending order.

## Features

* Zero-shot image classification using CLIP.
* Image and text feature extraction.
* Natural-language prompts for candidate labels.
* Similarity-based ranking of text descriptions for each image.
* Softmax-normalized similarity scores.
* Automatic device selection between CUDA and CPU.
* Batch processing of images stored in a local directory.

## Technologies

* **Python**
* **PyTorch** — Model inference and tensor operations.
* **CLIP** — Image–text representation and similarity.
* **Pillow (PIL)** — Image loading and preprocessing.

## Project Structure

```text
clip-zero-shot/
├── clip-zero-shot.py
├── images/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── requirements.txt
└── README.md
```

The `images/` directory contains the images to analyze. The example filenames above are illustrative; replace them with your own images.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/GhazalCh/clip-zero-shot-classification
cd clip-zero-shot-classification
```

Replace the placeholders with your actual GitHub repository URL and directory name.

### 2. Create a virtual environment (optional)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Install PyTorch following the instructions for your system, available on the official [PyTorch website](https://pytorch.org/get-started/locally/).

Install the remaining dependencies:

```bash
pip install pillow ftfy regex tqdm
```

Install the CLIP implementation:

```bash
pip install git+https://github.com/openai/CLIP.git
```

**Note:** The original OpenAI CLIP package depends on PyTorch and additional libraries. Install a compatible PyTorch version before running the project.

## Usage

### 1. Add images

Place the images you want to analyze in the `images/` directory.

Supported file extensions:

* `.jpg`
* `.jpeg`
* `.png`

### 2. Configure candidate descriptions

Open `clip-zero-shot.py` and edit the `class_names` list.

For example:

```python
class_names = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car",
    "a photo of a building"
]
```

These descriptions are candidate labels against which the model evaluates each image.

**Important:** The model compares an image with the descriptions you provide. It does not automatically discover every object or category present in an image.

### 3. Run the program

```bash
python clip-zero-shot.py
```

The program loads CLIP, processes the images in the `images/` directory, and prints the candidate descriptions ranked by their normalized similarity scores.

## How It Works

The program follows these steps:

1. **Device selection:** Selects CUDA when available; otherwise, uses the CPU.
2. **Model loading:** Loads the pretrained CLIP `ViT-B/32` model.
3. **Text encoding:** Converts the candidate descriptions into text features.
4. **Image preprocessing:** Loads and preprocesses each image.
5. **Image encoding:** Converts the image into a visual feature representation.
6. **Similarity calculation:** Computes similarity scores between the image features and the text features.
7. **Score normalization:** Applies softmax to the similarity scores.
8. **Result ranking:** Prints the descriptions in descending order of their normalized scores.

## Understanding the Results

For each image, the program prints the candidate descriptions alongside their scores.

Example output format:

```text
Image: image1.jpg
  a photo of a cat              => 0.7123
  a photo of a dog              => 0.1834
  a photo of a car              => 0.0643
  a photo of a building         => 0.0400
```

*The values above are illustrative examples, not actual experimental results.*

A higher score indicates that a description receives a higher relative score among the candidate descriptions for that image.

**Important:** These softmax-normalized scores should not be interpreted as calibrated probabilities that a label is objectively correct. Their values depend on the candidate descriptions included in the comparison.

## Limitations

* Predictions depend on the wording and selection of candidate descriptions.
* The model can produce incorrect rankings, particularly for ambiguous or unfamiliar images.
* Softmax scores are relative to the supplied descriptions.
* The current implementation processes images one at a time.
* Only PNG and JPEG images are included by the file-extension filter.
* The project does not include a labeled evaluation dataset or quantitative accuracy measurements.

## Potential Improvements

* Add command-line arguments for the image directory and candidate descriptions.
* Save classification results to a CSV file.
* Display the top-k predictions for each image.
* Compare alternative prompts and CLIP model architectures.
* Add an evaluation pipeline using a labeled dataset.
* Implement batch processing for improved efficiency.

## Learning Objectives

This project provides practical experience with:

* Vision–language models.
* Zero-shot learning and image classification.
* Pretrained model inference with PyTorch.
* Image and text feature extraction.
* Similarity-based ranking.
* Natural-language prompt design.

## References

* [OpenAI CLIP Repository](https://github.com/openai/CLIP)
* [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

This repository contains an educational implementation. The licensing terms of the CLIP implementation, pretrained model, and dependencies remain applicable to their respective components.
