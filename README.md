# Fruits and Vegetables Image Classification with VGG16

An image-classification project that uses transfer learning to recognize 24
classes of fruits and vegetables from the Fruits-360 original-size dataset.
The model is built with TensorFlow/Keras on top of a pretrained VGG16
convolutional neural network.

This repository documents my independent implementation and learning process
based on a Coursera project. The notebook covers the complete workflow, from
downloading and preparing the data to training, fine-tuning, evaluating, and
visualizing predictions.

## Project Overview

The goal is to train a multiclass image classifier that can distinguish
different fruit and vegetable categories, including several visually similar
apple varieties.

The workflow includes:

- downloading and extracting the dataset;
- loading images from train, validation, and test directories;
- applying image augmentation to the training data;
- using ImageNet-pretrained VGG16 as a feature extractor;
- training a custom classification head;
- fine-tuning the final VGG16 layers;
- evaluating the model on the test set;
- plotting learning curves; and
- visualizing predictions on individual images.

## Dataset

The notebook uses a 24-class subset of the Fruits-360 original-size dataset.

| Split | Images | Classes |
|---|---:|---:|
| Training | 6,231 | 24 |
| Validation | 3,114 | 24 |
| Test | 3,110 | 24 |

The classes include apple varieties as well as cabbage, carrot, cucumber,
eggplant, pear, and zucchini categories. The dataset is downloaded by the
notebook and is not intended to be committed to this repository.

## Model

The classifier uses:

- VGG16 pretrained on ImageNet, without its original classification head;
- `GlobalAveragePooling2D`;
- a fully connected layer with 256 ReLU units;
- batch normalization;
- dropout with a rate of 0.5; and
- a 24-unit softmax output layer.

During the first training stage, the VGG16 base is frozen. During fine-tuning,
the final five VGG16 layers are unfrozen and trained with a learning rate of
`1e-5`.

## Data Preprocessing

All images are resized to `64 × 64` pixels and normalized to the `[0, 1]`
range. The training pipeline also applies:

- random rotation;
- width and height shifts;
- shear;
- zoom; and
- horizontal flipping.

Validation and test images are only resized and normalized.

## Results

The recorded notebook run produced:

| Metric | Value |
|---|---:|
| Test accuracy | 69.23% |
| Test loss | 0.8926 |
| Best recorded fine-tuning validation accuracy | 80.00% |

These results come from one learning-oriented run. The training stages use a
limited number of batches per epoch, so the metrics should not be treated as a
fully optimized benchmark.

## Getting Started

### Prerequisites

- Python 3.12
- JupyterLab or VS Code with the Jupyter extension
- `wget` (used by the current download cell)

On macOS, `wget` can be installed with Homebrew:

```bash
brew install wget
```

### Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  tensorflow==2.16.2 \
  matplotlib==3.9.2 \
  numpy==1.26.4 \
  scipy==1.14.1 \
  scikit-learn==1.5.2 \
  jupyterlab
```

### Run the notebook

```bash
jupyter lab
```

Open the project notebook and run the cells from top to bottom. The dataset will
be downloaded and extracted into:

```text
fruits-360-original-size/
```

The first model run may also download the pretrained VGG16 weights.

## Suggested Repository Structure

```text
.
├── notebooks/
│   └── fruits360_vgg16.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

Large generated files should be excluded from Git:

```gitignore
.venv/
.ipynb_checkpoints/
fruits-360-original-size/
*.h5
*.keras
__pycache__/
```

## Limitations and Future Improvements

- Train on every batch in each epoch instead of using reduced step counts.
- Add a confusion matrix and per-class precision, recall, and F1-score.
- Investigate errors between visually similar apple varieties.
- Compare VGG16 with a lighter architecture such as MobileNetV2 or EfficientNet.
- Tune image size, learning rate, augmentation, and dropout.
- Save the best model with `ModelCheckpoint`.
- Add a reusable prediction script or a small interactive demo.
- Make experiments reproducible with fixed random seeds.

## Technologies

- Python
- TensorFlow and Keras
- VGG16
- NumPy
- Matplotlib
- Jupyter

## Acknowledgements

This project was developed as an independent learning implementation based on a
Coursera exercise. The image data comes from the Fruits-360 dataset. The
pretrained VGG16 weights are provided through TensorFlow/Keras.

## License

The source code in this repository may be shared under the MIT License. Dataset
images and course materials remain subject to their original licenses and terms.

