# Cell Counter PyTorch

This project provides tools for training, validating, and visualizing deep learning models for cell counting tasks using PyTorch. It supports multiple architectures and includes a graphical visualizer for inspecting results.

## Features

- **Multiple Architectures**: Support for FPN, ResUNet++, and CResUnet models.
- **Training Pipeline**: Complete training script with data augmentation, validation split, and early stopping.
- **Visualization**: GUI tool to visualize model predictions and ground truth.
- **Custom Losses**: Uses a combination of Weighted BCE and Dice loss for robust training.

## Installation

It was developed and tested with Python 3.11. To set up the environment, follow these steps:

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a model, use the `train.py` script. You can specify the architecture, data paths, and training hyperparameters.

**Basic Usage:**

```bash
python train.py --arch FPN --save_name my_fpn_model
```

**Arguments:**

-   `--arch`: Model architecture to use. Choices: `FPN` (default), `ResUNetPP`, `CResUnet`.
-   `--train_images`: Path to training images folder. Default: `data/train/images_original`.
-   `--train_masks`: Path to training masks folder. Default: `data/train/masks_original`.
-   `--train_weights`: Path to training weights folder. Default: `data/train/weights`.
-   `--val_images`: Path to validation images folder (for testing). Default: `data/eval/images`.
-   `--val_masks`: Path to validation masks folder (for testing). Default: `data/eval/masks`.
-   `--save_name`: Name for the saved model file. Default: `best_model`.
-   `--epochs`: Number of training epochs. Default: `200`.
-   `--batch_size`: Batch size for training. Default: `3`.

The script automatically splits the training data into training (80%) and validation (20%) sets. The best model (based on validation loss) is saved to `model/trained/` with the suffix `_early_stopping.pth`.

### Visualization for Model Inspection

To launch the visualizer GUI:

```bash
python start_visualizer.py
```

This will open a window where you can load images and models to inspect the cell counting performance. It is inteded to be used with grounds truth masks and weight maps generated during training.

### Generating Masks from Images

To generate masks for a set of images using a trained model, use the `generate_masks.py` script:

```bashpython generate_masks.py --model path/to/trained_model.pth --input_dir path/to/input_images --output_dir path/to/save_masks
```

**Arguments:**
-   `--model`: Path to the trained model file.
-   `--input_dir`: Directory containing input images.
-   `--output_dir`: Directory to save the generated masks.

## Project Structure

-   `train.py`: Main training script.
-   `start_visualizer.py`: Entry point for the visualization tool.
-   `generate_masks.py`: Script to generate masks from images using a trained model.
-   `model/`: Contains model architectures (`fpn.py`, `resunetpp.py`, `c_resunet.py`), loss functions, and training logic.
-   `data_handler/`: Handles dataset loading and augmentation.
-   `visualizer/`: Contains the source code for the GUI application.
-   `data/`: Directory structure for storing training and evaluation datasets.
