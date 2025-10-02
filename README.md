Here is the full content for your `README.md` file, ready to be copied and pasted into your code repository.

-----

# DuMM Bacteria Tracker (GNN-Based Cell Lineage Tracking)

This repository contains the source code for the **DuMM Bacteria Tracker**, a Python package (`mmtrack`) dedicated to highly accurate cell segmentation and lineage tracking for single-cell time-lapse microscopy data, particularly within **Mother Machine** devices.

The cell tracking component utilizes a custom **Graph Neural Network (GNN)** architecture for robust link prediction across time steps. The trained model weights are hosted separately on the [Hugging Face Hub](https://www.google.com/search?q=https://huggingface.co/nvivanco/DuMM_bacteria_track).

-----

## 1\. Installation and Setup

We use **Poetry** for dependency management to ensure a reproducible environment.

### Prerequisites

  * Python **3.11.8**
  * **Poetry** (install via `pip install poetry`)

### Setup Steps

1.  **Clone the Repository:**

    ```bash
    git clone <YOUR_REPO_URL>
    cd <repo-name>
    ```

2.  **Install Dependencies:** Poetry will create a new virtual environment using the specified Python version and install all packages defined in `pyproject.toml`.

    ```bash
    poetry install
    ```

3.  **Download Trained GNN Model Weights:** The trained GNN weights are hosted on the Hugging Face Hub to keep this repository small. Run the included script to download the model into the local `models/` directory.

    ```bash
    poetry run python download_assets.py
    ```

    *Note: This command saves the `best_link_prediction_model.pt` file to your local machine for inference.*

-----

## 2\. Usage and Tutorial 

The primary entry point for processing data is the `mm_raw_tiffs_to_tracked_cells.py` script.

### Workflow

The package is designed to handle the complete workflow:

1.  **Segmentation:** Identifying cell boundaries from raw microscopy TIFF files.
2.  **Feature Extraction:** Generating morphological and intensity features for each cell.
3.  **Candidate Graph Creation:** Forming plausible links (1-to-1 continuation and 1-to-2 division) between cells in adjacent frames based on distance and area heuristics.
4.  **Link Prediction:** Using the trained GNN model to score and select the true cell lineages.

### Running the Tracker

**To be completed by user:** Provide the final command and input/output examples for running the main script.

```bash
# Example command (Replace paths with your actual data locations):
poetry run python mm_raw_tiffs_to_tracked_cells.py --input-dir <raw_data_path> --output-dir <results_path>
```

-----

## 3\. GNN Model and Training Details

The core tracking logic relies on a custom Graph Neural Network architecture. Full details on the architecture and performance metrics are available on the [Hugging Face Model Card](https://www.google.com/search?q=https://huggingface.co/nvivanco/DuMM_bacteria_track).

### Architecture (`LineageLinkPredictionGNN`)

  * **Type:** Edge-Propagation Message Passing Neural Network (EP-MPNN) used for Link Prediction.
  * **Layers:** 2 custom `EP_MPNN_Block` layers followed by a **Jumping Knowledge (JK)** aggregation layer.
  * **Prediction:** The decoder predicts the probability of a link by combining the aggregated node embeddings and the current edge attributes.

### Training Methodology

| Component | Detail |
| :--- | :--- |
| **Node Features** | **10 scalar features** (area, centroid position, axis lengths, mean/max/min intensity from Phase Contrast and Fluorescence channels). |
| **Data Split** | **Time-based temporal split:** 60% Train, 20% Validation, 20% Test (based on chronological time frames). |
| **Normalization** | **Standard Scaling (`StandardScaler`)** fitted exclusively on the training set. |
| **Key Hyperparameters** | **Hidden Channels:** 128; **Optimizer:** Adam (LR: 0.001, Weight Decay: 0.0005). |
| **Stopping Criteria** | Early stopping based on **Validation Loss** with a patience of 10 epochs. |

-----

## 4\. Licensing and Citation 📜

### License

This project is released under the **MIT License**. For the full license text, see the `LICENSE` file.

### Citation

If you use this code or the trained model in your research, please cite the following original works that inspired the methodology:

  * **Cell Segmentation (adapted from napari-mm3):**

    > R. Thiermann et al., "Tools and methods for high-throughput single-cell imaging with the mother machine," *eLife*, vol. 12, p. RP88463, 2023.

  * **Cell Tracking GNN (inspired by Cell-tracker-GNN):**

    > T. Ben-Haim and T. Riklin-Raviv, "Graph Neural Network for Cell Tracking in Microscopy Videos," in *Proceedings of the European Conference on Computer Vision (ECCV)*, 2022.
