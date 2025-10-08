# DuMM Bacteria Tracker (GNN-Based Cell Lineage Tracking)

This repository contains the source code for the **DuMM Bacteria Tracker**, a Python package (`mmtrack`) dedicated to highly accurate cell segmentation and lineage tracking for single-cell time-lapse microscopy data, particularly within **Mother Machine** devices.

The cell tracking component utilizes a custom **Graph Neural Network (GNN)** architecture for robust link prediction across time steps. The trained model weights are hosted separately on the [Hugging Face Hub](https://huggingface.co/nvivanco/DuMM_bacteria_track).

-----

## 1\. Installation and Setup

We use **Poetry** for dependency management to ensure a reproducible environment.

### Prerequisites

  * Python **3.11.8**
  * **Poetry** 
  For Linux/macOS (using curl):
  ```bash
  curl -sSL https://install.python-poetry.org | python3 - --version 2.2.1
  ```
  For Windows (using PowerShell):
  ```PowerShell
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python - --version 2.2.1
  ```


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

    *Note: This command saves the `mm_link_prediction_model.pt` file to your local machine for inference.*

-----

## 2\. Usage and Tutorial

## Running Stage 1: Image Correction

The script `01_image_correction.py` executes the initial pre-processing steps: drift correction, rotation, and channel extraction. It saves the resulting image files for use in the next pipeline stage.

### Basic Execution

Run the script from your project root, replacing the placeholder paths with the actual locations of your raw TIFF data and where you want the processed images to be saved.

```bash
# Example: Use default parameters
poetry run python 01_image_correction.py \
    --input-dir '<path/to/raw/data/DIMM_CL008_072225>' 
```
An interacitve window will pop up with the detected lines used for image rotation, this serves a check for any potenetial downstream troubleshooting if rotation of image does not go as expected.

Close the window to proceed with pipeline.

A second interactive window will pop up with the identified microfluidic channels/trenches that capture cells. Each channel has a label in red at the bottom that needs to be used for analysis in the next step. This image will also be saved as a tif file.


## Directory Structure Overview

The project uses a structured directory where raw data is sequentially processed into hyperstacks, drift-corrected, rotated, and finally segmented.

```
<path/to/raw/data>/
└── <exp_folder>/
    ├── Pos0/
    ├── Pos1/
    ├── ...
    │
    └── hyperstacked/
        ├── hyperstacked_xy0.tif
        ├── ...
        │
        └── drift_corrected/
            ├── drift_cor_<exp>_xy0.tif
            ├── ...
            │
            └── rotated/
                ├── rotated_xy0.tif
                ├── ...
                │
                └── mm_channels/
                    ├── FOV###_mm_channel_mask.tif
                    ├── FOV###_region_##.tif
                    └── ...
```

-----

## Detailed File Descriptions

| File/Directory | Location | Purpose |
| :--- | :--- | :--- |
| **`Pos0/`, `Pos1/`** | `<exp_folder>/` | Directory of **Raw TIFFs** for each Field of View (FOV). Contains individual TIFF files for each channel and timepoint. |
| **`hyperstacked/`** | `.../<exp_folder>/` | Contains hyperstacks of the entire, **unprocessed** FOV (all channels/timepoints). |
| **`drift_corrected/`** | `.../hyperstacked/` | Contains hyperstacks after **X-Y drift correction**. |
| **`rotated/`** | `.../drift_corrected/` | Contains hyperstacks after **rotation correction**. |
| **`FOV###_mm_channel_mask.tif`** | `.../mm_channels/` | **Segmented Mask** file containing the identified microfluidic channels with unique, labeled IDs needed for subsequent analysis steps. |
| **`FOV###_region_##.tif`** | `.../mm_channels/` | Hyperstack file containing only the cropped time-series data for a **single identified microfluidic channel/trench** (includes all phase/fluorescent channels). |

### User Action and Output

Upon successful completion, the script will output the path to the extracted image files. **You must inspect these files** to determine the necessary IDs for background subtraction in Stage 2.

```markdown
#################################################################
# USER ACTION REQUIRED: Inspect TRENCH MASKS and TIME-LAPSE   
# Next Step: Determine 'empty_stack_id' and 'ana_peak_ids'. 
# Use <path_to_mm_channels> as the input for Stage 2.         
#################################################################
```

### Overriding Default Parameters

The script provides several optional arguments to handle different experimental setups. You can view all options using the `--help` flag:

```bash
poetry run python 01_image_correction.py --help
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input-dir` | `str` | *N/A* | **(Required)** Path to the directory containing the raw TIFF stacks. |
| `--phase-channel-idx` | `int` | `0` | Index of the phase contrast channel (`c` parameter). |
| `--fast-drift-correction` | `bool` | `True` | Use the fast drift correction method (`fast4`). |
| `--growth-channel-length` | `int` | `400` | Approx. pixel length for the Mother Machine channel. |
| `--trench-ends-orientation` | `str` | `'down'` | Orientation of the closed trench ends (`closed_ends` parameter). Choices: `up`, `down`, `none`. |

**Example of overriding defaults:**

If your phase channel is index 1 and you are not using the fast drift correction:

```bash
poetry run python 01_image_correction.py \
    --input-dir '<path/to/raw/data>' \
    --phase-channel-idx 1 \
    --fast-drift-correction False
```


## Running Stage 2: Background Subtraction

The script `02_background_subtraction.py` performs phase contrast and fluorescence background subtraction using an empty trench identified by the user in the previous stage.

This script requires several **mandatory** parameters determined by visual inspection of the images output from Stage 1.

### Basic Execution

Run the script from your project root.

You must provide the following **mandatory** arguments:

1.  **`--input-path`**: The output directory generated by `01_image_correction.py` (e.g., `./processed_images/.../mm_channels`).
2.  **`--fov`**: The Field of View ID you are analyzing (e.g., `'007'`).
3.  **`--empty-stack-id`**: The ID of the empty trench chosen for background subtraction (e.g., `'765'`).
4.  **`--ana-peak-ids`**: A space-separated list of the trench IDs that contain cells to be analyzed (e.g., `'992' '1219' '1749'`).

<!-- end list -->

```bash
# Example: Replace values based on your inspection
poetry run python 02_background_subtraction.py \
    --input-path './processed_images/DIMM_CL008_072225/mm_channels' \
    --fov '007' \
    --empty-stack-id '765' \
    --ana-peak-ids '992' '1219' '1749'
```

### Overriding Default Parameters

The script provides optional arguments to specify the channel indices if they differ from the standard configuration.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input-path` | `str` | *N/A* | **(Required)** Path to the extracted channel data (output of Stage 1). |
| `--fov` | `str` | *N/A* | **(Required)** Field of View ID (e.g., '007'). |
| `--empty-stack-id` | `str` | *N/A* | **(Required)** ID of the empty stack for background. |
| `--ana-peak-ids` | `str` (list) | *N/A* | **(Required)** Space-separated list of trench IDs to analyze. |
| `--phase-index` | `int` | `0` | The channel index for **Phase Contrast** subtraction. |
| `--fluor-index` | `int` | `1` | The channel index for **Fluorescence** subtraction. |

**Example of overriding channel indices:**

If Phase is at index 2 and Fluorescence is at index 3:

```bash
poetry run python 02_background_subtraction.py \
    --input-path './processed_images/mm_channels' \
    --fov '007' \
    --empty-stack-id '765' \
    --ana-peak-ids '992' '1219' '1749' \
    --phase-index 2 \
    --fluor-index 3
```

## Running Stage 3: Cell Segmentation

The script `03_cell_segmentation.py` performs the actual cell segmentation using `cell_segmentation.segment_chnl_stack()`. Since segmentation results are highly sensitive to image brightness, this script exposes all key segmentation parameters for tuning.

It calculates the `end-frame` dynamically based on the input file size unless the value is explicitly provided.

### Prerequisites

You must first visually inspect the images processed in Stage 2 to determine the correct segmentation parameters for your data.

### 1\. Defining the Target Data

This script iterates over multiple experiments, FOVs, and trenches, so you must define which data to process using the **`--exp-dict`** argument.

The value of this argument must be a single, properly formatted **JSON string** with the following structure:

```json
{
   "exp_directory_name_1": {
      "FOV_ID_A": ["trench_ID_1", "trench_ID_2"],
      "FOV_ID_B": ["trench_ID_3"]
   },
   "exp_directory_name_2": {
      "FOV_ID_C": ["trench_ID_4"]
   }
}
```

### 2\. Basic Execution

Run the script from your project root, replacing the placeholder path and providing the experiment dictionary as a JSON string.

```bash
# Example: Using a single experiment
EXP_DICT='{"DUMM_CL008_giTG068_072925": {"007": ["992", "1219", "1749"]}}'

poetry run python 03_cell_segmentation.py \
    --base-dir '/path/to/DuMM_image_analysis' \
    --exp-dict "${EXP_DICT}"
```

### 3\. Tuning Segmentation Parameters

The script defaults to parameters suitable for low phase-contrast exposure (`OTSU_threshold=0.5`). If your images are brighter, you will need to override these defaults.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--base-dir` | `str` | *N/A* | **(Required)** Root path containing all experiment folders. |
| `--exp-dict` | `str` | *N/A* | **(Required)** JSON string of the experiments/trenches to segment. |
| `--otsu-threshold` | `float` | `0.5` | Threshold for segmentation (range 0.5 to 1.5). **Adjust based on image brightness.** |
| `--first-opening` | `int` | `4` | Size of the first morphological opening (e.g., 3, 4, or 5). |
| `--distance-threshold` | `float` | `3.0` | Distance threshold for watershed seeding (e.g., 1.0 to 3.0). |
| `--second-opening-size` | `int` | `3` | Size of the second morphological opening (e.g., 1, 2, or 3). |
| `--min-cell-area` | `int` | `100` | Minimum cell area in pixels. |
| `--max-cell-area` | `int` | `1000` | Maximum cell area in pixels. |
| `--small-merge-area-threshold`| `int` | `100` | Threshold for merging small segmented regions. |
| `--end-frame` | `int` | `None` | Last time frame index to consider. **If omitted, calculated dynamically.** |
| `--phase-channel` | `str` | `'0'` | Phase channel index as string. |

**Example of overriding for high phase exposure (brighter images):**

```bash
# Define the experiment dictionary (same as above)
EXP_DICT='{"DUMM_CL008_giTG068_072925": {"007": ["992", "1219", "1749"]}}'

poetry run python 03_cell_segmentation.py \
    --base-dir '/path/to/DuMM_image_analysis' \
    --exp-dict "${EXP_DICT}" \
    --otsu-threshold 1.5 \
    --first-opening 3 \
    --distance-threshold 1.5 \
    --second-opening-size 1 \
    --max-cell-area 800 \
    --small-merge-area-threshold 50
```

This part of the pipeline applies the samme segmentation parameters to all specified experiments in the dictionary. Therefore, all these experiments should have consistent phase exposure settings. If different exposure setting have been applied, it is recommended to apply different segmentations settings to those experiments.

### 4\. Required User Action

The script automatically displays the segmented output for visual confirmation. After running this stage, you **must visually inspect the segmentation masks** to ensure they correctly identify cell regions before proceeding to the next stage of feature extraction.


## Running Stage 4: Feature Extraction and Data Consolidation 

The script `04_feature_extraction.py` performs the crucial step of converting segmented masks into quantifiable cellular data. It creates kymographs, extracts morphological and intensity features using `skimage.measure.regionprops_table`, and consolidates all results into a single pandas `.pkl` file per experiment.

This stage is designed for maximum automation: if no specific time ranges are provided, it automatically finds all experiment directories under a base directory, identifies all available segmented files and uses the full time range of each stack.


### Basic Automated Execution (Recommended)

If you intend to use the entire time-lapse for every segmented trench and analyze all experiment directories under a base directory, simply provide the `--base-dir`. The script will automatically discover all segmented TIFF files (`mm3_segmented*.tif`) and calculate the `start` (0) and `end` (last frame index) for each.

```bash
# Example: Full automation mode
poetry run python 04_feature_extraction.py \
    --base-dir '/path/to/DuMM_image_analysis'
```

-----

### Custom Time Range Override

If you need to analyze only a specific time window, for a given trench (e.g., excluding frames where a cell was lost or the trench clogged), or only a specific set of experiment directories, you can pass the **`--time-range-dict`** as a JSON string.

Only experiment directories in this dictionary will be processed. Any trench **omitted** from this dictionary will not be processed. You can also set the `end` frame to `null` to use a custom `start` frame while still defaulting the `end` to the size of the TIFF file.

#### `--time-range-dict` Format

The dictionary must be a single, valid JSON string:

```json
{
   "experiment_name": {
      "FOV_ID": {
         "trench_ID_A": {"start": 0, "end": 75},
         "trench_ID_B": {"start": 10, "end": null} 
      }
   }
}
```

#### Example of Override Execution

```bash
# Define custom time ranges for specific trenches
TIME_DICT='{"DUMM_CL008_giTG068_072925": {"007": {"992": {"start": 10, "end": 60}, "1219": {"start": 0, "end": null}}}}'

poetry run python 04_feature_extraction.py \
    --base-dir '/path/to/DuMM_image_analysis' \
    --time-range-dict "${TIME_DICT}"
```

### Output

The script generates the following:

1.  **Kymograph Images:** Saves kymographs for phase, mask, and fluorescence channels in subdirectories (`mask_kymos/`, `fluor_kymos/`).
2.  **Consolidated DataFrame:** Saves a single pickled pandas file (`.pkl`) per experiment folder (e.g., `all_cell_data_DUMM_CL008_giTG068_072925.pkl`) in the `--base-dir`. This file contains all extracted features and is the primary input for the final GNN tracking stage.

-----

### Optional Arguments

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--base-dir` | `str` | *N/A* | **(Required)** Root path containing all experiment folders. |
| `--time-range-dict` | `str` | `''` | **(Optional)** JSON string for custom trench time ranges. |
| `--phase-channel` | `str` | `'0'` | Phase channel index string used in file naming. |
| `--fluor-channel` | `str` | `'1'` | Fluor channel index string used in file naming. |


## Running Stage 5: GNN Lineage Tracking

The script `05_lineage_tracking.py` is the final step, which aggregates the feature data from all experiments, normalizes the features, prepares the data for the Graph Neural Network (GNN) model, runs the lineage link prediction, and saves the final tracked results.

### Prerequisites

1.  **Stage 4 Data:** You must have successfully run `04_feature_extraction.py`, resulting in `all_cell_data_*.pkl` files in your `--base-dir`.
2.  **Trained Model:** You must have the trained GNN model file (default name: `mm_link_prediction_model.pt`) accessible to the script.

### 1\. Defining the Strain/Experiment Map

This script requires the **`--strain-dict`** to map which feature file (`.pkl`) corresponds to which genetic strain and which FOVs within that file should be considered for that strain.

The value of this argument must be a single, properly formatted **JSON string** with the following structure:

```json
{
   "gene_A": {
      "exp_directory_name_1": ["FOV_ID_A", "FOV_ID_B"],
      "exp_directory_name_2": ["FOV_ID_C"]
   },
   "gene_B": {
      "exp_directory_name_3": ["FOV_ID_D"]
   }
}
```

### 2\. Execution

The script will automatically look for the GNN model at the default path: **`models/mm_link_prediction_model.pt`**.

```bash
# Example: Using the required strain dictionary
STRAIN_DICT='{"chpS": {"DUMM_giTG62_Glucose_012925": ["005"]}, "baeS": {"DUMM_giTG66_Glucose_012325": ["003"]}}'

poetry run python 05_lineage_tracking.py \
    --base-dir '/path/to/DuMM_image_analysis' \
    --strain-dict "${STRAIN_DICT}"
```

### 3\. Optional Arguments

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--base-dir` | `str` | *N/A* | **(Required)** Root path containing all experiment folders and the `.pkl` feature files. |
| `--strain-dict` | `str` | *N/A* | **(Required)** JSON string mapping strains/genes to experiment directories and FOVs. |
| `--model-path` | `str` | `models/mm_link_prediction_model.pt` | Path to the trained GNN model weights. |
| `--prob-threshold` | `float` | `0.8` | Minimum GNN prediction probability required to accept a cell link as a valid lineage connection. |
| `--output-filename` | `str` | `tracked_all_cell_data.pkl` | The filename for the final DataFrame containing the predicted lineages, saved to the `--base-dir`. |

**Example of overriding the probability threshold:**

```bash
STRAIN_DICT='{...}' # Use the full dictionary as defined above

poetry run python 05_lineage_tracking.py \
    --base-dir '/path/to/DuMM_image_analysis' \
    --strain-dict "${STRAIN_DICT}" \
    --prob-threshold 0.95
```

### Output

1.  **Tracked DataFrame:** A final pickled pandas file (default: `tracked_all_cell_data.pkl`) saved in the `--base-dir`. This file contains all original features plus a new column, **`predicted_lineage`**, used for plotting.
2.  **Kymograph Plots:** The script plots the predicted lineages over the corresponding phase and fluorescence kymographs for visual confirmation, saving the resulting images to the relevant experiment subdirectories.


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

## 4\. Licensing and Citation

### License

This project is released under the **MIT License**. For the full license text, see the `LICENSE` file.

### Citation

If you use this code or the trained model in your research, please cite the following original works that inspired the methodology:

  * **Cell Segmentation (adapted from napari-mm3):**

    > R. Thiermann et al., "Tools and methods for high-throughput single-cell imaging with the mother machine," *eLife*, vol. 12, p. RP88463, 2023.

  * **Cell Tracking GNN (inspired by Cell-tracker-GNN):**

    > T. Ben-Haim and T. Riklin-Raviv, "Graph Neural Network for Cell Tracking in Microscopy Videos," in *Proceedings of the European Conference on Computer Vision (ECCV)*, 2022.
