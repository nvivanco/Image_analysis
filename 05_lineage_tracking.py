import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
import tifffile
import glob
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from mmtrack import bh_track, lineage_detection, plot_cells
from datetime import datetime

# --- Local Model Path Constants (From your environment) ---
HF_FILENAME = "mm_link_prediction_model.pt"
LOCAL_MODEL_DIR = "models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, HF_FILENAME)

# --- Configuration Constants ---
NODE_FEATURE_COLS = ['area', 'centroid_y', 'axis_major_length', 'axis_minor_length',
                     'intensity_mean_phase', 'intensity_max_phase', 'intensity_min_phase',
                     'intensity_mean_fluor', 'intensity_max_fluor', 'intensity_min_fluor']
HIDDEN_CHANNELS = 128
NUM_BLOCKS = 2


def run_lineage_tracking(base_dir, model_path, strain_dict_json, prob_threshold, output_filename):
    """
    Aggregates cell feature data, prepares the GNN dataset, performs lineage prediction,
    and saves the final tracked data and kymograph plots.
    """

    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Strain Dictionary
    try:
        strain_exp_dict = json.loads(strain_dict_json)
    except json.JSONDecodeError:
        print("ERROR: Could not parse the strain dictionary. Ensure it is valid JSON.")
        return

    # 3. Aggregate Cell Feature DataFrames
    df_all = pd.DataFrame()
    for gene, exps_fov in strain_exp_dict.items():
        for exp_view, fovs in exps_fov.items():
            print(f"  -> Loading {exp_view} for gene {gene}")
            all_cells_filename = os.path.join(base_dir, f'all_cell_data_{exp_view}.pkl')

            try:
                all_cells_pd = pd.read_pickle(all_cells_filename)
            except FileNotFoundError:
                print(f"  WARNING: File not found: {all_cells_filename}. Skipping.")
                continue

            all_cells_pd['gene'] = None
            all_cells_pd.loc[all_cells_pd['FOV'].isin(fovs), 'gene'] = gene

            df_all = pd.concat([df_all, all_cells_pd], ignore_index=True)

    # FIX: This block was likely outside the function or incorrectly indented
    if df_all.empty:
        print("ERROR: No cell data found to process. Check base directory and dictionary.")
        return

    print(f"Total aggregated cells for tracking: {len(df_all)}")

    # 4. Standardize and Prepare Data
    df_all['node_id'] = df_all.index
    df_all.rename(columns={'centroid-0': 'centroid_y', 'centroid-1': 'centroid_x'}, inplace=True)

    # Convert node feature columns to float32
    for col in NODE_FEATURE_COLS:
        df_all[col] = df_all[col].astype(np.float32)

    # Fit Scaler and Create Transform
    all_train_node_features_df = df_all[NODE_FEATURE_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(all_train_node_features_df)
    print("Scaler fitted on aggregated data.")
    transform = bh_track.StandardScalerTransform(scaler)

    # Create Dataset and DataLoader
    dataset_root_base = os.path.join(base_dir, 'gnn_dataset') # Define base root here

    # FIX: These variables and logic were outside the function scope or poorly placed.
    dataset_root_timestamped = create_timestamped_subdir(dataset_root_base)

    if dataset_root_timestamped is None:
        print("FATAL: Failed to create unique dataset directory. Exiting or using fallback.")
        dataset_root_timestamped = dataset_root_base

    dataset = bh_track.CellTrackingDataset(root=dataset_root_timestamped,
                                           df_cells=df_all,
                                           node_feature_cols=NODE_FEATURE_COLS,
                                           device=device,
                                           pre_transform=transform)

    batch_size = 32
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Number of test graphs: {len(dataset)}")

    # 5. Initialize and Load GNN Model
    num_node_features = len(NODE_FEATURE_COLS)
    initial_edge_feature_dim = len(NODE_FEATURE_COLS) + 1  # delta features + time

    model = bh_track.LineageLinkPredictionGNN(
        in_channels=num_node_features,
        initial_edge_channels=initial_edge_feature_dim,
        hidden_channels=HIDDEN_CHANNELS,
        num_blocks=NUM_BLOCKS
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded GNN model weights from: {model_path}")
    except Exception as e:
        print(f"ERROR loading model state dict from {model_path}: {e}")
        return

    # 6. Predict Linkages
    model.eval()
    with torch.no_grad():
        lineage_predictions_df = bh_track.predict_cell_linkages(model, test_loader, device)

    # 7. Post-process Lineages
    df_consolidated_lineages = lineage_detection.process_all_fovs(
        lineage_predictions_df, df_all, prob_threshold=prob_threshold
    )
    df_for_kymograph_plot = lineage_detection.consolidate_lineages_to_node_df(
        df_consolidated_lineages, df_all
    )

    # 8. Save Final Tracked Data
    tracked_all_cells_filename = os.path.join(base_dir, output_filename)
    df_for_kymograph_plot.to_pickle(tracked_all_cells_filename)
    print(f"\nSaved final tracked data to: {tracked_all_cells_filename}")

    # 9. Plot Tracked Kymographs
    plot_kymographs(base_dir, df_for_kymograph_plot)


def plot_kymographs(base_dir, df_for_kymograph_plot):
    """Helper function to load kymograph images and plot predicted lineages."""
    print("\n--- Plotting Predicted Lineages on Kymographs ---")

    # Use pandas to get unique experiment/FOV/trench combinations
    unique_views = df_for_kymograph_plot[['experiment_name', 'FOV', 'trench_id']].drop_duplicates()

    if unique_views.empty:
        print("No unique views to plot.")
        return

    for index, cell_view in unique_views.iterrows():
        exp, fov, trench = cell_view['experiment_name'], cell_view['FOV'], cell_view['trench_id']

        # Path construction based on Stage 4's expected output
        kymo_base_path = os.path.join(base_dir, exp, 'hyperstacked', 'drift_corrected', 'rotated', 'mm_channels',
                                      'subtracted')

        # NOTE: Using the simple naming convention (e.g., '007_992.tif')
        path_to_phase_kymograph = os.path.join(kymo_base_path, f'{fov}_{trench}.tif')
        path_to_fluor_kymograph = os.path.join(kymo_base_path, 'fluor_kymos',
                                               f'{fov}_{trench}.tif')  # Assuming 'fluor' subdirectory

        # Fallback check for file existence
        if not (os.path.exists(path_to_phase_kymograph) and os.path.exists(path_to_fluor_kymograph)):
            print(f"  Skipping {exp}/{fov}/{trench}: Kymograph files not found.")
            continue

        print(f"  -> Plotting {exp}/{fov}/{trench}")

        phase_kymograph = tifffile.imread(path_to_phase_kymograph)
        fluor_kymograph = tifffile.imread(path_to_fluor_kymograph)

        df_view = df_for_kymograph_plot[
            (df_for_kymograph_plot['experiment_name'] == exp) &
            (df_for_kymograph_plot['FOV'] == fov) &
            (df_for_kymograph_plot['trench_id'] == trench)
            ].copy()

        plot_cells.plot_kymograph_cells_id(
            phase_kymograph, fluor_kymograph,
            df_view,
            exp, fov, trench,
            track_id_col='predicted_lineage'
        )


def create_timestamped_subdir(gnn_dataset_base_path):
    """Creates a new subdirectory with a timestamped name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir_name = f"run_{timestamp}"
    new_dir_path = os.path.join(gnn_dataset_base_path, new_dir_name)
    try:
        os.makedirs(new_dir_path, exist_ok=True)
        print(f"Created new timestamped directory for run data: {new_dir_path}")
        return new_dir_path
    except Exception as e:
        print(f"ERROR: Could not create directory {new_dir_path}. {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 5: Aggregate data, run GNN Lineage Link Prediction, and save tracked results."
    )

    # --- Mandatory I/O Arguments ---
    parser.add_argument(
        '--base-dir',
        required=True,
        type=str,
        help="Root directory containing all experiment folders and 'all_cell_data_*.pkl' files from Stage 4."
    )
    parser.add_argument(
        '--model-path',
        # Set required=False and provide a default using the constant
        required=False,
        type=str,
        default=LOCAL_MODEL_PATH,
        help=f"Path to the trained PyTorch GNN model state dictionary. Default: {LOCAL_MODEL_PATH}"
    )
    # --- Strain Dictionary (Mandatory for aggregation) ---
    parser.add_argument(
        '--strain-dict',
        required=True,
        type=str,
        help='JSON string defining experiment/FOV associations for each gene/strain. Format: {"gene": {"exp_dir": ["FOV_ID", ...]}}.'
    )

    # --- Optional Tracking/Output Arguments ---
    parser.add_argument(
        '--prob-threshold',
        type=float,
        default=0.8,
        help="Probability threshold for link acceptance in lineage detection. Default: 0.8."
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='tracked_all_cell_data.pkl',
        help="Name of the final pickled DataFrame file containing predicted lineages. Saved to --base-dir."
    )

    args = parser.parse_args()

    run_lineage_tracking(
        base_dir=args.base_dir,
        model_path=args.model_path,
        strain_dict_json=args.strain_dict,
        prob_threshold=args.prob_threshold,
        output_filename=args.output_filename
    )