import os

import numpy as np
import pandas as pd
import tifffile

import pre_process_mm
import mm3_adapted_cell_segmentation as mm3seg
import plot_cells
import lineage_detection
import gnn_ben_haim as bh_track

from skimage import measure
from skimage.measure import regionprops, label

import torch
torch.set_default_dtype(torch.float32)
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler


exp_directory = 'DUMM_CL008_giTG068_072925'
dir_files_to_correct = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{exp_directory}'
drift_corrected_path = pre_process_mm.drift_correct(dir_files_to_correct, 'DuMM', c=0)

path_to_rotated_images = pre_process_mm.rotate_stack(drift_corrected_path, c=0, # phase channel index, check in metadata
													 growth_channel_length=400, # approx pixel length for 1x1 binning in DuMM
													 closed_ends ='down') # check trench orientation in image preview
path_to_mm_channels = pre_process_mm.extract_mm_channels(path_to_rotated_images)

# Inspect trench mask and time-lapse to determine an empty trench to use
# as background subtraction aka empty_stack_id and which trenches have cell to analyze aka ana_peak_ids

FOV = '007'
empty_stack_id = '765'
ana_peak_ids = ['992','1219', '1749']

# subtract phase background
pre_process_mm.subtract_fov_stack(path_to_mm_channels,
								  FOV,
								  empty_stack_id,
								  ana_peak_ids,
								  method = 'phase',
								  channel_index= 0) # phase index

# subtract fluorescent background
pre_process_mm.subtract_fov_stack(path_to_mm_channels,
								  FOV,
								  empty_stack_id,
								  ana_peak_ids,
								  method = 'fluor', channel_index= 1) # fluor index

# Make a dictionary for the experimental directories to do cell segmentation on
# structure of dictionary
# dict = {'exp_directory': {'FOV': ana_peak_ids}} ana_peak_ids is a list of strings

seg_FOV_dict = {
	'DUMM_CL008_giTG068_072925':
		{
			'007': ['992', '1219', '1749'],

		}
}

# Visually confirm that the parameters used for the  mm3seg.segment_chnl_stack() function identify cell regions correctly
# parameter might change depending on brightness of phase image, OTSU_threshold parameter can range between 0.5 to 1.5
start = 0 # start time frame
end = 77 # last time frame to consider for segmentation
phase_channel = '0'
fluor_channel = '1'
for folder, fov_dict in seg_FOV_dict.items():
    for fov_id in fov_dict.keys():
        ana_peak_ids = fov_dict[fov_id]
        for peak_id in ana_peak_ids:
            path_to_phase_stack = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_channel}.tif'
            path_to_fluor_stack = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/subtracted_FOV_{fov_id}_region_{peak_id}_c_{fluor_channel}.tif'
            output_path = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/outputs'
            # the following worked for low phase exposure
            mm3seg.segment_chnl_stack(path_to_phase_stack,
                   output_path,
                   OTSU_threshold=0.5, # choose 0.5, 1, or 1.5
                   first_opening=4, # choose 3, 4, or 5
                   distance_threshold=3, # choose 1-3
                   second_opening_size=3, # choose 1, 2, 3
                   min_cell_area=100, # dependent on binning, these are for 1x1
				   max_cell_area=1000, # dependent on binning, these are for 1x1
                   small_merge_area_threshold=100) # dependent on binning, these are for 1x1
					# 	another example of paremeters that could work for higher phase exposure
					# 	OTSU_threshold = 1.5,
					# 	first_opening = 3,
					# 	distance_threshold = 1.5,
					# 	second_opening_size = 1,
					# 	min_cell_area = 100,
					# 	max_cell_area = 800,
					#   small_merge_area_threshold = 50
            labeled_stack = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/mm3_segmented_subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_channel}.tif'
            plot_cells.display_segmentation(path_to_phase_stack, mask_path = labeled_stack,
											start=start,
											end=end,
											alpha=0.5)


# create kymographs and load properties of segmented images into a dataframe

# sometime only part of a set of dataframes has usable data, where cells stay in trench for example.
# To track specific time ranges in a trench by trench basis, use this dictionary set up

time_range_dict = {'DUMM_CL008_giTG068_072925':
                   {'007':{'992':{'start':0,
                                   'end':75},
                           '1219':{'start':0,
                                   'end':75},
                            '1749':{'start':0,
                                   'end':75}
                          }
                    }
                   }

phase_channel = '0'
fluor_channel = '1'
for folder, fov_dict in seg_FOV_dict.items():
	all_cells_pd = pd.DataFrame()
	for fov_id in fov_dict.keys():
		ana_peak_ids = fov_dict[fov_id]
		for peak_id in ana_peak_ids:
			path_to_phase_stack = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_channel}.tif'
			path_to_fluor_stack = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/subtracted_FOV_{fov_id}_region_{peak_id}_c_{fluor_channel}.tif'
			path_to_labeled_stack = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/mm3_segmented_subtracted_FOV_{fov_id}_region_{peak_id}_c_{phase_channel}.tif'
			phase_stack = tifffile.imread(path_to_phase_stack)
			labeled_stack = tifffile.imread(path_to_labeled_stack)
			fluor_stack = tifffile.imread(path_to_fluor_stack)
			start = time_range_dict[folder][fov_id][peak_id]['start']
			end = time_range_dict[folder][fov_id][peak_id]['end']
			phase_output_dir = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted'
			phase_kymograph = plot_cells.create_kymograph(phase_stack, start, end, fov_id, peak_id, phase_output_dir)
			mask_output_dir = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/mask'

			os.makedirs(mask_output_dir, exist_ok=True)
			mask_kymograph = plot_cells.create_kymograph(labeled_stack, start, end, fov_id, peak_id, mask_output_dir)
			fluor_output_dir = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{folder}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/fluor'
			os.makedirs(fluor_output_dir, exist_ok=True)
			fluor_kymograph = plot_cells.create_kymograph(fluor_stack, start, end, fov_id, peak_id, fluor_output_dir)
			labeled_kymograph_mask = label(mask_kymograph)
			props = measure.regionprops_table(labeled_kymograph_mask, phase_kymograph,
											  properties=['label', 'area', 'coords', 'centroid', 'axis_major_length',
														  'axis_minor_length',
														  'intensity_mean', 'intensity_max', 'intensity_min'])
			props_fluor = measure.regionprops_table(labeled_kymograph_mask, fluor_kymograph,
													properties=['label',
																'intensity_mean', 'intensity_max', 'intensity_min'])
			region_phase_df = pd.DataFrame(props)
			region_fluor_df = pd.DataFrame(props_fluor)
			full_region_df = region_phase_df.merge(region_fluor_df, how='inner', on='label',
												   suffixes=('_phase', '_fluor'))
			labeled_stack_px_width = labeled_stack.shape[-1]
			mask_kymograph_px_width = mask_kymograph.shape[-1]
			plot_cells.add_time_frame_df(full_region_df, labeled_stack_px_width, mask_kymograph_px_width,
							   x_centroid_col='centroid-1')
			full_region_df['experiment_name'] = folder
			full_region_df['FOV'] = fov_id
			full_region_df['trench_id'] = peak_id
			if all_cells_pd.empty:
				all_cells_pd = full_region_df
			else:
				all_cells_pd = pd.concat([all_cells_pd, full_region_df], ignore_index=True)
	all_cells_filename = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/all_cell_data_{folder}.pkl'
	all_cells_pd.to_pickle(all_cells_filename)

# track cell lineages, must have trained GNN model

# create a dictionary with all the directories of segmented cells, specifying gene per directory, example dictionary:

strain_exp_dict = {'chpS': ['DUMM_giTG62_Glucose_012925'],
                  'baeS':['DUMM_giTG66_Glucose_012325'],
                  'lacZ':['DUMM_giTG059_noKan_Glucose_031125'],
                  'gfcE': ['DUMM_giTG064_Glucose_022625'],
                  'gldA': ['DUMM_giTG69_Glucose_013025'],
                  'alkA': ['DUMM_giTG068_063_061725_v2','DUMM_giTG63_giTG67_Glucose_121724_1_v2'],
                  'mazF': ['DUMM_giTG059_060_061125'], # constitutive
                  'hupA':['DUMM_giTG068_052925', 'DUMM_giTG068_063_061725'] # constitutive
                  }

df_all = pd.DataFrame()
for gene, exps in strain_exp_dict.items():
    for exp_view in exps:
        print(exp_view)
        all_cells_filename = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/all_cell_data_{exp_view}.pkl'
        all_cells_pd = pd.read_pickle(all_cells_filename)
        all_cells_pd['gene'] = gene
        if df_all.empty:
            df_all = all_cells_pd
        else:
            df_all = pd.concat([df_all, all_cells_pd], ignore_index=True)
df_all['node_id'] = df_all.index

df_all.rename(columns = {'centroid-0': 'centroid_y','centroid-1': 'centroid_x'}, inplace = True)
all_cells_filename = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/20250829_all_cell_data.pkl'
df_all.to_pickle(all_cells_filename)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
node_feature_cols = ['area', 'centroid_y',
       'axis_major_length', 'axis_minor_length', 'intensity_mean_phase',
       'intensity_max_phase', 'intensity_min_phase', 'intensity_mean_fluor',
       'intensity_max_fluor', 'intensity_min_fluor']
for col in node_feature_cols:
    df_all[col] = df_all[col].astype(np.float32);

# Access the node features from the single train_df DataFrame.
all_train_node_features_df = df_all[node_feature_cols].values.astype(np.float32)

# Initialize and fit the scaler on the training data.
scaler = StandardScaler()
scaler.fit(all_train_node_features_df)
print("Scaler fitted on training data.")

# Create an instance of the transform with the fitted scaler.
transform = bh_track.StandardScalerTransform(scaler)

dataset = bh_track.CellTrackingDataset(root='./processed_data_bh/full_dataset',
                                   df_cells=df_all,
                                   node_feature_cols=node_feature_cols,
                                   device=device,
                                   pre_transform=transform)

batch_size = 32
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print(f"Number of test graphs: {len(dataset)}")

num_node_features = len(node_feature_cols)
initial_edge_feature_dim = len(node_feature_cols) + 1
hidden_channels = 128 # balance between expressiveness and compute

model = bh_track.LineageLinkPredictionGNN(in_channels=num_node_features,
    initial_edge_channels=initial_edge_feature_dim,
    hidden_channels=128,
    num_blocks=2).to(device)
model.load_state_dict(torch.load('best_link_prediction_model.pt')) # Nora has this
print("Loaded best model cell linkage.")

lineage_predictions_df = bh_track.predict_cell_linkages(model,
                                                       test_loader,
                                                       device)

df_consolidated_lineages =  lineage_detection.process_all_fovs(lineage_predictions_df, df_all, prob_threshold=0.8)
df_for_kymograph_plot = lineage_detection.consolidate_lineages_to_node_df(df_consolidated_lineages, df_all)
tracked_all_cells_filename = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/20250829_tracked_all_cell_data.pkl'
df_for_kymograph_plot.to_pickle(tracked_all_cells_filename)

df_for_kymograph_plot = pd.read_pickle(tracked_all_cells_filename)
unique_fovs = df_for_kymograph_plot[['experiment_name', 'FOV', 'trench_id']].drop_duplicates().to_records(index=False)

for cell in unique_fovs:
	exp, fov, trench = cell
	path_to_phase_kymograph = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{exp}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/{fov}_{trench}.tif'
	path_to_fluor_kymograph = f'/Users/noravivancogonzalez/Documents/DuMM_image_analysis/{exp}/hyperstacked/drift_corrected/rotated/mm_channels/subtracted/fluor/{fov}_{trench}.tif'

	if os.path.exists(path_to_phase_kymograph) and os.path.exists(path_to_fluor_kymograph):
		print(cell)

		phase_kymograph = tifffile.imread(path_to_phase_kymograph)
		fluor_kymograph = tifffile.imread(path_to_fluor_kymograph)
		df_view = df_for_kymograph_plot[df_for_kymograph_plot['experiment_name'].isin([exp]) &
										df_for_kymograph_plot['FOV'].isin([fov]) &
										df_for_kymograph_plot['trench_id'].isin([trench])].copy()
		plot_cells.plot_kymograph_cells_id(phase_kymograph, fluor_kymograph,
								df_view,
								exp, fov, trench,
								track_id_col='predicted_lineage')