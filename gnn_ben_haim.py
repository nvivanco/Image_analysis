import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Batch, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import JumpingKnowledge
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils

from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial import KDTree
from torch.nn import BCEWithLogitsLoss


def train_dynamic(model, loader, optimizer, neg_sample_ratio=1, device=None):
	"""
	Trains the model for one epoch, adapted to handle the corrected model output.

	Args:
		model: The GNN model with a forward (encoder) and decode method.
		loader: The PyG DataLoader for training graphs.
		optimizer: The optimizer.
		neg_sample_ratio: The ratio of negative to positive samples.
		device: The device to run on ('cuda' or 'cpu').

	Returns:
		float: The average loss for the epoch.
	"""
	model.train()
	total_loss = 0
	num_batches_processed = 0

	pos_weight = torch.tensor(neg_sample_ratio, device=device)
	criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

	for batch_data in loader:
		optimizer.zero_grad()
		batch_data = batch_data.to(device)

		if batch_data.edge_index.numel() == 0 or batch_data.num_nodes == 0:
			continue

		# Correctly get the single tensor returned by the model's forward pass.
		# 'z' is now a tensor containing the final node embeddings.
		z, _ = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)

		pos_links = batch_data.edge_index.long()
		# Get the original edge attributes for positive links.
		# These were already projected within the model's forward pass.
		pos_edge_attr_raw = batch_data.edge_attr
		pos_edge_attr_projected = model.initial_edge_proj(pos_edge_attr_raw)

		# Generate negative links
		num_neg_samples = int(pos_links.size(1) * neg_sample_ratio)
		neg_links = pyg_utils.negative_sampling(
			pos_links, num_nodes=batch_data.num_nodes, num_neg_samples=num_neg_samples
		).to(device).long()

		# Generate negative edge attributes for the new links.
		# These need to be projected to the correct dimension before passing to the decoder.
		neg_src_features = batch_data.x[neg_links[0]]
		neg_tgt_features = batch_data.x[neg_links[1]]
		neg_edge_attr_raw = DS_block(neg_src_features, neg_tgt_features)

		# Project the raw negative edge attributes using the model's initial edge projector.
		neg_edge_attr_projected = model.initial_edge_proj(neg_edge_attr_raw)

		# Pass the final node embeddings ('z') and the correct edge attributes to the decoder.
		pos_logits = model.decode(z, pos_links, pos_edge_attr_projected)
		neg_logits = model.decode(z, neg_links, neg_edge_attr_projected)

		pos_labels = torch.ones(pos_logits.size(0), device=device)
		neg_labels = torch.zeros(neg_logits.size(0), device=device)

		logits = torch.cat([pos_logits, neg_logits])
		labels = torch.cat([pos_labels, neg_labels])

		loss = criterion(logits.squeeze(), labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		num_batches_processed += 1

	return total_loss / num_batches_processed if num_batches_processed > 0 else 0.0



def evaluate_dynamic(model, loader, criterion, device, neg_sample_ratio=1, node_lineage_map=None):
	"""
	Performs a single epoch of dynamic evaluation and returns comprehensive results.
	Corrected to handle the new model architecture.
	"""
	model.eval()
	total_loss = 0
	all_preds_logits = []
	all_labels_agg = []
	all_predicted_labels_individual = []
	all_probabilities_individual = []
	all_true_labels_individual = []
	all_evaluated_edge_indices_global = []
	derived_lineage_labels_final = []
	num_batches_processed_eval = 0

	with torch.no_grad():
		for data in loader:
			for key, value in data.items():
				if torch.is_tensor(value) and value.dtype == torch.float64:
					data[key] = value.to(torch.float32)

			data = data.to(device)

			if not hasattr(data, 'original_global_node_ids'):
				raise AttributeError("The 'data' object (batch) does not have 'original_global_node_ids'.")
			if not hasattr(data, 'edge_attr'):
				print(
					"Warning: 'edge_attr' not found in data batch. This function is designed to work with edge attributes.")
				continue

			if data.edge_index.numel() == 0 or data.num_nodes == 0:
				print(f"Skipping evaluation batch: No valid edges or nodes.")
				continue

			batch_original_global_node_ids = data.original_global_node_ids.cpu().numpy()

			pos_links = data.edge_index.long()

			# Project the positive edge attributes to match the hidden_channels dimension
			pos_edge_attr_projected = model.initial_edge_proj(data.edge_attr)

			num_neg_samples = int(pos_links.size(1) * neg_sample_ratio)
			neg_links = pyg_utils.negative_sampling(
				pos_links,
				num_nodes=data.num_nodes,
				num_neg_samples=num_neg_samples
			).to(device).long()

			if pos_links.numel() == 0 or neg_links.numel() == 0:
				print(f"Skipping evaluation batch: Insufficient positive or negative links after sampling.")
				continue

			# Generate and project the negative edge attributes
			neg_src_features = data.x[neg_links[0]]
			neg_tgt_features = data.x[neg_links[1]]
			neg_edge_attr_raw = DS_block(neg_src_features, neg_tgt_features)
			neg_edge_attr_projected = model.initial_edge_proj(neg_edge_attr_raw)

			# GNN Encoder and Decoder
			# The model's forward pass now returns only the node embeddings (z).
			z, _ = model(data.x, data.edge_index, data.edge_attr)

			# Pass the projected edge attributes to the decode method.
			pos_logits = model.decode(z, pos_links, pos_edge_attr_projected)
			neg_logits = model.decode(z, neg_links, neg_edge_attr_projected)

			# Create labels and concatenate logits
			pos_labels = torch.ones(pos_logits.size(0), device=device)
			neg_labels = torch.zeros(neg_logits.size(0), device=device)
			logits = torch.cat([pos_logits, neg_logits])
			labels = torch.cat([pos_labels, neg_labels])

			if logits.numel() == 0:
				print(f"Skipping evaluation batch: Combined logits tensor is empty.")
				continue

			loss = criterion(logits.squeeze(), labels)
			total_loss += loss.item()
			num_batches_processed_eval += 1

			all_preds_logits.append(logits.cpu())
			all_labels_agg.append(labels.cpu())
			probabilities = torch.sigmoid(logits)
			predicted_labels = (probabilities > 0.5).long()
			all_predicted_labels_individual.append(predicted_labels.cpu())
			all_probabilities_individual.append(probabilities.cpu())
			all_true_labels_individual.append(labels.cpu())

			# Map local indices back to global IDs
			current_evaluated_edge_indices_local = torch.cat([pos_links, neg_links], dim=-1)
			source_global_ids = batch_original_global_node_ids[current_evaluated_edge_indices_local[0].cpu().numpy()]
			target_global_ids = batch_original_global_node_ids[current_evaluated_edge_indices_local[1].cpu().numpy()]
			source_target_np = np.stack([source_global_ids, target_global_ids])
			current_evaluated_edge_indices_global = torch.from_numpy(source_target_np).long()
			all_evaluated_edge_indices_global.append(current_evaluated_edge_indices_global)

	if num_batches_processed_eval == 0:
		print("Warning: No batches were processed for evaluation. Returning default values.")
		return 0.0, 0.0, 0.0, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty((2, 0), dtype=torch.long), []

	# Aggregate results from all batches
	avg_loss = total_loss / num_batches_processed_eval
	all_preds_logits_agg = torch.cat(all_preds_logits)
	all_labels_np_agg = torch.cat(all_labels_agg).numpy()
	all_preds_proba_agg = torch.sigmoid(all_preds_logits_agg).numpy()

	accuracy = accuracy_score(all_labels_np_agg, (all_preds_proba_agg > 0.5).astype(int))
	if len(np.unique(all_labels_np_agg)) < 2:
		auc_score = 0.5
		print("Warning: Only one class present in true labels for AUC calculation. Setting AUC to 0.5.")
	else:
		auc_score = roc_auc_score(all_labels_np_agg, all_preds_proba_agg)

	final_predicted_labels = torch.cat(all_predicted_labels_individual, dim=0)
	final_probabilities = torch.cat(all_probabilities_individual, dim=0)
	final_true_labels = torch.cat(all_true_labels_individual, dim=0)
	final_evaluated_edge_indices_global = torch.cat(all_evaluated_edge_indices_global, dim=1)

	# Derive lineage labels if the map is provided
	if node_lineage_map:
		for i in range(final_evaluated_edge_indices_global.size(1)):
			src_global_id = final_evaluated_edge_indices_global[0, i].item()
			dst_global_id = final_evaluated_edge_indices_global[1, i].item()
			true_label_for_this_edge = final_true_labels[i].item()
			src_lineage = node_lineage_map.get(src_global_id, 'Unknown_Src_Node')
			dst_lineage = node_lineage_map.get(dst_global_id, 'Unknown_Dst_Node')
			if src_lineage.startswith('Unknown') or dst_lineage.startswith('Unknown'):
				edge_lineage = 'Unknown_Edge_Lineage'
			elif true_label_for_this_edge == 0:
				edge_lineage = f'NEG_({src_lineage}_to_{dst_lineage})'
			else:
				edge_lineage = f'POS_({src_lineage}_to_{dst_lineage})'
			derived_lineage_labels_final.append(edge_lineage)
	else:
		derived_lineage_labels_final = ['N/A'] * final_evaluated_edge_indices_global.size(1)

	source_nodes = final_evaluated_edge_indices_global[0].tolist()
	destination_nodes = final_evaluated_edge_indices_global[1].tolist()
	predicted_probabilities = final_probabilities.tolist()
	predicted_binary_labels = final_predicted_labels.tolist()
	true_binary_labels = final_true_labels.tolist()
	derived_lineage_labels = derived_lineage_labels_final

	data_for_df = {
		'Source_Node': source_nodes,
		'Destination_Node': destination_nodes,
		'Predicted_Probability': predicted_probabilities,
		'Predicted_Label': predicted_binary_labels,
		'True_Label': true_binary_labels,
		'Derived_lineage': derived_lineage_labels
	}
	df_predictions = pd.DataFrame(data_for_df)

	return (avg_loss, accuracy, auc_score, df_predictions)


def predict_cell_linkages(model, loader, device):
	"""
	Predicts cell linkages in candidate graphs by using the latest model architecture.

	Args:
		model (torch.nn.Module): The trained GNN model with the new architecture.
		loader (torch_geometric.data.DataLoader): Data loader for candidate graphs.
		device (str): Device to perform computations on ('cpu' or 'cuda').

	Returns:
		tuple: A tuple containing:
			- final_predicted_labels (torch.Tensor): Binary predictions (0 or 1).
			- final_probabilities (torch.Tensor): Predicted probabilities.
			- final_predicted_edge_indices_global (torch.Tensor): Global IDs of the predicted edges.
	"""
	model.eval()  # Set the model to evaluation mode

	all_probabilities = []
	all_predicted_labels = []  # Binary
	all_predicted_edge_indices_global = []  # Store global IDs of predicted edges

	with torch.no_grad():
		for data in loader:
			# Ensure data attributes are on the correct device
			for key, value in data.items():
				if torch.is_tensor(value) and value.dtype == torch.float64:
					data[key] = value.to(torch.float32)

			if not hasattr(data, 'original_global_node_ids'):
				raise AttributeError("The 'data' object (batch) lacks 'original_global_node_ids'.")

			if data.num_nodes == 0 or data.edge_index.numel() == 0:
				print("Skipping batch: No valid nodes or candidate edges.")
				continue

			# --- Core Prediction Logic ---

			# 1. Encode node features with the GNN encoder
			z, _ = model(data.x, data.edge_index, data.edge_attr)

			# 2. Get the candidate edges
			candidate_links = data.edge_index

			# 3. Use DS_block to generate raw edge attributes for these candidate links
			candidate_src_features = data.x[candidate_links[0]]
			candidate_tgt_features = data.x[candidate_links[1]]
			candidate_edge_attr_raw = DS_block(candidate_src_features, candidate_tgt_features)

			# 4. Project the raw edge attributes using the model's initial edge projection layer
			# This is crucial for matching your model's training pipeline
			candidate_edge_attr_projected = model.initial_edge_proj(candidate_edge_attr_raw)

			# 5. Decode the link logits using the node embeddings and projected edge attributes
			candidate_logits = model.decode(z, candidate_links, candidate_edge_attr_projected)

			# 6. Convert logits to probabilities and binary predictions
			probabilities = torch.sigmoid(candidate_logits)
			predicted_labels = (probabilities > 0.5).long()

			# --- Map local indices back to global IDs ---
			batch_original_global_node_ids = data.original_global_node_ids.cpu().numpy()

			current_predicted_edge_indices_local = data.edge_index.cpu().numpy()
			source_global_ids = batch_original_global_node_ids[current_predicted_edge_indices_local[0]]
			target_global_ids = batch_original_global_node_ids[current_predicted_edge_indices_local[1]]

			current_predicted_edge_indices_global = torch.tensor(
				[source_global_ids, target_global_ids], dtype=torch.long
			)

			# --- Accumulate results ---
			all_probabilities.append(probabilities.cpu())
			all_predicted_labels.append(predicted_labels.cpu())
			all_predicted_edge_indices_global.append(current_predicted_edge_indices_global)

	# Concatenate all results from batches
	final_probabilities = torch.cat(all_probabilities, dim=0)
	final_predicted_labels = torch.cat(all_predicted_labels, dim=0)
	final_predicted_edge_indices_global = torch.cat(all_predicted_edge_indices_global,
													dim=1)  # dim=1 for [2, num_edges]

	source_nodes = final_predicted_edge_indices_global[0].tolist()
	destination_nodes = final_predicted_edge_indices_global[1].tolist()
	exp_probabilities = final_probabilities.tolist()
	exp_binary_labels = final_predicted_labels.tolist()

	exp_data_for_df = {
		'Source_Node': source_nodes,
		'Destination_Node': destination_nodes,
		'Predicted_Probability': exp_probabilities,
		'Predicted_Label': exp_binary_labels,
	}
	df_exp_predictions = pd.DataFrame(exp_data_for_df)

	print("predict_cell_linkages: Prediction complete. Returning results.")
	return df_exp_predictions


def create_fov_graph(df_fov, node_feature_cols, device='cpu'):
	"""
	Creates a single PyG Data object for an entire Field of View (FOV) or trench,
	including all nodes and all ground-truth lineage edges, with edge attributes.

	Args:
		df_fov (pd.DataFrame): A DataFrame containing all cell data for a single FOV/trench.
		node_feature_cols (list): List of column names to use as node features.
		device (str): Device to place tensors on.

	Returns:
		torch_geometric.data.Data: A single PyG Data object for the entire FOV.
	"""
	if df_fov.empty:
		return None

	# Sort the DataFrame to ensure consistent node ordering
	df_fov = df_fov.sort_values(by=['node_id'])

	# Map original global IDs to new local indices
	original_global_node_ids = df_fov['node_id'].values
	global_id_to_local_idx = {global_id: i for i, global_id in enumerate(original_global_node_ids)}

	# Prepare node features and attributes
	x_data = df_fov[node_feature_cols].values.astype(np.float32)
	x = torch.tensor(x_data, dtype=torch.float32).to(device)
	pos_data = df_fov[['centroid_y']].values.astype(np.float32)
	pos = torch.tensor(pos_data, dtype=torch.float32).to(device)
	y = torch.tensor(df_fov['numeric_lineage'].values, dtype=torch.long).to(device)
	node_time_frames = torch.tensor(df_fov['time_frame'].values, dtype=torch.long).to(device)

	# Prepare edge_index and edge_attr
	source_nodes_local_idx = []
	target_nodes_local_idx = []

	sorted_time_frames = sorted(df_fov['time_frame'].unique())

	# Build the edge list for all lineages in the FOV
	for i in range(len(sorted_time_frames) - 1):
		current_t = sorted_time_frames[i]
		next_t = sorted_time_frames[i + 1]

		df_current_t = df_fov[df_fov['time_frame'] == current_t]
		df_next_t = df_fov[df_fov['time_frame'] == next_t]

		# Use dictionaries for efficient lookup
		current_lineage_to_node = df_current_t.set_index('ground_truth_lineage')['node_id'].to_dict()
		next_lineage_to_node = df_next_t.set_index('ground_truth_lineage')['node_id'].to_dict()

		for _, row in df_current_t.iterrows():
			current_ground_truth_lineage = row['ground_truth_lineage']
			current_global_node_id = row['node_id']

			# Case 1: Continuation (A -> A)
			if current_ground_truth_lineage in next_lineage_to_node:
				next_global_node_id = next_lineage_to_node[current_ground_truth_lineage]
				source_nodes_local_idx.append(global_id_to_local_idx[current_global_node_id])
				target_nodes_local_idx.append(global_id_to_local_idx[next_global_node_id])

			# Case 2: Division (A -> A.1, A.2)
			daughter1_lineage = f"{current_ground_truth_lineage}.1"
			daughter2_lineage = f"{current_ground_truth_lineage}.2"

			if daughter1_lineage in next_lineage_to_node:
				next_global_node_id = next_lineage_to_node[daughter1_lineage]
				source_nodes_local_idx.append(global_id_to_local_idx[current_global_node_id])
				target_nodes_local_idx.append(global_id_to_local_idx[next_global_node_id])
			if daughter2_lineage in next_lineage_to_node:
				next_global_node_id = next_lineage_to_node[daughter2_lineage]
				source_nodes_local_idx.append(global_id_to_local_idx[current_global_node_id])
				target_nodes_local_idx.append(global_id_to_local_idx[next_global_node_id])

	if not source_nodes_local_idx:
		edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
	else:
		unique_edges = list(set(zip(source_nodes_local_idx, target_nodes_local_idx)))
		source_nodes_unique, target_nodes_unique = zip(*unique_edges)
		edge_index = torch.tensor([list(source_nodes_unique), list(target_nodes_unique)], dtype=torch.long).to(device)

	# Create the single Data object without edge_attr
	data = Data(x=x,
				edge_index=edge_index,
				y=y,
				pos=pos,
				num_nodes=len(df_fov),
				time_frame=node_time_frames,
				original_global_node_ids=torch.tensor(original_global_node_ids, dtype=torch.long),
				experiment_name=df_fov['experiment_name'].iloc[0],
				fov=df_fov['FOV'].iloc[0],
				trench_id=df_fov['trench_id'].iloc[0]
				)
	return data



def create_smarter_candidate_graph(
		df_fov: pd.DataFrame,
		node_feature_cols: list,
		device: str = 'cpu',
		max_dist_growth: float = 150.0,
		max_dist_division: float = 150.0,
		min_area_ratio_division: float = 0.7,
		max_area_ratio_division: float = 1.3
) -> Data:
	"""
	Generates a smart candidate graph by applying biological and geometric rules
	and populating the PyTorch Geometric Data object with all specified features.

	Args:
		df_fov (pd.DataFrame): DataFrame containing cell data for a single FOV/trench.
							   Must include 'node_id', 'time_frame', 'centroid_y',
							   'centroid_x', 'area', 'major_axis_length', and
							   'minor_axis_length'.
		node_feature_cols (list): List of column names to use as node features.
		device (str): Device to place tensors on.
		max_dist_growth (float): Max centroid distance for a continuation edge.
		max_dist_division (float): Max centroid distance for a division edge.
		min_area_ratio_division (float): Min area ratio for a division to be considered valid.
		max_area_ratio_division (float): Max area ratio for a division to be considered valid.

	Returns:
		Data: A PyTorch Geometric Data object representing the smart candidate graph.
	"""
	if df_fov.empty:
		return None

	# Sort DataFrame for consistent node ordering
	df_fov = df_fov.sort_values(by=['node_id']).reset_index(drop=True)

	# --- 1. Prepare Node Features and Mapping ---
	original_global_node_ids = df_fov['node_id'].values
	global_id_to_local_idx = {global_id: i for i, global_id in enumerate(original_global_node_ids)}

	x_data = df_fov[node_feature_cols].values.astype(np.float32)
	x = torch.tensor(x_data, dtype=torch.float32).to(device)

	# Using centroid_y and centroid_x as 'pos' for 2D position
	pos_data = df_fov[['centroid_y', 'centroid_x']].values.astype(np.float32)
	pos = torch.tensor(pos_data, dtype=torch.float32).to(device)

	node_time_frames = torch.tensor(df_fov['time_frame'].values, dtype=torch.long).to(device)

	# --- 2. Generate Smart Edges ---
	edge_index_list = []
	sorted_time_frames = sorted(df_fov['time_frame'].unique())

	for i in range(len(sorted_time_frames) - 1):
		curr_frame, next_frame = sorted_time_frames[i], sorted_time_frames[i + 1]

		df_curr = df_fov[df_fov['time_frame'] == curr_frame].set_index('node_id')
		df_next = df_fov[df_fov['time_frame'] == next_frame].set_index('node_id')

		# Continuation Logic (1-to-1)
		for curr_id, curr_row in df_curr.iterrows():
			curr_pos = np.array([curr_row['centroid_y'], curr_row['centroid_x']])

			dists = np.linalg.norm(df_next[['centroid_y', 'centroid_x']].values - curr_pos, axis=1)
			if dists.size > 0:
				closest_idx = np.argmin(dists)
				closest_id = df_next.index[closest_idx]
				min_dist = dists[closest_idx]

				if min_dist < max_dist_growth:
					edge_index_list.append([global_id_to_local_idx[curr_id], global_id_to_local_idx[closest_id]])

		# Division Logic (1-to-2)
		for i, (next_id1, row1) in enumerate(df_next.iterrows()):
			for j, (next_id2, row2) in enumerate(df_next.iterrows()):
				if i >= j: continue

				pos1 = np.array([row1['centroid_y'], row1['centroid_x']])
				pos2 = np.array([row2['centroid_y'], row2['centroid_x']])

				daughter_dist = np.linalg.norm(pos1 - pos2)

				if daughter_dist < max_dist_division:
					midpoint = (pos1 + pos2) / 2

					dists_parent = np.linalg.norm(df_curr[['centroid_y', 'centroid_x']].values - midpoint, axis=1)
					if dists_parent.size > 0:
						closest_parent_idx = np.argmin(dists_parent)
						closest_parent_id = df_curr.index[closest_parent_idx]
						min_dist_parent = dists_parent[closest_parent_idx]

						if min_dist_parent < max_dist_division:
							parent_area = df_curr.loc[closest_parent_id]['area']
							total_daughter_area = row1['area'] + row2['area']

							if min_area_ratio_division <= total_daughter_area / parent_area <= max_area_ratio_division:
								edge_index_list.append(
									[global_id_to_local_idx[closest_parent_id], global_id_to_local_idx[next_id1]])
								edge_index_list.append(
									[global_id_to_local_idx[closest_parent_id], global_id_to_local_idx[next_id2]])

	# --- 3. Create the Final Graph ---
	if not edge_index_list:
		edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
	else:
		unique_edges = list(set(map(tuple, edge_index_list)))
		edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous().to(device)

	data = Data(
		x=x,
		edge_index=edge_index,
		pos=pos,
		num_nodes=len(df_fov),
		time_frame=node_time_frames,
		original_global_node_ids=torch.tensor(original_global_node_ids, dtype=torch.long).to(device),
		start_time_frame=df_fov['time_frame'].min(),
		experiment_name=df_fov['experiment_name'].iloc[0] if 'experiment_name' in df_fov.columns else 'N/A',
		fov=df_fov['FOV'].iloc[0] if 'FOV' in df_fov.columns else 'N/A',
		trench_id=df_fov['trench_id'].iloc[0] if 'trench_id' in df_fov.columns else 'N/A'
	)

	return data
def create_fov_candidate_graph(df_fov, node_feature_cols, device='cpu',
							   proximity_threshold_1d=50,  # Required for spatial search
							   max_neighbors=2  # To limit graph density
							   ):
	"""
	Creates a single PyG Data object for an entire FOV or trench, generating
	candidate edges based on spatial proximity between cells in consecutive time frames.

	Args:
		df_fov (pd.DataFrame): A DataFrame containing all cell data for a single FOV/trench.
		node_feature_cols (list): List of column names to use as node features.
		device (str): Device to place tensors on.
		proximity_threshold_1d (float, optional): Maximum 1D Euclidean distance (along centroid_y)
												  for a candidate link. REQUIRED.
		max_neighbors (int, optional): Limits the number of candidate edges a cell can have
									   to cells in the next time frame.

	Returns:
		torch_geometric.data.Data: A single PyG Data object with candidate edges.
	"""
	if df_fov.empty:
		return None

	if proximity_threshold_1d is None:
		raise ValueError("The 'proximity_threshold_1d' must be provided for candidate edge generation.")

	# Sort the DataFrame for consistent node ordering
	df_fov = df_fov.sort_values(by=['node_id']).reset_index(drop=True)

	# Map original global IDs to new local indices
	original_global_node_ids = df_fov['node_id'].values
	global_id_to_local_idx = {global_id: i for i, global_id in enumerate(original_global_node_ids)}

	# Prepare node features and attributes
	x = torch.tensor(df_fov[node_feature_cols].values, dtype=torch.float32).to(device)
	pos_data = df_fov[['centroid_y']].values  # Use centroid_y for 1D proximity search
	pos = torch.tensor(pos_data, dtype=torch.float32).to(device)
	y_data_for_search = pos_data.copy()
	y_min = y_data_for_search.min()
	y_max = y_data_for_search.max()
	y_range = y_max - y_min

	if y_range > 0:
		normalized_y_data = (y_data_for_search - y_min) / y_range
	else:
		normalized_y_data = y_data_for_search

	node_time_frames = torch.tensor(df_fov['time_frame'].values, dtype=torch.long).to(device)

	# --- Edge Generation Logic (Adapted from your other function) ---
	candidate_edges = []
	sorted_time_frames = sorted(df_fov['time_frame'].unique())
	time_frame_to_nodes_map = {}
	time_frame_to_kdtree_map = {}

	# Pre-build KDTree for each time frame
	for t_frame in sorted_time_frames:
		nodes_in_tf_mask = (node_time_frames.cpu().numpy() == t_frame)
		local_indices_in_tf = np.where(nodes_in_tf_mask)[0]
		time_frame_to_nodes_map[t_frame] = local_indices_in_tf

		if len(local_indices_in_tf) > 0:
			tf_pos_coords = pos_data[local_indices_in_tf].reshape(-1, 1)
			time_frame_to_kdtree_map[t_frame] = KDTree(tf_pos_coords)
		else:
			time_frame_to_kdtree_map[t_frame] = None

	# Iterate through all source time frames (except the last one)
	for i in range(len(sorted_time_frames) - 1):
		current_t = sorted_time_frames[i]
		next_t = sorted_time_frames[i + 1]

		current_nodes_local_indices = time_frame_to_nodes_map.get(current_t, [])
		kdtree_next_t = time_frame_to_kdtree_map.get(next_t)

		if (hasattr(current_nodes_local_indices, '__len__') and len(
				current_nodes_local_indices) == 0) or kdtree_next_t is None:
			continue

		for source_local_idx in current_nodes_local_indices:
			# Use normalized position for the query
			source_node_normalized_pos = normalized_y_data[source_local_idx]

			# Use the normalized radius
			neighbor_kdtree_indices_in_radius = kdtree_next_t.query_ball_point(
				source_node_normalized_pos.reshape(1, -1), r=proximity_threshold_1d
			)[0]

			if not neighbor_kdtree_indices_in_radius:
				continue

			distances_to_neighbors = []
			# Calculate distance using original (not normalized) y data for accuracy
			source_node_pos = pos_data[source_local_idx]
			for kdtree_idx in neighbor_kdtree_indices_in_radius:
				target_local_idx = time_frame_to_nodes_map[next_t][kdtree_idx]
				target_pos = pos_data[target_local_idx]
				dist = np.linalg.norm(source_node_pos - target_pos)
				distances_to_neighbors.append((dist, kdtree_idx))

			distances_to_neighbors.sort(key=lambda x: x[0])

			selected_neighbors_kdtree_indices = [
				item[1] for item in distances_to_neighbors[:max_neighbors]
			]

			for next_tf_kdtree_idx in selected_neighbors_kdtree_indices:
				target_local_idx = time_frame_to_nodes_map[next_t][next_tf_kdtree_idx]
				candidate_edges.append((source_local_idx, target_local_idx))

		if not candidate_edges:
			edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
		else:
			unique_edges = list(set(candidate_edges))
			edge_index = torch.tensor(unique_edges, dtype=torch.long).T.to(device)

	# --- Create PyG Data Object ---
	data = Data(
		x=x,
		edge_index=edge_index,
		pos=pos,
		num_nodes=len(df_fov),
		time_frame=node_time_frames,
		original_global_node_ids=torch.tensor(original_global_node_ids, dtype=torch.long),
		start_time_frame=df_fov['time_frame'].min(),
		experiment_name=df_fov['experiment_name'].iloc[0] if 'experiment_name' in df_fov.columns else 'N/A',
		fov=df_fov['FOV'].iloc[0] if 'FOV' in df_fov.columns else 'N/A',
		trench_id=df_fov['trench_id'].iloc[0] if 'trench_id' in df_fov.columns else 'N/A'
	)
	return data


# Helper MLP for f_PDN_edge
class PDNEdgeMLP(nn.Module):
	def __init__(self, edge_feature_dim, out_dim=1):
		super().__init__()
		# Simplified MLP for attention weights (scalar output)
		self.mlp = nn.Sequential(
			nn.Linear(edge_feature_dim, 32),  # Example hidden dim
			nn.ReLU(),
			nn.Linear(32, out_dim)
		)

	def forward(self, z):  # z is edge feature
		return self.mlp(z)


# Helper MLP for f_PDN_node
class PDNNodeMLP(nn.Module):
	def __init__(self, node_feature_dim, out_dim):
		super().__init__()
		# MLP for transforming node features before aggregation
		self.mlp = nn.Sequential(
			nn.Linear(node_feature_dim, out_dim),  # Typically out_dim = node_feature_dim
			nn.ReLU()
			# No final ReLU if you want negative values for weighted sum, or add Batch Norm
		)

	def forward(self, x):  # x is node feature
		return self.mlp(x)


# D-S Block determine similarity between nodes
def DS_block(v_i, v_j):
	"""
	Calculates Distance & Similarity vector for two batches of node feature vectors.

	Args:
	   v_i (torch.Tensor): Batch of source node features, shape [batch_size, d_v].
	   v_j (torch.Tensor): Batch of target node features, shape [batch_size, d_v].
	Returns:
	   torch.Tensor: Concatenated vector of absolute differences and cosine similarity, shape [batch_size, d_v + 1].
	"""
	# Calculate the absolute difference for the entire batch
	abs_diff = torch.abs(v_i - v_j)

	# Calculate cosine similarity across the feature dimension (dim=-1)
	cosine_similarity = F.cosine_similarity(v_i, v_j, dim=-1)

	# Unsqueeze the cosine similarity to add a feature dimension
	cosine_similarity = cosine_similarity.unsqueeze(-1)

	# Concatenate the two tensors along the feature dimension
	return torch.cat([abs_diff, cosine_similarity], dim=-1)


# EP-MPNN Block incorporates neighborhood info
class EP_MPNN_Block(MessagePassing):
	def __init__(self, node_channels, edge_channels):
		super().__init__(aggr='add',
						 flow='source_to_target')  # Aggregation for node update. source_to_target for N(i) being t-1 nodes.
		self.node_channels = node_channels
		self.edge_channels = edge_channels

		# Node Feature Update components (PDN-Conv)
		# f_PDN_edge: Maps edge features to scalar attention weights (omega)
		self.f_pdn_edge = PDNEdgeMLP(edge_channels, out_dim=1)
		# f_PDN_node: Transforms node features (tilde_x)
		self.f_pdn_node = PDNNodeMLP(node_channels, node_channels)  # Output dim same as input for residuals

		# Edge Feature Update components (Edge Encoder)
		# f_EE_edge: MLP to update edge features.
		# Input: current edge_features (edge_channels)
		#        + updated node_features from source (node_channels)
		#        + updated node_features from target (node_channels)
		#        + D-S block output (node_channels + 1)
		self.f_ee_edge = nn.Sequential(
			nn.Linear(edge_channels + 2 * node_channels + (node_channels + 1), 128),  # Example hidden size
			nn.ReLU(),
			nn.Linear(128, edge_channels)  # Output dim same as edge_channels
		)

		# BatchNorm (optional but often helpful for stability)
		self.bn_node = nn.BatchNorm1d(node_channels)
		self.bn_edge = nn.BatchNorm1d(edge_channels)

	def forward(self, x, edge_index, edge_attr):
		# x: node features X^(l-1)
		# edge_index: graph connectivity
		# edge_attr: edge features Z^(l-1)

		x_prev = x
		edge_attr_prev = edge_attr

		x_updated = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))

		# Add residual connection and apply BatchNorm/ReLU
		x = self.bn_node(x_prev + x_updated)  # Residual (assuming input and output dims are same)
		x = F.relu(x)  # x is now X^(l)

		# Get source and target node embeddings for edges
		row, col = edge_index
		src_node_features = x[row]  # x^(l) for source nodes
		tgt_node_features = x[col]  # x^(l) for target nodes

		# Compute D-S block output for each edge
		ds_outputs = []
		for i in range(edge_attr.size(0)):  # Iterate per edge
			ds_outputs.append(DS_block(src_node_features[i], tgt_node_features[i]))
		ds_outputs_tensor = torch.stack(ds_outputs, dim=0)  # Shape: [num_edges, node_channels + 1]

		# Concatenate inputs for f_EE_edge
		# current edge_features (Z^(l-1))
		# updated node_features from source (X^(l))
		# updated node_features from target (X^(l))
		# D-S block output (from X^(l), X^(l))
		edge_input_for_mlp = torch.cat([
			edge_attr_prev,  # Z^(l-1)
			src_node_features,  # X^(l)
			tgt_node_features,  # X^(l)
			ds_outputs_tensor  # D-S block on X^(l)
		], dim=-1)

		# Pass through edge encoder MLP
		edge_attr = self.f_ee_edge(edge_input_for_mlp)  # Z^(l)
		edge_attr = self.bn_edge(edge_attr)  # BatchNorm
		edge_attr = F.relu(edge_attr)  # ReLU

		return x, edge_attr  # Return updated nodes (X^(l)) and updated edges (Z^(l))

	def message(self, x_j, edge_attr):  # x_j is neighbor features, edge_attr_i is edge features to neighbor
		# x_j: x^(l-1)_j (features of neighbor j)
		# edge_attr: z^(l-1)_ji (features of edge (j,i))
		# Compute omega_ji = f_PDN_edge(z_ji) (attention weight for edge j,i)
		omega_ji = self.f_pdn_edge(edge_attr)
		# Compute tilde_x_j = f_PDN_node(x_j) (mapped feature vector of node j)
		tilde_x_j = self.f_pdn_node(x_j)

		# The message is omega_ji * tilde_x_j
		return omega_ji * tilde_x_j

	def aggregate(self, inputs, index, dim_size=None):
		# inputs: [num_messages, hidden_channels] (omega_ji * tilde_x_j for each edge)
		# index: target node index for each message
		# dim_size: total number of nodes
		out = super().aggregate(inputs, index, dim_size=dim_size)
		return out

	def update(self, aggr_out):

		return aggr_out  # This will be the x_updated in the forward pass


class LineageLinkPredictionGNN(nn.Module):
	def __init__(self, in_channels, initial_edge_channels, hidden_channels, num_blocks=2):
		super().__init__()
		self.num_blocks = num_blocks
		self.hidden_channels = hidden_channels

		self.jk = JumpingKnowledge('cat', hidden_channels, num_blocks)

		# Initial Linear layer to project input features to hidden_channels
		self.initial_node_proj = nn.Linear(in_channels, hidden_channels)

		# Initial Edge Feature Projector
		self.initial_edge_proj = nn.Linear(initial_edge_channels, hidden_channels)

		# Stack L EP-MPNN blocks
		self.ep_mpnn_blocks = nn.ModuleList()
		for _ in range(num_blocks):
			self.ep_mpnn_blocks.append(EP_MPNN_Block(hidden_channels, hidden_channels))

		# The output dimension of JK is num_blocks * hidden_channels for 'cat' mode.
		# The decoder needs to match this new dimension.
		# New input dimension for decoder:
		# (num_blocks * hidden_channels) + (num_blocks * hidden_channels) + hidden_channels
		# For a 2-layer GNN, this is (2*128) + (2*128) + 128 = 256 + 256 + 128 = 640
		decoder_in_channels = 2 * (num_blocks * hidden_channels) + hidden_channels

		self.decoder = nn.Sequential(
			nn.Linear(decoder_in_channels, 64),
			nn.ReLU(),
			nn.Linear(64, 1)  # Output a single logit for binary classification
		)


	def forward(self, x, edge_index, edge_attr):
		# Initial projection of node and edge features
		x = F.relu(self.initial_node_proj(x))
		edge_attr = F.relu(self.initial_edge_proj(edge_attr))

		# List to store node embeddings for Jumping Knowledge
		xs = []

		# Pass through L EP-MPNN blocks
		for block in self.ep_mpnn_blocks:
			x, edge_attr = block(x, edge_index, edge_attr)
			xs.append(x)  # Append the output of each block

		# Apply Jumping Knowledge to get the final, aggregated node embeddings
		z = self.jk(xs)

		return z, edge_attr


	def decode(self, z, edge_index, edge_attr):

		src_embed = z[edge_index[0, :]]
		tgt_embed = z[edge_index[1, :]]

		edge_features = torch.cat([src_embed, tgt_embed, edge_attr], dim=-1)

		logits = self.decoder(edge_features).squeeze(-1)

		return logits

class CellTrackingDataset_real_data(InMemoryDataset):
	def __init__(self, root, df_cells, node_feature_cols, device, transform=None, pre_transform=None,
				 force_reload=False):
		self.df_cells = df_cells
		self.node_feature_cols = node_feature_cols
		self.device = device

		# This will automatically call self.process() if the processed file doesn't exist.
		super().__init__(root, transform, pre_transform, force_reload=force_reload)

		data_object = torch.load(self.processed_paths[0], weights_only=False)
		self.data, self.slices = self.collate([data_object])

	@property
	def raw_file_names(self):
		# We don't have raw files to download in this manual pipeline.
		return []

	@property
	def processed_file_names(self):
		# The name of the processed file to be created.
		return ['cell_tracking_fovs.pt']

	def download(self):
		# This method is not needed as we are providing the data via a DataFrame.
		pass

	def process(self):
		print(f"Starting data processing for {self.root}...")

		all_fov_graphs = []
		unique_fovs = self.df_cells[['experiment_name', 'FOV', 'trench_id']].drop_duplicates().to_records(index=False)

		for exp, fov, trench in unique_fovs:
			df_fov_trench = self.df_cells[
				(self.df_cells['experiment_name'] == exp) &
				(self.df_cells['FOV'] == fov) &
				(self.df_cells['trench_id'] == trench)
				].copy()

			if not df_fov_trench.empty:
				# Use your existing create_fov_graph function
				fov_graph = create_fov_candidate_graph(df_fov_trench, self.node_feature_cols, self.device)
				if fov_graph is not None:
					all_fov_graphs.append(fov_graph)

		print(f"Finished processing. Created {len(all_fov_graphs)} PyG Data objects.")

		# Apply pre_transform (StandardScalerTransform)
		if self.pre_transform is not None:
			all_fov_graphs = [self.pre_transform(data) for data in all_fov_graphs]

		# Collate all graphs into a single object and save it.
		if len(all_fov_graphs) > 0:
			full_dataset_loader = DataLoader(all_fov_graphs, batch_size=len(all_fov_graphs))
			batched_data = next(iter(full_dataset_loader))
			torch.save(batched_data, self.processed_paths[0])
			print(f"Processed data saved to {self.processed_paths[0]}")
		else:
			# Handle case with no graphs
			torch.save(Data(), self.processed_paths[0])
			print(f"No graphs to process. Saved an empty Data object to {self.processed_paths[0]}")

class CellTrackingDataset(InMemoryDataset):
	def __init__(self, root, df_cells, node_feature_cols, device, transform=None, pre_transform=None,
				 force_reload=False):
		self.df_cells = df_cells
		self.node_feature_cols = node_feature_cols
		self.device = device

		# This will automatically call self.process() if the processed file doesn't exist.
		super().__init__(root, transform, pre_transform, force_reload=force_reload)

		data_object = torch.load(self.processed_paths[0], weights_only=False)
		self.data, self.slices = self.collate([data_object])

	@property
	def raw_file_names(self):
		# We don't have raw files to download in this manual pipeline.
		return []

	@property
	def processed_file_names(self):
		# The name of the processed file to be created.
		return ['cell_tracking_fovs.pt']

	def download(self):
		# This method is not needed as we are providing the data via a DataFrame.
		pass

	def process(self):
		print(f"Starting data processing for {self.root}...")

		all_fov_graphs = []
		unique_fovs = self.df_cells[['experiment_name', 'FOV', 'trench_id']].drop_duplicates().to_records(index=False)

		for exp, fov, trench in unique_fovs:
			df_fov_trench = self.df_cells[
				(self.df_cells['experiment_name'] == exp) &
				(self.df_cells['FOV'] == fov) &
				(self.df_cells['trench_id'] == trench)
				].copy()

			if not df_fov_trench.empty:
				# Use your existing create_fov_graph function
				fov_graph = create_fov_graph(df_fov_trench, self.node_feature_cols, self.device)
				if fov_graph is not None:
					all_fov_graphs.append(fov_graph)

		print(f"Finished processing. Created {len(all_fov_graphs)} PyG Data objects.")

		# Apply pre_transform (StandardScalerTransform)
		if self.pre_transform is not None:
			all_fov_graphs = [self.pre_transform(data) for data in all_fov_graphs]

		# Collate all graphs into a single object and save it.
		if len(all_fov_graphs) > 0:
			full_dataset_loader = DataLoader(all_fov_graphs, batch_size=len(all_fov_graphs))
			batched_data = next(iter(full_dataset_loader))
			torch.save(batched_data, self.processed_paths[0])
			print(f"Processed data saved to {self.processed_paths[0]}")
		else:
			# Handle case with no graphs
			torch.save(Data(), self.processed_paths[0])
			print(f"No graphs to process. Saved an empty Data object to {self.processed_paths[0]}")


class StandardScalerTransform(T.BaseTransform):
    def __init__(self, scaler):
        self.scaler = scaler

    def __call__(self, data: Data) -> Data:
        # Step 1: Scale node features (x)
        data.x = torch.from_numpy(self.scaler.transform(data.x.cpu().numpy())).to(data.x.device).to(torch.float32)

        # Step 2: Compute edge attributes using the now-scaled node features (x)
        if data.edge_index.numel() > 0:
            src_nodes_features = data.x[data.edge_index[0]]
            tgt_nodes_features = data.x[data.edge_index[1]]

            # Use your DS_block to compute the edge attributes
            def DS_block(v_i, v_j):
                abs_diff = torch.abs(v_i - v_j)
                cosine_similarity = F.cosine_similarity(v_i, v_j, dim=-1).unsqueeze(-1)
                return torch.cat([abs_diff, cosine_similarity], dim=-1)

            data.edge_attr = DS_block(src_nodes_features, tgt_nodes_features)
        else:
            # Handle empty graphs
            data.edge_attr = torch.empty((0, data.x.size(1) + 1), dtype=torch.float32, device=data.x.device)

        return data