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

from sklearn.metrics import roc_auc_score, accuracy_score
from torch.nn import BCEWithLogitsLoss

def train_dynamic(model, loader, optimizer, device=None):
	"""
	Trains the model for one epoch using a comprehensive candidate graph.
	The function now expects the DataLoader to provide graphs with a 'y_edge'
	attribute containing ground-truth labels for each candidate edge.

	Args:
		model: The GNN model with a forward (encoder) and decode method.
		loader: The PyG DataLoader for training graphs.
		optimizer: The optimizer.
		device: The device to run on ('cuda' or 'cpu').

	Returns:
		float: The average loss for the epoch.
	"""
	model.train()
	total_loss = 0
	num_batches_processed = 0

	# The pos_weight is no longer necessary as the loss function will handle
	# the class imbalance directly from the `y_edge` tensor.
	criterion = BCEWithLogitsLoss()

	for batch_data in loader:
		optimizer.zero_grad()
		batch_data = batch_data.to(device)

		# Skip batches with no nodes or edges.
		if batch_data.edge_index.numel() == 0 or batch_data.num_nodes == 0:
			continue

		# Get all candidate links and their ground-truth labels.
		candidate_links = batch_data.edge_index.long()
		labels = batch_data.y_edge.float()

		# Get the node embeddings from the GNN encoder.
		z, _ = model(batch_data.x, candidate_links, batch_data.edge_attr)

		# Project the edge attributes before passing them to the decoder
		projected_edge_attr = model.initial_edge_proj(batch_data.edge_attr)

		# Pass the embeddings and projected edge attributes to the decoder
		logits = model.decode(z, candidate_links, projected_edge_attr)

		# Calculate the loss on all candidate links (both positive and negative).
		loss = criterion(logits.squeeze(), labels)

		# Backpropagation and optimization.
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		num_batches_processed += 1

	return total_loss / num_batches_processed if num_batches_processed > 0 else 0.0


def evaluate_dynamic(model, loader, device, node_lineage_map=None):
	"""
	Performs a single epoch of dynamic evaluation on a comprehensive candidate graph.
	The function now expects the DataLoader to provide graphs with a 'y_edge'
	attribute containing ground-truth labels for each candidate edge.
	"""
	model.eval()
	total_loss = 0
	all_preds_logits = []
	all_labels_agg = []
	all_evaluated_edge_indices_global = []
	num_batches_processed_eval = 0

	criterion = BCEWithLogitsLoss()

	with torch.no_grad():
		for data in loader:
			# Ensure tensors are on the correct device
			data = data.to(device)

			# Skip batches with no nodes or edges, or missing crucial attributes
			if data.edge_index.numel() == 0 or data.num_nodes == 0 or not hasattr(data, 'y_edge'):
				print(f"Skipping evaluation batch: No valid edges, nodes, or 'y_edge' attribute.")
				continue

			# Get all candidate links and their ground-truth labels directly from the graph
			candidate_links = data.edge_index.long()
			labels = data.y_edge.float()

			if candidate_links.numel() == 0:
				print(f"Skipping evaluation batch: Combined logits tensor is empty.")
				continue

			# GNN Encoder and Decoder
			# The model's forward pass now returns only the node embeddings (z).
			z, _ = model(data.x, candidate_links, data.edge_attr)

			# Project the edge attributes before passing them to the decoder
			projected_edge_attr = model.initial_edge_proj(data.edge_attr)

			# Pass the embeddings and projected edge attributes to the decoder
			logits = model.decode(z, candidate_links, projected_edge_attr)

			# Calculate the loss on all candidate links (both positive and negative)
			loss = criterion(logits.squeeze(), labels)
			total_loss += loss.item()
			num_batches_processed_eval += 1

			# Store aggregated results for final metrics
			all_preds_logits.append(logits.cpu())
			all_labels_agg.append(labels.cpu())

			# Map local indices back to global IDs
			batch_original_global_node_ids = data.original_global_node_ids.cpu().numpy()
			source_global_ids = batch_original_global_node_ids[candidate_links[0].cpu().numpy()]
			target_global_ids = batch_original_global_node_ids[candidate_links[1].cpu().numpy()]
			source_target_np = np.stack([source_global_ids, target_global_ids])
			current_evaluated_edge_indices_global = torch.from_numpy(source_target_np).long()
			all_evaluated_edge_indices_global.append(current_evaluated_edge_indices_global)

	if num_batches_processed_eval == 0:
		print("Warning: No batches were processed for evaluation. Returning default values.")
		return 0.0, 0.0, 0.0, pd.DataFrame()  # Return DataFrame instead of old tuple

	# Aggregate results from all batches
	avg_loss = total_loss / num_batches_processed_eval
	all_preds_logits_agg = torch.cat(all_preds_logits)
	all_labels_agg = torch.cat(all_labels_agg)
	all_preds_proba_agg = torch.sigmoid(all_preds_logits_agg).numpy()
	all_labels_np_agg = all_labels_agg.numpy()

	# Calculate performance metrics
	accuracy = accuracy_score(all_labels_np_agg, (all_preds_proba_agg > 0.5).astype(int))
	if len(np.unique(all_labels_np_agg)) < 2:
		auc_score = 0.5
		print("Warning: Only one class present in true labels for AUC calculation. Setting AUC to 0.5.")
	else:
		auc_score = roc_auc_score(all_labels_np_agg, all_preds_proba_agg)

	# Prepare data for final DataFrame
	final_evaluated_edge_indices_global = torch.cat(all_evaluated_edge_indices_global, dim=1)

	data_for_df = {
		'Source_Node': final_evaluated_edge_indices_global[0].tolist(),
		'Destination_Node': final_evaluated_edge_indices_global[1].tolist(),
		'Predicted_Probability': all_preds_proba_agg.flatten().tolist(),
		'Predicted_Label': (all_preds_proba_agg > 0.5).astype(int).flatten().tolist(),
		'True_Label': all_labels_np_agg.flatten().tolist()
	}

	# Add derived lineage labels
	derived_lineage_labels_final = ['N/A'] * final_evaluated_edge_indices_global.size(1)
	if node_lineage_map:
		for i in range(final_evaluated_edge_indices_global.size(1)):
			src_global_id = final_evaluated_edge_indices_global[0, i].item()
			dst_global_id = final_evaluated_edge_indices_global[1, i].item()
			true_label_for_this_edge = all_labels_np_agg[i].item()
			src_lineage = node_lineage_map.get(src_global_id, 'Unknown_Src_Node')
			dst_lineage = node_lineage_map.get(dst_global_id, 'Unknown_Dst_Node')
			if src_lineage.startswith('Unknown') or dst_lineage.startswith('Unknown'):
				derived_lineage_labels_final[i] = 'Unknown_Edge_Lineage'
			elif true_label_for_this_edge == 0:
				derived_lineage_labels_final[i] = f'NEG_({src_lineage}_to_{dst_lineage})'
			else:
				derived_lineage_labels_final[i] = f'POS_({src_lineage}_to_{dst_lineage})'

	data_for_df['Derived_lineage'] = derived_lineage_labels_final

	df_predictions = pd.DataFrame(data_for_df)

	return avg_loss, accuracy, auc_score, df_predictions


def predict_cell_linkages(model, loader, device):
    """
    Predicts cell linkages in candidate graphs by using the trained model.
    This function is for inference on data that has no ground truth labels.

    Args:
        model (torch.nn.Module): The trained GNN model.
        loader (torch_geometric.data.DataLoader): Data loader for candidate graphs.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        pd.DataFrame: A DataFrame containing predictions for all candidate links.
    """
    model.eval()
    all_probabilities = []
    all_predicted_edge_indices_global = []

    with torch.no_grad():
        for data in loader:
            # Ensure tensors are on the correct device
            data = data.to(device)

            # Skip batches with no nodes or edges, or missing crucial attributes
            if data.edge_index.numel() == 0 or data.num_nodes == 0 or not hasattr(data, 'original_global_node_ids'):
                print(f"Skipping prediction batch: No valid edges, nodes, or 'original_global_node_ids' attribute.")
                continue

            # Get all candidate links
            candidate_links = data.edge_index.long()

            # GNN Encoder and Decoder
            z, _ = model(data.x, candidate_links, data.edge_attr)

            # Project the edge attributes before passing them to the decoder
            projected_edge_attr = model.initial_edge_proj(data.edge_attr)

            # Pass the embeddings and projected edge attributes to the decoder to get logits
            logits = model.decode(z, candidate_links, projected_edge_attr)

            # Convert logits to probabilities and binary predictions
            probabilities = torch.sigmoid(logits)

            # Map local indices back to global IDs
            batch_original_global_node_ids = data.original_global_node_ids.cpu().numpy()
            source_global_ids = batch_original_global_node_ids[candidate_links[0].cpu().numpy()]
            target_global_ids = batch_original_global_node_ids[candidate_links[1].cpu().numpy()]
            source_target_np = np.stack([source_global_ids, target_global_ids])
            current_evaluated_edge_indices_global = torch.from_numpy(source_target_np).long()

            # Store aggregated results
            all_probabilities.append(probabilities.cpu())
            all_predicted_edge_indices_global.append(current_evaluated_edge_indices_global)

    # If no batches were processed, return an empty DataFrame
    if not all_probabilities:
        print("Warning: No batches were processed for prediction. Returning empty DataFrame.")
        return pd.DataFrame()

    # Aggregate results from all batches
    final_probabilities = torch.cat(all_probabilities, dim=0)
    final_evaluated_edge_indices_global = torch.cat(all_predicted_edge_indices_global, dim=1)

    # Prepare data for final DataFrame
    data_for_df = {
        'Source_Node': final_evaluated_edge_indices_global[0].tolist(),
        'Destination_Node': final_evaluated_edge_indices_global[1].tolist(),
        'Predicted_Probability': final_probabilities.flatten().tolist(),
        'Predicted_Label': (final_probabilities > 0.5).int().flatten().tolist()
    }

    df_predictions = pd.DataFrame(data_for_df)

    return df_predictions

def create_comprehensive_candidate_graph(
      df_fov: pd.DataFrame,
      node_feature_cols: list,
      device: str = 'cpu',
      max_dist_link: float = 50.0,
      min_area_ratio_division: float = 1.8,
      max_area_ratio_division: float = 2.2,
      min_area_ratio_continuation: float = 0.8,
      max_area_ratio_continuation: float = 1.2
) -> Data:
   if df_fov.empty:
      return None

   df_fov = df_fov.sort_values(by=['node_id']).reset_index(drop=True)
   original_global_node_ids = df_fov['node_id'].values
   global_id_to_local_idx = {global_id: i for i, global_id in enumerate(original_global_node_ids)}

   # Prepare node features and attributes
   x_data = df_fov[node_feature_cols].values.astype(np.float32)
   x = torch.tensor(x_data, dtype=torch.float32).to(device)
   pos_data = df_fov[['centroid_y']].values.astype(np.float32)
   pos = torch.tensor(pos_data, dtype=torch.float32).to(device)
   node_time_frames = torch.tensor(df_fov['time_frame'].values, dtype=torch.long).to(device)

   # Dictionaries for efficient ground truth lookup
   gt_links = {}
   for i, row in df_fov.iterrows():
      next_ids = []
      if 'ground_truth_link_next_id' in row:
         next_ids = row['ground_truth_link_next_id'] if isinstance(row['ground_truth_link_next_id'], list) else [
            row['ground_truth_link_next_id']]
      gt_links[row['node_id']] = next_ids

   edge_index_list = []
   edge_label_list = []

   sorted_time_frames = sorted(df_fov['time_frame'].unique())

   for i in range(len(sorted_time_frames) - 1):
      curr_frame, next_frame = sorted_time_frames[i], sorted_time_frames[i + 1]
      df_curr = df_fov[df_fov['time_frame'] == curr_frame]
      df_next = df_fov[df_fov['time_frame'] == next_frame]

      # Convert to numpy arrays for efficient distance calculation
      curr_y_pos = df_curr['centroid_y'].values.reshape(-1, 1)
      next_y_pos = df_next['centroid_y'].values.reshape(-1, 1)

      # 1-to-1 and 1-to-2 links
      for curr_idx, curr_row in df_curr.iterrows():
         curr_y = curr_row['centroid_y']

         # --- Generate plausible 1-to-1 links with area check ---
         dists = np.abs(next_y_pos - curr_y)
         potential_next_indices = np.where(dists < max_dist_link)[0]

         for next_local_idx in potential_next_indices:
            next_row = df_next.iloc[next_local_idx]

            parent_area = curr_row['area']
            child_area = next_row['area']
            area_ratio = child_area / parent_area

            if min_area_ratio_continuation <= area_ratio <= max_area_ratio_continuation:
               source_idx = global_id_to_local_idx[curr_row['node_id']]
               target_idx = global_id_to_local_idx[next_row['node_id']]
               edge_index_list.append([source_idx, target_idx])

               is_true_link = next_row['node_id'] in gt_links.get(curr_row['node_id'], [])
               edge_label_list.append(int(is_true_link))

         # --- Generate plausible 1-to-2 links ---
         for next_idx1, next_row1 in df_next.iterrows():
            for next_idx2, next_row2 in df_next.iterrows():
               if next_idx1 >= next_idx2: continue

               pos1_y = next_row1['centroid_y']
               pos2_y = next_row2['centroid_y']
               midpoint_y = (pos1_y + pos2_y) / 2

               dist_to_midpoint = np.abs(curr_y - midpoint_y)

               if dist_to_midpoint < max_dist_link:
                  parent_area = curr_row['area']
                  total_daughter_area = next_row1['area'] + next_row2['area']
                  area_ratio = total_daughter_area / parent_area

                  if min_area_ratio_division <= area_ratio <= max_area_ratio_division:
                     source_idx = global_id_to_local_idx[curr_row['node_id']]
                     target1_idx = global_id_to_local_idx[next_row1['node_id']]
                     target2_idx = global_id_to_local_idx[next_row2['node_id']]

                     # Add first daughter link
                     edge_index_list.append([source_idx, target1_idx])
                     is_true_link1 = next_row1['node_id'] in gt_links.get(curr_row['node_id'], [])
                     edge_label_list.append(int(is_true_link1))

                     # Add second daughter link
                     edge_index_list.append([source_idx, target2_idx])
                     is_true_link2 = next_row2['node_id'] in gt_links.get(curr_row['node_id'], [])
                     edge_label_list.append(int(is_true_link2))

   # --- Create the Final Graph ---
   if not edge_index_list:
      edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
      edge_label = torch.empty(0, dtype=torch.float32).to(device)
   else:
      unique_edges_dict = {}

      # Iterate through all generated edges to build the dictionary
      for edge, label in zip(edge_index_list, edge_label_list):
         edge_tuple = tuple(edge)
         # If the edge is new or the new label is '1', update the dictionary
         if edge_tuple not in unique_edges_dict or label == 1:
            unique_edges_dict[edge_tuple] = label

      if not unique_edges_dict:
         edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
         edge_label = torch.empty(0, dtype=torch.float32).to(device)
      else:
         unique_edges = list(unique_edges_dict.keys())
         unique_labels = list(unique_edges_dict.values())
         edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous().to(device)
         edge_label = torch.tensor(unique_labels, dtype=torch.float32).to(device)

   data = Data(
      x=x,
      edge_index=edge_index,
      y_edge=edge_label,
      pos=pos,
      num_nodes=len(df_fov),
      time_frame=node_time_frames,
      original_global_node_ids=torch.tensor(original_global_node_ids, dtype=torch.long).to(device),
      start_time_frame=torch.tensor(df_fov['time_frame'].min(), dtype=torch.long).to(device),
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
				fov_graph = create_comprehensive_candidate_graph(df_fov_trench, self.node_feature_cols, self.device)
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