import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, Dataset, Batch, InMemoryDataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
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
		z = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)

		pos_links = batch_data.edge_index
		# Get the original edge attributes for positive links.
		# These were already projected within the model's forward pass.
		pos_edge_attr_raw = batch_data.edge_attr
		pos_edge_attr_projected = model.initial_edge_proj(pos_edge_attr_raw)

		# Generate negative links
		num_neg_samples = int(pos_links.size(1) * neg_sample_ratio)
		neg_links = pyg_utils.negative_sampling(
			pos_links, num_nodes=batch_data.num_nodes, num_neg_samples=num_neg_samples
		).to(device)

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

			# --- OPTIMIZED NEGATIVE SAMPLING ---
			pos_links = data.edge_index

			# --- Correctly generate and project edge attributes ---
			# Project the positive edge attributes to match the hidden_channels dimension
			pos_edge_attr_projected = model.initial_edge_proj(data.edge_attr)

			num_neg_samples = int(pos_links.size(1) * neg_sample_ratio)
			neg_links = pyg_utils.negative_sampling(
				pos_links,
				num_nodes=data.num_nodes,
				num_neg_samples=num_neg_samples
			).to(device)

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
			z = model(data.x, data.edge_index, data.edge_attr)

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
	derived_lineage_labels_final = []
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

	return (avg_loss, accuracy, auc_score,
			final_predicted_labels, final_probabilities, final_true_labels,
			final_evaluated_edge_indices_global, derived_lineage_labels_final)


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

	# --- New section for calculating edge attributes ---
	if not source_nodes_local_idx:
		edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
		edge_attr = torch.empty((0, len(node_feature_cols) * 2), dtype=torch.float).to(
			device)  # Placeholder for zero edges
	else:
		# Get unique edges and create edge_index
		unique_edges = list(set(zip(source_nodes_local_idx, target_nodes_local_idx)))
		source_nodes_unique, target_nodes_unique = zip(*unique_edges)
		edge_index = torch.tensor([list(source_nodes_unique), list(target_nodes_unique)], dtype=torch.long).to(device)

		# Calculate edge attributes for the unique edges
		edge_attr_list = []
		for i in range(edge_index.size(1)):
			src_idx = edge_index[0, i].item()
			tgt_idx = edge_index[1, i].item()

			v_i = x[src_idx]
			v_j = x[tgt_idx]

			# Use a simple difference-sum block for edge features
			abs_diff = torch.abs(v_i - v_j)
			cosine_similarity = F.cosine_similarity(v_i.unsqueeze(0), v_j.unsqueeze(0)).squeeze(0)

			# This now matches your model's expected input dimension (len(node_feature_cols) + 1)
			edge_attr_list.append(torch.cat([abs_diff, cosine_similarity.unsqueeze(0)], dim=-1))

		edge_attr = torch.stack(edge_attr_list, dim=0).to(device)

	# Create the single Data object, now including edge_attr
	data = Data(x=x,
				edge_index=edge_index,
				edge_attr=edge_attr,  # Added edge_attr
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


# --- Helper MLP for f_PDN_edge (Attention Weights) ---
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


# --- Helper MLP for f_PDN_node (Node Feature Transformation) ---
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


# --- D-S Block (Distance & Similarity) ---
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


# --- The EP-MPNN Block ---
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

		# 1. Edge Feature Update (first for this block, as per paper's description "In the l-th block vi = x(l)i and vj = x(l)j")
		# However, the paper implies x(l) is used. Let's assume for simplicity first block
		# uses x(l-1) and subsequent blocks use x(l).
		# To align with: "In the l-th block vi = x(l)i and vj = x(l)j." and "the features of an edge ej,i are updated based on the features of νi and νj"
		# This means edge update uses nodes *after* they are potentially updated by previous block.
		# For l=0 (initial), x(0) are raw features. For l>0, x(l) comes from PDN-Conv.
		# For simplicity, let's make it work sequentially: update nodes, THEN update edges using new nodes.
		# Or, as paper implies "alternately updated", meaning within the block loop:
		# Step A: Compute updated nodes x^(l) from x^(l-1) and z^(l-1)
		# Step B: Compute updated edges z^(l) from x^(l) and z^(l-1)
		# Let's follow this:

		# Cache inputs for edge update after node update
		x_prev = x
		edge_attr_prev = edge_attr

		# 2. Node Feature Update (PDN-Conv: Eq. 2)
		# Message passing step

		x_updated = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))

		# Add residual connection and apply BatchNorm/ReLU
		x = self.bn_node(x_prev + x_updated)  # Residual (assuming input and output dims are same)
		x = F.relu(x)  # x is now X^(l)

		# 3. Edge Feature Update (Edge Encoder: based on x^(l) and z^(l-1))
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
		# Summation aggregation (as per Eq. 2)
		out = super().aggregate(inputs, index, dim_size=dim_size)
		return out

	def update(self, aggr_out):
		# This is where the output of aggregation (sum_j omega_ji * tilde_x_j)
		# is combined with the current node feature.
		# But per Eq. 2, the residual is handled in the forward pass.
		# So we just return the aggregated messages here.
		return aggr_out  # This will be the x_updated in the forward pass



class LineageLinkPredictionGNN(nn.Module):
	def __init__(self, in_channels, initial_edge_channels, hidden_channels, num_blocks=2):
		super().__init__()
		self.num_blocks = num_blocks
		self.hidden_channels = hidden_channels

		# Initial Linear layer to project input features to hidden_channels
		self.initial_node_proj = nn.Linear(in_channels, hidden_channels)

		# Initial Edge Feature Projector
		self.initial_edge_proj = nn.Linear(initial_edge_channels, hidden_channels)

		# Stack L EP-MPNN blocks
		self.ep_mpnn_blocks = nn.ModuleList()
		for _ in range(num_blocks):
			# Assumes EP_MPNN_Block is defined elsewhere
			self.ep_mpnn_blocks.append(EP_MPNN_Block(hidden_channels, hidden_channels))

		# hidden_channels (src node) + hidden_channels (tgt node) + hidden_channels (edge)
		self.decoder = nn.Sequential(
			nn.Linear(3 * hidden_channels, 64),
			nn.ReLU(),
			nn.Linear(64, 1)  # Output a single logit for binary classification
		)

	def forward(self, x, edge_index, edge_attr):
		# Initial projection of node features
		x = F.relu(self.initial_node_proj(x))

		edge_attr = F.relu(self.initial_edge_proj(edge_attr))

		# Pass through L EP-MPNN blocks
		for block in self.ep_mpnn_blocks:
			x, edge_attr = block(x, edge_index, edge_attr)

		return x

	def decode(self, z, edge_index, edge_attr):
		src_embed = z[edge_index[0]]
		tgt_embed = z[edge_index[1]]

		edge_features = torch.cat([src_embed, tgt_embed, edge_attr], dim=-1)

		logits = self.decoder(edge_features).squeeze(-1)  # Use the correct decoder

		return logits


class CellTrackingDataset(InMemoryDataset):
    def __init__(self, root, df_cells, node_feature_cols, device, transform=None, pre_transform=None):
        self.df_cells = df_cells
        self.node_feature_cols = node_feature_cols
        self.device = device
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['cell_tracking_fovs.pt']  # Changed filename

    def download(self):
        pass

    def process(self):
        print("Starting data processing to create FOV graphs...")

        all_fov_graphs = []

        # Identify unique FOV/trench combinations
        unique_fovs = self.df_cells[['experiment_name', 'FOV', 'trench_id']].drop_duplicates().to_records(index=False)

        for exp, fov, trench in unique_fovs:
            # Filter the DataFrame to get ALL cells within this FOV/trench
            df_fov_trench = self.df_cells[
                (self.df_cells['experiment_name'] == exp) &
                (self.df_cells['FOV'] == fov) &
                (self.df_cells['trench_id'] == trench)
                ].copy()

            if not df_fov_trench.empty:
                # Call the new function
                fov_graph = create_fov_graph(df_fov_trench, self.node_feature_cols, self.device)

                if fov_graph is not None:
                    all_fov_graphs.append(fov_graph)

        print(f"Finished processing. Created {len(all_fov_graphs)} PyG Data objects (FOV graphs).")

        if self.pre_filter is not None:
            all_fov_graphs = [data for data in all_fov_graphs if self.pre_filter(data)]
        if self.pre_transform is not None:
            all_fov_graphs = [self.pre_transform(data) for data in all_fov_graphs]

        print("Processing data... Creating new data file.")
        torch.save(self.collate(all_fov_graphs), self.processed_paths[0])
        print(f"Processed data saved to {self.processed_paths[0]}")
