import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.utils as pyg_utils
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, Dataset, Batch, InMemoryDataset
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import deque
import numpy as np
from torch.nn import BCEWithLogitsLoss

def find_lineage_branches(graph):
    print("--- Function find_lineage_branches_optimized_v3 started ---")
    all_lineage_segments = []

    print("Step 1: Identifying root nodes...")
    root_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]
    print(f"Step 1 Complete: Found {len(root_nodes)} root nodes. Roots: {root_nodes[:5]}...")

    print("Step 2: Initializing stack...")
    stack = deque()
    # A set to keep track of (node, start_node_of_this_segment) pairs already processed
    # This prevents infinite loops if a node can be reached by multiple paths that don't constitute a cycle.
    processed_segments_from_node = set()

    if not root_nodes:
        print("WARNING: No root nodes found in the graph! Returning empty list.")
        return []

    for root in root_nodes:
        # Each root starts a new path segment.
        stack.append((root, [root]))
        processed_segments_from_node.add((root, root)) # Add (current_node, start_node_of_this_segment)

    print(f"Step 2 Complete: Stack initialized with {len(stack)} items.")

    print("Step 3: Entering main traversal loop...")

    iteration_count = 0
    while stack:
        iteration_count += 1
        if iteration_count % 1000 == 0:
            print(f"  Iteration {iteration_count}: Stack size: {len(stack)}")

        current_node, current_path_builder = stack.pop()
        # The 'current_path_builder' is mutable. Its first element is the start of *this* segment.
        segment_start_node = current_path_builder[0]

        # Debugging the node being processed (uncomment for verbose output)
        print(f"Processing node: {current_node}, current segment length: {len(current_path_builder)}")
        print(f"  Segment started from: {segment_start_node}")

        out_degree = graph.out_degree(current_node)

        # Case 1: Leaf Node (end of a lineage segment)
        if out_degree == 0:
            print(f"  Node {current_node} is a leaf.")
            all_lineage_segments.append({"type": "segment_to_leaf", "path": list(current_path_builder)})
            # No need to add to processed_segments_from_node here, as it's a terminal point for this branch.
            continue

        # Case 2: Division Node (more than one outgoing edge)
        if out_degree > 1:
            print(f"  Node {current_node} is a division point.")
            all_lineage_segments.append({"type": "segment_to_division", "path": list(current_path_builder)})

            # For each successor, start a NEW lineage segment.
            # Push a new path_builder list onto the stack for each branch.
            for neighbor in graph.neighbors(current_node):
                new_segment_path = [current_node, neighbor] # New segment starts from current_node
                # Only push if this (neighbor, new_segment_start) hasn't been processed
                if (neighbor, current_node) not in processed_segments_from_node:
                    stack.append((neighbor, new_segment_path))
                    processed_segments_from_node.add((neighbor, current_node))
            continue

        # Case 3: Straight Line (exactly one outgoing edge)
        if out_degree == 1:
            print(f"  Node {current_node} is a straight-line node.")
            neighbor = next(iter(graph.neighbors(current_node)))

            # If this (neighbor, segment_start_node) has already been processed, don't re-add to stack
            # This is key to stopping the redundant processing.
            if (neighbor, segment_start_node) not in processed_segments_from_node:
                current_path_builder.append(neighbor) # Append in place
                stack.append((neighbor, current_path_builder))
                processed_segments_from_node.add((neighbor, segment_start_node))
            else:
                print(f"  Skipping re-processing ({neighbor}, {segment_start_node})") # Debug print for skipped nodes

    print(f"Step 3 Complete: Main traversal loop finished after {iteration_count} iterations.")
    print(f"Total lineage segments collected: {len(all_lineage_segments)}")
    print("--- Function find_lineage_branches_optimized_v3 finished ---")
    return all_lineage_segments


def create_fov_graph(df_fov, node_feature_cols, device='cpu'):
    """
    Creates a single PyG Data object for an entire Field of View (FOV) or trench,
    including all nodes and all ground-truth lineage edges.

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

    # Prepare edge_index
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

    # Convert edge lists to a PyG tensor
    if not source_nodes_local_idx:
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
    else:
        edge_index = torch.tensor([source_nodes_local_idx, target_nodes_local_idx], dtype=torch.long).to(device)

    # Create the single Data object
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

def create_candidate_lineage_graph(df_cells, node_feature_cols, device='cpu',
                                   proximity_threshold_1d=50,  # For 1D distance (centroid_y)
                                   max_neighbors=2  # Limit edges per node to avoid too many
                                   ):
    """
    Creates a PyTorch Geometric Data object for a new dataset. It generates candidate edges based on spatial proximity
    along the y axis between cells in consecutive time frames.

    Args:
        df_cells (pd.DataFrame): DataFrame of an FOV containing cell information for the new dataset.
                                 Must include 'node_id', 'time_frame', 'centroid_y',
                                 and other node features.
        device (str): Device to put tensors on ('cpu' or 'cuda').
        proximity_threshold_1d (float, optional): Maximum 1D Euclidean distance (along centroid_y)
                                                  for a candidate link between cells in consecutive
                                                  time frames. This parameter is REQUIRED.
        max_neighbors (int, optional): Limits the number of candidate edges a cell can have
                                       to cells in the next time frame. Helps manage graph density.

    Returns:
        torch_geometric.data.Data: A PyG Data object with node features, original global node IDs,
                                   time frames, and candidate edges.
    """

    if proximity_threshold_1d is None:
        raise ValueError("The 'proximity_threshold_1d' must be provided when only using centroid_y.")

    original_global_node_ids = df_cells['node_id'].values
    global_id_to_local_idx = {global_id: i for i, global_id in enumerate(original_global_node_ids)}

    # Ensure all required columns are present in df_cells for feature extraction and node properties
    required_cols = list(node_feature_cols) + ['node_id', 'time_frame']
    if not all(col in df_cells.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_cells.columns]
        raise ValueError(f"Missing required columns in df_cells: {missing}. "
                         f"Please ensure node_feature_cols are correctly defined and present, "
                         f"along with 'node_id' and 'time_frame'")

    x = torch.tensor(df_cells[node_feature_cols].values, dtype=torch.float).to(device)

    pos_data = df_cells[['centroid_y']].values  # Keep as a 2D array for consistency [N, 1]
    pos = torch.tensor(pos_data, dtype=torch.float).to(device)

    node_time_frames = torch.tensor(df_cells['time_frame'].values, dtype=torch.long).to(device)

    num_nodes = len(df_cells)
    if num_nodes == 0:
        print("Warning: Input DataFrame is empty. Returning None.")
        return None

    candidate_edges = []

    sorted_time_frames = sorted(df_cells['time_frame'].unique())

    proximity_threshold = proximity_threshold_1d

    print(f"Generating candidate edges using 1D centroids (centroid_y) and threshold: {proximity_threshold}")

    # Pre-build KDTree for each time frame for efficiency
    time_frame_to_nodes_map = {}  # Maps time_frame to list of global local indices
    time_frame_to_kdtree_map = {}  # Maps time_frame to KDTree

    # Collect nodes for each time frame and build KDTrees
    for t_frame in sorted_time_frames:
        nodes_in_tf_mask = (node_time_frames.cpu().numpy() == t_frame)
        local_indices_in_tf = np.where(nodes_in_tf_mask)[0]
        time_frame_to_nodes_map[t_frame] = local_indices_in_tf

        if len(local_indices_in_tf) > 0:
            # KDTree expects 2D array, so reshape if input is flat 1D (which it is for 'centroid_y' only)
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
            continue  # No source nodes or no target nodes in next time frame

        for source_local_idx in current_nodes_local_indices:
            source_node_pos = pos_data[source_local_idx]

            # Reshape for KDTree query if original pos_data was 1D
            source_node_pos_query = source_node_pos.reshape(1, -1)

            # Use query_ball_point to get all neighbors within radius first
            neighbor_kdtree_indices_in_radius = kdtree_next_t.query_ball_point(
                source_node_pos_query, r=proximity_threshold
            )[0]

            if not neighbor_kdtree_indices_in_radius:
                continue

            # Now, get the actual distances for these neighbors to sort them
            distances_to_in_radius_neighbors = []
            for kdtree_idx in neighbor_kdtree_indices_in_radius:
                target_pos = kdtree_next_t.data[kdtree_idx]
                dist = np.linalg.norm(source_node_pos_query - target_pos)  # Calculate distance
                distances_to_in_radius_neighbors.append((dist, kdtree_idx))

            # Sort by distance
            distances_to_in_radius_neighbors.sort(key=lambda x: x[0])

            # Take only the top 'max_neighbors' from the sorted list
            selected_neighbors_kdtree_indices = [
                item[1] for item in distances_to_in_radius_neighbors[:max_neighbors]
            ]

            for next_tf_kdtree_idx in selected_neighbors_kdtree_indices:
                target_local_idx = time_frame_to_nodes_map[next_t][next_tf_kdtree_idx]
                candidate_edges.append((source_local_idx, target_local_idx))

    if not candidate_edges:
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
    else:
        # Use a set to ensure uniqueness
        unique_edges = list(set(candidate_edges))
        # Transpose to get [2, num_edges] format
        edge_index = torch.tensor(unique_edges, dtype=torch.long).T.to(device)

    print(f"Generated {edge_index.size(1)} candidate temporal edges.")

    # Create the Data object. 'y' is omitted.
    data = Data(x=x,
                edge_index=edge_index,  # The generated candidate edges
                pos=pos,  # centroid_y
                num_nodes=num_nodes,
                time_frame=node_time_frames,
                original_global_node_ids=torch.tensor(original_global_node_ids, dtype=torch.long),
                start_time_frame=df_cells['time_frame'].min(),
                experiment_name=df_cells['experiment_name'].iloc[0] if 'experiment_name' in df_cells.columns else 'N/A',
                fov=df_cells['FOV'].iloc[0] if 'FOV' in df_cells.columns else 'N/A',
                trench_id=df_cells['trench_id'].iloc[0] if 'trench_id' in df_cells.columns else 'N/A'
                )
    return data


def get_all_plausible_negative_candidates(data, radius_threshold, device, num_neg_samples_per_pos_edge=None,
                                                    positive_edges_to_exclude=None):
    if positive_edges_to_exclude is None:
        positive_edges_to_exclude = data.edge_index

    existing_edges_overall_batch_set = set(tuple(e) for e in positive_edges_to_exclude.T.tolist())

    plausible_negative_candidates_list = []

    is_batch = hasattr(data, 'ptr')

    if is_batch:
        num_graphs_in_batch = len(data.ptr) - 1
    else:
        num_graphs_in_batch = 1

    for graph_idx in range(num_graphs_in_batch):
        # Determine nodes for this graph (handling batch and single-graph cases)
        if is_batch:
            start_node_idx = data.ptr[graph_idx].item()
            end_node_idx = data.ptr[graph_idx + 1].item()
            nodes_in_graph = torch.arange(start_node_idx, end_node_idx, device=device)
        else:
            nodes_in_graph = torch.arange(data.num_nodes, device=device)

        # Get positions for nodes in the current graph
        pos_in_graph = data.x[nodes_in_graph, 0:2]  # Assumes x,y positions are first two features

        # Vectorized distance calculation for ALL pairs in the current graph
        # This is a key optimization.
        diff_matrix = pos_in_graph.unsqueeze(1) - pos_in_graph.unsqueeze(0)
        distances = torch.linalg.norm(diff_matrix, dim=-1)

        # Vectorized filtering for distance threshold
        within_radius_indices = torch.nonzero(distances < radius_threshold, as_tuple=True)

        # Convert local indices to global batch indices
        source_indices_in_batch = nodes_in_graph[within_radius_indices[0]]
        target_indices_in_batch = nodes_in_graph[within_radius_indices[1]]

        candidate_links_in_batch = torch.stack([source_indices_in_batch, target_indices_in_batch], dim=0)

        # Remove self-loops
        candidate_links_in_batch = candidate_links_in_batch[:,
                                   candidate_links_in_batch[0] != candidate_links_in_batch[1]]

        # Filter out existing positive links
        valid_negatives = []
        for i in range(candidate_links_in_batch.size(1)):
            src_global = data.original_global_node_ids[candidate_links_in_batch[0, i]].item()
            tgt_global = data.original_global_node_ids[candidate_links_in_batch[1, i]].item()
            if (src_global, tgt_global) not in existing_edges_overall_batch_set:
                valid_negatives.append(candidate_links_in_batch[:, i].tolist())

        if valid_negatives:
            plausible_negative_candidates_list.append(torch.tensor(valid_negatives, dtype=torch.long).T.to(device))

    if not plausible_negative_candidates_list:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    neg_candidates = torch.cat(plausible_negative_candidates_list, dim=1)

    # Sample from the plausible negatives to get the desired ratio
    if num_neg_samples_per_pos_edge is not None and positive_edges_to_exclude.size(1) > 0:
        num_pos_edges = positive_edges_to_exclude.size(1)
        num_to_sample = num_pos_edges * num_neg_samples_per_pos_edge

        if neg_candidates.size(1) > num_to_sample:
            indices = torch.randperm(neg_candidates.size(1))[:num_to_sample]
            neg_candidates = neg_candidates[:, indices]

    return neg_candidates


def train_one_epoch_cell_tracking_hnm(model, loader, optimizer, criterion, device, radius_threshold: float, num_hard_neg_per_positive=None):
    model.train()
    total_loss = 0
    num_batches = 0

    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()

        if data.edge_index.numel() == 0:
            print(f"Skipping training batch {i+1}: No positive edges for this timeframe.")
            continue

        if data.edge_index.dtype != torch.long: data.edge_index = data.edge_index.long()

        # Compute node embeddings
        z = model(data.x, data.edge_index) # Or model.encode(data.x) if your GNN is set up that way

        # --- HNM Specific Steps ---
        # 1. Generate ALL plausible negative candidates using your adapted function
        candidate_neg_edge_index = get_all_plausible_negative_candidates(
            data,
            radius_threshold=radius_threshold, # Pass your radius_threshold here
            device=device
        )

        # Handle cases where no negative candidates are left after filtering (very rare if your data is structured)
        if candidate_neg_edge_index.numel() == 0:
            print(f"Warning (Batch {i+1}): No plausible negative candidates generated for this timeframe. Training with positive only.")
            logits = model.decode(z, data.edge_index)
            labels = torch.ones(logits.size(0), device=device)
        else:
            # 2. Compute logits for positive and ALL candidate negative samples
            pos_logits = model.decode(z, data.edge_index)
            candidate_neg_logits = model.decode(z, candidate_neg_edge_index)

            # 3. Identify Hard Negatives (select top N highest logits from candidates)
            sorted_neg_logits, _ = torch.sort(candidate_neg_logits, descending=True)

            if num_hard_neg_per_positive is None:
                # Default: take a fixed number of hardest negatives, up to the total available
                num_hard_neg_to_select = min(5, sorted_neg_logits.numel()) # Example: Top 5 hardest negatives
            else:
                # Use the ratio: ensures at least one positive is matched with ratio_val hard negatives
                num_hard_neg_to_select = min(int(num_hard_neg_per_positive * pos_logits.size(0)), sorted_neg_logits.numel())

            if num_hard_neg_to_select == 0:
                print(f"Warning (Batch {i+1}): No hard negative samples selected based on criteria. Training with positive only.")
                selected_hard_neg_logits = torch.empty(0, device=device)
            else:
                selected_hard_neg_logits = sorted_neg_logits[:num_hard_neg_to_select]

            # 4. Construct final batch logits and labels
            logits = torch.cat([pos_logits, selected_hard_neg_logits])
            labels = torch.cat([
                torch.ones(pos_logits.size(0), device=device),
                torch.zeros(selected_hard_neg_logits.size(0), device=device)
            ])

        # 5. Calculate loss and backpropagate
        if logits.numel() == 0:
            print(f"Skipping batch {i+1}: Final logits tensor is empty after HNM. (This should be rare if pos_edge_index exists).")
            continue

        loss = criterion(logits.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_dynamic(model, loader, optimizer, neg_sample_ratio=1, device=None):
    """
    Final optimized version addressing class imbalance with both sampling and loss weighting.

    Args:
        model: The GNN model.
        loader: The PyG DataLoader.
        optimizer: The optimizer.
        neg_sample_ratio: The ratio of negative to positive samples.
        device: The device.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    total_loss = 0
    num_batches_processed = 0

    # The pos_weight is constant per batch since neg_sample_ratio is constant
    pos_weight = torch.tensor(neg_sample_ratio, device=device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    for batch_data in loader:
        optimizer.zero_grad()
        batch_data = batch_data.to(device)

        if batch_data.edge_index.numel() == 0 or batch_data.num_nodes == 0:
            continue

        z = model(batch_data.x, batch_data.edge_index)

        pos_links = batch_data.edge_index
        num_neg_samples = int(pos_links.size(1) * neg_sample_ratio)
        neg_links = pyg_utils.negative_sampling(
            pos_links,
            num_nodes=batch_data.num_nodes,
            num_neg_samples=num_neg_samples
        ).to(device)

        # Check for valid links after sampling
        if pos_links.numel() == 0 or neg_links.numel() == 0:
            continue

        pos_logits = model.decode(z, pos_links)
        neg_logits = model.decode(z, neg_links)

        pos_labels = torch.ones(pos_logits.size(0), device=device)
        neg_labels = torch.zeros(neg_logits.size(0), device=device)

        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])

        # Loss calculation is now handled by the weighted criterion
        loss = criterion(logits.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches_processed += 1

    return total_loss / num_batches_processed if num_batches_processed > 0 else 0.0


def evaluate_link_prediction(model, loader, criterion, device, neg_sample_ratio=1, radius_threshold=None,
                             node_lineage_map=None):
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
                raise AttributeError("The 'data' object (batch) does not have 'original_global_node_ids'. "
                                     "Ensure your create_lineage_graph function attaches this attribute "
                                     "to the Data object before putting it into the DataLoader.")

            # Ensure data.edge_index is long
            if data.edge_index.dtype != torch.long:
                data.edge_index = data.edge_index.long()

            # Handle cases where the batch has no positive edges
            if data.edge_index.numel() == 0:
                print(f"Skipping evaluation batch: No positive edges in the current batch.")
                continue

            batch_original_global_node_ids = data.original_global_node_ids.cpu().numpy()

            neg_edge_index = get_all_plausible_negative_candidates(
                data=data,  # Pass the entire Batch object
                num_neg_samples_per_pos_edge=neg_sample_ratio,
                radius_threshold=radius_threshold,
                device=device
            )

            # Ensure neg_edge_index is long
            if neg_edge_index is not None and neg_edge_index.dtype != torch.long:
                neg_edge_index = neg_edge_index.long()

            if neg_edge_index.numel() == 0:
                print(
                    "Warning: No local negative samples generated for evaluation batch. Falling back to random sampling.")
                if data.num_nodes < 2 or data.edge_index.size(1) == 0:
                    print(
                        f"Skipping evaluation batch: Not enough nodes ({data.num_nodes}) or no positive edges to sample random negatives.")
                    continue

                neg_edge_index = torch_geometric.utils.negative_sampling(
                    data.edge_index, num_nodes=data.num_nodes, num_neg_samples=data.edge_index.size(1)
                ).to(device)

                if neg_edge_index.numel() == 0:
                    print("Skipping evaluation batch: Random negative sampling also yielded no samples.")
                    continue

                if neg_edge_index.dtype != torch.long:
                    neg_edge_index = neg_edge_index.long()

            z = model(data.x, data.edge_index)

            pos_logits = model.decode(z, data.edge_index)
            neg_logits = model.decode(z, neg_edge_index)

            if pos_logits.numel() == 0 and neg_logits.numel() == 0:
                print(f"Skipping evaluation batch: Both positive and negative logits are empty after decoding.")
                continue

            pos_labels = torch.ones(pos_logits.size(0), device=device)
            neg_labels = torch.zeros(neg_logits.size(0), device=device)

            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([pos_labels, neg_labels])

            if logits.numel() == 0:
                print(f"Skipping evaluation batch: Combined logits tensor is empty. No valid edges for evaluation.")
                continue

            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches_processed_eval += 1

            all_preds_logits.append(logits.cpu())
            all_labels_agg.append(labels.cpu())

            probabilities = torch.sigmoid(logits)
            predicted_labels = (probabilities > 0.5).long()

            all_predicted_labels_individual.append(predicted_labels.cpu())
            all_probabilities_individual.append(probabilities.cpu())
            all_true_labels_individual.append(labels.cpu())

            current_evaluated_edge_indices_local = torch.cat([data.edge_index, neg_edge_index], dim=-1)

            source_global_ids = batch_original_global_node_ids[current_evaluated_edge_indices_local[0].cpu().numpy()]
            target_global_ids = batch_original_global_node_ids[current_evaluated_edge_indices_local[1].cpu().numpy()]

            # Concatenate the NumPy arrays first, then convert to a single tensor
            source_target_np = np.stack([source_global_ids, target_global_ids])
            current_evaluated_edge_indices_global = torch.from_numpy(source_target_np).long()

            all_evaluated_edge_indices_global.append(current_evaluated_edge_indices_global)

    if num_batches_processed_eval == 0:
        print("Warning: No batches were processed for evaluation. Returning default values.")
        return 0.0, 0.0, 0.0, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty((2, 0), dtype=torch.long), []

    avg_loss = total_loss / num_batches_processed_eval
    all_preds_logits_agg = torch.cat(all_preds_logits)
    all_labels_np_agg = torch.cat(all_labels_agg).numpy()
    all_preds_proba_agg = torch.sigmoid(all_preds_logits_agg).numpy()

    accuracy = accuracy_score(all_labels_np_agg, (all_preds_proba_agg > 0.5).astype(int))

    # Handle case where only one class is present in labels, which breaks roc_auc_score
    if len(np.unique(all_labels_np_agg)) < 2:
        auc_score = 0.5  # Or handle as appropriate for your context
        print("Warning: Only one class present in true labels for AUC calculation. Setting AUC to 0.5.")
    else:
        auc_score = roc_auc_score(all_labels_np_agg, all_preds_proba_agg)

    final_predicted_labels = torch.cat(all_predicted_labels_individual, dim=0)
    final_probabilities = torch.cat(all_probabilities_individual, dim=0)
    final_true_labels = torch.cat(all_true_labels_individual, dim=0)
    final_evaluated_edge_indices_global = torch.cat(all_evaluated_edge_indices_global, dim=1)

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
        if not derived_lineage_labels_final:
            derived_lineage_labels_final = ['N/A'] * final_evaluated_edge_indices_global.size(1)

    print("evaluate_link_prediction: Function about to return results.")
    return (avg_loss, accuracy, auc_score,
            final_predicted_labels, final_probabilities, final_true_labels,
            final_evaluated_edge_indices_global, derived_lineage_labels_final)


def evaluate_dynamic(model, loader, criterion, device, neg_sample_ratio=1, node_lineage_map=None):
    """
    Performs a single epoch of dynamic evaluation and returns comprehensive results.

    Args:
        model: The GNN model with a forward (encoder) and decode method.
        loader: The PyG DataLoader for validation/test graphs.
        criterion: The loss function.
        device: The device to run on ('cuda' or 'cpu').
        neg_sample_ratio: The ratio of negative to positive samples to generate.
        node_lineage_map: A dictionary mapping global node IDs to their ground truth lineage.

    Returns:
        tuple: (avg_loss, accuracy, auc_score, final_predicted_labels,
                final_probabilities, final_true_labels,
                final_evaluated_edge_indices_global, derived_lineage_labels_final)
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

            if data.edge_index.numel() == 0 or data.num_nodes == 0:
                print(f"Skipping evaluation batch: No valid edges or nodes.")
                continue

            batch_original_global_node_ids = data.original_global_node_ids.cpu().numpy()

            # --- OPTIMIZED NEGATIVE SAMPLING ---
            pos_links = data.edge_index
            num_neg_samples = int(pos_links.size(1) * neg_sample_ratio)
            neg_links = pyg_utils.negative_sampling(
                pos_links,
                num_nodes=data.num_nodes,
                num_neg_samples=num_neg_samples
            ).to(device)

            if pos_links.numel() == 0 or neg_links.numel() == 0:
                print(f"Skipping evaluation batch: Insufficient positive or negative links after sampling.")
                continue

            # GNN Encoder and Decoder
            z = model(data.x, data.edge_index)
            pos_logits = model.decode(z, pos_links)
            neg_logits = model.decode(z, neg_links)

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


def predict_cell_linkages(model, loader, device):
    model.eval() # Set the model to evaluation mode

    all_probabilities = []
    all_predicted_labels = [] # Binary
    all_predicted_edge_indices_global = [] # Store global IDs of predicted edges

    with torch.no_grad():
        for data in loader: # data is a PyG Batch object
            data = data.to(device)

            if not hasattr(data, 'original_global_node_ids'):
                raise AttributeError("The 'data' object (batch) does not have 'original_global_node_ids'. "
                                     "Ensure your data preparation for prediction attaches this attribute.")

            batch_original_global_node_ids = data.original_global_node_ids.cpu().numpy()

            # --- Core Prediction Logic ---
            # Encode node features
            z = model(data.x, data.edge_index) # data.edge_index here represents your *candidate* edges

            # Decode link logits for the candidate edges
            # You only have 'positive' (candidate) edges for prediction, no 'true' negatives
            candidate_logits = model.decode(z, data.edge_index)

            # Convert logits to probabilities
            probabilities = torch.sigmoid(candidate_logits)
            predicted_labels = (probabilities > 0.5).long() # Binary predictions based on a threshold

            # --- Map local indices back to global IDs for the *current batch's* edges ---
            # These are the edges that were actually processed for prediction
            current_predicted_edge_indices_local = data.edge_index

            source_global_ids = batch_original_global_node_ids[current_predicted_edge_indices_local[0].cpu().numpy()]
            target_global_ids = batch_original_global_node_ids[current_predicted_edge_indices_local[1].cpu().numpy()]

            current_predicted_edge_indices_global = torch.tensor(
                [source_global_ids, target_global_ids], dtype=torch.long
            )

            # Accumulate results
            all_probabilities.append(probabilities.cpu())
            all_predicted_labels.append(predicted_labels.cpu())
            all_predicted_edge_indices_global.append(current_predicted_edge_indices_global)

    # Concatenate all results from batches
    final_probabilities = torch.cat(all_probabilities, dim=0)
    final_predicted_labels = torch.cat(all_predicted_labels, dim=0)
    final_predicted_edge_indices_global = torch.cat(all_predicted_edge_indices_global, dim=1) # dim=1 for [2, num_edges]

    print("predict_cell_linkages: Function about to return results.")
    return final_predicted_labels, final_probabilities, final_predicted_edge_indices_global


class LineageLinkPredictionGNN_dp(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.decode_layer = nn.Linear(2 * hidden_channels, 1)
        self.dropout = nn.Dropout(p=dropout_rate) # Add a dropout layer

    def forward(self, x, edge_index):
        # GNN Encoder
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)  # Apply dropout after activation
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index).relu()
        x = self.dropout(x)
        return x

    def decode(self, z, edge_index):
        # Decoder
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        h = self.decode_layer(h)
        return h

class LineageLinkPredictionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LineageLinkPredictionGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Decoder/multilayer perceptron for link prediction:
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output a single logit for binary classification
        )

    def forward(self, x, edge_index):
        # GNN Encoder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index) # x are now node embeddings

        return x # Return node embeddings

    def decode(self, z, edge_index, neg_edge_index=None):  # Added neg_edge_index for clarity based on usage
        # --- THIS IS THE CRITICAL SECTION ---
        # Ensure all edge indices are long tensors
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        if neg_edge_index is not None and neg_edge_index.dtype != torch.long:
            neg_edge_index = neg_edge_index.long()

        all_edge_indices = torch.cat([edge_index, neg_edge_index], dim=-1) if neg_edge_index is not None else edge_index

        # The error occurs here because all_edge_indices[0] is not a long tensor
        source_embed = z[all_edge_indices[0]]
        target_embed = z[all_edge_indices[1]]

        edge_features = torch.cat([source_embed, target_embed], dim=-1)
        logits = self.decoder(edge_features)
        return logits.squeeze(-1)  # Squeeze to get a 1D tensor of logits

class StandardScalerTransform(BaseTransform):
    def __init__(self, scaler):
        super().__init__()
        self.scaler = scaler

    def __call__(self, data):
        if hasattr(data, 'x') and data.x is not None:
            # Convert to numpy, transform, and explicitly cast to float32
            x_np = data.x.cpu().numpy()
            x_scaled_np = self.scaler.transform(x_np).astype(np.float32)

            # Create the tensor with the correct data type and device
            data.x = torch.from_numpy(x_scaled_np).to(data.x.device)
        return data


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
