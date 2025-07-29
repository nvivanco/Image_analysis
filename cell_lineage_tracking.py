import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import deque
import numpy as np

def find_lineage_branches_optimized_v3(graph):
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

def create_lineage_graph(df_lineage, node_feature_cols, device='cpu'):
    # The function expects a sub-DataFrame already filtered for a specific lineage branch
    original_global_node_ids = df_lineage['node_id'].values
    global_id_to_local_idx = {global_id: i for i, global_id in enumerate(original_global_node_ids)}

    x = torch.tensor(df_lineage[node_feature_cols].values, dtype=torch.float).to(device)
    y = torch.tensor(df_lineage['numeric_lineage'].values, dtype=torch.long).to(device)
    pos = torch.tensor(df_lineage['centroid_y'].values, dtype=torch.float).to(device)
    node_time_frames = torch.tensor(df_lineage['time_frame'].values, dtype=torch.long).to(device)


    num_nodes = len(df_lineage)
    if num_nodes == 0:
        return None

    source_nodes_local_idx = []
    target_nodes_local_idx = []

    sorted_time_frames = sorted(df_lineage['time_frame'].unique())

    for i in range(len(sorted_time_frames) - 1):
        current_t = sorted_time_frames[i]
        next_t = sorted_time_frames[i+1]

        df_current_t = df_lineage[df_lineage['time_frame'] == current_t]
        df_next_t = df_lineage[df_lineage['time_frame'] == next_t]

        current_lineage_to_node = df_current_t.set_index('ground_truth_lineage')['node_id'].to_dict()
        next_lineage_to_node = df_next_t.set_index('ground_truth_lineage')['node_id'].to_dict()

        for idx, row in df_current_t.iterrows():
            current_global_node_id = row['node_id']
            current_ground_truth_lineage = row['ground_truth_lineage']

            if current_ground_truth_lineage in next_lineage_to_node:
                next_global_node_id = next_lineage_to_node[current_ground_truth_lineage]
                source_nodes_local_idx.append(global_id_to_local_idx[current_global_node_id])
                target_nodes_local_idx.append(global_id_to_local_idx[next_global_node_id])

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

    data = Data(x=x,
                edge_index=edge_index,
                y=y,
                pos=pos,
                num_nodes=num_nodes,
                time_frame=node_time_frames,
                original_global_node_ids=torch.tensor(original_global_node_ids, dtype=torch.long),
                root_lineage_branch=df_lineage['ground_truth_lineage'].iloc[0], # The GTL that defines this subgraph
                start_time_frame=df_lineage['time_frame'].min(),
                experiment_name=df_lineage['experiment_name'].iloc[0],
                fov=df_lineage['FOV'].iloc[0],
                trench_id=df_lineage['trench_id'].iloc[0]
               )
    return data

def generate_local_temporal_negative_samples(data: Data, num_neg_samples_per_pos_edge: float, radius_threshold: float, device='cpu'):
    """
    Generates negative samples by considering only cells in consecutive time frames
    and within a certain spatial radius of potential source nodes, excluding true positives.

    Args:
        data (torch_geometric.data.Data): A single graph batch containing x, edge_index, pos, time_frame.
        num_neg_samples_per_pos_edge (float): Ratio of negative samples to positive samples.
                                                e.g., 1.0 for 1:1, 2.0 for 2:1.
        radius_threshold (float): Maximum spatial distance for a potential negative connection.
        device (str): Device to put tensors on.

    Returns:
        torch.Tensor: edge_index of sampled negative connections, shape [2, num_neg_samples].
    """
    if data.edge_index.numel() == 0: # No positive edges, no negative samples possible this way
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # Convert tensors to CPU for easier numpy/list processing if needed, then back to device
    pos_coords = data.pos.cpu().numpy() # Assuming pos is [num_nodes, 2] (y,x) or [num_nodes, 1] (y)
    time_frames = data.time_frame.cpu().numpy()
    num_nodes = data.num_nodes
    existing_edges = set(tuple(e) for e in data.edge_index.cpu().T.tolist()) # Convert to set for fast lookup

    potential_neg_samples = []

    # Iterate through all possible source nodes
    for i in range(num_nodes):
        current_node_time = time_frames[i]
        current_node_pos = pos_coords[i]

        # Iterate through all possible target nodes (j)
        for j in range(num_nodes):
            # 1. Temporal Constraint: Only consider next time frame
            if time_frames[j] != current_node_time + 1:
                continue

            # 2. Local Constraint: Check spatial proximity (Euclidean distance)
            target_node_pos = pos_coords[j]
            # Adjust distance calculation based on your 'pos' dimension
            if pos_coords.ndim == 1: # If 'pos' is just centroid_y (1D)
                distance = np.abs(current_node_pos - target_node_pos)
            else: # If 'pos' is (y, x) or (x, y) etc. (2D or more)
                distance = np.linalg.norm(current_node_pos - target_node_pos)

            if distance > radius_threshold:
                continue

            # 3. Exclude existing positive edges
            if (i, j) not in existing_edges:
                potential_neg_samples.append((i, j))

    # Convert to tensor
    if not potential_neg_samples:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    potential_neg_samples_tensor = torch.tensor(potential_neg_samples, dtype=torch.long).T.to(device)

    # Sample a subset of potential negative samples based on ratio
    num_positive_edges = data.edge_index.size(1)
    desired_neg_samples = int(num_positive_edges * num_neg_samples_per_pos_edge)

    if desired_neg_samples >= potential_neg_samples_tensor.size(1):
        # If not enough potential negatives, take all of them
        return potential_neg_samples_tensor
    else:
        # Randomly sample the desired number of negative edges
        indices = torch.randperm(potential_neg_samples_tensor.size(1), device=device)[:desired_neg_samples]
        return potential_neg_samples_tensor[:, indices]


def train_link_prediction(model, train_loader, optimizer, criterion, device, neg_sample_ratio=3.0,
                          radius_threshold=None):
    model.train()
    total_loss = 0
    num_batches_processed = 0  # Track processed batches for accurate average loss

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Ensure data.edge_index is long
        if data.edge_index.dtype != torch.long:
            data.edge_index = data.edge_index.long()

        # Handle cases where the batch has no positive edges (no ground truth for training)
        if data.edge_index.numel() == 0:
            print(f"Skipping batch: No positive edges in the current batch for training.")
            continue  # Skip to the next batch

        # 1. Generate negative edges
        neg_edge_index = generate_local_temporal_negative_samples(
            data,
            num_neg_samples_per_pos_edge=neg_sample_ratio,  # How many neg samples per positive edge
            radius_threshold=radius_threshold,  # Spatial threshold (e.g., 50.0 units)
            device=device
        )

        # Ensure neg_edge_index is long
        if neg_edge_index is not None and neg_edge_index.dtype != torch.long:
            neg_edge_index = neg_edge_index.long()

        # Fallback if no local negatives are found or if initial neg_edge_index is empty
        if neg_edge_index.numel() == 0:
            print("Warning: No local negative samples generated for a batch. Falling back to random sampling.")
            # Ensure num_nodes is greater than 1 for negative_sampling to work
            if data.num_nodes < 2 or data.edge_index.size(1) == 0:
                print(
                    f"Skipping batch: Not enough nodes ({data.num_nodes}) or no positive edges to sample random negatives.")
                continue

            neg_edge_index = torch_geometric.utils.negative_sampling(
                data.edge_index, num_nodes=data.num_nodes, num_neg_samples=data.edge_index.size(1)).to(device)

            if neg_edge_index.numel() == 0:  # Still empty after fallback? Skip.
                print("Skipping batch: Random negative sampling also yielded no samples.")
                continue

            # Re-ensure the fallback also produces long tensors
            if neg_edge_index.dtype != torch.long:
                neg_edge_index = neg_edge_index.long()

        # Now that we are sure both pos and neg edges exist, proceed
        # 2. Get node embeddings from GNN encoder
        z = model(data.x,
                  data.edge_index)  # This might also fail if data.x is empty for some reason, but less likely for now

        # 3. Decode edges (both positive and negative)
        pos_logits = model.decode(z, data.edge_index)  # Logits for true edges
        neg_logits = model.decode(z, neg_edge_index)  # Logits for sampled negative edges

        # Ensure pos_logits and neg_logits are not empty before creating labels
        if pos_logits.numel() == 0 and neg_logits.numel() == 0:
            print(f"Skipping batch: Both positive and negative logits are empty after decoding.")
            continue

        # 4. Create labels: 1 for positive edges, 0 for negative edges
        pos_labels = torch.ones(pos_logits.size(0), device=device)
        neg_labels = torch.zeros(neg_logits.size(0), device=device)

        # 5. Concatenate logits and labels
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([pos_labels, neg_labels])

        # Ensure combined logits/labels are not empty before calculating loss
        if logits.numel() == 0:
            print(f"Skipping batch: Combined logits tensor is empty. This batch has no valid edges to train on.")
            continue

        # 6. Calculate loss
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches_processed += 1  # Increment only for successfully processed batches

    if num_batches_processed > 0:
        avg_loss = total_loss / num_batches_processed
    else:
        avg_loss = 0.0  # Or raise an error if no batches were processed at all

    return avg_loss


def evaluate_link_prediction(model, loader, criterion, device, neg_sample_ratio=1.0, radius_threshold=None,
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

            neg_edge_index = generate_local_temporal_negative_samples(
                data,
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

            current_evaluated_edge_indices_global = torch.tensor(
                [source_global_ids, target_global_ids], dtype=torch.long
            )
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


def identify_sub_lineage_roots(df):
    # Ensure relevant columns are present
    required_cols = ['experiment_name', 'FOV', 'trench_id', 'ground_truth_lineage', 'time_frame', 'node_id']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain all required columns: {required_cols}")
    df_temp_sorted = df.sort_values(by=['time_frame', 'node_id'])
    first_appearances = df_temp_sorted.drop_duplicates(
        subset=['experiment_name', 'FOV', 'trench_id', 'ground_truth_lineage'],
        keep='first'
    )

    # Extract the necessary information for each root
    # Convert to list of tuples as in the original function's output format
    sub_lineage_roots = list(first_appearances[[
        'experiment_name',
        'FOV',
        'trench_id',
        'ground_truth_lineage',
        'time_frame'
    ]].itertuples(index=False, name=None))

    return sub_lineage_roots

def identify_exp_fov(df):
    # Ensure relevant columns are present
    required_cols = ['experiment_name', 'FOV', 'trench_id', 'time_frame', 'node_id']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain all required columns: {required_cols}")
    df_sorted = df.sort_values(by=['time_frame', 'node_id'], ascending=True)
    unique_contexts = df_sorted.drop_duplicates(
        subset=['experiment_name', 'FOV', 'trench_id'],
        keep='first')

    # Extract the necessary information for each fov
    exp_fov_info = list(unique_contexts[[
        'experiment_name',
        'FOV',
        'trench_id',
        'time_frame']].itertuples(index=False, name=None))

    return exp_fov_info


class LineageDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


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



