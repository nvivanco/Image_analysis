import pandas as pd
import networkx as nx
from collections import deque


def consolidate_lineages_to_node_df(df_consolidated_lineages, df_for_training):
	"""
	Transforms a parent-daughter lineage DataFrame into a node-centric DataFrame
	in the same format as the original cell data.

	Args:
		df_consolidated_lineages (pd.DataFrame): The lineage DataFrame with parent-daughter pairs.
		df_for_training (pd.DataFrame): The original cell data with all attributes.

	Returns:
		pd.DataFrame: A new DataFrame in the original format, with a new 'predicted_lineage' column.
	"""
	if df_consolidated_lineages.empty:
		print("Input lineage DataFrame is empty. Returning original DataFrame with 'predicted_lineage' set to None.")
		df_for_training['predicted_lineage'] = None
		return df_for_training

	print("--- Consolidating lineage data to a node-centric format... ---")

	# Step 1: Prepare data for parent nodes
	parent_cols = [col for col in df_consolidated_lineages.columns if
				   col.startswith('Parent_') or col in ['Lineage_ID']]
	df_parents = df_consolidated_lineages[parent_cols].copy()
	df_parents.rename(columns={'Parent_Node': 'node_id', 'Lineage_ID': 'predicted_lineage'}, inplace=True)
	df_parents.rename(
		columns={col: col.replace('Parent_', '') for col in df_parents.columns if col.startswith('Parent_')},
		inplace=True)
	df_parents['is_parent'] = True
	df_parents['is_daughter'] = False

	# Step 2: Prepare data for daughter nodes
	daughter_cols = [col for col in df_consolidated_lineages.columns if
					 col.startswith('Daughter_') or col in ['Lineage_ID']]
	df_daughters = df_consolidated_lineages[daughter_cols].copy()
	df_daughters.rename(columns={'Daughter_Node': 'node_id', 'Lineage_ID': 'predicted_lineage'}, inplace=True)
	df_daughters.rename(
		columns={col: col.replace('Daughter_', '') for col in df_daughters.columns if col.startswith('Daughter_')},
		inplace=True)
	df_daughters['is_parent'] = False
	df_daughters['is_daughter'] = True

	# Step 3: Combine parent and daughter data.
	df_combined = pd.concat([df_parents, df_daughters], ignore_index=True)

	# Use a groupby to handle cases where a node is both a parent and a daughter
	# We want to keep the lineage_id and set the flags correctly
	agg_dict = {
		'predicted_lineage': 'first',  # The lineage ID should be the same
		'is_parent': 'any',
		'is_daughter': 'any'
	}
	df_lineage_info = df_combined.groupby('node_id').agg(agg_dict).reset_index()

	# Step 4: Merge the lineage information back to the original DataFrame
	# This is the most crucial step to get the final result
	final_df = df_for_training.merge(
		df_lineage_info[['node_id', 'predicted_lineage']],
		on='node_id',
		how='right'
	)

	# Handle nodes that were not part of any predicted lineage
	final_df['predicted_lineage'] = final_df['predicted_lineage'].fillna('No Predicted Link')

	print("--- Consolidation complete. Final DataFrame is ready for plotting. ---")
	return final_df

def process_all_fovs(test_lineage_predictions_df, df_for_training, prob_threshold=0.8):
	"""
	Processes all FOVs from a prediction DataFrame, builds cleaned graphs with
	experimental info, and generates a single consolidated lineage DataFrame.

	Args:
		test_lineage_predictions_df (pd.DataFrame): DataFrame with initial model predictions.
		df_for_training (pd.DataFrame): The original cell data with experimental info.
		prob_threshold (float): The minimum predicted probability to consider an edge.

	Returns:
		pd.DataFrame: A single DataFrame with all lineages and experimental data from all FOVs.
	"""
	# 1. Prepare cell data for quick lookup
	# Start with the list of required columns
	required_cols = ['experiment_name', 'FOV', 'trench_id', 'node_id', 'gene']

	# Check if 'ground_truth_lineage' column is present and add it to the list if so
	if 'ground_truth_lineage' in df_for_training.columns:
		required_cols.append('ground_truth_lineage')
		print("Column 'ground_truth_lineage' found. Including it in node attributes.")
	else:
		print("Column 'ground_truth_lineage' not found. Omitting it.")

	# Select the columns based on the updated list
	node_exp_info = df_for_training[required_cols].copy()

	# 2. Merge predictions with experimental info
	df_merged = test_lineage_predictions_df.merge(
		node_exp_info,
		left_on='Source_Node',
		right_on='node_id',
		how='left'
	)

	# 3. Identify unique FOVs
	unique_fovs = df_merged[['experiment_name', 'FOV', 'trench_id']].drop_duplicates().to_records(index=False)

	final_lineage_dfs = []

	for i, fov_info in enumerate(unique_fovs):
		exp, fov, trench = fov_info
		print(f"--- Processing FOV {i + 1}/{len(unique_fovs)}: {exp}, {fov}, {trench} ---")

		# 4. Filter predictions for the current FOV and probability threshold
		df_filtered_predictions = df_merged[
			(df_merged['experiment_name'] == exp) &
			(df_merged['FOV'] == fov) &
			(df_merged['trench_id'] == trench) &
			(df_merged['Predicted_Probability'] >= prob_threshold)
			].copy()

		# 5. Get the subset of cell data for the current FOV to build the graph nodes
		nodes_in_fov = set(df_filtered_predictions['Source_Node']).union(
			set(df_filtered_predictions['Destination_Node']))
		df_fov_cells = node_exp_info[node_exp_info['node_id'].isin(nodes_in_fov)].copy()

		if df_filtered_predictions.empty or df_fov_cells.empty:
			print("No valid edges or nodes found. Skipping.")
			continue

		# 6. Build the cleaned graph with all experimental attributes
		try:
			G = build_cleaned_lineage_graph_with_attributes(df_filtered_predictions, df_fov_cells)

			# 7. Find lineages from the cleaned graph, including all attributes
			lineage_df = find_lineage_df_with_attributes(G)

			if not lineage_df.empty:
				lineage_df['experiment_name'] = exp
				lineage_df['FOV'] = fov
				lineage_df['trench_id'] = trench
				final_lineage_dfs.append(lineage_df)
				print(f"Successfully generated {len(lineage_df)} lineage rows.")
			else:
				print("No lineages found in the cleaned graph.")

		except Exception as e:
			print(f"An error occurred while processing FOV {fov_info}: {e}")
			continue

	if not final_lineage_dfs:
		print("No lineages were processed successfully across all FOVs.")
		return pd.DataFrame()

	# 8. Concatenate all lineage DataFrames into one
	df_consolidated_lineages = pd.concat(final_lineage_dfs, ignore_index=True)
	return df_consolidated_lineages


def build_cleaned_lineage_graph_with_attributes(df_predictions, df_cells):
	"""
	Builds a lineage graph with node attributes from a DataFrame and removes cycles.

	Args:
		df_predictions (pd.DataFrame): DataFrame with 'Source_Node', 'Destination_Node',
									   and 'Predicted_Probability'.
		df_cells (pd.DataFrame): DataFrame with all cell data, including experimental info
								 and 'node_id', to be added as node attributes.

	Returns:
		nx.DiGraph: The final, cleaned, and cycle-free graph with all attributes.
	"""
	G = nx.DiGraph()

	# Step 1: Add nodes with all cell-level attributes
	print("--- Adding nodes with all experimental data... ---")
	node_data = df_cells.set_index('node_id').to_dict('index')
	for node_id, attributes in node_data.items():
		G.add_node(node_id, **attributes)

	# Step 2: Add edges with prediction attributes
	print("--- Building initial graph with edges... ---")
	for _, row in df_predictions.iterrows():
		source = row['Source_Node']
		dest = row['Destination_Node']
		prob = row['Predicted_Probability']
		# Add a conditional check to ensure nodes exist
		if G.has_node(source) and G.has_node(dest):
			G.add_edge(source, dest, prob=prob)

	print(f"Initial graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

	# Your existing cycle removal logic remains the same
	while True:
		try:
			cycle = next(nx.simple_cycles(G))
			weakest_edge = None
			min_prob = float('inf')

			for i in range(len(cycle)):
				u, v = cycle[i], cycle[(i + 1) % len(cycle)]
				if G.has_edge(u, v):
					current_prob = G.edges[u, v]['prob']
					if current_prob < min_prob:
						min_prob = current_prob
						weakest_edge = (u, v)

			if weakest_edge:
				G.remove_edge(*weakest_edge)
		except StopIteration:
			break

	print("\nAll cycles have been removed.")
	print(f"Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
	return G



def find_lineage_df_with_attributes(graph):
	"""
	Traverses a graph to identify lineages and includes node attributes in the output DataFrame.

	Args:
		graph (nx.DiGraph): The input graph with node and edge attributes.

	Returns:
		pd.DataFrame: DataFrame with lineage relationships and node attributes.
	"""
	parent_daughter_rows = []
	lineage_id_counter = 1
	root_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]

	if not root_nodes:
		return pd.DataFrame()

	stack = deque()
	processed_edges = set()
	for i, root in enumerate(root_nodes):
		stack.append((root, str(i + 1)))

	while stack:
		current_node, current_lineage_id = stack.pop()
		neighbors = list(graph.neighbors(current_node))
		num_neighbors = len(neighbors)
		relationship_type = 'Division' if num_neighbors > 1 else 'Continuation'

		# Retrieve the node's experimental attributes here
		current_node_attributes = graph.nodes[current_node]

		for i, neighbor in enumerate(neighbors):
			edge = (current_node, neighbor)
			if edge in processed_edges:
				continue
			processed_edges.add(edge)

			new_lineage_id = f"{current_lineage_id}.{i + 1}" if relationship_type == 'Division' else current_lineage_id

			# Retrieve the edge's attributes
			edge_attributes = graph.edges[edge]

			# Retrieve the daughter node's attributes
			daughter_node_attributes = graph.nodes[neighbor]

			row = {
				'Parent_Node': current_node,
				'Daughter_Node': neighbor,
				'Relationship_Type': relationship_type,
				'Lineage_ID': new_lineage_id,
				'Prediction_Probability': edge_attributes.get('prob', None)  # Get prob from edge attr
			}
			# Add all parent and daughter node attributes with prefixes
			for key, value in current_node_attributes.items():
				row[f'Parent_{key}'] = value
			for key, value in daughter_node_attributes.items():
				row[f'Daughter_{key}'] = value

			parent_daughter_rows.append(row)
			stack.append((neighbor, new_lineage_id))

	return pd.DataFrame(parent_daughter_rows)