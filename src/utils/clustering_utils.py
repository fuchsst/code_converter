# src/utils/clustering_utils.py
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from src.logger_setup import get_logger
import itertools
import time # For potential timing/debugging
from collections import defaultdict

logger = get_logger(__name__)

def _calculate_cluster_tokens(cluster: Set[str], node_weights: Dict[str, int]) -> int:
    """Calculates the total token count for a cluster."""
    return sum(node_weights.get(file, 0) for file in cluster)

def _calculate_jaccard_score(set1: Set[str], set2: Set[str]) -> float:
    """Calculates the Jaccard index (overlap score) between two sets."""
    if not set1 and not set2:
        return 1.0 # Define score for two empty sets? Or 0.0? Let's say 1.0 if both empty.
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    return intersection_size / union_size if union_size > 0 else 0.0

def _get_cluster_dependencies(cluster: Set[str], graph: nx.DiGraph, dependency_cache: Dict[str, Set[str]]) -> Set[str]:
    """Gets the set of all unique direct dependencies for all files in a cluster using a cache."""
    all_deps = set()
    for file_node in cluster:
        if file_node in dependency_cache:
            all_deps.update(dependency_cache[file_node])
    return all_deps

def cluster_files_by_dependency(
    graph: nx.DiGraph,
    node_weights: Dict[str, int],
    max_package_tokens: int
) -> Dict[int, List[str]]:
    """
    Clusters files by iteratively merging pairs with the highest shared dependency
    score, respecting a token limit. Then assigns remaining single-file "orphan"
    clusters to all larger clusters that import them, if token limits allow (the single file cluster are removed if they could be added to another cluster).

    Args:
        graph: The dependency graph (DiGraph where edge A -> B means A includes B).
        node_weights: Dictionary mapping file paths to their token counts.
        max_package_tokens: The maximum allowed token size for merging clusters.

    Returns:
        A dictionary mapping cluster index (int) to a sorted list of file paths (str).
    """
    logger.info("Starting unified iterative dependency clustering...")
    all_files = list(graph.nodes())
    if not all_files:
        logger.warning("Input graph has no nodes. Returning empty clustering.")
        return {}

    # --- Step 1: Initialization & Precomputation ---
    clusters: List[Set[str]] = [{file} for file in all_files]
    logger.info(f"Step 1: Initialized {len(clusters)} single-file clusters.")

    logger.info("Precomputing dependencies for all files...")
    dependency_cache: Dict[str, Set[str]] = {}
    for file_node in all_files:
         if file_node in graph:
              dependency_cache[file_node] = set(graph.successors(file_node))
         else:
              dependency_cache[file_node] = set()
    logger.info("Finished precomputing dependencies.")

    cluster_details = {} # Store details keyed by original index for stable merging
    for i, cluster in enumerate(clusters):
        cluster_deps = _get_cluster_dependencies(cluster, graph, dependency_cache)
        cluster_tokens = _calculate_cluster_tokens(cluster, node_weights)
        cluster_details[i] = {"deps": cluster_deps, "tokens": cluster_tokens, "cluster_set": cluster}

    # --- Step 2: Iterative Cluster Merging (Based on Dependency Overlap & Token Limit) ---
    logger.info("Step 2: Starting iterative cluster merging based on dependency overlap and token limit...")
    merged_in_iteration = True
    iteration_count = 0
    max_iterations = len(clusters) * 2 # Heuristic safety break

    active_indices = list(cluster_details.keys()) # Track indices of active clusters

    while merged_in_iteration and iteration_count < max_iterations and len(active_indices) > 1:
        merged_in_iteration = False
        iteration_count += 1
        start_time = time.time()
        logger.debug(f"Merge iteration {iteration_count} starting with {len(active_indices)} clusters.")

        best_merge_candidate: Optional[Tuple[float, int, int]] = None # (score, index1, index2)
        candidate_pairs_count = 0

        # Find the best pair of *active* clusters to merge
        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                idx1 = active_indices[i]
                idx2 = active_indices[j]

                details1 = cluster_details[idx1]
                details2 = cluster_details[idx2]
                tokens1 = details1["tokens"]
                tokens2 = details2["tokens"]

                if tokens1 + tokens2 > max_package_tokens:
                    continue

                candidate_pairs_count += 1
                deps1 = details1["deps"]
                deps2 = details2["deps"]
                score = _calculate_jaccard_score(deps1, deps2)

                if score > 0:
                    if best_merge_candidate is None or score > best_merge_candidate[0]:
                        best_merge_candidate = (score, idx1, idx2)

        if best_merge_candidate is not None:
            score, idx1, idx2 = best_merge_candidate

            cluster1_set = cluster_details[idx1]["cluster_set"]
            cluster2_set = cluster_details[idx2]["cluster_set"]
            merged_cluster_set = cluster1_set.union(cluster2_set)
            merged_tokens = _calculate_cluster_tokens(merged_cluster_set, node_weights)
            merged_deps = _get_cluster_dependencies(merged_cluster_set, graph, dependency_cache)

            logger.debug(f"  Merging cluster {idx1} ({len(cluster1_set)} files) and cluster {idx2} ({len(cluster2_set)} files). Score: {score:.4f}. New size: {len(merged_cluster_set)} files, {merged_tokens} tokens.")

            # Add the new merged cluster details (using a new index perhaps, or reusing one?)
            # Let's reuse idx1 and mark idx2 as inactive.
            cluster_details[idx1] = {"deps": merged_deps, "tokens": merged_tokens, "cluster_set": merged_cluster_set}
            # Remove idx2 from active list
            active_indices.remove(idx2)
            # Mark idx2 details as inactive (optional, helps debugging)
            # cluster_details[idx2]["active"] = False
            merged_in_iteration = True
        else:
            logger.debug("No suitable merge candidates found in this iteration.")

        end_time = time.time()
        logger.debug(f"Merge iteration {iteration_count} finished in {end_time - start_time:.2f}s. Found {candidate_pairs_count} potential pairs. Clusters remaining: {len(active_indices)}")

    # Extract final clusters from the details dict based on active indices
    merged_clusters = [cluster_details[idx]["cluster_set"] for idx in active_indices]

    if iteration_count == max_iterations:
        logger.warning(f"Merging stopped after reaching max iterations ({max_iterations}).")
    else:
         logger.info(f"Primary merging finished after {iteration_count} iterations, resulting in {len(merged_clusters)} clusters.")

    # --- Step 3: Assign Orphan Files ---
    logger.info("Step 3: Assigning orphan files to importing clusters...")
    multi_file_clusters = [c for c in merged_clusters if len(c) > 1]
    orphan_clusters = [c for c in merged_clusters if len(c) == 1]
    orphans = {list(o)[0] for o in orphan_clusters} # Set of orphan file paths

    if not orphans:
        logger.info("No orphan files found to assign.")
        final_clusters = multi_file_clusters # All clusters are multi-file
    else:
        logger.info(f"Found {len(orphans)} orphan files: {orphans}")
        # Build reverse dependency map (importers)
        importer_map = defaultdict(set)
        for u, v in graph.edges(): # u -> v (u imports v)
            importer_map[v].add(u)

        # Keep track of which orphans were successfully added to *any* cluster
        successfully_assigned_orphans = set()
        # Work with mutable copies of the multi-file cluster sets
        final_cluster_sets = [set(c) for c in multi_file_clusters]

        logger.debug(f"Attempting to assign {len(orphans)} orphans...")
        for orphan in list(orphans): # Iterate over a copy of the orphan set
            orphan_token = node_weights.get(orphan, 0)
            importers = importer_map.get(orphan, set())
            if not importers:
                logger.debug(f"Orphan '{orphan}' has no importers in the graph. Skipping assignment.")
                continue

            orphan_was_added = False # Track if this specific orphan was added anywhere
            # Check against each *current* multi-file cluster set
            for i in range(len(final_cluster_sets)):
                cluster_set = final_cluster_sets[i]
                # Check if any file in the cluster imports the orphan
                if any(importer in cluster_set for importer in importers):
                    cluster_tokens = _calculate_cluster_tokens(cluster_set, node_weights)
                    # Check token limit *before* adding
                    if cluster_tokens + orphan_token <= max_package_tokens:
                        logger.debug(f"  Adding orphan '{orphan}' ({orphan_token} tokens) to cluster {i} ({cluster_tokens} tokens) which imports it.")
                        cluster_set.add(orphan) # Modify the set in the list
                        orphan_was_added = True
                        # Note: Orphan might be added to multiple clusters if conditions met.
                        # We only need to know it was added *at least once*.
                    else:
                        logger.debug(f"  Skipping addition of orphan '{orphan}' to cluster {i}: exceeds token limit ({cluster_tokens + orphan_token} > {max_package_tokens}).")

            # If the orphan was added to at least one cluster, mark it as successfully assigned
            if orphan_was_added:
                successfully_assigned_orphans.add(orphan)
            else:
                 logger.debug(f"Orphan '{orphan}' could not be added to any importing cluster (due to token limits or no importing clusters).")

        # Determine which orphans remain unassigned
        unassigned_orphans = orphans - successfully_assigned_orphans

        # Final list includes the potentially modified multi-file clusters
        # and only the *unassigned* orphans as single-file clusters.
        final_clusters = final_cluster_sets + [{o} for o in unassigned_orphans]

        if unassigned_orphans:
             logger.warning(f"{len(unassigned_orphans)} orphan files will remain in their own clusters: {unassigned_orphans}")
        else:
             logger.info("All orphan files were successfully assigned to importing clusters.")

    logger.info(f"Orphan assignment complete. Total final clusters: {len(final_clusters)}")

    # --- Final Formatting ---
    final_packages: Dict[int, List[str]] = {}
    # Sort final clusters by size (optional, but can be nice)
    final_clusters.sort(key=len, reverse=True)
    for idx, cluster_set in enumerate(final_clusters):
        if cluster_set: # Avoid adding empty sets if something went wrong
            final_packages[idx] = sorted(list(cluster_set))

    logger.info(f"Clustering complete. Generated {len(final_packages)} final packages.")
    # Log final package sizes for verification
    for idx, files in final_packages.items():
         pkg_tokens = _calculate_cluster_tokens(set(files), node_weights)
         logger.debug(f"  - Package {idx+1}: {len(files)} files, {pkg_tokens} tokens")

    return final_packages
