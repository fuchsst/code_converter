# src/partitioning/balance_refiner.py
import networkx as nx
from typing import Dict, List, Any, Tuple, Set
import random
import math
import statistics
from collections import defaultdict

from src.logger_setup import get_logger
import src.config as config # For default thresholds

logger = get_logger(__name__)

def calculate_partition_stats(partitions: Dict[int, List[str]],
                              node_weights: Dict[str, int]) -> Dict[int, Dict[str, Any]]:
    """Calculates token counts and other stats for each partition."""
    stats = {}
    for part_id, nodes in partitions.items():
        total_tokens = sum(node_weights.get(node, 0) for node in nodes)
        stats[part_id] = {
            "nodes": nodes,
            "node_count": len(nodes),
            "total_tokens": total_tokens
        }
    return stats

def get_boundary_nodes(graph: nx.DiGraph, partition_id: int, partitions: Dict[int, List[str]]) -> Set[str]:
    """Finds nodes in a partition that have neighbors in other partitions."""
    boundary = set()
    nodes_in_partition = set(partitions.get(partition_id, []))
    if not nodes_in_partition:
        return boundary

    node_to_partition = {node: pid for pid, nodes in partitions.items() for node in nodes}

    for node in nodes_in_partition:
        # Check both predecessors and successors for neighbors in other partitions
        neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
        for neighbor in neighbors:
            if neighbor in node_to_partition and node_to_partition[neighbor] != partition_id:
                boundary.add(node)
                break # Found one neighbor outside, node is boundary
    return boundary


def refine_partitions_for_balance(
    initial_partitions: Dict[int, List[str]],
    node_weights: Dict[str, int],
    graph: nx.DiGraph,
    config: Dict[str, Any]
) -> Dict[int, List[str]]:
    """
    Attempts to refine partitions generated by an initial clustering (like Louvain)
    to improve balance based on node weights (token counts).

    Args:
        initial_partitions (Dict[int, List[str]]): The initial partitioning (community_id -> [nodes]).
        node_weights (Dict[str, int]): Mapping of node names to their token counts.
        graph (nx.DiGraph): The dependency graph, used to check adjacency for moves.
        config (Dict[str, Any]): Configuration dictionary, containing parameters like:
            - 'TARGET_PACKAGE_SIZE_TOKENS': Ideal average size (optional, calculated if not given).
            - 'MAX_PACKAGE_SIZE_TOKENS': Strict upper limit for a package.
            - 'MIN_PACKAGE_SIZE_FILES': Minimum number of files allowed in a package.
            - 'BALANCE_MAX_ITERATIONS': Max refinement iterations.
            - 'BALANCE_MOVE_STRATEGY': 'random' or 'best_fit' (heuristic).
            - 'BALANCE_RANDOM_STATE': Seed for random moves.

    Returns:
        Dict[int, List[str]]: The refined partitioning, aiming for better balance.
    """
    logger.info("Starting partition refinement for token balance...")
    if not initial_partitions:
        logger.warning("Initial partitions are empty, skipping refinement.")
        return {}

    max_iterations = config.get('BALANCE_MAX_ITERATIONS', 100)
    max_package_tokens = config.get('MAX_PACKAGE_SIZE_TOKENS', config.MAX_PACKAGE_SIZE_TOKENS)
    min_package_files = config.get('MIN_PACKAGE_SIZE_FILES', config.MIN_PACKAGE_SIZE_FILES)
    move_strategy = config.get('BALANCE_MOVE_STRATEGY', 'best_fit') # 'random' or 'best_fit'
    random_seed = config.get('BALANCE_RANDOM_STATE', None)
    if random_seed is not None:
        random.seed(random_seed)

    current_partitions = {pid: list(nodes) for pid, nodes in initial_partitions.items()} # Deep copy

    # Calculate target size (can be average or based on max limit)
    total_nodes = sum(len(nodes) for nodes in current_partitions.values())
    total_tokens_all = sum(node_weights.values())
    num_partitions = len(current_partitions)
    target_package_size = config.get('TARGET_PACKAGE_SIZE_TOKENS')
    if target_package_size is None:
        target_package_size = total_tokens_all / num_partitions if num_partitions > 0 else 0
    logger.info(f"Refinement target: {num_partitions} partitions. Target avg tokens/package: ~{target_package_size:.0f}. Max tokens/package: {max_package_tokens}.")

    for iteration in range(max_iterations):
        logger.debug(f"Balance refinement iteration {iteration + 1}/{max_iterations}")
        moved_node = False
        stats = calculate_partition_stats(current_partitions, node_weights)
        partition_ids = list(stats.keys())
        random.shuffle(partition_ids) # Process in random order

        # Identify overweight and underweight partitions
        overweight = {pid: s['total_tokens'] for pid, s in stats.items() if s['total_tokens'] > max_package_tokens}
        # Consider underweight relative to target, but also ensure min file count isn't violated
        underweight = {pid: s['total_tokens'] for pid, s in stats.items() if s['total_tokens'] < target_package_size * 0.9} # Example threshold

        if not overweight:
            logger.info(f"No overweight partitions found in iteration {iteration + 1}. Stopping refinement.")
            break

        # Try moving nodes from overweight partitions
        for source_pid in list(overweight.keys()): # Iterate over copy as dict might change
             if source_pid not in current_partitions: continue # Partition might have been emptied

             source_nodes = current_partitions[source_pid]
             source_tokens = stats[source_pid]['total_tokens']

             # Cannot move from partitions that are already too small in file count
             if len(source_nodes) <= min_package_files:
                 continue

             # Find boundary nodes of the overweight partition
             boundary_nodes = get_boundary_nodes(graph, source_pid, current_partitions)
             if not boundary_nodes:
                 logger.debug(f"Overweight partition {source_pid} has no boundary nodes to move.")
                 continue

             potential_moves: List[Tuple[str, int, float]] = [] # (node_to_move, target_pid, score)

             # Evaluate potential moves for each boundary node
             for node_to_move in boundary_nodes:
                 node_token_cost = node_weights.get(node_to_move, 0)

                 # Find adjacent partitions (partitions containing neighbors)
                 neighbors = set(graph.predecessors(node_to_move)) | set(graph.successors(node_to_move))
                 node_to_partition_map = {node: pid for pid, nodes in current_partitions.items() for node in nodes}
                 adjacent_pids = {node_to_partition_map[n] for n in neighbors if n in node_to_partition_map and node_to_partition_map[n] != source_pid}

                 for target_pid in adjacent_pids:
                     if target_pid not in stats: continue # Target partition might have been removed if empty
                     target_tokens = stats[target_pid]['total_tokens']

                     # Check if move is valid:
                     # 1. Target partition won't exceed max token limit
                     # 2. Source partition won't become too small (file count)
                     if (target_tokens + node_token_cost <= max_package_tokens and
                         len(source_nodes) - 1 >= min_package_files):

                         # Calculate a score (lower is better) - e.g., how much it reduces imbalance
                         # Simple score: absolute difference from target size after move
                         new_source_tokens = source_tokens - node_token_cost
                         new_target_tokens = target_tokens + node_token_cost
                         # Score based on sum of squared deviations from target?
                         current_dev_sq = (source_tokens - target_package_size)**2 + (target_tokens - target_package_size)**2
                         new_dev_sq = (new_source_tokens - target_package_size)**2 + (new_target_tokens - target_package_size)**2
                         score = new_dev_sq - current_dev_sq # Negative score means improvement

                         # Only consider moves that improve balance (score < 0)
                         if score < 0:
                              potential_moves.append((node_to_move, target_pid, score))

             # Select and perform the best move (or a random valid move)
             if potential_moves:
                 if move_strategy == 'best_fit':
                     potential_moves.sort(key=lambda x: x[2]) # Sort by score (lowest first)
                     best_move = potential_moves[0]
                 else: # 'random'
                     best_move = random.choice(potential_moves)

                 node_to_move, target_pid, move_score = best_move
                 logger.debug(f"Moving node '{node_to_move}' ({node_weights.get(node_to_move, 0)} tokens) from partition {source_pid} to {target_pid}. Score: {move_score:.2f}")

                 # Perform the move
                 current_partitions[source_pid].remove(node_to_move)
                 current_partitions[target_pid].append(node_to_move)
                 moved_node = True

                 # Optional: Re-calculate stats immediately for next iteration within the loop?
                 # Or recalculate at the start of the next outer iteration (simpler).
                 # For simplicity, break after one move per outer iteration to avoid complex state updates
                 break # Move to next iteration after one successful move

        if not moved_node:
            logger.info(f"No balancing moves made in iteration {iteration + 1}. Stopping refinement.")
            break # No improvement possible in this iteration

    # --- Final Cleanup: Remove empty partitions and re-index ---
    final_partitions_list = [nodes for nodes in current_partitions.values() if len(nodes) >= min_package_files]
    # Filter based on min file count again after moves
    final_valid_partitions = {i: sorted(nodes) for i, nodes in enumerate(final_partitions_list)}

    final_stats = calculate_partition_stats(final_valid_partitions, node_weights)
    logger.info(f"Balance refinement finished after {iteration + 1} iterations.")
    logger.info(f"Final partition count: {len(final_valid_partitions)}")
    if final_valid_partitions:
        tokens = [s['total_tokens'] for s in final_stats.values()]
        logger.info(f"Final token counts: Min={min(tokens)}, Max={max(tokens)}, Avg={statistics.mean(tokens):.0f}, StdDev={statistics.stdev(tokens) if len(tokens) > 1 else 0:.0f}")

    return final_valid_partitions
