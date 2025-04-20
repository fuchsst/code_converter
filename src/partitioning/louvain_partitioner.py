# src/partitioning/louvain_partitioner.py
from typing import Dict, List, Any
import networkx as nx
import community as community_louvain # python-louvain library
from collections import defaultdict

from .base_partitioner import BasePartitioner
from src.logger_setup import get_logger

logger = get_logger(__name__)

class LouvainPartitioner(BasePartitioner):
    """
    Partitions a graph using the Louvain method for community detection,
    optimizing for weighted directed modularity.
    """

    def partition(self,
                  graph: nx.DiGraph,
                  node_weights: Dict[str, int], # Not used by Louvain itself, but passed for interface consistency
                  config: Dict[str, Any]) -> Dict[int, List[str]]:
        """
        Partitions the graph using the Louvain algorithm.

        Args:
            graph (nx.DiGraph): The weighted directed dependency graph.
                                Edge A -> B means A includes B.
                                Edges must have a 'weight' attribute.
            node_weights (Dict[str, int]): Node weights (token counts), ignored by this partitioner.
            config (Dict[str, Any]): Configuration dictionary. May contain:
                                     - 'LOUVAIN_RESOLUTION': Resolution parameter for Louvain.
                                     - 'LOUVAIN_RANDOM_STATE': Seed for reproducibility.

        Returns:
            Dict[int, List[str]]: A dictionary mapping partition IDs (integers)
                                  to lists of node names (file paths).
        """
        if not graph.nodes:
            logger.warning("LouvainPartitioner received an empty graph. Returning empty partitioning.")
            return {}

        # Louvain works best on undirected graphs for standard modularity.
        # However, the python-louvain library can handle directed graphs if weights are present,
        # implicitly using a directed modularity variant (though documentation is sparse on specifics).
        # For robustness, we'll use the directed graph directly with weights.
        # Ensure weights exist, default to 1 if not.
        for u, v, data in graph.edges(data=True):
            if 'weight' not in data:
                data['weight'] = 1.0

        logger.info("Starting Louvain community detection on the directed graph...")
        resolution = config.get('LOUVAIN_RESOLUTION', 1.0)
        random_state = config.get('LOUVAIN_RANDOM_STATE', None)

        try:
            # Convert the directed graph to undirected for Louvain compatibility.
            # NetworkX's to_undirected() merges parallel edges; by default,
            # it keeps the attributes of the first edge encountered. This is acceptable here.
            undirected_graph = graph.to_undirected(as_view=False)
            logger.debug(f"Converted DiGraph to Undirected Graph with {undirected_graph.number_of_nodes()} nodes and {undirected_graph.number_of_edges()} edges.")

            # Use best_partition on the undirected graph
            partition_map = community_louvain.best_partition(
                undirected_graph,
                weight='weight', # Use the merged edge weights
                resolution=resolution,
                random_state=random_state
            )
            logger.info(f"Louvain completed. Found {len(set(partition_map.values()))} initial communities.")

        except Exception as e:
            logger.error(f"Louvain partitioning failed: {e}", exc_info=True)
            # Fallback: return each node as its own community? Or raise error?
            # Returning empty dict signals failure upstream.
            return {}

        # Convert the node-to-community map into a community-to-nodes map
        communities: Dict[int, List[str]] = defaultdict(list)
        for node, community_id in partition_map.items():
            communities[community_id].append(node)

        # Sort files within each community for consistency
        for community_id in communities:
            communities[community_id].sort()

        logger.info(f"Formatted Louvain results into {len(communities)} partitions.")
        # Log sizes for debugging
        # for cid, nodes in communities.items():
        #     logger.debug(f"  Community {cid}: {len(nodes)} nodes")

        return dict(communities)
