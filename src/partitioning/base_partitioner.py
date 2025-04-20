# src/partitioning/base_partitioner.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import networkx as nx

class BasePartitioner(ABC):
    """
    Abstract base class for graph partitioning algorithms used in Step 2.
    """

    @abstractmethod
    def partition(self,
                  graph: nx.DiGraph,
                  node_weights: Dict[str, int],
                  config: Dict[str, Any]) -> Dict[int, List[str]]:
        """
        Partitions the given graph into communities or packages.

        Args:
            graph (nx.DiGraph): The weighted directed dependency graph.
                                Edge A -> B means A includes B.
                                Edges should have a 'weight' attribute.
            node_weights (Dict[str, int]): A dictionary mapping node names (file paths)
                                           to their weights (e.g., token counts).
            config (Dict[str, Any]): Configuration dictionary, potentially containing
                                     algorithm-specific parameters (e.g., resolution,
                                     target package size, balance tolerance).

        Returns:
            Dict[int, List[str]]: A dictionary where keys are partition IDs (integers)
                                  and values are lists of node names (file paths)
                                  belonging to that partition.
                                  This represents the initial, potentially unbalanced partitioning.
        """
        pass
