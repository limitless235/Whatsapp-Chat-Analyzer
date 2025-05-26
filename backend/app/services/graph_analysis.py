from typing import List, Dict, Any
from collections import defaultdict
import networkx as nx

try:
    import community as community_louvain  # from python-louvain
except ImportError:
    community_louvain = None

from ..utils.cleaning import clean_text

class GraphAnalysisService:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.communities = {}

    def build_interaction_graph(self, messages: List[Dict[str, Any]], weight_by: str = "count") -> nx.DiGraph:
        """
        Build a directed interaction graph from messages with optional weighting.
        Each message: {"sender": str, "receivers": List[str], "text": str, "toxicity": float, "sentiment": str}
        """
        self.graph.clear()
        interaction_data = defaultdict(list)

        for msg in messages:
            sender = msg.get("sender")
            receivers = msg.get("receivers", [])
            text = msg.get("text", "")
            if not sender or not receivers:
                continue

            # Clean text to avoid noise
            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue  # skip empty cleaned texts

            # Optionally, you could store the cleaned text back or use it in weighting

            for receiver in receivers:
                interaction_data[(sender, receiver)].append(msg)

        for (sender, receiver), msgs in interaction_data.items():
            weight = self._calculate_weight(msgs, weight_by)
            self.graph.add_edge(sender, receiver, weight=weight, count=len(msgs))

        return self.graph

    def _calculate_weight(self, msgs: List[Dict], weight_by: str) -> float:
        if weight_by == "toxicity":
            values = [m.get("toxicity", 0.0) for m in msgs]
        elif weight_by == "sentiment":
            values = [self._sentiment_to_score(m.get("sentiment", "neutral")) for m in msgs]
        else:
            return len(msgs)

        return round(sum(values) / len(values), 4) if values else 0.0

    def _sentiment_to_score(self, label: str) -> float:
        return {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(label.lower(), 0.0)

    def get_node_degrees(self) -> Dict[str, int]:
        return dict(self.graph.degree())

    def get_most_connected_pairs(self, top_n: int = 10) -> List[Dict]:
        edges = sorted(self.graph.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)
        return [
            {"sender": u, "receiver": v, "weight": round(data["weight"], 3)}
            for u, v, data in edges[:top_n]
        ]

    def get_isolated_users(self) -> List[str]:
        return [node for node, degree in self.graph.degree() if degree == 0]

    def detect_communities(self) -> Dict[str, int]:
        """
        Detect user communities using Louvain algorithm (on undirected version).
        Returns a mapping: user -> community_id
        """
        if community_louvain is None:
            raise ImportError("Install `python-louvain` to use community detection.")

        undirected_graph = self.graph.to_undirected()
        self.communities = community_louvain.best_partition(undirected_graph)
        return self.communities

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the graph in a JSON-ready format (D3.js / Plotly-friendly).
        Includes nodes, links, optional communities.
        """
        nodes = []
        for node in self.graph.nodes():
            node_data = {"id": node}
            if self.communities:
                node_data["community"] = self.communities.get(node, -1)
            nodes.append(node_data)

        links = [
            {"source": u, "target": v, "weight": round(data.get("weight", 1.0), 3)}
            for u, v, data in self.graph.edges(data=True)
        ]

        return {
            "nodes": nodes,
            "links": links
        }
