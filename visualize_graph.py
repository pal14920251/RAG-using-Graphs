import networkx as nx
from pyvis.network import Network

def visualize_graph(graph, output_file="graph.html", max_nodes=50):
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )

    G = nx.DiGraph()

    # Add edges (limit for readability)
    added_nodes = set()
    for s, r, o in graph.edges:
        if len(added_nodes) >= max_nodes:
            break
        G.add_edge(s, o, label=r)
        added_nodes.add(s)
        added_nodes.add(o)

    # Add nodes with colors
    for node, data in graph.nodes.items():
        if node not in added_nodes:
            continue

        color = {
            "Chunk": "#FFA500",     # orange
            "Entity": "#87CEEB",    # blue
            "Concept": "#90EE90"    # green
        }.get(data["type"], "#D3D3D3")

        net.add_node(
            node,
            label=node,
            title=f"Type: {data['type']}",
            color=color
        )

    # Add edges with labels
    for s, o, data in G.edges(data=True):
        net.add_edge(s, o, label=data["label"])

    net.write_html(output_file)
    print(f"Graph saved to {output_file}")
