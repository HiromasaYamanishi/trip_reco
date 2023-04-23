import pandas as pd
import osmnx
import csv

G = osmnx.graph_from_place(f'Tokyo, Japan', network_type='drive')

# nodes
node=pd.DataFrame(osmnx.utils_graph.graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)[0])
# edges
edge=pd.DataFrame(osmnx.utils_graph.graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)[1])
# csv形式で出力。
node.to_csv(f'data/road/tokyo_nodes.csv')
edge.to_csv(f'data/road/tokyo_edges.csv')