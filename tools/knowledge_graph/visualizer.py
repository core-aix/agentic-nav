import networkx as nx
import plotly.graph_objects as go
import pickle

from pyvis.network import Network

from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent


class GraphVisualizer:

    def __init__(self):
        pass

    @staticmethod
    def __write_plotly_image(path_to_graph, *args, **kwargs):
        file_name = str(path_to_graph).split("/")[-1].split(".")[0]

        with open(path_to_graph, 'rb') as f:
            graph = pickle.load(f)
            f.close()

        pos = nx.spring_layout(graph)
        edge_trace = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                         mode='lines', line=dict(width=0.5, color='#888')))

        node_trace = go.Scatter(x=[pos[node][0] for node in graph.nodes()],
                                y=[pos[node][1] for node in graph.nodes()],
                                mode='markers+text', text=list(graph.nodes()),
                                marker=dict(size=10, color='lightblue'))

        fig = go.Figure(data=edge_trace + [node_trace])
        file_path = Path(f"{PROJECT_ROOT_PATH}/graphs/outputs/")
        if not file_path.exists():
            file_path.mkdir(parents=True, exist_ok=True)

        output_path = f"{file_path}/{file_name}_plotly.html"
        fig.write_html(file=output_path)
        print(f"Find the graph visualization under: {output_path}")

    @staticmethod
    def __write_pyvis_html_page(path_to_graph, *args, **kwargs):

        file_name = str(path_to_graph).split("/")[-1].split(".")[0]
        with open(path_to_graph, 'rb') as f:
            graph = pickle.load(f)
            f.close()

        net = Network(notebook=True)
        net.from_nx(graph)

        file_path = Path(f"{PROJECT_ROOT_PATH}/graphs/outputs/")
        if not file_path.exists():
            file_path.mkdir(parents=True, exist_ok=True)

        output_path = f"{file_path}/{file_name}_pyvis.html"
        net.show(output_path)
        print(f"Find the graph visualization under: {output_path}")



    def __call__(self, path_to_graph, engine: str = "plotly", *args, **kwargs):
        if engine == "plotly":
            self.__write_plotly_image(path_to_graph=path_to_graph, *args, **kwargs)
        elif engine == "pyvis":
            self.__write_pyvis_html_page(path_to_graph=path_to_graph, *args, **kwargs)
        else:
            raise NotImplementedError("Make sure to implement the visualization engine you'd like to call.")
