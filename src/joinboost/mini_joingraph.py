

class MiniJoinGraph:
    def __init__(self):
        self.nodes = []
        self.edges = {}

    def add_node(self, node):
        self.nodes.append(node)
        self.edges[node] = []

    def add_edge(self, node1, node2):
        self.edges[node1].append(node2)
        self.edges[node2].append(node1)

    def get_dfs_order(self):
        def dfs(parent, node, visited, visited_edges):
            visited.append(node)
            if parent is not None:
                visited_edges.append((parent, node))
            for neighbor in self.edges[node]:
                if neighbor not in visited:
                    dfs(node, neighbor, visited, visited_edges)

        visited_nodes = list()
        visited_edges = list()
        
        # Find a leaf node
        start_node = None
        for node in self.nodes:
            if len(self.edges[node]) == 1:
                start_node = node
                break

        visited_nodes = list()
        visited_edges = list()
        dfs(None, start_node, visited_nodes, visited_edges)
        for node in self.nodes:
            if node not in visited_nodes:
                dfs(None, node, visited_nodes, visited_edges)

        return visited_nodes, visited_edges
