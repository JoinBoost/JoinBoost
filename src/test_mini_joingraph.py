import unittest
from joinboost.mini_joingraph import MiniJoinGraph


class TestMiniJoingraph(unittest.TestCase):



    def test_dfs_order(self):
        join_graph = MiniJoinGraph()
        join_graph.add_node("R")
        join_graph.add_node("S")
        join_graph.add_node("T")
        join_graph.add_edge("R", "S")
        join_graph.add_edge("S", "T")
        actual_nodes, actual_edges = join_graph.get_dfs_order()
        self.assertEqual(actual_nodes, ["R", "S", "T"])
        self.assertEqual(actual_edges, [("R", "S"), ("S", "T")])

    #     A
    #    / \
    #   B   C
    #  / \ / \
    # D  E F  G
    def test_dfs_order_large_acyclic(self):
        join_graph = MiniJoinGraph()
        for node in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            join_graph.add_node(node)
        join_graph.add_edge('A', 'B')
        join_graph.add_edge('A', 'C')
        join_graph.add_edge('B', 'D')
        join_graph.add_edge('B', 'E')
        join_graph.add_edge('C', 'F')
        join_graph.add_edge('C', 'G')

        actual_nodes, actual_edges = join_graph.get_dfs_order()

        # DFS order depends on the order of the neighbors
        # Assuming the order is as the edges were added, one possible output is:
        expected_nodes = ['D', 'B', 'A', 'C', 'F', 'G', 'E']
        expected_edges = [('D', 'B'), ('B', 'A'), ('A', 'C'), ('C', 'F'), ('C', 'G'), ('B', 'E')]

        self.assertEqual(actual_nodes, expected_nodes)
        self.assertEqual(actual_edges, expected_edges)


if __name__ == '__main__':
    unittest.main()