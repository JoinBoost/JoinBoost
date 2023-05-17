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

if __name__ == '__main__':
    unittest.main()