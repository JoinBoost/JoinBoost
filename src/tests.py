import unittest
import math
import pandas as pd
import duckdb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from executor import Executor, DuckdbExecutor
from join_graph import JoinGraph
import app


class TestDecision(unittest.TestCase):
    join = pd.read_csv("../data/synthetic/RST.csv")
    con = duckdb.connect(database=':memory:')
    con.execute("CREATE TABLE R AS SELECT * FROM '../data/synthetic/R.csv'")
    con.execute("CREATE TABLE S AS SELECT * FROM '../data/synthetic/S.csv'")
    con.execute("CREATE TABLE T AS SELECT * FROM '../data/synthetic/T.csv'")
    con.execute("CREATE TABLE test AS SELECT * FROM '../data/synthetic/RST.csv'")
        
    def test_multiple_tables_attribute_as_join_keys(self):
        x = ["A", "B", "D", "E", "F"]
        y = "C"
        target_var = 'C'
        target_var_relation = 'R'

        exe = DuckdbExecutor(self.con, debug=True)
    
        dataset = JoinGraph(exe=exe)
        dataset.add_relation_attrs('R', ['B', 'D'], y = 'C')
        dataset.add_relation_attrs('S', ['A', 'E'])
        dataset.add_relation_attrs('T', ['F'])
        dataset.add_join("R", "S", ["A"], ["A"])
        dataset.add_join("R", "T", ["B"], ["B"])
        
        depth = 3
        gb = app.DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)
        
        gb.fit(dataset)
        
        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(self.join[x], self.join[y])
        mse = mean_squared_error(self.join[y], clf.predict(self.join[x]))
        self.assertTrue(abs(gb.compute_predict_se('test')[0] - math.sqrt(mse)) < 1e-3)


if __name__ == '__main__':
    unittest.main()