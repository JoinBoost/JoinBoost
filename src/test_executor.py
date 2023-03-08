import math
import unittest
import pandas as pd
import duckdb
from pandas import testing
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from joinboost.aggregator import Aggregator
from joinboost.app import DecisionTree
from joinboost.executor import PandasExecutor, DuckdbExecutor
from joinboost.joingraph import JoinGraph


class TestExecutor(unittest.TestCase):
    def test_synthetic(self):
        join = pd.read_csv("../data/synthetic/RST.csv")
        x = ["A", "B", "D", "E", "F"]
        y = "H"

        exe = PandasExecutor(None)
        # exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph()
        dataset.add_relation('R', ['B', 'D'], y='H', relation_address='../data/synthetic/R.csv')
        dataset.add_relation('S', ['A', 'E'], relation_address='../data/synthetic/S.csv')
        dataset.add_relation('T', ['F'], relation_address='../data/synthetic/T.csv')
        dataset.add_join("R", "S", ["A"], ["A"])
        dataset.add_join("R", "T", ["B"], ["B"])

        depth = 3
        gb = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)

        gb.fit(dataset)

        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(join[x], join[y])
        mse = mean_squared_error(join[y], clf.predict(join[x]))
        model_def1 = gb.model_def


        con = duckdb.connect(database=':memory:')
        con.execute("CREATE TABLE R AS SELECT * FROM '../data/synthetic/R.csv'")
        con.execute("CREATE TABLE S AS SELECT * FROM '../data/synthetic/S.csv'")
        con.execute("CREATE TABLE T AS SELECT * FROM '../data/synthetic/T.csv'")
        con.execute("CREATE TABLE test AS SELECT * FROM '../data/synthetic/RST.csv'")
        x = ["A", "B", "D", "E", "F"]
        y = "H"

        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', ['B', 'D'], y='H')
        dataset.add_relation('S', ['A', 'E'])
        dataset.add_relation('T', ['F'])
        dataset.add_join("R", "S", ["A"], ["A"])
        dataset.add_join("R", "T", ["B"], ["B"])

        depth = 3
        gb = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)

        gb.fit(dataset)

        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(join[x], join[y])
        mse = mean_squared_error(join[y], clf.predict(join[x]))
        model_def2 = gb.model_def
        
        # should be the same
        print(model_def1)
        print(model_def2)
        
        # TODO: compute_rmse is not implemented
        # self.assertTrue(abs(gb.compute_rmse('test')[0] - math.sqrt(mse)) < 1e-3)

    # TODO
#     def test_favorita(self):
#         con = duckdb.connect(database=':memory:')
#         con.execute("CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../data/favorita/holidays.csv';")
#         con.execute("CREATE OR REPLACE TABLE oil AS SELECT * FROM '../data/favorita/oil.csv';")
#         con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM '../data/favorita/transactions.csv';")
#         con.execute("CREATE OR REPLACE TABLE stores AS SELECT * FROM '../data/favorita/stores.csv';")
#         con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM '../data/favorita/items.csv';")
#         con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM '../data/favorita/sales_small.csv';")
#         con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM '../data/favorita/train_small.csv';")

#         y = "Y"
#         x = ["htype", "locale", "locale_name", "transferred", "f2", "dcoilwtico", "f3", "transactions",
#              "f5", "city", "state", "stype", "cluster", "f4", "family", "class", "perishable", "f1"]

#         exe = PandasExecutor(con, debug=False)

#         dataset = JoinGraph(exe=exe)
#         dataset.add_relation("sales", [], y='Y', relation_address='../data/favorita/sales_small.csv')
#         dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred", "f2"], relation_address='../data/favorita/holidays.csv')
#         dataset.add_relation("oil", ["dcoilwtico", "f3"], relation_address='../data/favorita/oil.csv')
#         dataset.add_relation("transactions", ["transactions", "f5"], relation_address='../data/favorita/transactions.csv')
#         dataset.add_relation("stores", ["city", "state", "stype", "cluster", "f4"], relation_address='../data/favorita/stores.csv')
#         dataset.add_relation("items", ["family", "class", "perishable", "f1"], relation_address='../data/favorita/items.csv')
#         dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
#         dataset.add_join("sales", "transactions", ["tid"], ["tid"])
#         dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
#         dataset.add_join("transactions", "holidays", ["date"], ["date"])
#         dataset.add_join("holidays", "oil", ["date"], ["date"])

#         depth = 3
#         reg = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)

#         reg.fit(dataset)

#         data = pd.read_csv('../data/favorita/train_small.csv')
#         clf = DecisionTreeRegressor(max_depth=depth)
#         clf = clf.fit(data[x], data[y])
#         mse = mean_squared_error(data[y], clf.predict(data[x]))
#         self.assertTrue(abs(reg.compute_rmse('train')[0] - math.sqrt(mse)) < 1e-3)


if __name__ == '__main__':
    unittest.main()
