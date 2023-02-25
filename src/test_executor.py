import math
import unittest
import pandas as pd
import duckdb
from pandas import testing
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from src.joinboost.aggregator import Aggregator
from src.joinboost.app import DecisionTree
from src.joinboost.executor import PandasExecutor
from src.joinboost.joingraph import JoinGraph


class TestExecutor(unittest.TestCase):

    # def test_pandas_executor_agg(self):
    #     # initialize PandaExecutor
    #     con = duckdb.connect(database=':memory:')
    #     executor = PandasExecutor(con)
    #     # add tables
    #     R = pd.read_csv("../data/synthetic-one-to-many/R.csv")
    #     S = pd.read_csv("../data/synthetic-one-to-many/S.csv")
    #     T = pd.read_csv("../data/synthetic-one-to-many/T.csv")
    #     executor.add_table('R', R)
    #     executor.add_table('S', S)
    #     executor.add_table('T', T)
    #     # generate aggregate expressions to sum R.A
    #     agg_exprs = {'B': ('B', Aggregator.SUM)}
    #
    #     actual_result = executor.execute_spja_query(aggregate_expressions=agg_exprs, from_tables=['R', 'S', 'T'],
    #                                                 join_conds=['R.B IS NOT DISTINCT FROM T.B', 'S.F IS NOT DISTINCT FROM T.F'],
    #                                                 select_conds=['R.A >= 1'], order_by=[('A', 'ASC')],
    #                                                 group_by=['A'], mode=3)
    #     print(actual_result)
    #     # renaming columns because that's what the executor does
    #     # for col in R.columns:
    #     #     R = R.rename(columns={col: 'R.' + col})
    #     # filter pandas dataframe R with A = 1
    #     # expected_result = R[R['R.A'] == 1]
    #     # # check if the data of actual_result and expected_result are the same
    #     # testing.assert_frame_equal(actual_result, expected_result)

    def test_synthetic(self):
        join = pd.read_csv("../data/synthetic/RST.csv")
        con = duckdb.connect(database=':memory:')
        pd.read_csv('../data/synthetic/R.csv')
        pd.read_csv("../data/synthetic/S.csv")
        pd.read_csv("../data/synthetic/T.csv")
        pd.read_csv("../data/synthetic/RST.csv")
        x = ["A", "B", "D", "E", "F"]
        y = "H"

        exe = PandasExecutor(con)
        # exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
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
        print(gb.model_def)
        self.assertTrue(abs(gb.compute_rmse('test')[0] - math.sqrt(mse)) < 1e-3)


    def test_favorita(self):
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../data/favorita/holidays.csv';")
        con.execute("CREATE OR REPLACE TABLE oil AS SELECT * FROM '../data/favorita/oil.csv';")
        con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM '../data/favorita/transactions.csv';")
        con.execute("CREATE OR REPLACE TABLE stores AS SELECT * FROM '../data/favorita/stores.csv';")
        con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM '../data/favorita/items.csv';")
        con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM '../data/favorita/sales_small.csv';")
        con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM '../data/favorita/train_small.csv';")

        y = "Y"
        x = ["htype", "locale", "locale_name", "transferred", "f2", "dcoilwtico", "f3", "transactions",
             "f5", "city", "state", "stype", "cluster", "f4", "family", "class", "perishable", "f1"]

        exe = PandasExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation("sales", [], y='Y', relation_address='../data/favorita/sales_small.csv')
        dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred", "f2"], relation_address='../data/favorita/holidays.csv')
        dataset.add_relation("oil", ["dcoilwtico", "f3"], relation_address='../data/favorita/oil.csv')
        dataset.add_relation("transactions", ["transactions", "f5"], relation_address='../data/favorita/transactions.csv')
        dataset.add_relation("stores", ["city", "state", "stype", "cluster", "f4"], relation_address='../data/favorita/stores.csv')
        dataset.add_relation("items", ["family", "class", "perishable", "f1"], relation_address='../data/favorita/items.csv')
        dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("transactions", "holidays", ["date"], ["date"])
        dataset.add_join("holidays", "oil", ["date"], ["date"])

        depth = 3
        reg = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)

        reg.fit(dataset)

        data = pd.read_csv('../data/favorita/train_small.csv')
        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(data[x], data[y])
        mse = mean_squared_error(data[y], clf.predict(data[x]))
        self.assertTrue(abs(reg.compute_rmse('train')[0] - math.sqrt(mse)) < 1e-3)


if __name__ == '__main__':
    unittest.main()
