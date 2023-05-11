import unittest
import math
import pandas as pd
import numpy as np
import duckdb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree, GradientBoosting, RandomForest


class TestApp(unittest.TestCase):
    def test_predict_api_full_join(self):
        con = duckdb.connect(database=":memory:")
        con.execute(
            "CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../data/favorita/holidays.csv';"
        )
        con.execute(
            "CREATE OR REPLACE TABLE oil AS SELECT * FROM '../data/favorita/oil.csv';"
        )
        con.execute(
            "CREATE OR REPLACE TABLE transactions AS SELECT * FROM '../data/favorita/transactions.csv';"
        )
        con.execute(
            "CREATE OR REPLACE TABLE stores AS SELECT * FROM '../data/favorita/stores.csv';"
        )
        con.execute(
            "CREATE OR REPLACE TABLE items AS SELECT * FROM '../data/favorita/items.csv';"
        )
        con.execute(
            "CREATE OR REPLACE TABLE sales AS SELECT * FROM '../data/favorita/sales_small.csv';"
        )
        con.execute(
            "CREATE OR REPLACE TABLE train AS SELECT * FROM '../data/favorita/train_small.csv';"
        )

        y = "Y"
        x = [
            "htype",
            "locale",
            "locale_name",
            "transferred",
            "f2",
            "dcoilwtico",
            "f3",
            "transactions",
            "f5",
            "city",
            "state",
            "stype",
            "cluster",
            "f4",
            "family",
            "class",
            "perishable",
            "f1",
        ]

        exe = DuckdbExecutor(con, debug=False)
        depth = 3
        iteration = 3
        dataset = JoinGraph(exe=exe)
        dataset.add_relation("sales", [], y="Y", relation_address="../data/favorita/sales_small.csv")
        dataset.add_relation(
            "holidays", ["htype", "locale", "locale_name", "transferred", "f2"], relation_address="../data/favorita/holidays.csv"
        )
        dataset.add_relation("oil", ["dcoilwtico", "f3"], relation_address="../data/favorita/oil.csv")
        dataset.add_relation("transactions", ["transactions", "f5"], relation_address="../data/favorita/transactions.csv")
        dataset.add_relation("stores", ["city", "state", "stype", "cluster", "f4"], relation_address="../data/favorita/stores.csv")
        dataset.add_relation("items", ["family", "class", "perishable", "f1"], relation_address="../data/favorita/items.csv")
        dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("transactions", "holidays", ["date"], ["date"])
        dataset.add_join("holidays", "oil", ["date"], ["date"])

        reg = GradientBoosting(
            learning_rate=1, max_leaves=2**depth, max_depth=depth, iteration=iteration
        )

        reg.fit(dataset)
        
        
        dataset.target_relation = "train"
        reg_prediction = reg.predict(joingraph=dataset, input_mode="FULL_JOIN_JG")

        data = pd.read_csv("../data/favorita/train_small.csv")
        clf = GradientBoostingRegressor(
            max_depth=depth, learning_rate=1, n_estimators=iteration
        )
        clf = clf.fit(data[x], data[y])
        clf_prediction = clf.predict(data[x])
        mse = mean_squared_error(data[y], clf_prediction)
        _reg_rmse = reg.compute_rmse("train")[0]

        self.assertTrue(abs(_reg_rmse - math.sqrt(mse)) < 1e-3)
        self.assertTrue(np.sum(np.abs(reg_prediction - clf_prediction)) < 1e-3)

    def test_predict_api_joingraph(self):
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../data/favorita/holidays.csv';")
        con.execute("CREATE OR REPLACE TABLE oil AS SELECT * FROM '../data/favorita/oil.csv';")
        con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM '../data/favorita/transactions.csv';")
        con.execute("CREATE OR REPLACE TABLE stores AS SELECT * FROM '../data/favorita/stores.csv';")
        con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM '../data/favorita/items.csv';")
        con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM '../data/favorita/sales_small.csv';")

        y = "Y"
        x = ["htype", "locale", "locale_name", "transferred","f2","dcoilwtico","f3","transactions",
             "f5","city","state","stype","cluster","f4","family","class","perishable","f1"]

        exe = DuckdbExecutor(con, debug=False)
        depth = 3
        iteration = 3
        dataset = JoinGraph(exe=exe)
        dataset.add_relation("sales", [], y = 'Y', relation_address="../data/favorita/sales_small.csv")
        dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"], relation_address="../data/favorita/holidays.csv")
        dataset.add_relation("oil", ["dcoilwtico","f3"], relation_address="../data/favorita/oil.csv")
        dataset.add_relation("transactions", ["transactions","f5"], relation_address="../data/favorita/transactions.csv")
        dataset.add_relation("stores", ["city","state","stype","cluster","f4"], relation_address="../data/favorita/stores.csv")
        dataset.add_relation("items", ["family","class","perishable","f1"], relation_address="../data/favorita/items.csv")
        dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("transactions", "holidays", ["date"], ["date"])
        dataset.add_join("holidays", "oil", ["date"], ["date"])
        
        dataset2 = JoinGraph(exe=exe)
        dataset2.add_relation("sales", [], y = 'Y', relation_address="../data/favorita/sales_small.csv")
        dataset2.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"], relation_address="../data/favorita/holidays.csv")
        dataset2.add_relation("oil", ["dcoilwtico","f3"], relation_address="../data/favorita/oil.csv")
        dataset2.add_relation("transactions", ["transactions","f5"], relation_address="../data/favorita/transactions.csv")
        dataset2.add_relation("stores", ["city","state","stype","cluster","f4"], relation_address="../data/favorita/stores.csv")
        dataset2.add_relation("items", ["family","class","perishable","f1"], relation_address="../data/favorita/items.csv")
        dataset2.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset2.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset2.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset2.add_join("transactions", "holidays", ["date"], ["date"])
        dataset2.add_join("holidays", "oil", ["date"], ["date"])

        reg = GradientBoosting(learning_rate=1, max_leaves=2 ** depth, max_depth=depth, iteration=iteration)

        reg.fit(dataset)
        reg_prediction = reg.predict(joingraph=dataset2, input_mode="JOIN_GRAPH")

        data = pd.read_csv('../data/favorita/train_small.csv')
        clf = GradientBoostingRegressor(max_depth=depth,learning_rate=1, n_estimators=iteration)
        clf = clf.fit(data[x], data[y])
        clf_prediction = clf.predict(data[x])

        print(np.sum(np.abs(reg_prediction - clf_prediction)))
        self.assertTrue(np.sum(np.abs(reg_prediction - clf_prediction)) < 1e-3)

    def test_predict_api_joingraph_rowid_column(self):
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../data/favorita/holidays.csv';")
        con.execute("CREATE OR REPLACE TABLE oil AS SELECT * FROM '../data/favorita/oil.csv';")
        con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM '../data/favorita/transactions.csv';")
        con.execute("CREATE OR REPLACE TABLE stores AS SELECT * FROM '../data/favorita/stores.csv';")
        con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM '../data/favorita/items.csv';")
        con.execute("CREATE OR REPLACE TABLE sales_original AS SELECT * FROM '../data/favorita/sales_small.csv';")
        con.execute("""CREATE OR REPLACE TABLE sales AS
                    SELECT item_nbr,unit_sales,onpromotion AS rowid,tid,Y
                    FROM sales_original;
                    """)

        y = "Y"
        x = ["htype", "locale", "locale_name", "transferred","f2","dcoilwtico","f3","transactions",
             "f5","city","state","stype","cluster","f4","family","class","perishable","f1"]

        exe = DuckdbExecutor(con, debug=False)
        depth = 3
        iteration = 3
        dataset = JoinGraph(exe=exe)
        dataset.add_relation("sales", [], y = 'Y', relation_address="../data/favorita/sales_small.csv")
        dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"], relation_address="../data/favorita/holidays.csv")
        dataset.add_relation("oil", ["dcoilwtico","f3"], relation_address="../data/favorita/oil.csv")
        dataset.add_relation("transactions", ["transactions","f5"], relation_address="../data/favorita/transactions.csv")
        dataset.add_relation("stores", ["city","state","stype","cluster","f4"], relation_address="../data/favorita/stores.csv")
        dataset.add_relation("items", ["family","class","perishable","f1"], relation_address="../data/favorita/items.csv")
        dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("transactions", "holidays", ["date"], ["date"])
        dataset.add_join("holidays", "oil", ["date"], ["date"])
        
        dataset2 = JoinGraph(exe=exe)
        dataset2.add_relation("sales", [], y = 'Y', relation_address="../data/favorita/sales_small.csv")
        dataset2.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"], relation_address="../data/favorita/holidays.csv")
        dataset2.add_relation("oil", ["dcoilwtico","f3"], relation_address="../data/favorita/oil.csv")
        dataset2.add_relation("transactions", ["transactions","f5"], relation_address="../data/favorita/transactions.csv")
        dataset2.add_relation("stores", ["city","state","stype","cluster","f4"], relation_address="../data/favorita/stores.csv")
        dataset2.add_relation("items", ["family","class","perishable","f1"], relation_address="../data/favorita/items.csv")
        dataset2.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset2.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset2.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset2.add_join("transactions", "holidays", ["date"], ["date"])
        dataset2.add_join("holidays", "oil", ["date"], ["date"])

        reg = GradientBoosting(learning_rate=1, max_leaves=2 ** depth, max_depth=depth, iteration=iteration)

        reg.fit(dataset)
        reg_prediction = reg.predict(dataset2, input_mode="JOIN_GRAPH")

        data = pd.read_csv('../data/favorita/train_small.csv')
        clf = GradientBoostingRegressor(max_depth=depth,learning_rate=1, n_estimators=iteration)
        clf = clf.fit(data[x], data[y])
        clf_prediction = clf.predict(data[x])

        print(np.sum(np.abs(reg_prediction - clf_prediction)))
        self.assertTrue(np.sum(np.abs(reg_prediction - clf_prediction)) < 1e-3)


if __name__ == "__main__":
    unittest.main()
