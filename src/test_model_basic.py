import unittest
import math
import pandas as pd
import numpy as np
import duckdb
import lightgbm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree, GradientBoosting, RandomForest


class TestModel(unittest.TestCase):

    # TODO: some table have the same attribute name, making it hard to predict
    def test_demo(self):
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE TABLE customer AS SELECT * FROM '../data/demo/customer.csv'")
        con.execute("CREATE TABLE lineorder AS SELECT * FROM '../data/demo/lineorder.csv'")
        con.execute("CREATE TABLE date AS SELECT * FROM '../data/demo/date.csv'")
        con.execute("CREATE TABLE part AS SELECT * FROM '../data/demo/part.csv'")
        con.execute("CREATE TABLE supplier AS SELECT * FROM '../data/demo/supplier.csv'")
        x = ["NAME", "ADDRESS", "CITY", "NAME", "MFGR", "CATEGORY", "BRAND1", "DATE", "DAYOFWEEK",
             "MONTH", "YEAR", "YEARMONTH", "YEARMONTHNUM", "DAYNUMINWEEK", "NAME", "ADDRESS", "CITY", "NATION"]
        y = "REVENUE"
        # delete rows beyond 1000 in lineorder using duckdb
        # con.execute("DELETE FROM lineorder WHERE rowid > 1000")

        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('lineorder', [], y='REVENUE', relation_address='../data/demo/lineorder.csv')
        dataset.add_relation('customer', ['NAME', 'ADDRESS', 'CITY'], relation_address='../data/demo/customer.csv')
        dataset.add_relation('part', ['NAME', 'MFGR', 'CATEGORY', 'BRAND1'], relation_address='../data/demo/part.csv')
        dataset.add_relation('date', ['DATE', 'DAYOFWEEK', 'MONTH', 'YEAR', 'YEARMONTH', 'YEARMONTHNUM', 'DAYNUMINWEEK'], relation_address='../data/demo/date.csv')
        dataset.add_relation('supplier', ['NAME', 'ADDRESS', 'CITY', 'NATION'], relation_address='../data/demo/supplier.csv')
        dataset.add_join("customer", "lineorder", ["CUSTKEY"], ["CUSTKEY"])
        dataset.add_join("part", "lineorder", ["PARTKEY"], ["PARTKEY"])
        dataset.add_join("date", "lineorder", ["DATEKEY"], ["ORDERDATE"])
        dataset.add_join("supplier", "lineorder", ["SUPPKEY"], ["SUPPKEY"])

        depth = 3
        gb = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)

        gb.fit(dataset)

        for line in gb.model_def:
            for subline in line:
                print(subline)
        # clf = DecisionTreeRegressor(max_depth=depth)
        # clf = clf.fit(join[x], join[y])
        # mse = mean_squared_error(join[y], clf.predict(join[x]))

        # self.assertTrue(abs(gb.compute_rmse('test')[0] - math.sqrt(mse)) < 1e-3)

    def test_synthetic(self):
        join = pd.read_csv("../data/synthetic/RST.csv")
        con = duckdb.connect(database=":memory:")
        con.execute("CREATE TABLE test AS SELECT * FROM '../data/synthetic/RST.csv'")
        x = ["A", "B", "D", "E", "F"]
        y = "H"

        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', ['B', 'D'], y='H', relation_address='../data/synthetic/R.csv')
        dataset.add_relation('S', ['A', 'E'], relation_address='../data/synthetic/S.csv')
        dataset.add_relation('T', ['F'], relation_address='../data/synthetic/T.csv')
        dataset.add_join("R", "S", ["A"], ["A"])
        dataset.add_join("R", "T", ["B"], ["B"])

        depth = 3
        gb = DecisionTree(learning_rate=1, max_leaves=2**depth, max_depth=depth)

        gb.fit(dataset)

        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(join[x], join[y])
        mse = mean_squared_error(join[y], clf.predict(join[x]))
        self.assertTrue(abs(gb.compute_rmse("test")[0] - math.sqrt(mse)) < 1e-3)

    def test_favorita(self):
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM '../data/favorita/train_small.csv';")

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

        dataset = JoinGraph(exe=exe)
        dataset.add_relation("sales", [], y = 'Y', relation_address='../data/favorita/sales_small.csv')
        dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"], relation_address='../data/favorita/holidays.csv')
        dataset.add_relation("oil", ["dcoilwtico","f3"], relation_address='../data/favorita/oil.csv')
        dataset.add_relation("transactions", ["transactions","f5"], relation_address='../data/favorita/transactions.csv')
        dataset.add_relation("stores", ["city","state","stype","cluster","f4"], relation_address='../data/favorita/stores.csv')
        dataset.add_relation("items", ["family","class","perishable","f1"], relation_address='../data/favorita/items.csv')
        dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("transactions", "holidays", ["date"], ["date"])
        dataset.add_join("holidays", "oil", ["date"], ["date"])

        depth = 3
        reg = DecisionTree(learning_rate=1, max_leaves=2**depth, max_depth=depth)

        reg.fit(dataset)

        data = pd.read_csv("../data/favorita/train_small.csv")
        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(data[x], data[y])
        mse = mean_squared_error(data[y], clf.predict(data[x]))
        self.assertTrue(abs(reg.compute_rmse("train")[0] - math.sqrt(mse)) < 1e-3)

    def test_gradient_boosting(self):
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM '../data/favorita/train_small.csv';")

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
        dataset.add_relation("sales", [], y = 'Y', relation_address='../data/favorita/sales_small.csv')
        dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"], relation_address='../data/favorita/holidays.csv')
        dataset.add_relation("oil", ["dcoilwtico","f3"], relation_address='../data/favorita/oil.csv')
        dataset.add_relation("transactions", ["transactions","f5"], relation_address='../data/favorita/transactions.csv')
        dataset.add_relation("stores", ["city","state","stype","cluster","f4"], relation_address='../data/favorita/stores.csv')
        dataset.add_relation("items", ["family","class","perishable","f1"], relation_address='../data/favorita/items.csv')
        dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("transactions", "holidays", ["date"], ["date"])
        dataset.add_join("holidays", "oil", ["date"], ["date"])

        reg = GradientBoosting(
            learning_rate=1, max_leaves=2**depth, max_depth=depth, iteration=iteration
        )

        reg.fit(dataset)
        
        dataset.set_target_relation("train")
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

    def test_sample_syn(self):
        data = pd.read_csv("../data/synthetic/RST.csv")
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE TABLE train AS SELECT * FROM '../data/synthetic/RST.csv'")
        x = ["A", "B", "D", "E", "F"]
        y = "H"

        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', ['B', 'D'], y = 'H', relation_address='../data/synthetic/R.csv')
        dataset.add_relation('S', ['A', 'E'], relation_address='../data/synthetic/S.csv')
        dataset.add_relation('T', ['F'], relation_address='../data/synthetic/T.csv')
        dataset.add_join("R", "S", ["A"], ["A"])
        dataset.add_join("R", "T", ["B"], ["B"])

        depth = 2
        reg = RandomForest(
            max_leaves=2**depth, max_depth=depth, subsample=0.5, iteration=2
        )

        reg.fit(dataset)

        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(data[x], data[y])
        mse = mean_squared_error(data[y], clf.predict(data[x]))
        # the training data is sampled, but the accuracy should still be similar
        print(reg.compute_rmse("train")[0])
        print(math.sqrt(mse))

    def test_lightgbm_catigorial(self):
        R = pd.read_csv("../data/synthetic-very-small/R.csv")
        con = duckdb.connect(database=":memory:")
        con.execute("CREATE TABLE R AS SELECT * FROM R")
        x = ["D"]
        y = "H"

        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', categorical_feature=['D'], y = 'H', relation_address='../data/synthetic-very-small/R.csv')

        iteration = 1
        depth = 1

        reg = GradientBoosting(learning_rate=1, max_depth=depth, iteration=iteration)

        reg.fit(dataset)

        clf = lightgbm.LGBMRegressor(
            learning_rate=1,
            max_depth=depth,
            min_child_samples=1,
            objective="RMSE",
            cat_l2=0,
            cat_smooth=0,
            deterministic=True,
            max_bin=20000,
            min_data_in_bin=1,
            n_estimators=iteration,
            max_cat_threshold=10000,
            max_cat_to_onehot=1,
            min_data_per_group=1,
        )

        clf.fit(R[x], R[y], feature_name=x, categorical_feature=x)
        mse = mean_squared_error(R[y], clf.predict(R[x]))

        self.assertTrue(reg.compute_rmse("R")[0] < math.sqrt(mse))


if __name__ == "__main__":
    unittest.main()
