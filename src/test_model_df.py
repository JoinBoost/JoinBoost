import math
import unittest

import numpy as np
import pandas as pd
import duckdb
import pytest
from pandas import testing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from src.joinboost.aggregator import Aggregator
from src.joinboost.app import DecisionTree, GradientBoosting, RandomForest
from src.joinboost.executor import PandasExecutor
from src.joinboost.joingraph import JoinGraph


class TestExecutor(unittest.TestCase):

    @pytest.mark.skip(reason="compute_rmse is not implemented yet")
    def test_synthetic(self):
        join = pd.read_csv("../data/synthetic/RST.csv")
        con = duckdb.connect(database=':memory:')
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
        for i in range(len(gb.model_def)):
            for j in range(len(gb.model_def[i])):
                print(gb.model_def[i][j])


        self.assertTrue(abs(gb.compute_rmse('test')[0] - math.sqrt(mse)) < 1e-3)


    @pytest.mark.skip(reason="compute_rmse is not implemented yet")
    def test_favorita(self):
        con = duckdb.connect(database=':memory:')

        y = "Y"
        x = ["htype", "locale", "locale_name", "transferred", "f2", "dcoilwtico", "f3", "transactions",
             "f5", "city", "state", "stype", "cluster", "f4", "family", "class", "perishable", "f1"]

        exe = PandasExecutor(con, debug=True)

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
        for i in range(len(reg.model_def)):
            for j in range(len(reg.model_def[i])):
                print(reg.model_def[i][j])
        mse = mean_squared_error(data[y], clf.predict(data[x]))
        self.assertTrue(abs(reg.compute_rmse('train')[0] - math.sqrt(mse)) < 1e-3)

    # def test_gradient_boosting(self):
    #     con = duckdb.connect(database=':memory:')
    #
    #     y = "Y"
    #     x = ["htype", "locale", "locale_name", "transferred", "f2", "dcoilwtico", "f3", "transactions",
    #          "f5", "city", "state", "stype", "cluster", "f4", "family", "class", "perishable", "f1"]
    #
    #     exe = PandasExecutor(con, debug=True)
    #     depth = 3
    #     iteration = 3
    #     dataset = JoinGraph(exe=exe)
    #     dataset.add_relation("sales", [], y='Y', relation_address='../data/favorita/sales_small.csv')
    #     dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred", "f2"], relation_address='../data/favorita/holidays.csv')
    #     dataset.add_relation("oil", ["dcoilwtico", "f3"], relation_address='../data/favorita/oil.csv')
    #     dataset.add_relation("transactions", ["transactions", "f5"], relation_address='../data/favorita/transactions.csv')
    #     dataset.add_relation("stores", ["city", "state", "stype", "cluster", "f4"], relation_address='../data/favorita/stores.csv')
    #     dataset.add_relation("items", ["family", "class", "perishable", "f1"], relation_address='../data/favorita/items.csv')
    #     dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
    #     dataset.add_join("sales", "transactions", ["tid"], ["tid"])
    #     dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
    #     dataset.add_join("transactions", "holidays", ["date"], ["date"])
    #     dataset.add_join("holidays", "oil", ["date"], ["date"])
    #
    #     reg = GradientBoosting(learning_rate=1, max_leaves=2 ** depth, max_depth=depth, iteration=iteration)
    #
    #     reg.fit(dataset)
    #     reg_prediction = reg.predict(data='train', input_mode=1)
    #
    #     data = pd.read_csv('../data/favorita/train_small.csv')
    #     clf = GradientBoostingRegressor(max_depth=depth, learning_rate=1, n_estimators=iteration)
    #     clf = clf.fit(data[x], data[y])
    #     clf_prediction = clf.predict(data[x])
    #     for i in range(len(reg.model_def)):
    #         for j in range(len(reg.model_def[i])):
    #             print(reg.model_def[i][j])
    #     mse = mean_squared_error(data[y], clf_prediction)
    #     _reg_rmse = reg.compute_rmse('train')[0]
    #
    #     self.assertTrue(abs(_reg_rmse - math.sqrt(mse)) < 1e-3)
    #     self.assertTrue(np.sum(np.abs(reg_prediction - clf_prediction)) < 1e-3)

    @pytest.mark.skip(reason="compute_rmse is not implemented yet")
    def test_sample_syn(self):
        data = pd.read_csv("../data/synthetic/RST.csv")
        con = duckdb.connect(database=':memory:')
        x = ["A", "B", "D", "E", "F"]
        y = "H"

        exe = PandasExecutor(con, debug=True)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', ['B', 'D'], y='H', relation_address='../data/synthetic/R.csv')
        dataset.add_relation('S', ['A', 'E'], relation_address='../data/synthetic/S.csv')
        dataset.add_relation('T', ['F'], relation_address='../data/synthetic/T.csv')
        dataset.add_join("R", "S", ["A"], ["A"])
        dataset.add_join("R", "T", ["B"], ["B"])

        depth = 2
        reg = RandomForest(max_leaves=2 ** depth,
                           max_depth=depth,
                           subsample=0.5,
                           iteration=2)

        reg.fit(dataset)

        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(data[x], data[y])
        for i in range(len(reg.model_def)):
            for j in range(len(reg.model_def[i])):
                print(reg.model_def[i][j])
        mse = mean_squared_error(data[y], clf.predict(data[x]))
        # the training data is sampled, but the accuracy should still be similar
        print(reg.compute_rmse('train')[0])
        print(math.sqrt(mse))


if __name__ == '__main__':
    unittest.main()
