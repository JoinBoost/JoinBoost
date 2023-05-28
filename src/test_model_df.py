import math
import unittest

import pandas as pd
import duckdb
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from joinboost.app import DecisionTree, GradientBoosting, RandomForest
from joinboost.executor import PandasExecutor
from joinboost.joingraph import JoinGraph


class TestExecutor(unittest.TestCase):

    def test_synthetic(self):
        join = pd.read_csv("../data/synthetic/RST.csv")
        con = duckdb.connect(database=":memory:")
        con.execute("CREATE TABLE test AS SELECT * FROM '../data/synthetic/RST.csv'")
        
        x = ["A", "B", "D", "E", "F"]
        y = "H"

        exe = PandasExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation("R", ["B", "D"], y="H", relation_address="../data/synthetic/R.csv")
        dataset.add_relation("S", ["A", "E"], relation_address="../data/synthetic/S.csv")
        dataset.add_relation("T", ["F"], relation_address="../data/synthetic/T.csv")
        dataset.add_join("R", "S", ["A"], ["A"])
        dataset.add_join("R", "T", ["B"], ["B"])

        depth = 3
        gb = DecisionTree(learning_rate=1, max_leaves=2**depth, max_depth=depth)

        gb.fit(dataset)
        gb._build_model_legacy()
        
        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(join[x], join[y])
        mse = mean_squared_error(join[y], clf.predict(join[x]))
        expected_model_def = [
            (-5671945.596612875, ['A <= 3197.0', 'A <= 1981.0', 'D <= 963.0']),
            (-8322336.378526217, ['A <= 3197.0', 'A <= 1981.0', 'D > 963.0']),
            (14692866.08994809, ['A > 3197.0', 'A > 4134.0', 'A > 4587.0']),
            (579584.8480680565, ['A <= 3197.0', 'A > 1981.0', 'A > 2770.0']),
        (10730215.776843822, ['A > 3197.0', 'A > 4134.0', 'A <= 4587.0']),
        (-2624682.6485183574, ['A <= 3197.0', 'A > 1981.0', 'A <= 2770.0']),
        (3546818.3957054005, ['A > 3197.0', 'A <= 4134.0', 'A <= 3667.0']),
        (6931916.088615404, ['A > 3197.0', 'A <= 4134.0', 'A > 3667.0']),
        ]
        for i in range(len(gb.model_def)):
            for j in range(len(gb.model_def[i])):
                self.assertTrue(abs(gb.model_def[i][j][0] - expected_model_def[j][0]) < 1e-3)
                print(gb.model_def[i][j])

#         self.assertTrue(abs(gb.compute_rmse("test")[0] - math.sqrt(mse)) < 1e-3)

    def test_favorita(self):
        con = duckdb.connect(database=":memory:")

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
#             "class",
            "perishable",
            "f1",
        ]

        exe = PandasExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation(
            "sales", [], y="Y", relation_address="../data/favorita/sales_small.csv"
        )
        dataset.add_relation(
            "holidays",
            ["htype", "locale", "locale_name", "transferred", "f2"],
            relation_address="../data/favorita/holidays.csv",
        )
        dataset.add_relation(
            "oil", ["dcoilwtico", "f3"], relation_address="../data/favorita/oil.csv"
        )
        dataset.add_relation(
            "transactions",
            ["transactions", "f5"],
            relation_address="../data/favorita/transactions.csv",
        )
        dataset.add_relation(
            "stores",
            ["city", "state", "stype", "cluster", "f4"],
            relation_address="../data/favorita/stores.csv",
        )
        dataset.add_relation(
            "items",
            # TODO: python reserved word can't be used as features
            # ["family", "class", "perishable", "f1"],
            ["family", "perishable", "f1"],
            relation_address="../data/favorita/items.csv",
        )
        dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("transactions", "holidays", ["date"], ["date"])
        dataset.add_join("holidays", "oil", ["date"], ["date"])

        depth = 3
        reg = DecisionTree(learning_rate=1, max_leaves=2**depth, max_depth=depth)

        reg.fit(dataset)
        reg._build_model_legacy()

        data = pd.read_csv("../data/favorita/train_small.csv")
        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(data[x], data[y])
        expected_model_def = [
        [(-2337.9228084177735, ['f4 > 496', 'f5 <= 553', 'f3 <= 475']), (
        -7160.971070371713, ['f4 > 496', 'f5 <= 553', 'f3 > 475']), (
        -2060.958149762831, ['f4 <= 496', 'f5 <= 557', 'f3 > 475']), (
        -1953.1974409598786, ['f4 > 496', 'f5 > 553', 'f3 > 450']), (
        3039.5826387332845, ['f4 <= 496', 'f5 <= 557', 'f3 <= 475']), (
        3077.0322614426113, ['f4 > 496', 'f5 > 553', 'f3 <= 450']), (
        8135.210187401269, ['f4 <= 496', 'f5 > 557', 'f3 <= 463']), (
        3187.9458062845265, ['f4 <= 496', 'f5 > 557', 'f3 > 463'])],
        [(1283.702144280435, ['f5 > 229', 'f4 <= 797', 'f3 <= 774']), (
        -478.6770702231717, ['f5 > 229', 'f4 <= 797', 'f3 > 774']), (
        -868.3545071360177, ['f5 <= 229', 'f4 <= 829', 'f3 <= 815']), (
        -2596.802972185131, ['f5 > 229', 'f4 > 797', 'f3 > 781']), (
        -553.9671029322164, ['f5 > 229', 'f4 > 797', 'f3 <= 781']), (
        -2889.972196029417, ['f5 <= 229', 'f4 <= 829', 'f3 > 815']), (
        -2787.542235779109, ['f5 <= 229', 'f4 > 829', 'f3 <= 777']), (
        -4727.273559858574, ['f5 <= 229', 'f4 > 829', 'f3 > 777'])],
        [(-635.0822083124633, ['f4 > 115', 'f4 <= 496', 'f4 <= 326']), (
        -2566.2580076237236, ['f4 > 115', 'f4 <= 496', 'f4 > 326']), (
        38.48773229020168, ['f4 > 115', 'f4 > 496', 'f4 > 604']), (
        1446.281210767631, ['f4 > 115', 'f4 > 496', 'f4 <= 604']), (
        1634.9714863857482, ['f4 <= 115', 'f3 > 166', 'f3 > 471']), (
        26.7753578224185, ['f4 <= 115', 'f3 > 166', 'f3 <= 471']), (
        2314.8470149268637, ['f4 <= 115', 'f3 <= 166', 'f5 <= 866']), (
        3684.8484795115146, ['f4 <= 115', 'f3 <= 166', 'f5 > 866'])]
            ]
        for i in range(len(reg.model_def)):
            for j in range(len(reg.model_def[i])):
                self.assertTrue(abs(reg.model_def[i][j][0] - expected_model_def[i][j][0]) < 1e-3)
                print(reg.model_def[i][j])
        # mse = mean_squared_error(data[y], clf.predict(data[x]))
        # self.assertTrue(abs(reg.compute_rmse("train")[0] - math.sqrt(mse)) < 1e-3)

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
        con = duckdb.connect(database=":memory:")
        x = ["A", "B", "D", "E", "F"]
        y = "H"

        exe = PandasExecutor(con, debug=True)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation(
            "R", ["B", "D"], y="H", relation_address="../data/synthetic/R.csv"
        )
        dataset.add_relation(
            "S", ["A", "E"], relation_address="../data/synthetic/S.csv"
        )
        dataset.add_relation("T", ["F"], relation_address="../data/synthetic/T.csv")
        dataset.add_join("R", "S", ["A"], ["A"])
        dataset.add_join("R", "T", ["B"], ["B"])

        depth = 2
        reg = RandomForest(
            max_leaves=2**depth, max_depth=depth, subsample=0.5, iteration=2
        )

        reg.fit(dataset)
        reg._build_model_legacy()

        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(data[x], data[y])
        for i in range(len(reg.model_def)):
            for j in range(len(reg.model_def[i])):
                print(reg.model_def[i][j])
        mse = mean_squared_error(data[y], clf.predict(data[x]))
        # the training data is sampled, but the accuracy should still be similar
        print(reg.compute_rmse("train")[0])
        print(math.sqrt(mse))


if __name__ == "__main__":
    unittest.main()
