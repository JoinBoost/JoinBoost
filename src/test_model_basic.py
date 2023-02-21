import unittest
import math
import time
import pandas as pd
import numpy as np
import duckdb
import lightgbm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree,GradientBoosting,RandomForest

class TestModel(unittest.TestCase):
        
    def test_synthetic(self):
        join = pd.read_csv("../data/synthetic/RST.csv")
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE TABLE R AS SELECT * FROM '../data/synthetic/R.csv'")
        con.execute("CREATE TABLE S AS SELECT * FROM '../data/synthetic/S.csv'")
        con.execute("CREATE TABLE T AS SELECT * FROM '../data/synthetic/T.csv'")
        con.execute("CREATE TABLE test AS SELECT * FROM '../data/synthetic/RST.csv'")
        x = ["A", "B", "D", "E", "F"]
        y = "H"

        exe = DuckdbExecutor(con, debug=False)
    
        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', ['B', 'D'], y = 'H')
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
        x = ["htype", "locale", "locale_name", "transferred","f2","dcoilwtico","f3","transactions",
             "f5","city","state","stype","cluster","f4","family","class","perishable","f1"]

        exe = DuckdbExecutor(con, debug=False)
    
        dataset = JoinGraph(exe=exe)
        dataset.add_relation("sales", [], y = 'Y')
        dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"])
        dataset.add_relation("oil", ["dcoilwtico","f3"])
        dataset.add_relation("transactions", ["transactions","f5"])
        dataset.add_relation("stores", ["city","state","stype","cluster","f4"])
        dataset.add_relation("items", ["family","class","perishable","f1"])
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
        
    def test_gradient_boosting(self):
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../data/favorita/holidays.csv';")
        con.execute("CREATE OR REPLACE TABLE oil AS SELECT * FROM '../data/favorita/oil.csv';")
        con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM '../data/favorita/transactions.csv';")
        con.execute("CREATE OR REPLACE TABLE stores AS SELECT * FROM '../data/favorita/stores.csv';")
        con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM '../data/favorita/items.csv';")
        con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM '../data/favorita/sales_small.csv';")
        con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM '../data/favorita/train_small.csv';")

        y = "Y"
        x = ["htype", "locale", "locale_name", "transferred","f2","dcoilwtico","f3","transactions",
             "f5","city","state","stype","cluster","f4","family","class","perishable","f1"]

        exe = DuckdbExecutor(con, debug=False)
        depth = 3
        iteration = 3
        dataset = JoinGraph(exe=exe)
        dataset.add_relation("sales", [], y = 'Y')
        dataset.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"])
        dataset.add_relation("oil", ["dcoilwtico","f3"])
        dataset.add_relation("transactions", ["transactions","f5"])
        dataset.add_relation("stores", ["city","state","stype","cluster","f4"])
        dataset.add_relation("items", ["family","class","perishable","f1"])
        dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset.add_join("transactions", "holidays", ["date"], ["date"])
        dataset.add_join("holidays", "oil", ["date"], ["date"])

        reg = GradientBoosting(learning_rate=1, max_leaves=2 ** depth, max_depth=depth, iteration = iteration)

        reg.fit(dataset)
        reg_prediction = reg.predict(data='train', input_mode=1)
        
        data = pd.read_csv('../data/favorita/train_small.csv')
        clf = GradientBoostingRegressor(max_depth=depth,learning_rate=1, n_estimators=iteration)
        clf = clf.fit(data[x], data[y])
        clf_prediction = clf.predict(data[x])
        mse = mean_squared_error(data[y], clf_prediction)
        _reg_rmse = reg.compute_rmse('train')[0]

        self.assertTrue(abs(_reg_rmse - math.sqrt(mse)) < 1e-3)
        self.assertTrue(np.sum(np.abs(reg_prediction - clf_prediction)) < 1e-3)
    
    def test_sample_syn(self):
        data = pd.read_csv("../data/synthetic/RST.csv")
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE TABLE R AS SELECT * FROM '../data/synthetic/R.csv'")
        con.execute("CREATE TABLE S AS SELECT * FROM '../data/synthetic/S.csv'")
        con.execute("CREATE TABLE T AS SELECT * FROM '../data/synthetic/T.csv'")
        con.execute("CREATE TABLE train AS SELECT * FROM '../data/synthetic/RST.csv'")
        x = ["A", "B", "D", "E", "F"]
        y = "H"

        exe = DuckdbExecutor(con, debug=False)
    
        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', ['B', 'D'], y = 'H')
        dataset.add_relation('S', ['A', 'E'])
        dataset.add_relation('T', ['F'])
        dataset.add_join("R", "S", ["A"], ["A"])
        dataset.add_join("R", "T", ["B"], ["B"])

        depth = 2
        reg = RandomForest(max_leaves=2 ** depth, 
                           max_depth=depth, 
                           subsample=0.5, 
                           iteration = 2)

        reg.fit(dataset)

        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(data[x], data[y])
        mse = mean_squared_error(data[y], clf.predict(data[x]))
        # the training data is sampled, but the accuracy should still be similar
        print(reg.compute_rmse('train')[0])
        print(math.sqrt(mse))
    
    def test_lightgbm_catigorial(self):
        R = pd.read_csv("../data/synthetic-very-small/R.csv")
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE TABLE R AS SELECT * FROM R")
        x = ["D"]
        y = "H"

        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', categorical_feature=['D'], y = 'H')

        iteration = 1
        depth = 1

        reg = GradientBoosting(learning_rate=1, 
                               max_depth=depth, 
                               iteration = iteration)

        reg.fit(dataset)

        clf = lightgbm.LGBMRegressor(learning_rate=1, 
                                     max_depth=depth, 
                                     min_child_samples=1, 
                                     objective='RMSE', 
                                     cat_l2=0, 
                                     cat_smooth=0, 
                                     deterministic=True,
                                     max_bin=20000, 
                                     min_data_in_bin=1, 
                                     n_estimators=iteration, 
                                     max_cat_threshold=10000, 
                                     max_cat_to_onehot=1, 
                                     min_data_per_group=1)

        clf.fit(R[x], R[y], feature_name=x, categorical_feature=x)
        mse = mean_squared_error(R[y], clf.predict(R[x]))
        
        self.assertTrue(reg.compute_rmse('R')[0] < math.sqrt(mse))

if __name__ == '__main__':
    unittest.main()