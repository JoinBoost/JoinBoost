import unittest
import math
import pandas as pd
import duckdb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree,GradientBoosting

class TestDecision(unittest.TestCase):
        
    def test_synthetic(self):
        join = pd.read_csv("../data/synthetic/RST.csv")
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE TABLE R AS SELECT * FROM '../data/synthetic/R.csv'")
        con.execute("CREATE TABLE S AS SELECT * FROM '../data/synthetic/S.csv'")
        con.execute("CREATE TABLE T AS SELECT * FROM '../data/synthetic/T.csv'")
        con.execute("CREATE TABLE test AS SELECT * FROM '../data/synthetic/RST.csv'")
        x = ["A", "B", "D", "E", "F"]
        y = "C"

        exe = DuckdbExecutor(con, debug=False)
    
        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', ['B', 'D'], y = 'C')
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
        math.sqrt(mse)
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

        
        data = pd.read_csv('../data/favorita/train_small.csv')
        clf = GradientBoostingRegressor(max_depth=depth,learning_rate=1, n_estimators=iteration)
        clf = clf.fit(data[x], data[y])
        mse = mean_squared_error(data[y], clf.predict(data[x]))
        math.sqrt(mse)
        self.assertTrue(abs(reg.compute_rmse('train')[0] - math.sqrt(mse)) < 1e-3)

if __name__ == '__main__':
    unittest.main()