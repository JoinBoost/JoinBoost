import duckdb

from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree,GradientBoosting
import unittest

class TestApp(unittest.TestCase):    
    # this test the case when s and c are already in the databases
    # semi-ring should choose a different name
    def test_add_prefix_to_preserved_sc_columns(self):
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../data/favorita/holidays.csv';")
        con.execute("CREATE OR REPLACE TABLE holidays_renamed_sc_cols AS SELECT * FROM '../data/favorita/holidays_renamed_sc_cols.csv';")
        con.execute("CREATE OR REPLACE TABLE oil AS SELECT * FROM '../data/favorita/oil.csv';")
        con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM '../data/favorita/transactions.csv';")
        con.execute("CREATE OR REPLACE TABLE stores AS SELECT * FROM '../data/favorita/stores.csv';")
        con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM '../data/favorita/items.csv';")
        con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM '../data/favorita/sales_small.csv';")
        con.execute("CREATE OR REPLACE TABLE sales_renamed_sc_cols AS SELECT * FROM '../data/favorita/sales_small_renamed_sc_cols.csv';")
        con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM '../data/favorita/train_small.csv';")
        exe = DuckdbExecutor(con, debug=False)
        dataset1 = JoinGraph(exe=exe)
        dataset2 = JoinGraph(exe=exe)

        dataset1.add_relation("sales", [], y = 'Y')
        dataset1.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"])
        dataset1.add_relation("oil", ["dcoilwtico","f3"])
        dataset1.add_relation("transactions", ["transactions","f5"])
        dataset1.add_relation("stores", ["city","state","stype","cluster","f4"])
        dataset1.add_relation("items", ["family","class","perishable","f1"])
        dataset1.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset1.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset1.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset1.add_join("transactions", "holidays", ["date"], ["date"])
        dataset1.add_join("holidays", "oil", ["date"], ["date"])
        
        dataset2.add_relation("sales", [], y = 'Y')
        dataset2.add_relation("holidays_renamed_sc_cols", ["joinboost_preserved_s", "c", "locale_name", "transferred","f2"])
        dataset2.add_relation("oil", ["dcoilwtico","f3"])
        dataset2.add_relation("transactions", ["transactions","f5"])
        dataset2.add_relation("stores", ["city","state","stype","cluster","f4"])
        dataset2.add_relation("items", ["family","class","perishable","f1"])
        dataset2.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset2.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset2.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset2.add_join("transactions", "holidays_renamed_sc_cols", ["date"], ["date"])
        dataset2.add_join("holidays_renamed_sc_cols", "oil", ["date"], ["date"])
        
        depth = 3     
        reg1 = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)
        reg1.fit(dataset1)
        reg2 = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)
        reg2.fit(dataset2)
        
        rmse1 = reg1.compute_rmse('train')[0]
        rmse2 = reg2.compute_rmse('train')[0]
        
        print(rmse1)
        print(rmse2)
        self.assertTrue(rmse1 == rmse2)
        # this test the case when s and c are already in the databases
        
    def test_add_prefix_to_target_variable(self):
        con = duckdb.connect(database=':memory:')
        con.execute("CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../data/favorita/holidays.csv';")
        con.execute("CREATE OR REPLACE TABLE holidays_renamed_sc_cols AS SELECT * FROM '../data/favorita/holidays_renamed_sc_cols.csv';")
        con.execute("CREATE OR REPLACE TABLE oil AS SELECT * FROM '../data/favorita/oil.csv';")
        con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM '../data/favorita/transactions.csv';")
        con.execute("CREATE OR REPLACE TABLE stores AS SELECT * FROM '../data/favorita/stores.csv';")
        con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM '../data/favorita/items.csv';")
        con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM '../data/favorita/sales_small.csv';")
        con.execute("CREATE OR REPLACE TABLE sales_renamed_sc_cols AS SELECT * FROM '../data/favorita/sales_small_renamed_sc_cols.csv';")
        con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM '../data/favorita/train_small.csv';")
        con.execute("CREATE OR REPLACE TABLE train_renamed AS SELECT * FROM '../data/favorita/train_small_renamed.csv';")
        
        exe = DuckdbExecutor(con, debug=False)
        dataset1 = JoinGraph(exe=exe)
        dataset2 = JoinGraph(exe=exe)

        dataset1.add_relation("sales", [], y = 'Y')
        dataset1.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"])
        dataset1.add_relation("oil", ["dcoilwtico","f3"])
        dataset1.add_relation("transactions", ["transactions","f5"])
        dataset1.add_relation("stores", ["city","state","stype","cluster","f4"])
        dataset1.add_relation("items", ["family","class","perishable","f1"])
        dataset1.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset1.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset1.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset1.add_join("transactions", "holidays", ["date"], ["date"])
        dataset1.add_join("holidays", "oil", ["date"], ["date"])
        
        dataset2.add_relation("sales_renamed_sc_cols", [], y = 's')
        dataset2.add_relation("holidays_renamed_sc_cols", ["joinboost_preserved_s", "c", "locale_name", "transferred","f2"])
        dataset2.add_relation("oil", ["dcoilwtico","f3"])
        dataset2.add_relation("transactions", ["transactions","f5"])
        dataset2.add_relation("stores", ["city","state","stype","cluster","f4"])
        dataset2.add_relation("items", ["family","class","perishable","f1"])
        dataset2.add_join("sales_renamed_sc_cols", "items", ["item_nbr"], ["item_nbr"])
        dataset2.add_join("sales_renamed_sc_cols", "transactions", ["tid"], ["tid"])
        dataset2.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset2.add_join("transactions", "holidays_renamed_sc_cols", ["date"], ["date"])
        dataset2.add_join("holidays_renamed_sc_cols", "oil", ["date"], ["date"])

        depth = 3
        reg1 = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)
        reg1.fit(dataset1)  
        reg2 = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)
        reg2.fit(dataset2)

        rmse1 = reg1.compute_rmse('train')[0]
        
        # TODO: this is hard code. Remove it.
        # keep track of the column updated because of the reserved words
        dataset2.exe.rename('train_renamed', 's', 'joinboost_reserved_s')
        rmse2 = reg2.compute_rmse('train_renamed')[0]

        print()
        print(rmse1)
        print(rmse2)
        self.assertTrue(rmse1 == rmse2)

if __name__ == '__main__':
    unittest.main()
