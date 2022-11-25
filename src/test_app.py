import duckdb

from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree,GradientBoosting
import unittest

class TestApp(unittest.TestCase):
	def setUp(self):
		con = duckdb.connect(database=':memory:')
		con.execute("CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../test_data/favorita/holidays.csv';")
		con.execute("CREATE OR REPLACE TABLE holidays_renamed_sc_cols AS SELECT * FROM '../test_data/favorita/holidays_renamed_sc_cols.csv';")
		con.execute("CREATE OR REPLACE TABLE oil AS SELECT * FROM '../test_data/favorita/oil.csv';")
		con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM '../test_data/favorita/transactions.csv';")
		con.execute("CREATE OR REPLACE TABLE stores AS SELECT * FROM '../test_data/favorita/stores.csv';")
		con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM '../test_data/favorita/items.csv';")
		con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM '../test_data/favorita/sales_small.csv';")
		con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM '../test_data/favorita/train_small.csv';")
		exe = DuckdbExecutor(con, debug=False)
		self.dataset1 = JoinGraph(exe=exe)
		self.dataset2 = JoinGraph(exe=exe)

		self.dataset1.add_relation("sales", [], y = 'Y')
		self.dataset1.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"])
		self.dataset1.add_relation("oil", ["dcoilwtico","f3"])
		self.dataset1.add_relation("transactions", ["transactions","f5"])
		self.dataset1.add_relation("stores", ["city","state","stype","cluster","f4"])
		self.dataset1.add_relation("items", ["family","class","perishable","f1"])
		self.dataset1.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
		self.dataset1.add_join("sales", "transactions", ["tid"], ["tid"])
		self.dataset1.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
		self.dataset1.add_join("transactions", "holidays", ["date"], ["date"])
		self.dataset1.add_join("holidays", "oil", ["date"], ["date"])
		
		self.dataset2.add_relation("sales", [], y = 'Y')
		self.dataset2.add_relation("holidays_renamed_sc_cols", ["joinboost_preserved_s", "c", "locale_name", "transferred","f2"])
		self.dataset2.add_relation("oil", ["dcoilwtico","f3"])
		self.dataset2.add_relation("transactions", ["transactions","f5"])
		self.dataset2.add_relation("stores", ["city","state","stype","cluster","f4"])
		self.dataset2.add_relation("items", ["family","class","perishable","f1"])
		self.dataset2.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
		self.dataset2.add_join("sales", "transactions", ["tid"], ["tid"])
		self.dataset2.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
		self.dataset2.add_join("transactions", "holidays_renamed_sc_cols", ["date"], ["date"])
		self.dataset2.add_join("holidays_renamed_sc_cols", "oil", ["date"], ["date"])

	def test_add_prefix_to_preserved_sc_columns(self):
		depth = 3
		reg1 = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)
		reg1.fit(self.dataset1)
		reg2 = DecisionTree(learning_rate=1, max_leaves=2 ** depth, max_depth=depth)
		reg2.fit(self.dataset2)
		
		print(reg1.compute_rmse('train')[0])
		print(reg2.compute_rmse('train')[0])
		self.assertTrue(reg1.compute_rmse('train')[0] == reg2.compute_rmse('train')[0])

if __name__ == '__main__':
	unittest.main()
