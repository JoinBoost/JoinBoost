import duckdb

from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree,GradientBoosting


con = duckdb.connect(database=':memory:')
# con.execute("CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../test_data/favorita/holidays_renamed_sc_cols.csv';")
con.execute("CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../test_data/favorita/holidays.csv';")

con.execute("CREATE OR REPLACE TABLE oil AS SELECT * FROM '../test_data/favorita/oil.csv';")
con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM '../test_data/favorita/transactions.csv';")
con.execute("CREATE OR REPLACE TABLE stores AS SELECT * FROM '../test_data/favorita/stores.csv';")
con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM '../test_data/favorita/items.csv';")
con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM '../test_data/favorita/sales_small.csv';")
con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM '../test_data/favorita/train_small.csv';")

exe = DuckdbExecutor(con, debug=False)
dataset = JoinGraph(exe=exe)
dataset.add_relation("sales", [], y = 'Y')
# dataset.add_relation("holidays", ["s", "c", "locale_name", "transferred","f2"])
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
print(reg.compute_rmse('train')[0])
