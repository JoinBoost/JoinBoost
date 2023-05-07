import duckdb

from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree, GradientBoosting
import unittest


def create_tables(con):
    table_files = {
        "holidays": "../data/favorita/holidays.csv",
        "oil": "../data/favorita/oil.csv",
        "transactions": "../data/favorita/transactions.csv",
        "stores": "../data/favorita/stores.csv",
        "items": "../data/favorita/items.csv",
        "sales": "../data/favorita/sales_small.csv",
        "train": "../data/favorita/train_small.csv"
    }

    for table_name, file_path in table_files.items():
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM '{file_path}';")

def create_renamed_tables(con):
    con.execute(
        """CREATE OR REPLACE TABLE holidays_renamed_sc_cols AS 
                SELECT date, htype AS joinboost_reserved_s, locale AS c, locale_name, transferred, f2
                FROM holidays;
                """
    )
    con.execute(
        """CREATE OR REPLACE TABLE sales_renamed_sc_cols AS 
                SELECT item_nbr, unit_sales, onpromotion, tid, Y AS s
                FROM sales;
                """
    )
    con.execute(
        """CREATE OR REPLACE TABLE train_renamed AS 
                SELECT Y AS s, onpromotion, htype AS joinboost_reserved_s, locale AS c, locale_name, transferred, f2, date, dcoilwtico, f3, tid, transactions, f5, store_nbr, city, state, stype, cluster, f4, item_nbr, family, class, perishable, f1, unit_sales
                FROM train;
                """
    )

class TestApp(unittest.TestCase):
    
    def setUp(self):
        self.con = duckdb.connect(database=":memory:")
        create_tables(self.con)
        create_renamed_tables(self.con)
        self.exe = DuckdbExecutor(self.con, debug=False)
        
    # this test the case when s and c are already in the databases
    # semi-ring should choose a different name
    def test_add_prefix_to_reserved_sc_columns(self):
        dataset1, dataset2 = JoinGraph(exe=self.exe), JoinGraph(exe=self.exe)

        dataset1.add_relation("sales", [], y = 'Y', relation_address="../data/favorita/sales_small.csv")
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
        dataset2.add_relation("holidays_renamed_sc_cols", ["joinboost_reserved_s", "c", "locale_name", "transferred","f2"])
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
        reg1 = DecisionTree(learning_rate=1, max_leaves=2**depth, max_depth=depth)
        reg1.fit(dataset1)
        reg2 = DecisionTree(learning_rate=1, max_leaves=2**depth, max_depth=depth)
        reg2.fit(dataset2)

        rmse1 = reg1.compute_rmse("train")[0]
        rmse2 = reg2.compute_rmse("train")[0]

        print(rmse1)
        print(rmse2)
        self.assertTrue(rmse1 == rmse2)
        # this test the case when s and c are already in the databases

    def test_add_prefix_to_target_variable(self):
        dataset1, dataset2 = JoinGraph(exe=self.exe), JoinGraph(exe=self.exe)

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
        dataset2.add_relation("holidays_renamed_sc_cols", ["joinboost_reserved_s", "c", "locale_name", "transferred","f2"])
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
        reg1 = DecisionTree(learning_rate=1, max_leaves=2**depth, max_depth=depth)
        reg1.fit(dataset1)
        reg2 = DecisionTree(learning_rate=1, max_leaves=2**depth, max_depth=depth)
        reg2.fit(dataset2)

        rmse1 = reg1.compute_rmse("train")[0]
        rmse2 = reg2.compute_rmse("train_renamed")[0]

        print(rmse1)
        print(rmse2)
        print(reg2.preprocessor.get_history())
        self.assertTrue(rmse1 == rmse2)


if __name__ == "__main__":
    unittest.main()
