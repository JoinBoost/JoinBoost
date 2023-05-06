import duckdb

from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree, GradientBoosting
import unittest


class TestApp(unittest.TestCase):

    # this test the case when s and c are already in the databases
    # semi-ring should choose a different name
    def test_add_prefix_to_reserved_sc_columns(self):
        con = duckdb.connect(database=":memory:")
        con.execute(
            "CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../data/favorita/holidays.csv';"
        )
        con.execute(
            """CREATE OR REPLACE TABLE holidays_renamed_sc_cols AS 
                    SELECT date, htype AS joinboost_reserved_s, locale AS c, locale_name, transferred, f2
                    FROM holidays;
                    """
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
            """CREATE OR REPLACE TABLE sales_renamed_sc_cols AS 
                    SELECT item_nbr, unit_sales, onpromotion, tid, Y AS s
                    FROM sales;
                    """
        )
        con.execute(
            "CREATE OR REPLACE TABLE train AS SELECT * FROM '../data/favorita/train_small.csv';"
        )
        exe = DuckdbExecutor(con, debug=False)
        dataset1 = JoinGraph(exe=exe)
        dataset2 = JoinGraph(exe=exe)

        dataset1.add_relation("sales", [], y = 'Y', relation_address="../data/favorita/sales_small.csv")
        con.execute("""CREATE OR REPLACE TABLE sales_renamed_sc_cols AS 
                    SELECT item_nbr, unit_sales, onpromotion, tid, Y AS s
                    FROM sales;
                    """)
        dataset1.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"], relation_address="../data/favorita/holidays.csv")
        dataset1.add_relation("oil", ["dcoilwtico","f3"], relation_address="../data/favorita/oil.csv")
        dataset1.add_relation("transactions", ["transactions","f5"], relation_address="../data/favorita/transactions.csv")
        dataset1.add_relation("stores", ["city","state","stype","cluster","f4"], relation_address="../data/favorita/stores.csv")
        dataset1.add_relation("items", ["family","class","perishable","f1"], relation_address="../data/favorita/items.csv")
        dataset1.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset1.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset1.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset1.add_join("transactions", "holidays", ["date"], ["date"])
        dataset1.add_join("holidays", "oil", ["date"], ["date"])
        con.execute("""CREATE OR REPLACE TABLE holidays_renamed_sc_cols AS 
                    SELECT date, htype AS joinboost_reserved_s, locale AS c, locale_name, transferred, f2
                    FROM holidays;
                    """)

        dataset2.add_relation("sales", [], y = 'Y', relation_address="../data/favorita/sales_small.csv")
        dataset2.add_relation("holidays_renamed_sc_cols", ["joinboost_reserved_s", "c", "locale_name", "transferred","f2"], relation_address="../data/favorita/holidays.csv")
        dataset2.add_relation("oil", ["dcoilwtico","f3"], relation_address="../data/favorita/oil.csv")
        dataset2.add_relation("transactions", ["transactions","f5"], relation_address="../data/favorita/transactions.csv")
        dataset2.add_relation("stores", ["city","state","stype","cluster","f4"], relation_address="../data/favorita/stores.csv")
        dataset2.add_relation("items", ["family","class","perishable","f1"], relation_address="../data/favorita/items.csv")
        dataset2.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset2.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset2.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset2.add_join(
            "transactions", "holidays_renamed_sc_cols", ["date"], ["date"]
        )
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
        con = duckdb.connect(database=":memory:")
        con.execute(
            "CREATE OR REPLACE TABLE holidays AS SELECT * FROM '../data/favorita/holidays.csv';"
        )
        con.execute(
            """CREATE OR REPLACE TABLE holidays_renamed_sc_cols AS 
                    SELECT date, htype AS joinboost_reserved_s, locale AS c, locale_name, transferred, f2
                    FROM holidays;
                    """
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
            """CREATE OR REPLACE TABLE sales_renamed_sc_cols AS 
                    SELECT item_nbr, unit_sales, onpromotion, tid, Y AS s
                    FROM sales;
                    """
        )
        con.execute(
            "CREATE OR REPLACE TABLE train AS SELECT * FROM '../data/favorita/train_small.csv';"
        )
        con.execute(
            """CREATE OR REPLACE TABLE train_renamed AS 
                    SELECT Y AS s,onpromotion,htype AS joinboost_reserved_s,locale AS c,locale_name,transferred,f2,date,dcoilwtico,f3,tid,transactions,f5,store_nbr,city,state,stype,cluster,f4,item_nbr,family,class,perishable,f1,unit_sales
                    FROM train;
                    """
        )

        exe = DuckdbExecutor(con, debug=False)
        dataset1 = JoinGraph(exe=exe)
        dataset2 = JoinGraph(exe=exe)

        dataset1.add_relation("sales", [], y = 'Y', relation_address="../data/favorita/sales_small.csv")
        dataset1.add_relation("holidays", ["htype", "locale", "locale_name", "transferred","f2"], relation_address="../data/favorita/holidays.csv")
        dataset1.add_relation("oil", ["dcoilwtico","f3"], relation_address="../data/favorita/oil.csv")
        dataset1.add_relation("transactions", ["transactions","f5"], relation_address="../data/favorita/transactions.csv")
        dataset1.add_relation("stores", ["city","state","stype","cluster","f4"], relation_address="../data/favorita/stores.csv")
        dataset1.add_relation("items", ["family","class","perishable","f1"], relation_address="../data/favorita/items.csv")
        dataset1.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
        dataset1.add_join("sales", "transactions", ["tid"], ["tid"])
        dataset1.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset1.add_join("transactions", "holidays", ["date"], ["date"])
        dataset1.add_join("holidays", "oil", ["date"], ["date"])

        dataset2.add_relation("sales_renamed_sc_cols", [], y = 's', relation_address="../data/favorita/sales_small.csv")
        dataset2.add_relation("holidays_renamed_sc_cols", ["joinboost_reserved_s", "c", "locale_name", "transferred","f2"], relation_address="../data/favorita/holidays.csv")
        dataset2.add_relation("oil", ["dcoilwtico","f3"], relation_address="../data/favorita/oil.csv")
        dataset2.add_relation("transactions", ["transactions","f5"], relation_address="../data/favorita/transactions.csv")
        dataset2.add_relation("stores", ["city","state","stype","cluster","f4"], relation_address="../data/favorita/stores.csv")
        dataset2.add_relation("items", ["family","class","perishable","f1"], relation_address="../data/favorita/items.csv")
        dataset2.add_join("sales_renamed_sc_cols", "items", ["item_nbr"], ["item_nbr"])
        dataset2.add_join("sales_renamed_sc_cols", "transactions", ["tid"], ["tid"])
        dataset2.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
        dataset2.add_join(
            "transactions", "holidays_renamed_sc_cols", ["date"], ["date"]
        )
        dataset2.add_join("holidays_renamed_sc_cols", "oil", ["date"], ["date"])

        depth = 3
        reg1 = DecisionTree(learning_rate=1, max_leaves=2**depth, max_depth=depth)
        reg1.fit(dataset1)
        reg2 = DecisionTree(learning_rate=1, max_leaves=2**depth, max_depth=depth)
        reg2.fit(dataset2)

        rmse1 = reg1.compute_rmse("train")[0]
        rmse2 = reg2.compute_rmse("train_renamed")[0]

        print()
        print(rmse1)
        print(rmse2)
        print(reg2.preprocessor.get_history())
        self.assertTrue(rmse1 == rmse2)


if __name__ == "__main__":
    unittest.main()
