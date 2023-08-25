import unittest
import pandas as pd
import duckdb
from joinboost.executor import DuckdbExecutor
from joinboost.joingraph import JoinGraph, JoinGraphException
from joinboost.app import DecisionTree, GradientBoosting, RandomForest


class TestModel(unittest.TestCase):

    def test_missing_target(self):
        con = duckdb.connect(database=':memory:')
        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', X=["A", "D", "H"], relation_address='../data/synthetic-one-to-many/R.csv')
        dataset.add_relation('S', X=["E"], relation_address='../data/synthetic-one-to-many/S.csv')
        dataset.add_relation('T', X=["K"], relation_address='../data/synthetic-one-to-many/T.csv')
        dataset.add_join('R', 'T', ['B'], ['B'])
        dataset.add_join('S', 'T', ['F'], ['F'])

        depth = 2
        reg = RandomForest(
            max_leaves=2**depth, max_depth=depth, subsample=0.5, iteration=2
        )
        
        try:
            reg.fit(dataset)
            raise Exception("Missing target relation is allowed!")
        except JoinGraphException:
            pass
        
    def test_no_missing_target(self):
        con = duckdb.connect(database=':memory:')
        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', X=["A", "D"], y="H", relation_address='../data/synthetic-one-to-many/R.csv')
        dataset.add_relation('S', X=["E"], relation_address='../data/synthetic-one-to-many/S.csv')
        dataset.add_relation('T', X=["K"], relation_address='../data/synthetic-one-to-many/T.csv')
        dataset.add_join('R', 'T', ['B'], ['B'])
        dataset.add_join('S', 'T', ['F'], ['F'])

        depth = 2
        reg = RandomForest(
            max_leaves=2**depth, max_depth=depth, subsample=0.5, iteration=2
        )
        reg.fit(dataset)
        
    def test_missing_feature(self):
        con = duckdb.connect(database=':memory:')
        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', y="H", relation_address='../data/synthetic-one-to-many/R.csv')
        dataset.add_relation('S', relation_address='../data/synthetic-one-to-many/S.csv')
        dataset.add_relation('T', relation_address='../data/synthetic-one-to-many/T.csv')
        dataset.add_join('R', 'T', ['B'], ['B'])
        dataset.add_join('S', 'T', ['F'], ['F'])

        depth = 2
        reg = DecisionTree(learning_rate=1, num_leaves=2**depth, max_depth=depth)
        reg.fit(dataset)
        reg.compute_rmse("R")[0]

    def test_missing_join1(self):
        con = duckdb.connect(database=':memory:')
        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', X=["A", "D"], y="H", relation_address='../data/synthetic-missing1/R.csv')
        dataset.add_relation('S', X=["E"], relation_address='../data/synthetic-missing1/S.csv')
        dataset.add_join('R', 'S', ['B'], ['B'])

        depth = 2
        reg = DecisionTree(learning_rate=1, num_leaves=2**depth, max_depth=depth)
        reg.fit(dataset)
        
    def test_missing_join2(self):
        con = duckdb.connect(database=':memory:')
        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', X=["A", "D"], y="H", relation_address='../data/synthetic-missing2/R.csv')
        dataset.add_relation('S', X=["E"], relation_address='../data/synthetic-missing2/S.csv')
        dataset.add_join('R', 'S', ['B'], ['B'])

        depth = 2
        reg = DecisionTree(learning_rate=1, num_leaves=2**depth, max_depth=depth)
        
        try:
            reg.fit(dataset)
            raise Exception("Missing join key is allowed!")
        except JoinGraphException:
            pass
        
    def test_many_to_many(self):
        con = duckdb.connect(database=':memory:')
        exe = DuckdbExecutor(con, debug=False)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', X=["D"], y="H",
                     relation_address='../data/synthetic-many-to-many/R.csv')
        dataset.add_relation('S', X=["E"],
                     relation_address='../data/synthetic-many-to-many/S.csv')
        dataset.add_relation('T', X=["F"],
                     relation_address='../data/synthetic-many-to-many/T.csv')

        depth = 2
        reg = DecisionTree(learning_rate=1, num_leaves=2**depth, max_depth=depth)
        
        try:
            reg.fit(dataset)
            raise Exception("Many-to-many is allowed!")
        except JoinGraphException:
            pass
        
    
        
if __name__ == "__main__":
    unittest.main()
