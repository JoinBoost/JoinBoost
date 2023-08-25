import unittest
import math
import pandas as pd
import numpy as np
import duckdb
import lightgbm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joinboost.executor import DuckdbExecutor, PandasExecutor, SparkExecutor
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree, GradientBoosting, RandomForest
from pyspark.sql import SparkSession



class TestModel(unittest.TestCase):

    def test_synthetic(self):
        spark = SparkSession.builder.appName("myApp").getOrCreate()

        exe = SparkExecutor(spark, debug=True)

        dataset = JoinGraph(exe=exe)
        dataset.add_relation('R', ['B', 'D'], y='H', relation_address='../data/synthetic/R.csv')
        dataset.add_relation('S', ['A', 'E'], relation_address='../data/synthetic/S.csv')
        dataset.add_relation('T', ['F'], relation_address='../data/synthetic/T.csv')
        dataset.add_join("R", "S", ["A"], ["A"])
        dataset.add_join("R", "T", ["B"], ["B"])

        depth = 2
        gb = DecisionTree(learning_rate=1, num_leaves=2**depth, max_depth=depth, debug=True)

        gb.fit(dataset)
        
        spark.read.csv("../data/synthetic/RST.csv", header=True).createOrReplaceTempView("test")
        
        join = pd.read_csv("../data/synthetic/RST.csv")
        x = ["A", "B", "D", "E", "F"]
        y = "H"
        clf = DecisionTreeRegressor(max_depth=depth)
        clf = clf.fit(join[x], join[y])
        mse = mean_squared_error(join[y], clf.predict(join[x]))
        self.assertTrue(abs(gb.compute_rmse("test")[0] - math.sqrt(mse)) < 1e-3)

if __name__ == "__main__":
    unittest.main()
