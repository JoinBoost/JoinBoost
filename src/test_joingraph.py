import unittest
import math
import time
import pandas as pd
import duckdb
import lightgbm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joinboost.executor import DuckdbExecutor, ExecutorException
from joinboost.joingraph import JoinGraph, JoinGraphException
from joinboost.app import DecisionTree,GradientBoosting,RandomForest

class TestJoingraph(unittest.TestCase):
        
    def test_cycle(self):
        R = pd.DataFrame(columns=['A', 'B'])
        S = pd.DataFrame(columns=['B', 'C'])
        T = pd.DataFrame(columns=['A', 'C'])
    
        dataset = JoinGraph()
        try:
            dataset.add_relation('R', ['B', 'E'], relation_address=R)
            raise JoinGraphException('Attribute not in the relation but is allowed!')
        except:
            pass
        dataset.add_relation('R', ['B', 'A'], y='B', relation_address=R)
        dataset.add_relation('S', ['B', 'C'], relation_address=S)
        dataset.add_relation('T', ['A', 'C'], relation_address=T)
        
        try:
            dataset._preprocess()
            raise Exception('Disjoint join graph is allowed!')
        except JoinGraphException:
            pass
        
        dataset.add_join("R", "S", ["B"], ["B"])
        dataset.add_join("S", "T", ["C"], ["C"])
        dataset._preprocess()
        
        # TODO: check join keys available
        # dataset.add_join("R", "S", ["A"], ["A"])
        
        dataset.add_join("R", "T", ["A"], ["A"])
        try:
            dataset._preprocess()
            raise Exception('Cyclic join graph is allowed!')
        except JoinGraphException:
            pass
    
if __name__ == '__main__':
    unittest.main()