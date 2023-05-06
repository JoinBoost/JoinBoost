import duckdb

from joinboost.semiring import AvgSemiRing
from joinboost.joingraph import JoinGraph
from joinboost.cjt import CJT

"""
Join Graph for data/synthetic-one_to_many/
R(ABDH) - T(BFK) - S(FE)
"""
def initialize_synthetic_one_to_many(semi_ring=AvgSemiRing()):
    duck_db_conn = duckdb.connect(database=':memory:')
    join_graph = JoinGraph(duck_db_conn)
    cjt = CJT(semi_ring=semi_ring, join_graph=join_graph)
    cjt.add_relation('R', X=["A", "D", "H"], relation_address='../data/synthetic-one-to-many/R.csv')
    cjt.add_relation('S', X=["E"], relation_address='../data/synthetic-one-to-many/S.csv')
    cjt.add_relation('T', relation_address='../data/synthetic-one-to-many/T.csv')
    cjt.add_join('R', 'T', ['B'], ['B'])
    cjt.add_join('S', 'T', ['F'], ['F'])
    return cjt

"""
Join Graph for data/synthetic-many-to-many/
S(BE) - T(BF) - R(ABDH) 
"""

def initialize_synthetic_many_to_many(semi_ring=AvgSemiRing()):
    duck_db_conn = duckdb.connect(database=':memory:')
    join_graph = JoinGraph(duck_db_conn)
    cjt = CJT(semi_ring=semi_ring, join_graph=join_graph)
    cjt.add_relation('R', X=["D", "H"],
                     relation_address='../data/synthetic-many-to-many/R.csv')
    cjt.add_relation('S', X=["E"],
                     relation_address='../data/synthetic-many-to-many/S.csv')
    cjt.add_relation('T', X=["F"],
                     relation_address='../data/synthetic-many-to-many/T.csv')
    cjt.add_join('R', 'S', ['B'], ['B'])
    cjt.add_join('S', 'T', ['B'], ['B'])
    return cjt
