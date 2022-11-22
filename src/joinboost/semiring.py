from abc import ABC, abstractmethod
from copy import deepcopy
from .aggregator import Aggregator, Message
from .joingraph import JoinGraph

'''Handle semi ring in DBMS'''
class SemiRing(ABC):
    type: str

    def __init__(self):
        pass
        
    def __add__(self, other):
        pass
    
    def __sub__(self, other):
        pass

    def multiplication(self, semi_ring):
        pass

    def lift_exp(self, attr):
        pass

    def lift_addition(self, attr, constant_):
        pass

    def get_sr_in_select(self, m_type: Message, f_table: str, in_msgs: list, f_table_attrs: list):
        pass


class varSemiRing(SemiRing):
    def __init__(self, s=0, c=0, s_name='s', c_name='c'):
        self.r_pair = (s, c)
        self.sum_column_name = s_name
        self.count_column_name = c_name

    def set_semi_ring(self, TS: float, TC: int):
        self.r_pair = (TS, TC)
    
    def set_sc_columns_name(self, s_name, c_name):
        self.sum_column_name = s_name
        self.count_column_name = c_name
    
    def init_sc_columns_name(self, relation_schema):
        # if any column has name 's' or 'c', name the sum column as 
        # 'joinboost_preserved_s' and count column as 'joinboost_preserved_c';
        # Recursively adding prefix until both s,c columns are unique
        cols = set()
        for _cols in relation_schema.values():
            for col in _cols.keys():
                cols.add(col)
        prefix = "joinboost_preserved_"
        
        while self.sum_column_name in cols or self.count_column_name in cols:
            self.sum_column_name = prefix + self.sum_column_name
            self.count_column_name = prefix + self.count_column_name
        
    def __add__(self, other):
        return varSemiRing(self.r_pair[0] + other.r_pair[0], self.r_pair[1] + other.r_pair[1])
        
    def __sub__(self, other):
        return varSemiRing(self.r_pair[0] - other.r_pair[0], self.r_pair[1] - other.r_pair[1])

    def multiplication(self, semi_ring):
        s, c = semi_ring.get_value()
        self.r_pair = (self.r_pair[0] * c + self.r_pair[1] * s, c * self.r_pair[1])

    def lift_exp(self, s = 's', c = '1'):
        s_after, c_after = self.sum_column_name, self.count_column_name
        return {s_after: (s, Aggregator.IDENTITY), c_after: (c, Aggregator.IDENTITY)}

    def col_sum(self, s = 's', c = 'c'):
        s_after, c_after = self.sum_column_name, self.count_column_name
        return {s_after: (s, Aggregator.SUM), c_after: (c, Aggregator.SUM)}

    def get_value(self):
        return self.r_pair

    def get_sc_columns_name(self):
        return (self.sum_column_name, self.count_column_name)
    
