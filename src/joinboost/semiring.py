from abc import ABC, abstractmethod
from copy import deepcopy
from .aggregator import Aggregator, Message
from .joingraph import JoinGraph

# MATT: use proper python arithmetic methods
# MATT: use proper abc stuff


"""Handle semi ring in DBMS"""


class SemiRing(ABC):
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

    def get_sr_in_select(
        self, m_type: Message, f_table: str, in_msgs: list, f_table_attrs: list
    ):
        pass


class GradientHessianSemiRing(SemiRing):
    def __init__(self, g=0, h=0, g_name="s", h_name="c"):
        self.pair = (g, h)
        self.gradient_column_name = g_name
        self.hessian_column_name = h_name


# for rmse, gradient is sum and hessian is count
class varSemiRing(GradientHessianSemiRing):
    def set_semi_ring(self, TS: float, TC: int):
        self.pair = (TS, TC)

    def copy(self):
        return deepcopy(self)

    def set_columns_name(self, g_name, h_name):
        self.gradient_column_name = g_name
        self.hessian_column_name = h_name

    def get_columns_name(self):
        return (self.gradient_column_name, self.hessian_column_name)

    def init_columns_name(self, relation_schema):
        # if any column has name 's' or 'c', name the sum column as
        # 'joinboost_preserved_s' and count column as 'joinboost_preserved_c';
        # Recursively adding prefix until both s,c columns are unique
        cols = set()
        for _cols in relation_schema.values():
            for col in _cols.keys():
                cols.add(col)
        prefix = "joinboost_preserved_"

        while self.gradient_column_name in cols or self.hessian_column_name in cols:
            self.gradient_column_name = prefix + self.gradient_column_name
            self.hessian_column_name = prefix + self.hessian_column_name

    def __add__(self, other):
        result = self.copy()
        result.set_semi_ring(self.pair[0] + other.pair[0], self.pair[1] + other.pair[1])
        return result

    def __sub__(self, other):
        result = self.copy()
        result.set_semi_ring(self.pair[0] - other.pair[0], self.pair[1] - other.pair[1])
        return result

    def multiplication(self, semi_ring):
        g, h = semi_ring.get_value()
        self.pair = (self.pair[0] * h + self.pair[1] * g, h * self.pair[1])

    def lift_exp(self, g="s", h="1"):
        g_after, h_after = self.gradient_column_name, self.hessian_column_name
        return {g_after: (g, Aggregator.IDENTITY), h_after: (h, Aggregator.IDENTITY)}

    def col_sum(self, pair=("s", "c")):
        g, h = pair
        g_after, h_after = self.gradient_column_name, self.hessian_column_name
        return {g_after: (g, Aggregator.SUM), h_after: (h, Aggregator.SUM)}

    def get_value(self):
        return self.pair
