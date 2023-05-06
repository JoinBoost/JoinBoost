from abc import ABC, abstractmethod
from copy import deepcopy
from .aggregator import Aggregator, Message
from .joingraph import JoinGraph


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


class SemiField(SemiRing):
    def division(self, dividend, divisor):
        pass

class GradientHessianSemiRing(SemiRing):
    def __init__(self, g=0, h=0, g_name="s", h_name="c"):
        self.pair = (g, h)
        self.gradient_column_name = g_name
        self.hessian_column_name = h_name
        self.target_rowid_colname = "rowid"


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


class AvgSemiRing(SemiField):

    def __init__(self, user_table="", attr=""):
        self.user_table = user_table
        self.attr = attr
        self.agg = "AVG"

    def lift_exp(self, s_after='s', c_after='c', user_table=""):
        if user_table == self.user_table:
            return {s_after: (self.attr, Aggregator.IDENTITY), c_after: ("1", Aggregator.IDENTITY)}
        else:
            return {s_after: ("0", Aggregator.IDENTITY), c_after: ("1", Aggregator.IDENTITY)}

    def col_sum(self, s='s', c='c', s_after='s', c_after='c'):
        return {s_after: (s, Aggregator.SUM), c_after: (c, Aggregator.SUM)}

    def sum_over_product(self, user_tables=[], s='s', c='c', s_after='s', c_after='c'):
        annotated_count = {}
        for i, user_table in enumerate(user_tables):
            annotated_count[f'"{user_table}"'] = f'"{c}"'

        sum_join_calculation = []
        for i, user_table in enumerate(user_tables):
            sum_join_calculation.append([f'"{str(user_table)}"."{s}"'] + \
                                        [f'"{rel}"."{c}"' for rel in (user_tables[:i] + user_tables[i + 1:])])

        return {s_after: (sum_join_calculation, Aggregator.DISTRIBUTED_SUM_PROD),
                c_after: (annotated_count, Aggregator.SUM_PROD)}

    def sum_col(self, user_table):
        return self.sum_over_product([user_table])

    # we assume that divisor.s is 0
    def division(self, dividend, divisor, s='s', c='c', s_after='s', c_after='c'):
        return {s_after: ([f'"{dividend}"."{s}"', f'"{divisor}"."{c}"'], Aggregator.DIV),
                c_after: ([f'"{dividend}"."{c}"', f'"{divisor}"."{c}"'], Aggregator.DIV)}

    def get_value(self):
        return self.r_pair

    def get_user_table(self):
        return self.user_table

    def __str__(self, relation=True):
        return f'AVG({(self.user_table + ".") if relation else ""}{self.attr})'

