from enum import Enum
import pandas as pd
import numpy as np

class QualifiedAttribute:
    def __init__(self, table_name, attribute_name):
        if not isinstance(table_name, str):
            raise TypeError("table_name must be a string.")
        if not isinstance(attribute_name, str):
            raise TypeError("attribute_name must be a string.")
        self.table_name = table_name
        self.attribute_name = attribute_name

    def to_str(self, qualified=True):
        if qualified:
            return self.table_name + "." + self.attribute_name
        else:
            return self.attribute_name
        
    def new_table(self, table_name):
        return QualifiedAttribute(table_name, self.attribute_name)

    def table(self):
        return self.table_name

    def attribute(self):
        return self.attribute_name

    def __str__(self):
        return f"Qualified Attribute({self.table_name},{self.attribute_name})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        # TODO: remove this if
        if isinstance(other, str):
            return self.attribute_name == other
        return self.table_name == other.table_name and self.attribute_name == other.attribute_name

    def __hash__(self):
        return hash(self.__str__())


class AggExpression:
    def __init__(self, agg, para):
        self.agg = agg
        self.para = para

    # this is to make sure we can unpack AggExpression as (para, agg)
    def __iter__(self):
        return iter((self.para, self.agg))
    
    def __str__(self):
        return f"Aggregation Expression({self.agg.name},{self.para})"
    
    def __repr__(self):
        return self.__str__()

    
SELECTION = Enum(
    'NULL', 'NULL NOT_NULL NOT_GREATER LESSER GREATER DISTINCT NOT_DISTINCT IN NOT_IN EQUAL NOT_EQUAL SEMI_JOIN')

class SelectionExpression:
    def __init__(self, selection, para):
        self.selection = selection
        self.para = para
        
    def __str__(self):
        return  f"Aggregation Expression({self.selection.name},{self.para})"
    
    def __repr__(self):
        return self.__str__()

# TODO: separate the aggregation (MIN/MAX...) from expression (ADD/SUB...)
Aggregator = Enum(
    'Aggregator',
    'SUM MAX MIN COUNT DISTINCT_COUNT AVG SUM_PROD DISTRIBUTED_SUM_PROD PROD '
    'SUB ADD DIV DISTINCT_IDENTITY IDENTITY IDENTITY_LAMBDA SQRT POW CAST '
    'CASE'
)

def is_aggregator(agg):
    desired_aggregators = {Aggregator.SUM.value, Aggregator.MAX.value, Aggregator.MIN.value, Aggregator.COUNT.value, Aggregator.DISTINCT_COUNT, Aggregator.AVG.value}
    return agg.value in desired_aggregators


# return a numpy array that applies it
# this is supposed to be used without group-by
def agg_to_np(agg_expr, df, qualified=False):
    agg = agg_expr.agg
    para = agg_expr.para

    if agg.value == Aggregator.IDENTITY.value or agg.value == Aggregator.IDENTITY_LAMBDA.value:
        return df.eval(f"res = {value_to_sql(para, False)}")["res"] 
    elif agg.value == Aggregator.DISTINCT_IDENTITY.value:
        return np.unique(df.eval(f"res = {value_to_sql(para, False)}")["res"])
    elif agg.value == Aggregator.COUNT.value:
        return np.array([len(df)])
    elif agg.value == Aggregator.MAX.value:
        return np.array([df.eval(f"res = {value_to_sql(para, False)}")["res"].max()])
    elif agg.value == Aggregator.SUM.value:
        return np.array([df.eval(f"res = {value_to_sql(para, False)}")["res"].sum()])
    elif agg.value == Aggregator.CASE.value:
        from functools import reduce
        # the para is a list of (value, condition) pairs
        # the condition is a list of SelectionExpression, that are to be ANDed together
        cases = []

        # if there is no condition, return 0
        if len(para) == 0:
            return "0"
        
        conditions = []
        choices = []
        for i in range(len(para)):
            val, cond = para[i]
            conditions.append(reduce(np.intersect1d, [selection_to_df(c, df, qualified) for c in cond]))
            choices.append(df.eval(value_to_sql(val, False)))
            
            
        return np.select(conditions, choices, default=0)
    else:
        raise Exception("Unsupported Semiring Expression")
        
# this is supposed to be used in df.groupby.agg(...)
def agg_to_df_exp(agg_expr, qualified=False):
    
    # if it is a qualified attribute, return the attribute name
    if isinstance(agg_expr, QualifiedAttribute):
        return agg_expr.to_str(qualified)
    
    agg = agg_expr.agg
    para = agg_expr.para

    if agg.value == Aggregator.SUM.value:
        return (agg_to_sql(para, qualified), 'sum') 

    elif agg.value == Aggregator.AVG.value:
        return (agg_to_sql(para, qualified), 'mean') 
    
    elif agg.value == Aggregator.COUNT.value:
        return (agg_to_sql(para, qualified), 'count') 
    
    elif agg.value == Aggregator.MAX.value:
        return (agg_to_sql(para, qualified), 'max') 
    
    else:
        raise Exception("Unsupported Semiring Expression")
        
def selection_to_df(sel, df, qualified=True):
    if sel.selection == SELECTION.EQUAL:
        attr1, attr2 = sel.para[0], sel.para[1]
        # assume attr is a number
        return df[value_to_sql(attr1, qualified)] == float(attr2) 

    elif sel.selection == SELECTION.NOT_EQUAL:
        attr1, attr2 = sel.para[0], sel.para[1]
        return df[value_to_sql(attr1, qualified)] != float(attr2)

    elif sel.selection == SELECTION.NOT_GREATER:
        attr1, attr2 = sel.para[0], sel.para[1]
        return df[value_to_sql(attr1, qualified)] <= float(attr2)

    elif sel.selection == SELECTION.GREATER:
        attr1, attr2 = sel.para[0], sel.para[1]
        return df[value_to_sql(attr1, qualified)] > float(attr2)
    
    elif sel.selection == SELECTION.LESSER:
        attr1, attr2 = sel.para[0], sel.para[1]
        return df[value_to_sql(attr1, qualified)] < float(attr2)
    
    else:
        raise Exception("Unsupported Selection Expression")

def agg_to_sql(agg_expr, qualified=True):
    # check if agg_expr is a string as the base SQL expression
    if isinstance(agg_expr, str):
        return agg_expr
    
    # if it is a qualified attribute, return the attribute name
    if isinstance(agg_expr, QualifiedAttribute):
        return agg_expr.to_str(qualified)
    
    agg = agg_expr.agg
    para = agg_expr.para

    if agg.value == Aggregator.SUM.value:
        # para is a single agg_expr
        return 'SUM(' + agg_to_sql(para, qualified) + ')'

    elif agg.value == Aggregator.DISTINCT_COUNT.value:
        return "COUNT(DISTINCT(" + agg_to_sql(para, qualified) + "))"
    
    elif agg.value == Aggregator.AVG.value:
        return "AVG(" + agg_to_sql(para, qualified) + ")"
    
    elif agg.value == Aggregator.COUNT.value:
        return "COUNT(" + agg_to_sql(para, qualified) + ")"

    elif agg.value == Aggregator.IDENTITY.value or agg.value == Aggregator.IDENTITY_LAMBDA.value:
        return agg_to_sql(para, qualified) 
    
    elif agg.value == Aggregator.DISTINCT_IDENTITY.value:
        return "DISTINCT(" + agg_to_sql(para, qualified) + ")"

    elif agg.value == Aggregator.PROD.value:
        assert isinstance(para, list)
        _tmp = ["CAST(" + val + " AS DOUBLE)" for val in para]
        return "*".join(_tmp)

    elif agg.value == Aggregator.MAX.value:
        return "MAX(" + agg_to_sql(para, qualified) + ")"

    elif agg.value == Aggregator.ADD.value:
        # addition has para as a list of abritary length, and add them together
        return "(" + " + ".join(["(" + agg_to_sql(val, qualified) + ")" for val in para]) + ")"

    elif agg.value == Aggregator.SUB.value:
        # substraction has para as a list of two
        return "(" + agg_to_sql(para[0], qualified) + ") - (" + agg_to_sql(para[1], qualified) + ")"

    elif agg.value == Aggregator.DIV.value:
        # division has para as a list of two
        return "(" + agg_to_sql(para[0], qualified) + ") / (" + agg_to_sql(para[1], qualified) + ")"

    # TODO: use addition and product to implement distributed sum product
    elif agg.value == Aggregator.DISTRIBUTED_SUM_PROD.value:
        # para structure: [[sum_column and list of annotated columns in other relations],[...]]
        # example para: [[R1.s, R2.c, R3.c], [R2.s, R1.c, R3.c], [R3.s, R1.c, R2.c]]
        _tmp = ['*'.join(elem) for elem in para]
        # expected: SUM(R1.s*R2.c*R3.c + R2.s*R1.c*R3.c + R3.s*R1.c*R2.c)
        return 'SUM(' + '+'.join(_tmp) + ')'

    elif agg.value == Aggregator.SUM_PROD.value:
        assert isinstance(para, dict)
        _tmp = [key + "." + value for key, value in para.items()]
        return "SUM(" + "*".join(_tmp) + ")"
    
    elif agg.value == Aggregator.SQRT.value:
        return "SQRT(" + agg_to_sql(para, qualified) + ")"
    
    elif agg.value == Aggregator.POW.value:
        # the first attribute is the base, the second is the power
        base, power = para[0], para[1]
        return "POW(" + agg_to_sql(base, qualified) + ", " + agg_to_sql(power, qualified) + ")"
    
    elif agg.value == Aggregator.CAST.value:
        # the first attribute is the attribute to be casted, the second is the type
        attr, type = para[0], para[1]
        return "CAST(" + agg_to_sql(attr, qualified) + " AS " + agg_to_sql(type, qualified) + ")"
    
    elif agg.value == Aggregator.CASE.value:
        # the para is a list of (value, condition) pairs
        # the condition is a list of SelectionExpression, that are to be ANDed together
        cases = []

        # if there is no condition, return 0
        if len(para) == 0:
            return "0"

        for val, cond in para:
            conds = " AND ".join([selection_to_sql(c, qualified) for c in cond])
            cases.append(f" WHEN {conds} THEN CAST({val} AS DOUBLE)")
        return f"CASE {' '.join(cases)} ELSE 0 END\n"
    else:
        raise Exception("Unsupported Semiring Expression")


# temp way to check if agg is aggregation or not
def is_agg(agg):
    if agg.value == Aggregator.SUM.value:
        return True
    elif agg.value == Aggregator.SUM_PROD.value:
        return True
    elif agg.value == Aggregator.DISTINCT_COUNT.value:
        return True
    elif agg.value == Aggregator.COUNT.value:
        return True
    elif agg.value == Aggregator.PROD.value:
        return True
    return False



Message = Enum('Message', 'IDENTITY SELECTION FULL UNDECIDED')


def value_to_sql(value, qualified=True):
    # if it is  a qualified attribute, return the attribute name
    if isinstance(value, QualifiedAttribute):
        return value.to_str(qualified)
    # if it is a string, return the string with quotes
    elif isinstance(value, str):
        return value
    else:
        raise Exception("Unsupported value type for", value)


def selection_to_sql(sel, qualified=True):

    if sel.selection == SELECTION.IN:
        # the first element in para is the attribute name
        # the second element in para is the list of values
        attr, values = sel.para[0], sel.para[1]
        _tmp = ["'" + str(value) + "'" for value in values]
        return value_to_sql(attr, qualified) + " IN (" + ",".join(_tmp) + ")"

    elif sel.selection == SELECTION.NOT_IN:
        attr, values = sel.para[0], sel.para[1]
        _tmp = ["'" + str(value) + "'" for value in values]
        return value_to_sql(attr, qualified) + " NOT IN (" + ",".join(_tmp) + ")"

    elif sel.selection == SELECTION.SEMI_JOIN:
        # the first element in para is the list of attributes in the left relation
        # the second element in para is the list of attributes in the right relation
        left_attrs, right_attrs = sel.para[0], sel.para[1]
        left_table, right_table = left_attrs[0].table(), right_attrs[0].table()
        # we want to write SQL that performs left semi join based on the attributes
        return "(" + ",".join([value_to_sql(attr, qualified) for attr in left_attrs]) + ") in (SELECT (" + ",".join([value_to_sql(attr, qualified) for attr in right_attrs]) + ") FROM " + right_table + ")"

    elif sel.selection == SELECTION.NOT_DISTINCT:
        # the first element in para is the attribute name/one value
        # the second element in para is the attribute name/one value
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " IS NOT DISTINCT FROM " + value_to_sql(attr2, qualified)

    elif sel.selection == SELECTION.DISTINCT:
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " IS DISTINCT FROM " + value_to_sql(attr2, qualified)

    elif sel.selection == SELECTION.EQUAL:
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " = " + value_to_sql(attr2, qualified)

    elif sel.selection == SELECTION.NOT_EQUAL:
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " != " + value_to_sql(attr2, qualified)

    elif sel.selection == SELECTION.NOT_GREATER:
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " <= " + value_to_sql(attr2, qualified)
    
    elif sel.selection == SELECTION.LESSER:
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " < " + value_to_sql(attr2, qualified) 

    elif sel.selection == SELECTION.GREATER:
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " > " + value_to_sql(attr2, qualified)

    elif sel.selection == SELECTION.NULL:
        attr = sel.para
        return value_to_sql(attr, qualified) + " IS NULL"

    elif sel.selection == SELECTION.NOT_NULL:
        attr = sel.para
        return value_to_sql(attr, qualified) + " IS NOT NULL"

    else:
        raise Exception("Unsupported Selection Expression")

# selection used by dataframe query
# has some difference. E.g., it doesn't use "=" but "==" instead
def selection_to_df_sql(sel, qualified=True):
    if sel.selection == SELECTION.EQUAL:
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " == " + value_to_sql(attr2, qualified)

    elif sel.selection == SELECTION.NOT_EQUAL:
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " != " + value_to_sql(attr2, qualified)

    elif sel.selection == SELECTION.NOT_GREATER:
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " <= " + value_to_sql(attr2, qualified)

    elif sel.selection == SELECTION.GREATER:
        attr1, attr2 = sel.para[0], sel.para[1]
        return value_to_sql(attr1, qualified) + " > " + value_to_sql(attr2, qualified)
    else:
        raise Exception("Unsupported Selection Expression")


def selections_to_sql(selectionExpressions, qualified=True):
    return [selection_to_sql(sel, qualified) for sel in selectionExpressions]

def selections_to_df_sql(selectionExpressions, qualified=True):
    return [selection_to_df_sql(sel, qualified) for sel in selectionExpressions]
