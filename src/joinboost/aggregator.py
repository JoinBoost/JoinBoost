from enum import Enum


class QualifiedAttribute:
    def __init__(self, table_name, attribute_name):
        self.table_name = table_name
        self.attribute_name = attribute_name

    def to_str(self, qualified=True):
        if qualified:
            return self.table_name + "." + self.attribute_name
        else:
            return self.attribute_name

    def table(self):
        return self.table_name

    def attribute(self):
        return self.attribute_name

    def __str__(self):
        return self.table_name + "." + self.attribute_name

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
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
        return self.agg.name + "(" + str(self.para) + ")"

class SelectionExpression:
    def __init__(self, selection, para):
        self.selection = selection
        self.para = para


Aggregator = Enum(
    'Aggregator',
    'SUM MAX MIN COUNT DISTINCT_COUNT AVG SUM_PROD DISTRIBUTED_SUM_PROD PROD '
    'SUB ADD DIV IDENTITY IDENTITY_LAMBDA SQRT POW CAST '
    'CASE'
)

# TODO: separate the aggregation (MIN/MAX...) from expression (ADD/SUB...)


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


SELECTION = Enum(
    'NULL', 'NULL NOT_NULL NOT_GREATER GREATER DISTINCT NOT_DISTINCT IN NOT_IN EQUAL NOT_EQUAL SEMI_JOIN')


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


def selections_to_sql(selectionExpressions, qualified=True):
    return [selection_to_sql(sel, qualified) for sel in selectionExpressions]
