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

class SelectionExpression:
    def __init__(self, selection, para):
        self.selection = selection
        self.para = para

Aggregator = Enum(
    'Aggregator', 'SUM MAX MIN SUB ADD SUM_PROD DISTRIBUTED_SUM_PROD PROD DIV COUNT DISTINCT_COUNT IDENTITY IDENTITY_LAMBDA')


def agg_to_sql(agg_expr):
    # check if agg_expr is a string as the base SQL expression
    if isinstance(agg_expr, str):
        return agg_expr

    agg = agg_expr.agg
    para = agg_expr.para

    if agg.value == Aggregator.SUM.value:
        # para is a single agg_expr
        return 'SUM(' + agg_to_sql(para) + ')'

    elif agg.value == Aggregator.DISTINCT_COUNT.value:
        return "COUNT(DISTINCT(" + agg_to_sql(para) + "))"

    elif agg.value == Aggregator.COUNT.value:
        return "COUNT(" + agg_to_sql(para) + ")"

    elif agg.value == Aggregator.IDENTITY.value or agg.value == Aggregator.IDENTITY_LAMBDA.value:
        return para

    elif agg.value == Aggregator.PROD.value:
        assert isinstance(para, list)
        _tmp = ["CAST(" + val + " AS DOUBLE)" for val in para]
        return "*".join(_tmp)

    elif agg.value == Aggregator.MAX.value:
        return "MAX(" + agg_to_sql(para) + ")"
    
    elif agg.value == Aggregator.ADD.value:
        # addition has para as a list of abritary length, and add them together
        return "(" + " + ".join(["(" + agg_to_sql(val) + ")" for val in para]) + ")"

    elif agg.value == Aggregator.SUB.value:
        # substraction has para as a list of two
        return "(" + agg_to_sql(para[0]) + ") - (" + agg_to_sql(para[1]) + ")"

    elif agg.value == Aggregator.DIV.value:
        # division has para as a list of two
        return "(" + agg_to_sql(para[0]) + ") / (" + agg_to_sql(para[1]) + ")"
    
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
    'NULL', 'NULL NOT_NULL NOT_GREATER GREATER DISTINCT NOT_DISTINCT IN NOT_IN EQUAL NOT_EQUAL')


Message = Enum('Message', 'IDENTITY SELECTION FULL UNDECIDED')

# refactor the codes to make it more readable

def parse_ann(selectionExpressions, qualified=True):
    select_conds = []
    print("selectionExpressions: ", selectionExpressions)

    for sel in selectionExpressions:

        if sel.selection == SELECTION.IN:
            # the first element in para is the attribute name
            # the second element in para is the list of values
            attr, values = sel.para[0], sel.para[1]
            _tmp = ["'" + str(value) + "'" for value in values]
            select_conds.append(attr.to_str(qualified) + " IN (" + ",".join(_tmp) + ")")

        elif sel.selection == SELECTION.NOT_IN:
            attr, values = sel.para[0], sel.para[1]
            _tmp = ["'" + str(value) + "'" for value in values]
            select_conds.append(attr.to_str(qualified) + " NOT IN (" + ",".join(_tmp) + ")")

        elif sel.selection == SELECTION.NOT_DISTINCT:
            # the first element in para is the attribute name
            # the second element in para is one value
            attr, value = sel.para[0], sel.para[1]
            select_conds.append(attr.to_str(qualified) + " IS NOT DISTINCT FROM '" + str(value) + "'")
        elif sel.selection == SELECTION.DISTINCT:
            attr, value = sel.para[0], sel.para[1]
            select_conds.append(attr.to_str(qualified) + " IS DISTINCT FROM '" + str(value) + "'")
        elif sel.selection == SELECTION.EQUAL:
            attr, value = sel.para[0], sel.para[1]
            select_conds.append(attr.to_str(qualified) + " == '" + str(value) + "'")
        elif sel.selection == SELECTION.NOT_EQUAL:
            attr, value = sel.para[0], sel.para[1]
            select_conds.append(attr.to_str(qualified) + " != '" + str(value) + "'")
        elif sel.selection == SELECTION.NOT_GREATER:
            attr, value = sel.para[0], sel.para[1]
            select_conds.append(attr.to_str(qualified) + " <= '" + str(value) + "'")
        elif sel.selection == SELECTION.GREATER:
            attr, value = sel.para[0], sel.para[1]
            select_conds.append(attr.to_str(qualified) + " > '" + str(value) + "'")
        elif sel.selection == SELECTION.NULL:
            attr = sel.para
            select_conds.append(attr.to_str(qualified) + " IS NULL")
        elif sel.selection == SELECTION.NOT_NULL:
            attr = sel.para
            select_conds.append(attr.to_str(qualified) + " IS NOT NULL")
        else:
            raise Exception("Unsupported Selection Expression")
    return select_conds
