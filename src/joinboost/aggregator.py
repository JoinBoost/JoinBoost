from enum import Enum

class AggExpression:
    def __init__(self, agg, para):
        self.agg = agg
        self.para = para
        
Aggregator = Enum('Aggregator', 'SUM MAX MIN SUB SUM_PROD DISTRIBUTED_SUM_PROD PROD DIV COUNT DISTINCT_COUNT IDENTITY IDENTITY_LAMBDA')

def agg_to_sql(agg_expr):
    agg = agg_expr.agg
    para = agg_expr.para
    if agg.value == Aggregator.SUM.value:
        assert isinstance(para, str)
        return 'SUM(' + para + ')'
    elif agg.value == Aggregator.DISTRIBUTED_SUM_PROD.value:
        assert isinstance(para, list)
        # para structure: [[sum_column and list of annotated columns in other relations],[...]]
        # example para: [[R1.s, R2.c, R3.c], [R2.s, R1.c, R3.c], [R3.s, R1.c, R2.c]]
        _tmp = ['*'.join(elem) for elem in para]
        # expected: SUM(R1.s*R2.c*R3.c + R2.s*R1.c*R3.c + R3.s*R1.c*R2.c)
        return 'SUM(' + '+'.join(_tmp) + ')'
    elif agg.value == Aggregator.SUM_PROD.value:
        assert isinstance(para, dict)
        _tmp = [key + "." + value for key, value in para.items()]
        return "SUM(" + "*".join(_tmp) + ")"
    elif agg.value == Aggregator.DISTINCT_COUNT.value:
        assert isinstance(para, str)
        return "COUNT(DISTINCT(" + para + "))"
    elif agg.value == Aggregator.COUNT.value:
        assert isinstance(para, str)
        return "COUNT(" + para + ")"
    elif agg.value == Aggregator.IDENTITY.value or agg.value == Aggregator.IDENTITY_LAMBDA.value:
        return str(para)
    elif agg.value == Aggregator.PROD.value:
        assert isinstance(para, list)
        _tmp = ["CAST(" + val + " AS DOUBLE)" for val in para]
        return "*".join(_tmp)
    elif agg.value == Aggregator.SUB.value:
        assert isinstance(para, tuple)
        return str(para[0]) + " - (" + str(para[1]) + ")"
    elif agg.value == Aggregator.DIV.value:
        assert isinstance(para, tuple)
        return str(para[0]) + " / (" + str(para[1]) + ")"
    elif agg.value == Aggregator.MAX.value:
        assert isinstance(para, str)
        return 'MAX(' + para + ')'
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

Annotation = Enum('NULL', 'NULL NOT_NULL NOT_GREATER GREATER DISTINCT NOT_DISTINCT IN NOT_IN')
Message = Enum('Message', 'IDENTITY SELECTION FULL UNDECIDED')

def parse_ann(annotations: dict, prepend_relation=False):
    select_conds = []
    for r_name, annotations in annotations.items():
        for ann in annotations:
            if prepend_relation:
                attr = ann[0]
            else:
                attr = r_name + "." + ann[0]
            if ann[1] == Annotation.IN:
                assert isinstance(ann[2], list)
                _tmp = ["'" + str(ele) + "'" for ele in ann[2]]
                select_conds.append(attr + " in (" + ",".join(_tmp) + ")")
            elif ann[1] == Annotation.NOT_IN:
                assert isinstance(ann[2], list)
                _tmp = ["'" + str(ele) + "'" for ele in ann[2]]
                select_conds.append(attr + " not in (" + ",".join(_tmp) + ")")
            elif ann[1] == Annotation.NOT_DISTINCT:
                select_conds.append(attr + " == '" + str(ann[2]) + "'")
            elif ann[1] == Annotation.DISTINCT:
                select_conds.append(attr + " != '" + str(ann[2]) + "'")
            elif ann[1] == Annotation.NOT_GREATER:
                select_conds.append(attr + " <= " + str(ann[2]))
            elif ann[1] == Annotation.GREATER:
                select_conds.append(attr + " > " + str(ann[2]))
            elif ann[1] == Annotation.NULL:
                select_conds.append(attr + " != " + attr)
            elif ann[1] == Annotation.NOT_NULL:
                select_conds.append(attr + " == " + attr)
    return select_conds
