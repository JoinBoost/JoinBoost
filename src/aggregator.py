from enum import Enum

# TODO: make aggregator class (like operator in database), so we can do simple composition and optimization
Aggregator = Enum('Aggregator', 'SUM MAX MIN SUB SUM_PROD PROD DIV COUNT DISTINCT_COUNT IDENTITY')
Annotation = Enum('NULL', 'NULL NOT_NULL NOT_GREATER GREATER DISTINCT NOT_DISTINCT IN NOT_IN')
Message = Enum('Message', 'IDENTITY SELECTION FULL UNDECIDED')

def parse_agg(agg, para):
    if agg == Aggregator.SUM:
        assert isinstance(para, str)
        return 'SUM(' + para + ')'
    elif agg == Aggregator.SUM_PROD:
        assert isinstance(para, dict)
        _tmp = [key + '.' + value for key, value in para.items()]
        return 'SUM(' + '*'.join(_tmp) + ')'
    elif agg == Aggregator.DISTINCT_COUNT:
        assert isinstance(para, str)
        return 'COUNT(DISTINCT(' + para + '))'
    elif agg == Aggregator.COUNT:
        assert isinstance(para, str)
        return 'COUNT(' + para + ')'
    elif agg == Aggregator.IDENTITY:
        return str(para)
    elif agg == Aggregator.PROD:
        assert isinstance(para, list)
        _tmp = ['CAST(' + val + ' AS DOUBLE)' for val in para]
        return '*'.join(_tmp) 
    elif agg == Aggregator.SUB:
        assert isinstance(para, tuple)
        return str(para[0]) + ' - ' + str(para[1])
    elif agg == Aggregator.DIV:
        assert isinstance(para, tuple)
        return str(para[0]) + ' / ' + str(para[1])
    else:
        raise Exception('Unsupported Semiring Expression')
        
# temp way to check if agg is aggregation or not
def is_agg(agg):
    if agg == Aggregator.SUM:
        return True
    elif agg == Aggregator.SUM_PROD:
        return True
    elif agg == Aggregator.DISTINCT_COUNT:
        return True
    elif agg == Aggregator.COUNT:
        return True
    elif agg == Aggregator.PROD:
        return True
    return False

def parse_ann(annotations: dict, prepend_relation=False):
    select_conds = []
    for r_name, annotations in annotations.items():
        for ann in annotations:
            if prepend_relation:
                attr = ann[0]
            else:
                attr = r_name + '.' + ann[0]
            if ann[1] == Annotation.IN:
                assert isinstance(ann[2], list)
                _tmp = ["'" + str(ele) + "'" for ele in ann[2]]
                select_conds.append(attr + ' in (' + ','.join(_tmp) + ')')
            elif ann[1] == Annotation.NOT_IN:
                assert isinstance(ann[2], list)
                _tmp = ["'" + str(ele) + "'" for ele in ann[2]]
                select_conds.append(attr + ' not in (' + ','.join(_tmp) + ')')
            elif ann[1] == Annotation.NOT_DISTINCT:
                select_conds.append(attr + " == '" + str(ann[2]) + "'")
            elif ann[1] == Annotation.DISTINCT:
                select_conds.append(attr + " != '" + str(ann[2]) + "'")
            elif ann[1] == Annotation.NOT_GREATER:
                select_conds.append(attr + ' <= ' + str(ann[2]))
            elif ann[1] == Annotation.GREATER:
                select_conds.append(attr + ' > ' + str(ann[2]))
            elif ann[1] == Annotation.NULL:
                select_conds.append(attr + ' != ' + attr)
            elif ann[1] == Annotation.NOT_NULL:
                select_conds.append(attr + ' == ' + attr)
    return select_conds