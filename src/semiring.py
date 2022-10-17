from abc import ABC, abstractmethod
from conn.conn import Conn
from copy import deepcopy
from aggregator import Aggregator
from join_graph import JoinGraph

'''Handle semi ring in DBML'''
class SemiRing(ABC):
    type: str

    def __init__(self, type: str):
        self.type = type

    @abstractmethod
    def build_semi_ring(self, join_graph: JoinGraph, conn: Conn, r_names: list):
        pass

    def get_semi_ring(self):
        pass

    def update_join_conds(self, f_table: str, left_join: dict, join_keys: tuple,
                          in_msg: dict, in_msgs: list, where_conds: list):
        pass

    # get the semi ring columns in message passing from selection
    def get_sr_in_select(self, m_type: int, f_table: str, in_msgs: list, f_table_attrs: list):
        pass


class gbSemiRing(SemiRing):
    r_pair: tuple = None  # (TS, TC)

    def __init__(self):
        super().__init__(type='gb')

    def set_semi_ring(self, TS: float, TC: int):
        self.r_pair = (TS, TC)

    def get_pred(self):
        return self.r_pair[0] / self.r_pair[1]

    # split the semi ring according to current split
    def split_semi_ring(self, s: float, c: int):
        l_semi_ring, r_semi_ring = gbSemiRing(), gbSemiRing()
        l_semi_ring.set_semi_ring(s, c)
        r_semi_ring.set_semi_ring(self.r_pair[0] - s, self.r_pair[1] - c)
        return l_semi_ring, r_semi_ring

    def update_join_conds(self, f_table: str, left_join: dict, join_keys: tuple,
                          in_msg: dict, in_msgs: list, where_conds: list):
        where_conds[in_msg['name']] = join_keys
        in_msgs.append(in_msg)
        return in_msgs, left_join, where_conds

    def build_semi_ring(self, join_graph: JoinGraph, conn: Conn, r_names: list):
        table, target_var = join_graph.get_target_var_relation(), join_graph.get_target_var()
        sr_cols = {'s': (target_var, Aggregator.SUM), 'c': ('1', Aggregator.COUNT)}
        view = conn.aggregation(semi_ring_selects=sr_cols, in_msgs=[], f_table=table, groupby=[],
                                where_conds={}, annotations=[], left_join={})
        TS, TC = conn.fetch_agg(view)[0]
        pred = TS / TC
        self.set_semi_ring(0, TC)
        for r_name in r_names:
            if r_name == table:
                sr_cols = {'s': ((target_var, pred), Aggregator.SUB), 'c': (1, Aggregator.IDENTITY)}
            else:
                sr_cols = {'s': ('CAST(0 AS DOUBLE)', Aggregator.IDENTITY),
                           'c': ('CAST(1 AS DOUBLE)', Aggregator.IDENTITY)}
            for attr in join_graph.get_relation_attrs_full()[r_name]:
                sr_cols[attr] = (attr, Aggregator.IDENTITY)
            conn.aggregation(semi_ring_selects=sr_cols, in_msgs=[], f_table=r_name, groupby=[],
                             where_conds={}, annotations=[], left_join={}, create_table=r_name)

    def get_semi_ring(self):
        return self.r_pair

    def get_sr_in_select(self, m_type: int, f_table: str, in_msgs: list, f_table_attrs: list):
        new_c = '(' + f_table + '.c)'
        new_s = '(' + f_table + '.s)'
        in_msgs_copy = deepcopy(in_msgs)
        while in_msgs_copy:
            cur_msg = in_msgs_copy.pop(0)
            new_s = '(' + cur_msg['name'] + '.c*' + new_s + '+' + new_c + '*' + cur_msg['name'] + '.s)'
            new_c = '(' + new_c + '*' + cur_msg['name'] + '.c)'
        sem_cols = {'c': (new_c, Aggregator.SUM), 's': (new_s, Aggregator.SUM)}
        # print(sem_cols)
        return sem_cols


class lrSemiRing(SemiRing):

    def __init__(self):
        super().__init__(type='lr')

    def build_semi_ring(self, join_graph: JoinGraph, conn: Conn, r_names: list):
        for r_name in r_names:
            full_attrs = join_graph.get_relation_attrs_full()[r_name]
            features = join_graph.get_relation_attrs(r_name)
            sr_cols = {attr: (attr, Aggregator.IDENTITY) for attr in full_attrs}
            sr_cols['cov_c'] = (1, Aggregator.IDENTITY)
            for i, attr in enumerate(features):
                sr_cols['cov_s_' + attr] = (attr, Aggregator.IDENTITY)
                for j in range(i, len(features)):
                    sr_cols['cov_Q_' + attr + '_' + features[j]] = ([attr, features[j]], Aggregator.PROD)
            conn.aggregation(semi_ring_selects=sr_cols, in_msgs=[], f_table=r_name,
                             groupby=[], where_conds={}, annotations=[], left_join={}, create_table=r_name)

    def update_join_conds(self, f_table: str, left_join: dict, join_keys: tuple,
                          in_msg: dict, in_msgs: list, where_conds: list):
        in_msgs.append(in_msg)
        where_conds[in_msg['name']] = join_keys
        return in_msgs, {}, where_conds

    def get_semi_ring(self):
        pass

    def get_sr_in_select(self, mtype: int, f_table: str, in_msgs: list, f_table_attrs: list):
        sem_cols = {}
        r_names = [f_table] + [msg['name'] for msg in in_msgs]
        all_attrs = [f_table_attrs] + [msg['attributes'] for msg in in_msgs]
        all_msg_cts = {r_name: 'cov_c' for r_name in r_names}
        for l, attrs in enumerate(all_attrs):
            other_msg_cts = {r_name: 'cov_c' for r_name in r_names if r_name != r_names[l]}
            for i, attr in enumerate(attrs):
                for j in range(i, len(attrs)):
                    cov_q = 'cov_Q_' + attr + '_' + attrs[j]
                    _tmp = deepcopy(other_msg_cts)
                    _tmp[r_names[l]] = cov_q
                    sem_cols[cov_q] = (_tmp, Aggregator.SUM_PROD)
                for m in range(l+1, len(all_attrs)):
                    for r_attr in all_attrs[m]:
                        _tmp = {r_names[l]: 'cov_s_' + attr, r_names[m]: 'cov_s_' + r_attr}
                        sem_cols['cov_Q_' + attr + '_' + r_attr] = (_tmp, Aggregator.SUM_PROD)
                _tmp = deepcopy(other_msg_cts)
                _tmp[r_names[l]] = 'cov_s_' + attr
                sem_cols['cov_s_' + attr] = (_tmp, Aggregator.SUM_PROD)
        sem_cols['cov_c'] = (all_msg_cts, Aggregator.SUM_PROD)
        return sem_cols


