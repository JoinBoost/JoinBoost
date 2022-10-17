import copy
from semiring import SemiRing
from join_graph import JoinGraph
from conn.conn import Conn


class CJT:
    semi_ring: SemiRing = None
    join_graph: JoinGraph = None
    target_var: str = ''
    messages: dict = {}
    message_id: int = 0
    cur_depth: int = 0
    annotations: dict = {}
    annotation_tables: list = []
    target_var_relation: str = ''
    conn: Conn = None

    def __init__(self, semi_ring: SemiRing, join_graph: JoinGraph, conn: Conn,
                 messages: dict = {}, annotations: dict = {}):
        self.message_id = 0
        self.semi_ring = semi_ring
        self.join_graph = join_graph
        self.messages = messages
        self.annotations = annotations
        self.target_var = join_graph.get_target_var()
        self.target_var_relation = join_graph.get_target_var_relation()
        self.conn = conn

    def get_messages(self):
        return self.messages

    def get_message(self, f_table: str, t_table: str):
        return self.messages[f_table][t_table]['name']

    def get_annotations(self, table):
        if table not in self.annotations:
            return []
        return self.annotations[table]

    def get_all_annotations(self):
        annotations = []
        for table in self.annotations:
            annotations += self.annotations[table]
        return annotations

    def add_annotations(self, r_name: str, annotation: str):
        if r_name not in self.annotations:
            self.annotations[r_name] = [annotation]
        else:
            self.annotations[r_name].append(annotation)

    def add_annotation_table(self, table: str):
        self.annotation_tables.append(table)

    def clean_message(self):
        m_names = []
        for f_table in self.messages:
            for t_table in self.messages[f_table]:
                m_name = self.messages[f_table][t_table]['name']
                m_names.append(m_name)
        self.conn.clean_message(m_names)

    def get_semi_ring(self): return self.semi_ring

    def copy_cjt(self, semi_ring: SemiRing):
        annotations = copy.deepcopy(self.annotations)
        messages = copy.deepcopy(self.messages)
        c_cjt = CJT(semi_ring=semi_ring, join_graph=self.join_graph, conn=self.conn,
                    messages=messages, annotations=annotations)
        return c_cjt

    def get_abs_msgs(self, abs_table):
        joins, neighbors = self.join_graph.get_joins(), {}
        for table in joins[abs_table]:
            neighbors[table] = (self.join_graph.get_join_keys(abs_table, table),
                                self.get_message(table, abs_table))
        return neighbors

    def downward_message_passing(self, root_table: str = None):
        msgs = []
        if not root_table:
            root_table = self.target_var_relation
        self._pre_dfs(root_table)
        return msgs

    def upward_message_passing(self, root_table: str = None, m_type: int = 0):
        if not root_table:
            root_table = self.target_var_relation
        self._post_dfs(root_table, m_type=m_type)

    def _post_dfs(self, c_table: str, p_table: str = None, m_type: int = 0):
        jg = self.join_graph.get_joins()
        if c_table not in jg:
            return
        for c_neighbor in jg[c_table]:
            if c_neighbor != p_table:
                self._post_dfs(c_neighbor, c_table, m_type=m_type)
        if p_table:
            self._send_message(f_table=c_table, t_table=p_table, m_type=m_type)

    def _pre_dfs(self, c_table: str, p_table: str = None):
        joins = self.join_graph.get_joins()
        if c_table not in joins:
            return
        for c_neighbor in joins[c_table]:
            if c_neighbor != p_table:
                self._send_message(f_table=c_table, t_table=c_neighbor)
                self._pre_dfs(c_neighbor, c_table)

    def absorption(self, table: str):
        f_table_attrs = self.join_graph.get_relation_attrs(table)
        in_msgs, left_join, where_conds, _ = self._get_income_messages(f_table=table, t_table='')
        return in_msgs, where_conds, left_join, f_table_attrs

    # key function for message passing, Sec 3.3 of CJT paper
    def _get_income_messages(self, f_table: str, t_table: str):
        joins, left_join, in_msg_attrs = self.join_graph.get_joins(), {}, []
        in_msgs, where_conds = [], {}
        for table in joins[f_table]:
            if table != t_table:
                in_msg = self.messages[table][f_table]
                join_keys = self.join_graph.get_join_keys(table, f_table)
                in_msg_attrs += [attr for attr in in_msg['attributes'] if attr not in join_keys[0]]
                in_msgs, left_join, where_conds = self.semi_ring.update_join_conds(f_table=f_table,
                                                                                   left_join=left_join,
                                                                                   join_keys=join_keys,
                                                                                   in_msg=in_msg,
                                                                                   in_msgs=in_msgs,
                                                                                   where_conds=where_conds)
        return in_msgs, left_join, where_conds, in_msg_attrs

    def _get_message_attrs(self, f_table_attrs: list, in_msgs: list):
        attrs = []
        for in_msg in in_msgs:
            attrs += in_msg['attributes']
        return f_table_attrs + attrs

    # 3 message types: identity, selection, variance
    def _send_message(self, f_table: str, t_table: str, m_type: int = 2):
        # print('--Sending Message from', f_table, 'to', t_table, 'm_type is', m_type)
        joins = self.join_graph.get_joins()
        if f_table not in joins and t_table not in joins[f_table]:
            raise Exception('Table', f_table, 'and table', t_table, 'are not connected')

        if f_table not in self.messages:
            self.messages[f_table] = {}

        m_name = 'm_' + str(id(self)) + '_' + str(self.message_id)
        self.message_id += 1
        in_msgs, left_join, where_conds, msg_attrs = self._get_income_messages(f_table, t_table)
        # if not left_join and f_table != self.target_var_relation:
        #     m_type = 1
        # only one version of attributes shall be here if join key is a feature
        f_table_attrs = self.join_graph.get_relation_attrs(f_table)
        msg_attrs = self._get_message_attrs(f_table_attrs, in_msgs)
        self.messages[f_table][t_table] = {'name': m_name, 'type': m_type, 'attributes': msg_attrs}
        groupby, _ = self.join_graph.get_join_keys(f_table, t_table)
        semi_ring_selects = self.semi_ring.get_sr_in_select(m_type, f_table, in_msgs, f_table_attrs)
        self.conn.aggregation(semi_ring_selects, in_msgs, f_table, groupby, where_conds,
                              self.get_annotations(f_table), left_join, create_table=m_name)
        # self.conn.debug(m_name)

    def data_augmentation(self, r_name: str, f_table: str):
        self._send_message(f_table, r_name)
        self.downward_message_passing(root_table=r_name)


