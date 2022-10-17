from conn.conn import Conn
from aggregator import Aggregator


class JoinGraph:
    relations: list = []
    target_var: str = ''
    target_var_relation: str = ''
    d_types: dict = {}
    r_attrs: dict = {}
    r_attrs_full: dict = {}
    joins: dict = {}
    threshold: int = 0
    test_attr_mapping: dict = {}

    def __init__(self, target_var: str, target_var_relation: str, conn: Conn, threshold: int = 0):
        # reference to conn.relation_attrs
        self.r_attrs_full = conn.relation_attrs
        self.threshold = threshold
        self.relations = conn.relations
        self.target_var_relation = self._target_var_check(target_var_relation, target_var)
        self.target_var = self.target_var_relation + '_' + target_var
        self.test_attr_mapping[target_var] = self.target_var

    def _target_var_check(self, target_var_relation: str, target_var: str):
        if target_var_relation + '_' + target_var in self.r_attrs_full[target_var_relation]:
            return target_var_relation
        raise Exception('Target variable not in schema!')

    def get_dtypes(self): return self.d_types

    def get_target_var(self): return self.target_var

    def get_target_var_relation(self): return self.target_var_relation

    def add_relation_attrs(self, r_name: str, attrs: list, attr_meta: list, conn: Conn):
        for i, attr in enumerate(attrs):
            if attr_meta[i] == 2:
                self.d_types[r_name + '_' + attr] = 'NUM'
            else:
                self.d_types[r_name + '_' + attr] = 'CAT'
                # view = conn.aggregation(semi_ring_selects={r_name + '_' + attr: (r_name + '_' + attr,
                #                                                                  Aggregator.DISTINCT_COUNT)},
                #                         in_msgs=[], f_table=r_name, groupby=[], where_conds={},
                #                         annotations=[], left_join={})
                # res = conn.fetch_agg(view)
                # if res[0][0] <= self.threshold:
                #     self.d_types[r_name + '_' + attr] = 'LCAT'
            self.test_attr_mapping[attr] = r_name + '_' + attr
        attrs = [r_name + '_' + attr for attr in attrs]
        if not set(attrs).issubset(set(self.r_attrs_full[r_name])):
            Exception('Key error in ', attrs + '. Attribute does not exist in table', r_name)
        self.r_attrs[r_name] = attrs

    def get_relations(self):
        return self.relations

    def get_relation_attrs_full(self):
        return self.r_attrs_full

    def print_join_tables(self):
        print(self.joins)

    # get features for each table
    def get_relation_attrs(self, r_name):
        if r_name not in self.r_attrs:
            Exception('Attribute not in ', r_name)
        return self.r_attrs[r_name]

    def get_join_keys(self, l_table: str, r_table: str):
        if l_table not in self.joins:
            Exception(l_table, 'not in join graph')
        if r_table:
            if r_table not in self.joins[l_table]:
                Exception(r_table, 'not connected to', l_table)
            return self.joins[l_table][r_table]
        else:
            keys = []
            for table in self.joins[l_table]:
                keys += self.joins[l_table][table]
            return keys

    def get_joins(self):
        return self.joins

    def data_aug(self, data_dir: str, conn: Conn, r_name: str, features: list, attr_meta: list,
                 f_table: str, l_join_keys: list, r_join_keys: list):
        conn.load_table(r_name, data_dir)
        self.add_relation_attrs(r_name=r_name, attrs=features, attr_meta=attr_meta, conn=conn)
        self.add_join(table_name_left=r_name, table_name_right=f_table, left_keys=l_join_keys, right_keys=r_join_keys)

    def add_join(self, table_name_left: str, table_name_right: str, left_keys: list, right_keys: list):
        if len(left_keys) != len(right_keys):
            raise Exception('Join keys have different lengths!')
        if table_name_left not in self.relations:
            raise Exception(table_name_left + 'table doesn\'t exit!')
        if table_name_right not in self.relations:
            raise Exception(table_name_right + 'table doesn\'t exit!')

        for l_key in left_keys:
            if table_name_left + '_' + l_key not in self.r_attrs_full[table_name_left]:
                raise Exception(l_key + ' doesn\'t exit in table ' + table_name_left)

        for r_key in right_keys:
            if table_name_right + '_' + r_key not in self.r_attrs_full[table_name_right]:
                raise Exception(r_key + 'doesn\'t exit in table ' + table_name_right)

        if table_name_left not in self.joins:
            self.joins[table_name_left] = dict()
        if table_name_right not in self.joins:
            self.joins[table_name_right] = dict()

        left_keys = [table_name_left + '_' + attr for attr in left_keys]
        right_keys = [table_name_right + '_' + attr for attr in right_keys]

        self.joins[table_name_left][table_name_right] = (left_keys, right_keys)
        self.joins[table_name_right][table_name_left] = (right_keys, left_keys)