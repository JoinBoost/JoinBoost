import duckdb
import sys
from conn.conn import Conn
sys.path.append('../')
from aggregator import Aggregator, Annotation


class Duckdb_Conn(Conn):
    conn = None
    aug_relations: dict = {}

    def __init__(self):
        super().__init__()
        con = duckdb.connect(database=':memory:')
        self.conn = con

    def _create_table(self, view_name: str, table_name: str):
        sql = 'CREATE OR REPLACE TABLE ' + table_name + ' AS SELECT * FROM ' + view_name
        self.execute_query(sql)

    def _gen_sql_case(self, leaf_conds: list):
        conds = []
        for leaf_cond in leaf_conds:
            cond = 'CASE\n'
            for (pred, annotations) in leaf_cond:
                cond += ' WHEN ' + ' AND '.join(self._translate_annotations(annotations)) + \
                        ' THEN CAST(' + str(pred) + ' AS DOUBLE)\n'
            cond += 'ELSE 0 END\n'
            conds.append(cond)
        return conds

    def debug(self, table):
        print(self.get_table_attrs(table))
        print(self.execute_query('SELECT * FROM ' + table + '\n limit 5;'))

    def fetch_agg(self, table: str):
        sql = 'SELECT * FROM ' + table
        return self.execute_query(sql)

    def compute_rmse(self, pred_col: str, true_col: str, table: str):
        sql = 'SELECT SQRT(AVG(POW(' + true_col + ' - ' + pred_col + ',2))) AS RMSE FROM ' + table
        return self.execute_query(sql)[0]

    def pred_by_cond(self, target_var: str, default_pred: float, leaf_conds: list, test_table: str):
        view = self.get_next_view_name()
        translated_leaf_cond = self._gen_sql_case(leaf_conds)
        sql = 'CREATE OR REPLACE VIEW ' + view + ' AS SELECT * FROM(\n'
        sql += 'SELECT ' + target_var + ', CAST(' + str(default_pred) + ' AS DOUBLE) +\n'
        sql += " + \n".join(translated_leaf_cond)
        sql += 'AS prediction\nFROM ' + test_table + ')\n'
        self.execute_query(sql)
        return view

    def load_table(self, table_name: str, data_dir: str, col_rename: dict = None):
        self._create_table("'" + data_dir + "'", table_name)
        if col_rename == {}:
            return
        elif col_rename is None:
            self.relations.append(table_name)
            attrs = self.get_table_attrs(table_name)
            self.relation_attrs[table_name] = [table_name + '_' + attr for attr in attrs]
            preprocess_cols = {table_name + '_' + attr: (attr, Aggregator.IDENTITY) for attr in attrs}
        else:
            preprocess_cols = {val: (key, Aggregator.IDENTITY) for key, val in col_rename.items()}
        self.aggregation(semi_ring_selects=preprocess_cols, in_msgs=[], f_table=table_name, groupby=[],
                         where_conds={}, annotations=[], left_join={}, create_table=table_name)

    def get_table_attrs(self, t_name: str):
        attrs = [x[1] for x in self.execute_query('PRAGMA table_info(' + t_name + ')')]
        return attrs

    def clean_message(self, messages: list):
        for m_name in messages:
            sql = 'DROP TABLE IF EXISTS ' + m_name + ';\n'
            self.execute_query(sql)

    def attrs_below_obj_threshold(self, attr: str, threshold: float, view: str):
        # try storing this in a table
        sql = 'SELECT ' + attr + ' FROM ' + view
        sql += '\nWHERE s/c <= ' + str(threshold)
        return ["'" + str(x[0]) + "'" for x in self.execute_query(sql)]

    # specifically for gb
    def aggregate_max(self, ts: float, tc: int, attr: str, view: str):
        sql = 'SELECT ' + attr + ',CASE WHEN ' + str(tc) + ' > c THEN (-(CAST(' + str(ts) + ' AS DOUBLE)/' + \
              str(tc) + ')* ' + str(ts) + ' + (s/c)*s + (' + str(ts) + '-s)/(' + str(tc) + '-c)*(' + \
              str(ts) + '-s)) ELSE 0 END as red_in_var, s as s, c as c\nFROM '
        sql += view
        sql += '\nORDER BY red_in_var DESC\n'
        sql += 'limit 1;'
        return self.execute_query(sql)

    def cumulative_sum_window(self, attr: str, attr_type: str, view: str):
        view_name = self.get_next_view_name()
        sql = 'CREATE OR REPLACE VIEW ' + view_name + ' AS SELECT * FROM\n'
        if attr_type == 'NUM':
            sql += '(\nSELECT ' + view + '.' + attr + ', SUM(' + view + '.c) OVER(ORDER BY ' + view + '.' + attr + \
                   ') as c, SUM(' + view + '.s) OVER(ORDER BY ' + view + '.' + attr + ') as s\nFROM '
        elif attr_type == 'LCAT':
            sql += '(\nSELECT R1.' + attr + ', SUM(R1.c) OVER(ORDER BY ' + \
                   'R1.object) as c, SUM(R1.s) OVER(ORDER BY R1.object) as s\nFROM (\n'
            sql += 'SELECT ' + view + '.' + attr + ', s/c as object, c as c, s as s\nFROM '
        sql += view
        if attr_type == 'NUM':
            sql += ' ORDER BY ' + view + '.' + attr + ' ASC\n)'
        elif attr_type == 'LCAT':
            sql += ' ORDER BY object ASC\n)'
            sql += ' as R1\n)'
        self.execute_query(sql)
        return view_name

    def _translate_annotations(self, annotations: list):
        join_conds = []
        for ann in annotations:
            attr = ann[0]
            if ann[1] == Annotation.IN:
                assert isinstance(ann[2], list)
                join_conds += [attr + " IN (" + ','.join(ann[2]) + ")"]
            elif ann[1] == Annotation.NOT_IN:
                assert isinstance(ann[2], list)
                join_conds += [attr + " NOT IN (" + ','.join(ann[2]) + ")"]
            elif ann[1] == Annotation.NOT_DISTINCT:
                join_conds += [attr + " IS NOT DISTINCT FROM " + "'" + str(ann[2]) + "'"]
            elif ann[1] == Annotation.DISTINCT:
                join_conds += [attr + " IS DISTINCT FROM " + "'" + str(ann[2]) + "'"]
            elif ann[1] == Annotation.NOT_GREATER:
                join_conds += [attr + ' <= ' + str(ann[2])]
            elif ann[1] == Annotation.GREATER:
                join_conds += [attr + ' > ' + str(ann[2])]
            elif ann[1] == Annotation.NULL:
                join_conds += [attr + " IS NULL"]
            elif ann[1] == Annotation.NOT_NULL:
                join_conds += [attr + " IS NOT NULL"]
        return join_conds

    def update_error(self, f_table: str, annotations: list, pred: float):
        update_conds = self._translate_annotations(annotations)
        sql = 'UPDATE ' + f_table + ' SET s=s-(' + str(pred) + ') \n' + \
              'WHERE ' + " AND ".join(update_conds) + '\n'
        # print(sql)
        self.execute_query(sql)

    # returns the view name
    def aggregation(self, semi_ring_selects: dict, in_msgs: list, f_table: str, groupby: list, where_conds: dict,
                    annotations: list, left_join: dict, create_table: str = ''):
        selection_msgs = [msg['name'] for msg in in_msgs]
        select_sql = []
        for col, agg in semi_ring_selects.items():
            if agg[1] == Aggregator.SUM:
                assert isinstance(agg[0], str)
                select_sql.append('SUM(' + agg[0] + ') as ' + col)
            elif agg[1] == Aggregator.SUM_PROD:
                assert isinstance(agg[0], dict)
                _tmp = [key + '.' + value for key, value in agg[0].items()]
                select_sql.append('SUM(' + '*'.join(_tmp) + ') as ' + col)
            elif agg[1] == Aggregator.MAX:
                select_sql.append('MAX(' + agg[0] + ') as ' + col)
            elif agg[1] == Aggregator.MIN:
                pass
            elif agg[1] == Aggregator.DISTINCT_COUNT:
                assert isinstance(agg[0], str)
                select_sql.append('COUNT(DISTINCT(' + agg[0] + ')) as ' + col)
            elif agg[1] == Aggregator.COUNT:
                assert isinstance(agg[0], str)
                select_sql.append('COUNT(' + agg[0] + ') as ' + col)
            elif agg[1] == Aggregator.IDENTITY:
                select_sql.append(str(agg[0]) + ' as ' + col)
            elif agg[1] == Aggregator.PROD:
                assert isinstance(agg[0], list)
                _tmp = ['CAST(' + val + ' AS DOUBLE)' for val in agg[0]]
                select_sql.append('*'.join(_tmp) + ' as ' + col)
            elif agg[1] == Aggregator.SUB:
                assert isinstance(agg[0], tuple)
                select_sql.append(str(agg[0][0]) + ' - ' + str(agg[0][1]) + ' as ' + col)
            else:
                pass
        view_name = self.get_next_view_name()
        sql = 'CREATE OR REPLACE VIEW ' + view_name + ' AS '
        sql += 'SELECT ' + ', '.join([f_table + '.' + attr for attr in groupby] + select_sql) + '\n'
        if not left_join:
            selection_msgs.append(f_table)
        else:
            for in_msg in left_join:
                l_join_keys, r_join_keys = left_join[in_msg]
                left_join_conds = [in_msg + "." + l_join_keys[i] + " = " + f_table + "."
                                   + r_join_keys[i] for i in range(len(l_join_keys))]
                selection_msgs += [in_msg + " LEFT JOIN " + f_table + " ON (" + " AND ".join(left_join_conds) + ")"]
        sql += "FROM " + ",".join(selection_msgs) + '\n'
        if where_conds or annotations:
            join_conds = []
            for in_msg in where_conds:
                l_join_keys, r_join_keys = where_conds[in_msg]
                join_conds += [in_msg + "." + l_join_keys[i] + " IS NOT DISTINCT FROM " +
                               f_table + "." + r_join_keys[i] for i in range(len(l_join_keys))]
            join_conds += self._translate_annotations(annotations)
            sql += "WHERE " + " AND ".join(join_conds) + '\n'
        if groupby:
            sql += "GROUP BY " + ",".join([f_table + '.' + attr for attr in groupby]) + '\n'
        self.execute_query(sql)
        if create_table:
            self._create_table(view_name, create_table)
        return view_name

    def execute_query(self, q):
        # print(q)
        self.conn.execute(q)
        result = None
        try:
            result = self.conn.fetchall()
        except Exception as e:
            print(e)
        return result

    def add_test_table(self, f_name):
        self.execute_query("CREATE OR REPLACE TABLE test AS SELECT * FROM '" + f_name + ".csv';")
