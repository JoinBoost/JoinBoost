import sys
import numpy as np
import pandas as pd
from conn import Conn
sys.path.append('../')
from aggregator import Aggregator, Annotation


class Pandas_Conn(Conn):
    data: dict = {}

    def __init__(self):
        super().__init__()

    def debug(self, table):
        print(list(self.data[table].columns))
        print(self.data[table].head().values)

    def load_table(self, table_name: str, data_dir: str, col_rename: dict = None):
        df = None
        try:
            df = pd.read_csv(data_dir)
        except Exception as e:
            print(e)
        if col_rename == {}:
            pass
        elif col_rename is None:
            self.relations.append(table_name)
            self.relation_attrs[table_name] = [table_name + '_' + attr for attr in df.columns]
            df = df.rename(columns={attr: table_name + '_' + attr for attr in df.columns})
        else:
            df = df.rename(columns=col_rename)
        self.data[table_name] = df

    def get_table_attrs(self, t_name: str):
        attrs = list(self.data[t_name].columns)
        return attrs

    def fetch_agg(self, table: str):
        res = self.data[table].values
        return np.reshape(res, (1, res.shape[0]))

    def pred_by_cond(self, target_var: str, default_pred: float, leaf_conds: list, test_table: str):
        assert test_table in self.data
        view = self.get_next_view_name()
        test_df = self.data[test_table]
        test_df['prediction'] = default_pred
        for i in range(len(leaf_conds)):
            for (pred, annotations) in leaf_conds[i]:
                translated_annotations = self._translate_annotations(annotations)
                if translated_annotations:
                    test_df.loc[test_df.query(' & '.join(translated_annotations)).index, 'prediction'] += pred
        self.data[view] = test_df
        return view

    def compute_rmse(self, pred_col: str, true_col: str, table: str):
        assert table in self.data
        df = self.data[table]
        assert true_col in df.columns and pred_col in df.columns
        df['error_square'] = (df[true_col] - df[pred_col]) ** 2
        rmse = np.sqrt(np.average(df['error_square'].values))
        return rmse

    def aggregate_max(self, ts: float, tc: int, attr: str, view: str):
        assert view in self.data
        df = self.data[view]
        df['red_in_var'] = np.select([df['c'] < tc],
                                     [-ts**2/tc+(df['s']**2/df['c'])+(ts-df['s'])**2/(tc-df['c'])],
                                     default=0)
        df = df.sort_values('red_in_var', ascending=False)[[attr, 'red_in_var', 's', 'c']]
        if not df.empty:
            return [tuple(df.values[0])]
        else:
            return [('', 0, 0, 0)]

    def cumulative_sum_window(self, attr: str, attr_type: str, view: str):
        view_name = self.get_next_view_name()
        assert view in self.data
        df = self.data[view]
        if attr_type == 'NUM':
            df = df.sort_values(by=[attr])
        elif attr_type == 'LCAT':
            df['object'] = df['s'] / df['c']
            df = df.sort_values(by=['object'])
        df['s'] = df['s'].cumsum()
        df['c'] = df['c'].cumsum()
        self.data[view_name] = df[[attr, 's', 'c']]
        return view_name

    def _translate_annotations(self, annotations: list):
        join_conds = []
        for ann in annotations:
            attr = ann[0]
            if ann[1] == Annotation.IN:
                assert isinstance(ann[2], list)
                _tmp = ["'" + str(ele) + "'" for ele in ann[2]]
                join_conds.append(attr + ' in (' + ','.join(_tmp) + ')')
            elif ann[1] == Annotation.NOT_IN:
                assert isinstance(ann[2], list)
                _tmp = ["'" + str(ele) + "'" for ele in ann[2]]
                join_conds.append(attr + ' not in (' + ','.join(_tmp) + ')')
            elif ann[1] == Annotation.NOT_DISTINCT:
                join_conds.append(attr + " == '" + str(ann[2]) + "'")
            elif ann[1] == Annotation.DISTINCT:
                join_conds.append(attr + " != '" + str(ann[2]) + "'")
            elif ann[1] == Annotation.NOT_GREATER:
                join_conds.append(attr + ' <= ' + str(ann[2]))
            elif ann[1] == Annotation.GREATER:
                join_conds.append(attr + ' > ' + str(ann[2]))
            elif ann[1] == Annotation.NULL:
                join_conds.append(attr + ' != ' + attr)
            elif ann[1] == Annotation.NOT_NULL:
                join_conds.append(attr + ' == ' + attr)
        return join_conds

    def update_error(self, target_var_neighbors: dict, f_table: str, annotations: list, pred: float):
        assert f_table in self.data
        df = self.data[f_table]
        cols = list(df.columns)
        # First do the selection
        for table in target_var_neighbors:
            (l_keys, r_keys), msg = target_var_neighbors[table]
            assert msg in self.data
            r_df = self.data[msg]
            r_df = r_df.drop_duplicates(subset=r_keys)
            df = pd.merge(left=df, right=r_df, how='inner', left_on=l_keys, right_on=r_keys)[cols]
        translated_annotations = self._translate_annotations(annotations)

        if translated_annotations:
            df = df.query(' & '.join(translated_annotations))
            # df.loc[df.query(' & '.join(translated_annotations)).index, 's'] = pred
        df = df.rename(columns={col: 'tmp_' + col for col in cols})
        update_table = self.data[f_table]
        update_table = pd.merge(left=update_table, right=df, how='left', left_on=cols,
                                right_on=['tmp_' + col for col in cols])
        update_table.loc[update_table.query('tmp_s == tmp_s').index, 's'] -= pred
        update_table = update_table[cols]
        self.data[f_table] = update_table

    def aggregation(self, semi_ring_selects: dict, in_msgs: list, f_table: str, groupby: list, where_conds: dict,
                    annotations: list, left_join: dict, create_table: str = ''):
        if f_table not in self.data:
            raise Exception(f_table + ' is not in the database!')
        df = self.data[f_table]
        f_table_cols = set(list(df.columns))
        selection_msgs = [msg['name'] for msg in in_msgs]
        msg_rename = {}
        for msg in selection_msgs:
            assert msg in self.data
            cols = set(list(self.data[msg].columns))
            overlaps = f_table_cols.intersection(cols)
            msg_rename[msg] = {attr: msg + '_' + attr for attr in overlaps}
        # First compute left joins
        if left_join:
            for in_msg in left_join:
                l_join_keys, r_join_keys = left_join[in_msg]
                assert in_msg in self.data
                # attributes has table name prefixes
                df = pd.merge(left=self.data[in_msg], right=df, how='left', left_on=l_join_keys, right_on=r_join_keys)
        # Second join all incoming messages
        for msg in selection_msgs:
            assert msg in where_conds and msg in self.data
            l_join_keys, r_join_keys = where_conds[msg]
            l_df = self.data[msg].rename(columns=msg_rename[msg])
            df = pd.merge(left=l_df, right=df, how='inner', left_on=l_join_keys, right_on=r_join_keys)
        translated_annotations = self._translate_annotations(annotations)
        if translated_annotations:
            df = df.query(' & '.join(translated_annotations))
        # do aggregation
        col_agg = {}
        for col, agg in semi_ring_selects.items():
            if agg[1] == Aggregator.SUM:
                assert isinstance(agg[0], str)
                if col != agg[0]:
                    df[col] = df[agg[0]]
                col_agg[col] = 'sum'
            elif agg[1] == Aggregator.SUM_PROD:
                assert isinstance(agg[0], dict)
                tmp = str(id(df))
                df[tmp] = 1
                for r_name, mult_col in agg[0].items():
                    if r_name + '_' + mult_col in df.columns:
                        df[tmp] *= df[r_name + '_' + mult_col]
                    else:
                        assert mult_col in df.columns
                        df[tmp] *= df[mult_col]
                df[col] = df[tmp]
                col_agg[col] = 'sum'
            elif agg[1] == Aggregator.MAX:
                pass
            elif agg[1] == Aggregator.MIN:
                pass
            elif agg[1] == Aggregator.DISTINCT_COUNT:
                assert isinstance(agg[0], str)
                df[col] = df[agg[0]]
                col_agg[col] = 'nunique'
            elif agg[1] == Aggregator.COUNT:
                assert isinstance(agg[0], str)
                df[col] = df[agg[0]]
                col_agg[col] = 'count'
            elif agg[1] == Aggregator.IDENTITY:
                if agg[0] in df.columns:
                    df[col] = df[agg[0]]
                else:
                    df[col] = agg[0]
                col_agg[col] = lambda x: x
            elif agg[1] == Aggregator.PROD:
                assert isinstance(agg[0], list)
                tmp = str(id(df))
                df[tmp] = 1
                for mult_col in agg[0]:
                    assert mult_col in df.columns
                    df[tmp] *= df[mult_col]
                df[col] = df[tmp]
                col_agg[col] = lambda x: x
            elif agg[1] == Aggregator.SUB:
                assert isinstance(agg[0], tuple) and agg[0][0] in df.columns
                if agg[0][1] in df.columns:
                    df[col] = df[agg[0][0]] - df[agg[0][1]]
                else:
                    df[col] = df[agg[0][0]] - agg[0][1]
                col_agg[col] = lambda x: x
            else:
                pass
        view_name = self.get_next_view_name()
        if col_agg:
            if groupby:
                df = df[list(col_agg.keys()) + groupby].groupby(groupby).agg(col_agg).reset_index()
            else:
                df = df[list(col_agg.keys())].agg(col_agg)
        if create_table:
            self.data[create_table] = df
        else:
            self.data[view_name] = df
        return view_name

    def attrs_below_obj_threshold(self, attr: str, threshold: float, view: str):
        df = self.data[view]
        df = df[df['s'] / df['c'] <= threshold]
        attrs = df[attr].values
        return list(attrs)

    def clean_message(self, messages: list):
        for m_name in messages:
            if m_name in self.data:
                del self.data[m_name]

    def execute_query(self, q):
        pass