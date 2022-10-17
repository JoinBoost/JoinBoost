import sys
import time
import math
from queue import PriorityQueue
sys.path.append('../')
from conn.conn import Conn
from conn.duckdb_conn import Duckdb_Conn
from cjt import CJT
from join_graph import JoinGraph
from semiring import gbSemiRing
from aggregator import Aggregator, Annotation


def load_test_table(f_dir: str, conn: Conn, rename_col: dict):
    conn.load_table('test', f_dir, rename_col)


class GradientBoosting():
    learning_rate: float = 0
    default_pred: float = 0
    cjts: dict = {}
    nodes: PriorityQueue = PriorityQueue()
    leaf_cjts: list = []
    leaf_conds: list = []

    def __init__(self, join_graph: JoinGraph, conn: Conn, learning_rate: float = 1, iteration: int = 1):
        self.learning_rate = learning_rate
        self.target_var = join_graph.get_target_var()
        self.target_var_relation = join_graph.get_target_var_relation()
        # treat the cjt as the root cjt
        semi_ring = gbSemiRing()
        semi_ring.build_semi_ring(join_graph=join_graph, conn=conn, r_names=join_graph.get_relations())
        for i in range(iteration):
            start = time.time()
            self.cjt = CJT(semi_ring=semi_ring, join_graph=join_graph, conn=conn, annotations={}, messages={})
            # calibration
            self.cjt.upward_message_passing()
            self.cjt.downward_message_passing()
            l_cjt, r_cjt, l_semi_ring, r_semi_ring, r_name = self._build_gradient_boosting()
            self._update_error(l_cjt, r_cjt, l_semi_ring, r_semi_ring, r_name)
            self._clean_messages()
            print('RMSE:', self.compute_predict_se(l_cjt, r_cjt, 'test')[0], 'Epoch Time:', time.time()-start)
            self.leaf_cjts = []

    def _get_split_cjt(self, expanding_cjt: CJT, l_semi_ring: gbSemiRing, r_semi_ring: gbSemiRing):
        l_cjt, r_cjt = expanding_cjt.copy_cjt(l_semi_ring), expanding_cjt.copy_cjt(r_semi_ring)
        return l_cjt, r_cjt

    def _build_gradient_boosting(self):
        self._get_best_split()

        red, r_name, attr, cur_value, s, c = self._get_best_split()
        l_semi_ring, r_semi_ring = self.cjt.get_semi_ring().split_semi_ring(s, c)
        l_cjt, r_cjt = self._get_split_cjt(expanding_cjt=self.cjt, l_semi_ring=l_semi_ring, r_semi_ring=r_semi_ring)
        l_annotations, r_annotations = self._comp_annotations(r_name=r_name, attr=attr, cur_value=cur_value,
                                                              objective=math.ceil(s / c * 100) / 100,
                                                              expanding_cjt=self.cjt)
        l_cjt.add_annotations(r_name, l_annotations)
        r_cjt.add_annotations(r_name, r_annotations)
        # print(r_name, attr, cur_value, s, c)

        l_cjt.downward_message_passing(r_name)

        r_cjt.downward_message_passing(r_name)
        return l_cjt, r_cjt, l_semi_ring, r_semi_ring, r_name

    def _update_error(self, l_cjt, r_cjt, l_semi_ring, r_semi_ring, r_name):
        l_pred, r_pred = l_semi_ring.get_pred() * self.learning_rate, r_semi_ring.get_pred() * self.learning_rate
        l_cjt.conn.update_error(r_name, l_cjt.get_annotations(r_name), l_pred)
        r_cjt.conn.update_error(r_name, r_cjt.get_annotations(r_name), r_pred)

    def compute_predict_se(self, l_cjt, r_cjt, test_table: str):
        cur_leaf_conds = []
        for cur_cjt in [l_cjt, r_cjt]:
            annotations = cur_cjt.get_all_annotations()
            pred = cur_cjt.get_semi_ring().get_pred() * self.learning_rate
            cur_leaf_conds.append((pred, annotations))
        if cur_leaf_conds:
            self.leaf_conds.append(cur_leaf_conds)
        view = self.cjt.conn.pred_by_cond(self.target_var, self.default_pred, self.leaf_conds, test_table)
        # self.cjt.conn.debug(view)
        return self.cjt.conn.compute_rmse('prediction', self.cjt.join_graph.get_target_var(), view)

    def _clean_messages(self):
        self.cjt.clean_message()

    def _comp_annotations(self, r_name: str, attr: str, cur_value: str, objective: float, expanding_cjt: CJT):
        attr_type = expanding_cjt.join_graph.get_dtypes()[attr]
        if attr_type == 'LCAT':
            groupby = [attr]
            in_msgs, where_conds, left_join, f_table_attrs = expanding_cjt.absorption(r_name)
            semi_ring_selects = expanding_cjt.get_semi_ring().get_sr_in_select(2, r_name, in_msgs, f_table_attrs)
            view = self.cjt.conn.aggregation(semi_ring_selects, in_msgs, r_name, groupby, where_conds,
                                         expanding_cjt.get_annotations(r_name), left_join)
            view_ord_by_obj = self.cjt.conn.cumulative_sum_window(attr, attr_type, view)
            result = self.cjt.conn.attrs_below_obj_threshold(attr, objective, view_ord_by_obj)
            l_annotation = (attr, Annotation.IN, result)
            r_annotation = (attr, Annotation.NOT_IN, result)
        elif cur_value == 'NULL':
            l_annotation = (attr, Annotation.NULL, Annotation.NULL)
            r_annotation = (attr, Annotation.NOT_NULL, Annotation.NOT_NULL)
        elif attr_type == 'NUM':
            l_annotation = (attr, Annotation.NOT_GREATER, cur_value)
            r_annotation = (attr, Annotation.GREATER, cur_value)
        else:
            l_annotation = (attr, Annotation.NOT_DISTINCT, cur_value)
            r_annotation = (attr, Annotation.DISTINCT, cur_value)
        return l_annotation, r_annotation

    # get best split of current cjt
    def _get_best_split(self):
        cur_semi_ring = self.cjt.get_semi_ring()
        attr_meta = self.cjt.join_graph.get_dtypes()
        best_red, best_red_ann = 0, ('', '', 0, 0, 0)
        for r_name in self.cjt.join_graph.get_relations():
            for attr in self.cjt.join_graph.get_relation_attrs(r_name):
                attr_type, groupby = attr_meta[attr], [attr]
                in_msgs, where_conds, left_join, f_table_attrs = self.cjt.absorption(r_name)
                semi_ring_selects = cur_semi_ring.get_sr_in_select(2, r_name, in_msgs, f_table_attrs)
                view = self.cjt.conn.aggregation(semi_ring_selects, in_msgs, r_name, groupby, where_conds,
                                                 self.cjt.get_annotations(r_name), left_join)
                ts, tc = cur_semi_ring.get_semi_ring()
                results = self.cjt.conn.aggregate_max(ts, tc, attr,
                                                      self.cjt.conn.cumulative_sum_window(attr, attr_type, view))
                if not results:
                    continue
                cur_value, cur_var_red, s, c = results[0]
                if cur_var_red > best_red:
                    best_red = cur_var_red
                    best_red_ann = (r_name, attr, str(cur_value), s, c)
        return (-best_red, ) + best_red_ann

def train_fav():
    threshold = 100
    target_var = 'unit_sales'
    target_var_relation = 'sales'
    con = Duckdb_Conn()
    con.load_table('holidays', '../../data/favorita/holidays.csv')
    con.load_table('sales', '../../data/favorita/sales_samp.csv')
    con.load_table('oil', '../../data/favorita/oil.csv')
    con.load_table('transactions', '../../data/favorita/transactions.csv')
    con.load_table('stores', '../../data/favorita/stores.csv')
    con.load_table('items', '../../data/favorita/items.csv')
    jg = JoinGraph(target_var, target_var_relation, con, threshold)
    jg.add_relation_attrs("sales", ["onpromotion"], [1], con)
    jg.add_relation_attrs("holidays", ["htype", "locale", "locale_name", "transferred"], [1, 1, 1, 1], con)
    jg.add_relation_attrs("oil", ["dcoilwtico"], [1], con)
    jg.add_relation_attrs("transactions", ["transactions"], [2], con)
    jg.add_relation_attrs("stores", ["store_nbr", "city", "state", "stype", "cluster"], [1, 1, 1, 1, 1], con)
    jg.add_relation_attrs("items", ["item_nbr", "family", "class", "perishable"], [1, 1, 1, 1], con)
    load_test_table('../../data/favorita/train.csv', con, {})
    # load_test_table('join_test2_sample.csv', con, jg.test_attr_mapping)
    jg.add_join("sales", "items", ["item_nbr"], ["item_nbr"])
    jg.add_join("sales", "transactions", ["date", "store_nbr"], ["date", "store_nbr"])
    jg.add_join("transactions", "stores", ["store_nbr"], ["store_nbr"])
    jg.add_join("transactions", "holidays", ["date"], ["date"])
    jg.add_join("holidays", "oil", ["date"], ["date"])
    start = time.time()
    gb = GradientBoosting(jg, con, learning_rate=1, iteration=20)
    print(time.time() - start)


def gen_imdb_train_join(con):
    sql = '''
CREATE OR REPLACE TABLE train AS 
SELECT movie_production_year, movie_season_nr, movie_episode_nr, title_kind_id, cast_info_movie_id, 
movie_key_keyword_id, key_type_keyword, movie_comp_company_id, movie_comp_company_type_id, 
movie_info_info_type_id, cast_info_person_id, info_type_info
FROM
movie, title, cast_info, movie_key, key_type, movie_comp, person, movie_info, comp, person_info, info_type
WHERE movie_kind_id = title_kind_id
AND movie_movie_id = movie_key_movie_id
AND movie_key_keyword_id = key_type_keyword_id
AND movie_info_movie_id = movie_movie_id
AND movie_movie_id = movie_comp_movie_id
AND movie_comp_company_id = comp_company_id
AND movie_info_info_type_id = info_type_info_type_id
AND movie_movie_id = cast_info_movie_id
AND cast_info_person_id = person_person_id
AND person_info_person_id = person_person_id
    '''
    res = con.execute_query(sql)
    sql = "COPY train TO '../../data/imdb/train.csv' (HEADER, DELIMITER ',');"
    con.execute_query(sql)


def train_imdb():
    threshold = 100
    target_var = 'person_id'
    target_var_relation = 'cast_info'
    con = Duckdb_Conn()
    con.load_table('movie', '../../data/imdb/title_samp.csv')
    con.load_table('title', '../../data/imdb/kind_type.csv')
    con.load_table('cast_info', '../../data/imdb/cast_info_samp.csv')
    con.load_table('movie_key', '../../data/imdb/movie_keyword.csv')
    con.load_table('key_type', '../../data/imdb/keyword.csv')
    con.load_table('movie_comp', '../../data/imdb/movie_companies.csv')
    con.load_table('person', '../../data/imdb/name.csv')
    con.load_table('movie_info', '../../data/imdb/movie_info.csv')
    con.load_table('comp', '../../data/imdb/company_name.csv')
    con.load_table('person_info', '../../data/imdb/person_info.csv')
    con.load_table('info_type', '../../data/imdb/info_type.csv')
    jg = JoinGraph(target_var, target_var_relation, con, threshold)
    jg.add_relation_attrs("movie", ["production_year", "season_nr", "episode_nr"], [1, 2, 2], con)
    jg.add_relation_attrs("title", ["kind_id"], [1], con)
    jg.add_relation_attrs("comp", [], [], con)
    jg.add_relation_attrs("cast_info", ["movie_id"], [1], con)
    jg.add_relation_attrs("info_type", ["info"], [1], con)
    jg.add_relation_attrs("movie_key", ["keyword_id"], [1], con)
    jg.add_relation_attrs("key_type", ["keyword"], [1], con)
    jg.add_relation_attrs("movie_comp", ["company_id", "company_type_id"], [1, 1], con)
    jg.add_relation_attrs("movie_info", ["info_type_id"], [1], con)
    jg.add_relation_attrs("person", [], [], con)
    jg.add_relation_attrs("person_info", [], [], con)
    gen_imdb_train_join(con)
    load_test_table('../../data/imdb/train.csv', con, {})
    # load_test_table('join_test2_sample.csv', con, jg.test_attr_mapping)
    jg.add_join("movie", "title", ["kind_id"], ["kind_id"])
    jg.add_join("movie", "movie_key", ["movie_id"], ["movie_id"])
    jg.add_join("movie_key", "key_type", ["keyword_id"], ["keyword_id"])
    jg.add_join("movie_info", "movie", ["movie_id"], ["movie_id"])
    jg.add_join("movie", "movie_comp", ["movie_id"], ["movie_id"])
    jg.add_join("movie_comp", "comp", ["company_id"], ["company_id"])
    jg.add_join("movie_info", "info_type", ["info_type_id"], ["info_type_id"])
    jg.add_join("movie", "cast_info", ["movie_id"], ["movie_id"])
    jg.add_join("cast_info", "person", ["person_id"], ["person_id"])
    jg.add_join("person_info", "person", ["person_id"], ["person_id"])
    gb = GradientBoosting(jg, con, learning_rate=0.5, iteration=20)

if __name__ == "__main__":
    train_imdb()