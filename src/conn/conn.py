from abc import ABC, abstractmethod

class Conn(ABC):
    view_id: int = 0
    relations: list = []
    relation_attrs: dict = {}

    def __init__(self):
        pass

    def get_next_view_name(self):
        view_name = 'v_' + str(self.view_id)
        self.view_id += 1
        return view_name

    @abstractmethod
    def debug(self, table):
        pass

    def get_table_attrs(self, t_name: str):
        pass

    def execute_query(self, q):
        pass

    def pred_by_cond(self, target_var: str, default_pred: float, leaf_conds: list, test_table: str):
        pass

    def compute_rmse(self, pred_col: str, true_col: str, table: str):
        pass

    def fetch_agg(self, table: str):
        pass

    def clean_message(self, messages: list):
        pass

    def load_table(self, table_name: str, data_dir: str, col_rename: dict = None):
        pass

    def aggregate_max(self, ts: float, tc: int, attr: str, view: str):
        pass

    def attrs_below_obj_threshold(self, attr: str, threshold: float, view: str):
        pass

    def cumulative_sum_window(self, attr: str, attr_type: str, view: str):
        pass

    def update_error(self, target_var_neighbors: dict, f_table: str, annotations: list, pred: float):
        pass

    def aggregation(self, semi_ring_selects: dict, in_msgs: list, f_table: str, groupby: list, where_conds: dict,
                    annotations: list, left_join: dict, create_table: str = ''):
        pass