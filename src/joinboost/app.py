import math
from abc import ABC
from .joingraph import JoinGraph
from .semiring import *
from .aggregator import Aggregator, Annotation, Message
from .cjt import CJT
from queue import PriorityQueue
import numpy as np
from typing import Union


class App(ABC):
    def __init__(self):
        pass
    
class DummyModel(App):
    def __init__(self):
        super().__init__()
        self.semi_ring = varSemiRing()
        self.prefix = "joinboost_tmp_"
        self.model_def = []
    
    def fit(self,
           jg: JoinGraph):
        
        jg._preprocess()
        self.semi_ring.init_columns_name(jg)
        
        # get the gradient and hessian
        # for rmse, g is the sum and h is the count
        agg_exp = self.semi_ring.col_sum((jg.get_target_var(), '1'))
        g, h = jg.exe.execute_spja_query(agg_exp,
                                          [jg.get_target_relation()],
                                          mode = 3)[0]
        
        prediction = g / h
        self.semi_ring.set_semi_ring(g, h)
        
        # below currently only works for rmse
        self.count_ = h
        self.constant_ = prediction
        
        # store full join sql
        self._full_join_sql = jg.get_full_join_sql()

    def predict(self, data: Union[str, JoinGraph], input_mode: int):
        return self.constant_
    

class DecisionTree(DummyModel):
    def __init__(self,                
                 max_leaves: int = 31,
                 learning_rate: float = 1, 
                 max_depth: int = 6,
                 subsample: float = 1,
                 debug: bool = False):
        
        super().__init__()
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.debug = debug
        
    def fit(self,
           jg: JoinGraph):
        # super().fit(jg)
        self.semi_ring.init_columns_name(jg)
        # shall we first sample then fit dummy model, or first fit dummy model then sample?
        # the current solution is to first sample than fit dummy model
        self.cjt = CJT(semi_ring=self.semi_ring, join_graph=jg)
        self.create_sample()
        super().fit(jg)
        
        self.cjt.lift(self.cjt.get_target_var() + "- (" + str(self.constant_) + ")")
        self.semi_ring.set_semi_ring(0, self.count_)
        
        self.train_one()
    
    def create_sample(self):
        if self.subsample < 1:
            # TODO: Possible to sample 0 tuples.
            # Add check to make sure the sampled table has tuples
            new_fact_name = self.cjt.exe.execute_spja_query(from_tables=[self.cjt.target_relation], 
                                                            sample_rate=self.subsample,
                                                            mode = 4)
            self.cjt.replace(self.cjt.target_relation, new_fact_name)
        
    
    def train_one(self, last = True):
        # store (node_id) -> cjt
        self.nodes = {}
        self.nodes[0] = self.cjt
        
        # store a pq of split_candidates, sorted on the criteria
        # this one is a bit complex. TODO: simplify or make it a class
        self.split_candidates = PriorityQueue()
        
        # leaf_nodes is used to compute the final models
        self.leaf_nodes = []
        
        self._build_tree()
        if last:
            self._update_error()
            
        self._build_model()
        # TODO: should clean all temp tables, not just messages
        if not self.debug:
            self._clean_messages()
        
    def _build_model(self):
        cur_model_def = []
        for cur_cjt in self.leaf_nodes:
            annotations = cur_cjt.get_all_parsed_annotations()
            g, h = cur_cjt.get_semi_ring().get_value()
            
            pred = float(g / h)* self.learning_rate
            if annotations:
                cur_model_def.append((pred, annotations))
        if cur_model_def:
            self.model_def.append(cur_model_def)
        
    def compute_rmse(self, test_table: str):
        # TODO: refactor
        view = self.cjt.exe.case_query(test_table, '+', 'prediction', str(self.constant_),
                                       self.model_def, [self.cjt.get_target_var()])
        predict_agg = {'RMSE': 
            (f'SQRT(AVG(POW({self.cjt.get_target_var()} - prediction, 2)))',
             Aggregator.IDENTITY)}
        predict = self.cjt.exe.execute_spja_query(predict_agg, [view], mode=4)
        return self.cjt.exe.execute_spja_query(from_tables=[predict], 
                                               mode=3)[0]
    
    # input_mode = 1 takes the full join's table name as input
    # input_mode = 2 takes the join graph as input (assume fact table)
    # input_mode = 3 takes the fact table's name as input (and automatically join it
    # with dimensional tables used in training)
    # TODO: support different outputs
    # output_mode = 1 returns a numpy array
    # output_mode = 2 stores the prediction in a table and returns table name
    def predict(self, data: Union[str, JoinGraph], 
                input_mode: int = 1,
                output_mode: int = 1,):
        
        if input_mode == 1:
            assert(isinstance(data, str))
            view = self.cjt.exe.case_query(data, '+', 'prediction',
                                           str(self.constant_), self.model_def,
                                           [self.cjt.get_target_var()])

        elif input_mode == 2:
            assert(isinstance(data, JoinGraph))
            # TODO
            pass

        elif input_mode == 3:
            assert(isinstance(data, str))
            view = self.cjt.exe.case_query(self._full_join_sql, '+', 'prediction', 
                                           str(self.constant_), self.model_def,
                                           [self.cjt.get_target_var()],
                                           order_by=f'{data}.rowid')

        preds = self.cjt.exe._execute_query(f"select prediction from {view};")
        return np.array(preds)[:, 0]

    def _clean_messages(self):
        for cjt in self.nodes.values():
            cjt.clean_message()

    def _comp_annotations(self, r_name: str, attr: str, cur_value: str, obj: float, expanding_cjt: CJT):
        attr_type = expanding_cjt.get_relation_schema()[r_name][attr]
        g_col, h_col = self.semi_ring.get_columns_name()

        # TODO: remove window_query and everything is spja
        if attr_type == 'LCAT':
            group_by = [attr]
            absoprtion_view = expanding_cjt.absorption(r_name, [attr])
            agg_exp = {attr: (attr, Aggregator.IDENTITY),
                       'object': ((g_col, h_col), Aggregator.DIV),
                       g_col: (g_col, Aggregator.IDENTITY),
                       h_col: (h_col, Aggregator.IDENTITY)}
            obj_view = self.cjt.exe.execute_spja_query(agg_exp, [absoprtion_view])
            view_ord_by_obj = self.cjt.exe.window_query(obj_view, [attr], 'object', [g_col, h_col])
            attr_view = self.cjt.exe.execute_spja_query({attr: (attr, Aggregator.IDENTITY)},
                                                        [view_ord_by_obj],
                                                        [f'{g_col}/{h_col} <=' + str(obj)])
            attrs = [str(x[0])  for x in self.cjt.exe.execute_spja_query(from_tables=[attr_view], mode=3)]
            l_annotation = (attr, Annotation.IN, attrs)
            r_annotation = (attr, Annotation.NOT_IN, attrs)
        elif cur_value == 'NULL':
            l_annotation = (attr, Annotation.NULL, Annotation.NULL)
            r_annotation = (attr, Annotation.NOT_NULL, Annotation.NOT_NULL)
        elif attr_type == 'NUM':
            l_annotation = (attr, Annotation.NOT_GREATER, cur_value)
            r_annotation = (attr, Annotation.GREATER, cur_value)
        elif attr_type == 'CAT':
            l_annotation = (attr, Annotation.NOT_DISTINCT, cur_value)
            r_annotation = (attr, Annotation.DISTINCT, cur_value)
        else:
            raise Exception('Unsupported Split')
        return l_annotation, r_annotation

    # get best split of current cjt
    def _get_best_split(self, cjt_id: int, cjt_depth: int):
        cjt = self.nodes[cjt_id]
        cur_semi_ring = cjt.get_semi_ring()
        attr_meta = self.cjt.get_relation_schema()
        g_col, h_col = self.semi_ring.get_columns_name()
        
        # criteria, (relation name, split attribute, split value, new s, new c)
        best_criteria, best_criteria_ann = 0, ('', '', 0, 0, 0)
        
        if cjt_depth == self.max_depth:
            self.split_candidates.put((-best_criteria, cjt_depth,) + best_criteria_ann + (cjt_id,))
            return
        
        g, h = cur_semi_ring.get_value()
        const_ = float((g**2)/h)
        for r_name in cjt.get_relations():
            for attr in cjt.get_relation_features(r_name):
                attr_type, group_by = self.cjt.get_type(r_name, attr), [attr]
                absoprtion_view = cjt.absorption(r_name, group_by, mode=4)
                if attr_type == 'NUM':
                    agg_exp = cur_semi_ring.col_sum((g_col, h_col))
                    agg_exp[attr] = (attr, Aggregator.IDENTITY)
                    view_to_max = self.cjt.exe.execute_spja_query(agg_exp,
                                                                  [absoprtion_view], 
                                                                  window_by=[attr],
                                                                  mode=4)
                
                elif attr_type == 'LCAT':
                    # TODO: further optimization. We don't need to keep the attr.
                    # The only thing we care for splitting is the sum_s/sum_c
                    agg_exp = {attr: (attr, Aggregator.IDENTITY),
                               'object': ((g_col, h_col), Aggregator.DIV),
                               g_col: (g_col, Aggregator.IDENTITY),
                               h_col: (h_col, Aggregator.IDENTITY)}
                    obj_view = self.cjt.exe.execute_spja_query(agg_exp, 
                                                               [absoprtion_view],
                                                               mode=4)
                    agg_exp = cur_semi_ring.col_sum((g_col, h_col))
                    agg_exp[attr] = (attr, Aggregator.IDENTITY)
                    agg_exp['object'] = ('object', Aggregator.IDENTITY)
                    view_to_max = self.cjt.exe.execute_spja_query(agg_exp,
                                                                  [obj_view], 
                                                                  window_by=['object'],
                                                                  mode=4)
                elif attr_type == 'CAT':
                    view_to_max = absoprtion_view
                    
                l2_agg_exp = {
                    attr: (attr, Aggregator.IDENTITY),
                    'criteria': (
                    f"""CASE WHEN {h} > {h_col}
                            THEN (
                                ({g_col} / {h_col}) * {g_col} +
                                ({g} - {g_col}) / ({h} - {h_col}) * ({g} - {g_col})
                            )
                            ELSE 0
                        END
                    """, Aggregator.IDENTITY),
                    
                    g_col: (g_col, Aggregator.IDENTITY),
                    h_col: (h_col, Aggregator.IDENTITY),
                    
                }
                results = self.cjt.exe.execute_spja_query(l2_agg_exp, 
                                                          [view_to_max], 
                                                          order_by='criteria DESC', 
                                                          limit=1, 
                                                          mode=3)
                if not results:
                    continue
                cur_value, cur_criteria, left_g, left_h = results[0]
                if cur_criteria > best_criteria:
                    best_criteria = cur_criteria
                    # relation name, split attribute, split value, left gradient, left hessian
                    best_criteria_ann = (r_name, attr, str(cur_value), left_g, left_h)
        self.split_candidates.put((const_-best_criteria, cjt_depth,) + best_criteria_ann + (cjt_id,))
        
    # split the semi-ring according to current split
    def split_semi_ring(self, 
                        total_semi_ring: varSemiRing, 
                        left_semi_ring: varSemiRing):
        return left_semi_ring, total_semi_ring - left_semi_ring
    
    # don't update error for single ecision tree 
    def _update_error(self):
        pass
    
    def _get_split_cjt(self, expanding_cjt: CJT, l_semi_ring: varSemiRing, r_semi_ring: varSemiRing):
        l_cjt, r_cjt = expanding_cjt.copy_cjt(l_semi_ring), expanding_cjt.copy_cjt(r_semi_ring)
        next_id = len(self.nodes)
        
        self.nodes[next_id] = l_cjt
        self.nodes[next_id + 1] = r_cjt
        
        return l_cjt, r_cjt, next_id, next_id + 1

    def _build_tree(self):
        self.cjt.calibration()
        self._get_best_split(0, 0)
        
        # while there are beneficial splits and doesn't read max leaves 
        while not self.split_candidates.empty() and \
              self.split_candidates.queue[0][0] < 0 and \
              self.split_candidates.qsize() < self.max_leaves:
            criteria, cur_level, r_name, attr, cur_value, left_g, left_h, c_id = self.split_candidates.get()
            expanding_cjt = self.nodes[c_id]
            
            l_semi_ring = expanding_cjt.semi_ring.copy()
            l_semi_ring.set_semi_ring(left_g, left_h)
            
            l_semi_ring, r_semi_ring = self.split_semi_ring(expanding_cjt.get_semi_ring(), l_semi_ring)

            l_cjt, r_cjt, l_id, r_id = self._get_split_cjt(expanding_cjt=expanding_cjt,
                                                           l_semi_ring=l_semi_ring,
                                                           r_semi_ring=r_semi_ring)
            # TODO: objective has some rounding problem
            # currently it has an ugly solution. find a better solution
            l_annotations, r_annotations = self._comp_annotations(r_name=r_name, 
                                                                  attr=attr,
                                                                  cur_value=cur_value,
                                                                  obj=math.ceil(left_g / left_h * 100) / 100,
                                                                  expanding_cjt=expanding_cjt)
            
            # add annotations according to split conditions
            l_cjt.add_annotations(r_name, l_annotations)
            r_cjt.add_annotations(r_name, r_annotations)
            
            # for the leaf split_candidates that can't be splitted (e.g. meet max depth)
            # we still need message passing to fact table for semi-join selection
            # but not necessarily downward_message_passing. 
            # Can be optimized to upward_message_passing(fact)
            l_cjt.downward_message_passing(r_name)
            r_cjt.downward_message_passing(r_name)
            
            self._get_best_split(l_id, cur_level + 1)
            self._get_best_split(r_id, cur_level + 1)
                
        self.leaf_nodes = [self.nodes[ele[-1]] for ele in self.split_candidates.queue]


        
class GradientBoosting(DecisionTree):
    # TODO: add some checks. E.g., paramters have to be positive
    def __init__(self,
                 max_leaves: int = 31,
                 learning_rate: float = 1, 
                 max_depth: int = 6,
                 iteration: int = 1,
                 debug: bool = False):
        super().__init__(max_leaves,learning_rate,max_depth,debug=debug)
        self.iteration = iteration
    
    def fit(self,
           jg: JoinGraph):
        super().fit(jg)
        
        for _ in range(self.iteration - 1):
            self.train_one()

    def _update_error(self):
        for cur_cjt in self.leaf_nodes:
            cur_cond = []
            target_relation = cur_cjt.get_target_relation()
            g, h = cur_cjt.get_semi_ring().get_value()
            pred = g / h * self.learning_rate
            _, join_conds = cur_cjt._get_income_messages(cur_cjt.get_target_relation(), condition=2)
            join_conds += cur_cjt.get_parsed_annotations(target_relation)
            g_col, _ = self.semi_ring.get_columns_name()
            self.cjt.exe.update_query(f"{g_col}={g_col}-({pred})",
                                      target_relation,
                                      join_conds)
            
class RandomForest(DecisionTree):
    def __init__(self,
                 max_leaves: int = 31,
                 learning_rate: float = 1, 
                 max_depth: int = 6,
                 subsample: float = 1,
                 iteration: int = 1,
                 debug: bool = False):
        super().__init__(max_leaves,learning_rate,max_depth,subsample,debug=debug)
        self.iteration = iteration
        self.learning_rate = 1/iteration
    
    def fit(self,
           jg: JoinGraph):
        
        for _ in range(self.iteration):
            super().fit(jg)
            
            
