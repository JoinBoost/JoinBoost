import math
from abc import ABC

from .joingraph import JoinGraphException
from .preprocessor import Preprocessor, RenameStep
from .executor import SPJAData, PandasExecutor, ExecuteMode
import pandas as pd
from .semiring import *
from .aggregator import *
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

    def fit(self, jg: JoinGraph):
        self._fit(jg)

    def _fit(self, jg: JoinGraph):
        jg._preprocess()

        # get the gradient and hessian
        # for rmse, g is the sum and h is the count
        agg_exp = self.semi_ring.col_sum((jg.target_var, "1"))
        spja_data = SPJAData(
            aggregate_expressions=agg_exp, from_tables=[jg.target_relation]
        )
        g, h = jg.exe.execute_spja_query(
            spja_data, mode=ExecuteMode.EXECUTE)[0]

        prediction = g / h
        self.semi_ring.set_semi_ring(g, h)

        # below currently only works for rmse
        self.count_ = h
        self.constant_ = prediction

    def predict(self, data: Union[str, JoinGraph], input_mode: int):
        return self.constant_


class DecisionTree(DummyModel):
    def __init__(
        self,
        max_leaves: int = 31,
        learning_rate: float = 1,
        max_depth: int = 6,
        subsample: float = 1,
        debug: bool = False,
        partition_early: bool = True,
        enable_batch_optimization: bool = False, # This is only applicable for pandas right now
    ):
        assert max_leaves > 0, "max_leaves should be positive"
        assert max_depth > 0, "max_depth should be positive"
        # sample ratio should be in (0, 1]
        assert 0 < subsample <= 1, "subsample should be in (0, 1]"
        # learning rate should be in (0, 1]
        assert 0 < learning_rate <= 1, "learning_rate should be in (0, 1]"

        super().__init__()
        # whether the fact table is partitioned early
        self.partition_early = partition_early
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.debug = debug
        self.enable_batch_optimization = enable_batch_optimization
        self.preprocessor = Preprocessor()

    def _fit(self, jg: JoinGraph):
        jg._preprocess()

        # First, we run preprocess to rename reserved column name
        # Create views for tables having conflicting column names with reserved words.
        g, h = self.semi_ring.get_columns_name()

        self.preprocessor.add_step(RenameStep(reserved_words=[g, h, "rowid"]))
        self.preprocessor.run_preprocessing(jg)
        jg = self.preprocessor.get_join_graph()

        # shall we first sample then fit dummy model, or first fit dummy model then sample?
        # the current solution is to first sample than fit dummy model
        self.cjt = CJT(semi_ring=self.semi_ring, join_graph=jg)
        self.create_sample()

        super()._fit(jg)

        # substracting the target variable by means
        exp = agg_to_sql(AggExpression(
            Aggregator.SUB, (self.cjt.target_var, str(self.constant_))))

#         if isinstance(self.cjt.exe, PandasExecutor):
#             exp = lambda row: row[self.cjt.target_var] - self.constant_

        self.cjt.lift(exp)
        self.semi_ring.set_semi_ring(0, self.count_)

        self.train_one()

    def create_sample(self, mode=ExecuteMode.WRITE_TO_TABLE):
        if self.subsample < 1:
            # TODO: Possible to sample 0 tuples.
            # Add check to make sure the sampled table has tuples
            spja_data = SPJAData(
                from_tables=[self.cjt.target_relation],
                sample_rate=self.subsample,
            )
            new_fact_name = self.cjt.exe.execute_spja_query(
                spja_data, mode=mode)
            self.cjt.replace(self.cjt.target_relation, new_fact_name)

    def train_one(self, last=True):
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
            annotations = cur_cjt.get_all_annotations()
            # annotations = selections_to_sql(list_of_ann, qualified=qualified)
            g, h = cur_cjt.get_semi_ring().get_value()
            pred = float(g / h) * self.learning_rate
            if annotations:
                cur_model_def.append((pred, annotations))

        # note that gradient boosting has multiple decision trees
        self.model_def.append(AggExpression(Aggregator.CASE, cur_model_def))

    # TODO: remove the test codes and rewrite test cases
    def _build_model_legacy(self, qualified=False):
        self.model_def = []
        cur_model_def = []
        for cur_cjt in self.leaf_nodes:
            list_of_ann = cur_cjt.get_all_annotations()
            annotations = selections_to_sql(list_of_ann, qualified=qualified)
            g, h = cur_cjt.get_semi_ring().get_value()
            pred = float(g / h) * self.learning_rate
            if annotations:
                cur_model_def.append((pred, annotations))
        if cur_model_def:
            self.model_def.append(cur_model_def)

    def compute_rmse(self, test_table: str):
        # The challenge is that the original target name may be renamed
        # during preprocessing. We need to get the original target name
        # target = self.cjt.target_var
        target = self.preprocessor.get_original_target_name()

        # TODO: make sure the target is not named as "prediction"
        compute_prediction = SPJAData(
            aggregate_expressions={"prediction": self.get_prediction_aggregate(),
                                   target: AggExpression(Aggregator.IDENTITY, target)},
            from_tables=[test_table], qualified=False
        )

        view = self.cjt.exe.execute_spja_query(
            compute_prediction, mode=ExecuteMode.NESTED_QUERY
        )

        predict_agg = {
            "RMSE": AggExpression(Aggregator.IDENTITY,  f"SQRT(AVG(POW({target} - prediction, 2)))")
        }

        prediction_query_data = SPJAData(
            aggregate_expressions=predict_agg, from_tables=[view]
        )

        predict = self.cjt.exe.execute_spja_query(
            prediction_query_data, mode=ExecuteMode.NESTED_QUERY
        )

        rmse_query_data = SPJAData(from_tables=[predict])
        return self.cjt.exe.execute_spja_query(rmse_query_data, mode=ExecuteMode.EXECUTE)[0]

    def get_prediction_aggregate(self):
        # for gradient boosting, the prediction is the base_val plus the sum of the tree predictions
        return AggExpression(Aggregator.ADD, [str(self.constant_)] + self.model_def)

    # input_mode = "FULL_JOIN_JG" takes the join graph as input, with the full join specified by JG._target_relation
    # input_mode = "FULL_JOIN_DF" takes the dataframe of full join's table name as input
    # input_mode = "JOIN_GRAPH" takes the join graph as input (assume the same schema as training data)
    # output_mode = "NUMPY" returns a numpy array
    # output_mode = "WRITE_TO_TABLE" stores the prediction in a table and returns table name
    def predict(
        self,
        joingraph: JoinGraph,
        input_mode: str = "FULL_JOIN_JG",
        output_mode: str = "NUMPY",
        qualified: bool = False,
    ):
        input_modes = ["FULL_JOIN_JG", "FULL_JOIN_DF", "JOIN_GRAPH"]
        output_modes = ["NUMPY", "WRITE_TO_TABLE"]
        if input_mode not in input_modes:
            raise Exception("Unsupported input_mode")
        if output_mode not in output_modes:
            raise Exception("Unsupported output_mode")

        if input_mode == "FULL_JOIN_JG":
            # TODO: one concern of full join is that, there would be ambiguity for features with the same name but from table
            # E.g., R(A,B), S(A,B). They join on A, and B is a feature name shared by both.
            # The full will have ambiguous naming, and may be renamed to (A, R.B, S.B)
            # To avoid this, requires a rename mapping from users. By default, we consider renaming mapping which prefixes the feature with relation name.
            # One solution is, to require the attributes in the join graph to be qualified with relation name.
            compute_prediction = SPJAData(
                aggregate_expressions={"prediction": self.get_prediction_aggregate(),
                                       self.cjt.target_var: AggExpression(Aggregator.IDENTITY, self.cjt.target_var)},
                from_tables=[joingraph.target_relation], qualified=qualified
            )

            view = self.cjt.exe.execute_spja_query(
                compute_prediction, mode=ExecuteMode.NESTED_QUERY
            )

        if input_mode == "JOIN_GRAPH":
            # TODO: reapply all the preprocessing steps
            self._update_fact_table_column_name(
                jg=joingraph, check_rowid_col=True)

            full_join = joingraph.get_full_join_sql()

            compute_prediction = SPJAData(
                aggregate_expressions={"prediction": self.get_prediction_aggregate(),
                                       self.cjt.target_var: AggExpression(Aggregator.IDENTITY, self.cjt.target_var)},
                from_tables=[full_join], order_by=[(f"{joingraph.target_relation}.rowid", "")], qualified=False
            )

            # have to be mode=ExecuteMode.WRITE_TO_TABLE, can't be NESTED_QUERY
            # TODO: fix this
            view = self.cjt.exe.execute_spja_query(
                compute_prediction, mode=ExecuteMode.WRITE_TO_TABLE
            )

            self._update_fact_table_column_name(
                jg=joingraph, resume_rowid_col=True)

        if output_mode == "NUMPY":
            preds = joingraph.exe._execute_query(
                f"select prediction from {view};")
            return np.array(preds)[:, 0]
        elif output_mode == "WRITE_TO_TABLE":
            return view
        

    # TODO: refactor this
    def _update_fact_table_column_name(
        self, jg, check_rowid_col=False, resume_rowid_col=False
    ):
        """Rename/resume fact table's rowid column(if exists)."""

        if jg.check_target_relation_contains_rowid_col():
            if check_rowid_col:
                old_name, new_name = "rowid", jg.target_rowid_colname

            if resume_rowid_col:
                old_name, new_name = jg.target_rowid_colname, "rowid"

            sql = f"ALTER TABLE {jg.target_relation} RENAME COLUMN {old_name} TO {new_name};"
            self.cjt.exe._execute_query(sql)

    def _clean_messages(self):
        for cjt in self.nodes.values():
            cjt.clean_message()

    def _comp_annotations(
        self, r_name: str, attr: str, cur_value: str, obj: float, expanding_cjt: CJT
    ):
        attr_type = expanding_cjt.get_feature_type(r_name, attr)
        g_col, h_col = self.semi_ring.get_columns_name()

        # Following https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support
        # this is to split the categorical feature based on whether the value is in the set or not
        if attr_type == "LCAT":

            absoprtion_view = expanding_cjt.absorption(r_name, [attr])

            # sort the absorption view by g_col/h_col according to the paper
            agg_exp = {
                attr: AggExpression(Aggregator.IDENTITY, 
                                    value_to_sql(attr,qualified=False)),
                "object": AggExpression(Aggregator.DIV, (g_col, h_col)),
                g_col: AggExpression(Aggregator.IDENTITY, g_col),
                h_col: AggExpression(Aggregator.IDENTITY, h_col),
            }
            spja_data = SPJAData(
                aggregate_expressions=agg_exp, from_tables=[absoprtion_view]
            )
            obj_view = self.cjt.exe.execute_spja_query(
                spja_data, mode=ExecuteMode.NESTED_QUERY
            )

            # use the prefix sum to get the cumulative gradient and hessian, just like for numerical features
            agg_exp = {
                attr: AggExpression(Aggregator.IDENTITY, 
                                    value_to_sql(attr,qualified=False)),
                g_col: AggExpression(Aggregator.SUM, g_col),
                h_col: AggExpression(Aggregator.SUM, h_col),
            }

            spja_data = SPJAData(
                aggregate_expressions=agg_exp, from_tables=[obj_view], window_by=["object"],
            )

            view_ord_by_obj = self.cjt.exe.execute_spja_query(
                spja_data, mode=ExecuteMode.NESTED_QUERY
            )

            # extract the set of categorical values that are less than the current value
            # currently, the set is stored in the model definition
            # maybe a better way is to store it in the database
            attr_spja_data = SPJAData(
                aggregate_expressions={attr: AggExpression(Aggregator.IDENTITY, 
                                                           value_to_sql(attr,qualified=False))},
                from_tables=[view_ord_by_obj],
                select_conds=[SelectionExpression(
                    SELECTION.NOT_GREATER, (f"{g_col}/{h_col}", str(obj)))]
            )

            attr_view = self.cjt.exe.execute_spja_query(
                attr_spja_data, mode=ExecuteMode.NESTED_QUERY
            )

            attrs = [
                str(x[0])
                for x in self.cjt.exe.execute_spja_query(
                    SPJAData(from_tables=[attr_view]), mode=ExecuteMode.EXECUTE
                )
            ]

            l_annotation = SelectionExpression(
                SELECTION.IN, (attr, attrs))
            r_annotation = SelectionExpression(
                SELECTION.NOT_IN, (attr, attrs))
            
        elif cur_value == "NULL":
            l_annotation = SelectionExpression(
                SELECTION.NULL, attr)
            r_annotation = SelectionExpression(
                SELECTION.NOT_NULL, attr)
        elif attr_type == "NUM":
            l_annotation = SelectionExpression(
                SELECTION.NOT_GREATER, (attr, cur_value))
            r_annotation = SelectionExpression(
                SELECTION.GREATER, (attr, cur_value))
        elif attr_type == "CAT":
            l_annotation = SelectionExpression(
                SELECTION.NOT_DISTINCT, (attr, cur_value))
            r_annotation = SelectionExpression(
                SELECTION.DISTINCT, (attr, cur_value))
        else:
            raise Exception("Unsupported Split")
        return l_annotation, r_annotation

    # get best split of current cjt
    def _get_best_split(self, cjt_id: int, cjt_depth: int):
        cjt = self.nodes[cjt_id]
        cur_semi_ring = cjt.get_semi_ring()
        g_col, h_col = self.semi_ring.get_columns_name()

        # criteria, (relation name, split attribute, split value, new s, new c)
        best_criteria, best_criteria_ann = 0, ("", "", 0, 0, 0)

        if cjt_depth == self.max_depth:
            self.split_candidates.put(
                (-best_criteria,cjt_depth,) + best_criteria_ann+ (cjt_id,))
            return

        g, h = cur_semi_ring.get_value()
        const_ = float((g**2) / h)

        # the next task is to compute the best split split among all the features
        # naively, we can iterate over all the features and compute the best split
        # alternatively, we batch the computation of the best split for all the features
        if not self.enable_batch_optimization:
            # if not batch optimization, we iterate over all the relations and their features
            for r_name in cjt.relations:
                for attr in cjt.get_relation_features(r_name):
                    attr_type, group_by = cjt.get_feature_type(r_name, attr), [attr]
                    absorption_view = cjt.absorption(r_name, group_by)

                    if attr_type == "NUM":
                        agg_exp = cur_semi_ring.col_sum((g_col, h_col))
                        # the query is over the absorption view, so attr is not qualified
                        agg_exp[attr] = AggExpression(Aggregator.IDENTITY,
                                                      value_to_sql(attr,qualified=False))
                        spja_data = SPJAData(aggregate_expressions=agg_exp,
                                             from_tables=[absorption_view],
                                             window_by=[attr])
                        view_to_max = self.cjt.exe.execute_spja_query(spja_data, mode=ExecuteMode.NESTED_QUERY)

                    elif attr_type == "LCAT":
                        # TODO: further optimization. We don't need to keep the attr.
                        # The only thing we care for splitting is the sum_s/sum_c
                        agg_exp = {
                            attr: AggExpression(Aggregator.IDENTITY,
                                                value_to_sql(attr,qualified=False)),
                            "object": AggExpression(Aggregator.DIV, (g_col, h_col)),
                            g_col: AggExpression(Aggregator.IDENTITY, g_col),
                            h_col: AggExpression(Aggregator.IDENTITY, h_col),
                        }
                        spja_data = SPJAData(
                            aggregate_expressions=agg_exp, from_tables=[
                                absorption_view]
                        )
                        obj_view = self.cjt.exe.execute_spja_query(
                            spja_data, mode=ExecuteMode.NESTED_QUERY
                        )
                        agg_exp = cur_semi_ring.col_sum((g_col, h_col))
                        agg_exp[attr] = AggExpression(Aggregator.IDENTITY,
                                                      value_to_sql(attr,qualified=False))
                        agg_exp["object"] = AggExpression(
                            Aggregator.IDENTITY, "object")

                        spja_data = SPJAData(
                            aggregate_expressions=agg_exp,
                            from_tables=[obj_view],
                            window_by=["object"],
                        )

                        view_to_max = self.cjt.exe.execute_spja_query(
                            spja_data, mode=ExecuteMode.NESTED_QUERY
                        )

                    elif attr_type == "CAT":
                        view_to_max = absorption_view

                    l2_agg_exp = {
                            attr: AggExpression(Aggregator.IDENTITY, value_to_sql(attr,qualified=False)),
                            # the case expression is for window functions
                            "criteria": AggExpression(Aggregator.CASE,
                                                      [(f"({g_col}/{h_col})*{g_col} + ({g}-{g_col})/({h}-{h_col})*({g}-{g_col})",
                                                        [SelectionExpression(SELECTION.LESSER, (str(h_col),str(h)))])]),
                            g_col: AggExpression(Aggregator.IDENTITY, g_col),
                            h_col: AggExpression(Aggregator.IDENTITY, h_col),
                        }

                    spja_data = SPJAData(
                        aggregate_expressions=l2_agg_exp,
                        from_tables=[view_to_max],
                        order_by=[("criteria", "DESC")],
                        limit=1
                        )

                    results = self.cjt.exe.execute_spja_query(spja_data, mode=ExecuteMode.EXECUTE)

                    if not results:
                        continue

                    cur_value, cur_criteria, left_g, left_h = results[0]

                    if cur_criteria > best_criteria:
                        best_criteria = cur_criteria
                        # relation name, split attribute, split value, left gradient, left hessian
                        best_criteria_ann = (
                            r_name, attr, str(cur_value), left_g, left_h)
                        
        # with batch optimization, we compute the best split for all the features              
        else:
            # TODO:  following currenly only work for Pandas, and split by numerical features            
            absorptions = []

            for relation in cjt.relations:
                features = cjt.get_relation_features(relation)
                absorption = cjt.absorption(relation, group_by=list(set(features)))
                absorption = cjt.exe.melt(absorption, 
                                          id_vars= self.semi_ring.get_columns_name(), 
                                          value_vars=features, 
                                          var_name='key',
                                          value_name='value')
                # set the relation name to dataframe
                absorption['relation'] = relation
                absorptions.append(absorption)

            result = cjt.exe.concat(absorptions)
            result = result.groupby(['relation', 'key', 'value']).sum().reset_index()
            result = result.sort_values(['relation', 'key', 'value'])
            result[[g_col, h_col]] = result.groupby(['relation', 'key'])[[g_col, h_col]].cumsum()

            if result[g_col].dtype != 'float64':
                result[g_col] = result[g_col].astype('float64')
            if result[h_col].dtype != 'float64':
                result[h_col] = result[h_col].astype('float64')

            
            result = result[result[h_col] < h]

            # these are for the total sum of g and h
            # this seems wasteful, but necessary for pandas eval
            result["ts"] = float(g)
            result["tc"] = float(h)

            result = result.reset_index().assign(criteria=result.eval('(s*s/c) + ((ts-s)* (ts - s))/(tc-c)'))
            idx = result.groupby(['relation', 'key'])['criteria'].idxmax()
            result = result.iloc[idx]

            max_row = result.nlargest(1, 'criteria')
            best_criteria = max_row["criteria"].iloc[-1]
            max_s = max_row[g_col].iloc[-1]
            max_c = max_row[h_col].iloc[-1]
            max_index = max_row["value"].iloc[-1]
            relation = max_row["relation"].iloc[-1]
            feature = max_row["key"].iloc[-1]


            # relation name, split attribute, split value, left gradient, left hessian
            best_criteria_ann = (relation, feature, str(max_index), max_s, max_c)

        self.split_candidates.put((const_ - float(best_criteria), cjt_depth,) + best_criteria_ann + (cjt_id,))

    # split the semi-ring according to current split
    def split_semi_ring(self, total_semi_ring: varSemiRing, left_semi_ring: varSemiRing):
        return left_semi_ring, total_semi_ring - left_semi_ring

    # don't update error for single decision tree
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

        while (
            # while there are still candidates to split
            not self.split_candidates.empty()
            # while the split is beneficial
            and self.split_candidates.queue[0][0] < 0
            # while the number of leaves is less than the max number of leaves
            and self.split_candidates.qsize() < self.max_leaves
        ):
            # get the best split
            (criteria, cur_level, r_name, attr, cur_value, left_g, left_h, c_id,) = self.split_candidates.get()
            # get the cjt of the best split to expand
            expanding_cjt = self.nodes[c_id]

            l_semi_ring = expanding_cjt.semi_ring.copy()
            l_semi_ring.set_semi_ring(left_g, left_h)

            l_semi_ring, r_semi_ring = self.split_semi_ring(
                expanding_cjt.get_semi_ring(), l_semi_ring
            )

            l_cjt, r_cjt, l_id, r_id = self._get_split_cjt(
                expanding_cjt=expanding_cjt,
                l_semi_ring=l_semi_ring,
                r_semi_ring=r_semi_ring,
            )

            # TODO: objective has some rounding problem
            l_annotation, r_annotation = self._comp_annotations(
                r_name=r_name,
                attr=attr,
                cur_value=cur_value,
                # currently it has an ugly solution. find a better solution
                obj=math.ceil(left_g / left_h * 100) / 100,
                expanding_cjt=expanding_cjt,
            )

            # add annotations according to split conditions
            l_cjt.add_annotation(r_name, l_annotation)
            r_cjt.add_annotation(r_name, r_annotation)

            # dim_relation_name = l_annotation.para[0].table_name

            l_cjt.downward_message_passing(r_name)
            r_cjt.downward_message_passing(r_name)

            # partition the target relation for the left subtree
            if self.partition_early:
                # naively, just get absorption
                # new_l_target_relation = l_cjt.absorption(l_cjt.target_relation, l_cjt.get_useful_attributes(l_cjt.target_relation), ExecuteMode.WRITE_TO_TABLE)
                # l_cjt.replace(l_cjt.target_relation, new_l_target_relation)

                # new_r_target_relation = r_cjt.absorption(r_cjt.target_relation, r_cjt.get_useful_attributes(r_cjt.target_relation), ExecuteMode.WRITE_TO_TABLE)
                # r_cjt.replace(r_cjt.target_relation, new_r_target_relation)

                # better, find which dimension attribute is used for split
                # then only perform semi-join with relation on that attribute
                new_l_target_relation = l_cjt.partition_target_relation(r_name)
                if new_l_target_relation is not None:
                    l_cjt.replace(l_cjt.target_relation, new_l_target_relation)
                    self.nodes[l_id] = l_cjt

                # partition the target relation for the right subtree
                # TODO: maybe this can be derived from left result instead of traversing tree again. Might not be worth it.
                new_r_target_relation = r_cjt.partition_target_relation(r_name)
                if new_r_target_relation is not None:
                    r_cjt.replace(r_cjt.target_relation, new_r_target_relation)
                    self.nodes[r_id] = r_cjt

            self._get_best_split(l_id, cur_level + 1)
            self._get_best_split(r_id, cur_level + 1)

        self.leaf_nodes = [self.nodes[ele[-1]] for ele in self.split_candidates.queue]


class GradientBoosting(DecisionTree):
    def __init__(
        self,
        max_leaves: int = 31,
        learning_rate: float = 1,
        max_depth: int = 6,
        iteration: int = 1,
        debug: bool = False,
        partition_early: bool = False,
    ):
        assert iteration > 0, "iteration should be positive"
        
        super().__init__(max_leaves, learning_rate, max_depth, debug=debug, partition_early=partition_early)
        self.iteration = iteration

    def _fit(self, jg: JoinGraph):
        super()._fit(jg)

        for _ in range(self.iteration - 1):
            self.train_one()

    def _update_error(self):

        # use the target_relation of the root node, because leaf nodes can be over partitioned
        target_relation = self.cjt.target_relation

        for cur_cjt in self.leaf_nodes:
            g, h = cur_cjt.get_semi_ring().get_value()
            pred = g / h * self.learning_rate
            _, join_conds = cur_cjt._get_income_messages(
                cur_cjt.target_relation, condition=2
            )

            g_col, _ = self.semi_ring.get_columns_name()
            self.cjt.exe.update_query(
                f"{g_col}={g_col}-({pred})", 
                target_relation, 
                join_conds + cur_cjt.get_annotations(target_relation),
                qualified=False,
            )


class RandomForest(DecisionTree):
    def __init__(
        self,
        max_leaves: int = 31,
        learning_rate: float = 1,
        max_depth: int = 6,
        subsample: float = 1,
        iteration: int = 1,
        debug: bool = False,
        partition_early: bool = False,
    ):
        assert iteration > 0, "iteration should be positive"

        super().__init__(max_leaves, learning_rate, max_depth, subsample, debug=debug, partition_early=partition_early)
        self.iteration = iteration
        self.learning_rate = 1 / iteration

    def _fit(self, jg: JoinGraph):

        for _ in range(self.iteration):
            super()._fit(jg)
