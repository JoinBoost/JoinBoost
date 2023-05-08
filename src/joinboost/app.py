import math
from abc import ABC

from .preprocessor import Preprocessor, RenameStep
from .executor import SPJAData, PandasExecutor, ExecuteMode
from .joingraph import JoinGraph
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
        g, h = jg.exe.execute_spja_query(spja_data, mode=ExecuteMode.EXECUTE)[0]

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
    ):

        super().__init__()
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.debug = debug
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
        exp = agg_to_sql(AggExpression(Aggregator.SUB, (self.cjt.target_var, str(self.constant_))))
        
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
            new_fact_name = self.cjt.exe.execute_spja_query(spja_data, mode=mode)
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
            annotations = cur_cjt.get_all_parsed_annotations()
            g, h = cur_cjt.get_semi_ring().get_value()

            pred = float(g / h) * self.learning_rate
            if annotations:
                cur_model_def.append((pred, annotations))
        if cur_model_def:
            self.model_def.append(cur_model_def)

    def compute_rmse(self, test_table: str):
        target = self.preprocessor.get_original_target_name()

        # TODO: refactor
        view = self.cjt.exe.case_query(
            test_table, "+", "prediction", str(self.constant_), self.model_def, [target]
        )

        predict_agg = {
            "RMSE": (f"SQRT(AVG(POW({target} - prediction, 2)))", Aggregator.IDENTITY)
        }
        prediction_query_data = SPJAData(
            aggregate_expressions=predict_agg, from_tables=[view]
        )

        predict = self.cjt.exe.execute_spja_query(
            prediction_query_data, mode=ExecuteMode.NESTED_QUERY
        )
        rmse_query_data = SPJAData(from_tables=[predict])
        return self.cjt.exe.execute_spja_query(
            rmse_query_data, mode=ExecuteMode.EXECUTE
        )[0]

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
            view = joingraph.exe.case_query(
                joingraph.target_relation,
                "+",
                "prediction",
                str(self.constant_),
                self.model_def,
                [self.cjt.target_var],
            )
        if input_mode == "JOIN_GRAPH":
            # TODO: reapply all the preprocessing steps
            self._update_fact_table_column_name(jg=joingraph, check_rowid_col=True)

            full_join = joingraph.get_full_join_sql()
            # the reason why we order by rowid is because of the set semantics of the relational models
            # e.g., for duckdb, join result has its row order shuffled, making it hard to decide the corresponding prediction
            # we therefore sort by rowid to enforce the correct ordering
            view = joingraph.exe.case_query(
                full_join,
                "+",
                "prediction",
                str(self.constant_),
                self.model_def,
                [self.cjt.target_var],
                order_by=f"{joingraph.target_relation}.rowid",
            )
            self._update_fact_table_column_name(jg=joingraph, resume_rowid_col=True)

        if output_mode == "NUMPY":
            preds = joingraph.exe._execute_query(f"select prediction from {view};")
            return np.array(preds)[:, 0]
        elif output_mode == "WRITE_TO_TABLE":
            return view

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
        attr_type = expanding_cjt.relation_schema[r_name][attr]
        g_col, h_col = self.semi_ring.get_columns_name()

        # TODO: remove window_query and everything is spja
        if attr_type == "LCAT":
            group_by = [attr]
            absoprtion_view = expanding_cjt.absorption(r_name, [attr])
            agg_exp = {
                attr: (attr, Aggregator.IDENTITY),
                "object": ((g_col, h_col), Aggregator.DIV),
                g_col: (g_col, Aggregator.IDENTITY),
                h_col: (h_col, Aggregator.IDENTITY),
            }
            spja_data = SPJAData(
                aggregate_expressions=agg_exp, from_tables=[absoprtion_view]
            )
            obj_view = self.cjt.exe.execute_spja_query(
                spja_data, mode=ExecuteMode.NESTED_QUERY
            )
            view_ord_by_obj = self.cjt.exe.window_query(
                obj_view, [attr], "object", [g_col, h_col]
            )
            attr_spja_data = SPJAData(
                aggregate_expressions={attr: (attr, Aggregator.IDENTITY)},
                from_tables=[view_ord_by_obj],
                # TODO: the {g_col}/{h_col} should be a qualified attribute 
                select_conds=[SelectionExpression(SELECTION.NOT_GREATER,(f"{g_col}/{h_col}",str(obj)))]
                
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
            agg_exp = {
                attr: (attr, Aggregator.IDENTITY),
                "object": ((g_col, h_col), Aggregator.DIV),
                g_col: (g_col, Aggregator.IDENTITY),
                h_col: (h_col, Aggregator.IDENTITY),
            }
            obj_spja_data = SPJAData(
                aggregate_expressions=agg_exp, from_tables=[absoprtion_view]
            )
            obj_view = self.cjt.exe.execute_spja_query(
                obj_spja_data, mode=ExecuteMode.NESTED_QUERY
            )
            view_ord_by_obj = self.cjt.exe.window_query(
                obj_view, [attr], "object", [g_col, h_col]
            )
            attr_view_data = SPJAData(
                aggregate_expressions={attr: (attr, Aggregator.IDENTITY)},
                from_tables=[view_ord_by_obj],
                select_conds=[SelectionExpression(SELECTION.NOT_GREATER,(f"{g_col}/{h_col}",str(obj)))]
            )
            attr_view = self.cjt.exe.execute_spja_query(
                attr_view_data, mode=ExecuteMode.NESTED_QUERY
            )

            attrs = [
                str(x[0])
                for x in self.cjt.exe.execute_spja_query(
                    SPJAData(from_tables=[attr_view]), mode=ExecuteMode.EXECUTE
                )
            ]
            l_annotation = SelectionExpression(SELECTION.IN, (QualifiedAttribute(r_name,attr), attrs))
            r_annotation = SelectionExpression(SELECTION.NOT_IN, (QualifiedAttribute(r_name,attr), attrs))
        elif cur_value == "NULL":
            l_annotation = SelectionExpression(SELECTION.NULL, QualifiedAttribute(r_name,attr))
            r_annotation = SelectionExpression(SELECTION.NOT_NULL, QualifiedAttribute(r_name,attr))
        elif attr_type == "NUM":
            l_annotation = SelectionExpression(SELECTION.NOT_GREATER, (QualifiedAttribute(r_name,attr), cur_value))
            r_annotation = SelectionExpression(SELECTION.GREATER, (QualifiedAttribute(r_name,attr), cur_value))
        elif attr_type == "CAT":
            l_annotation = SelectionExpression(SELECTION.NOT_DISTINCT, (QualifiedAttribute(r_name,attr), cur_value))
            r_annotation = SelectionExpression(SELECTION.DISTINCT, (QualifiedAttribute(r_name,attr), cur_value))
        else:
            raise Exception("Unsupported Split")
        return l_annotation, r_annotation

    # get best split of current cjt
    def _get_best_split(self, cjt_id: int, cjt_depth: int):
        cjt = self.nodes[cjt_id]
        cur_semi_ring = cjt.get_semi_ring()
        attr_meta = self.cjt.relation_schema
        g_col, h_col = self.semi_ring.get_columns_name()

        # criteria, (relation name, split attribute, split value, new s, new c)
        best_criteria, best_criteria_ann = 0, ("", "", 0, 0, 0)

        if cjt_depth == self.max_depth:
            self.split_candidates.put(
                (
                    -best_criteria,
                    cjt_depth,
                )
                + best_criteria_ann
                + (cjt_id,)
            )
            return

        g, h = cur_semi_ring.get_value()
        const_ = float((g**2) / h)
        for r_name in cjt.relations:
            for attr in cjt.get_relation_features(r_name):
                attr_type, group_by = self.cjt.get_type(r_name, attr), [attr]
                absoprtion_view = cjt.absorption(r_name, group_by)
                if attr_type == "NUM":
                    agg_exp = cur_semi_ring.col_sum((g_col, h_col))
                    agg_exp[attr] = (attr, Aggregator.IDENTITY)
                    spja_data = SPJAData(
                        aggregate_expressions=agg_exp,
                        from_tables=[absoprtion_view],
                        window_by=[attr],
                    )
                    view_to_max = self.cjt.exe.execute_spja_query(
                        spja_data, mode=ExecuteMode.NESTED_QUERY
                    )

                elif attr_type == "LCAT":
                    # TODO: further optimization. We don't need to keep the attr.
                    # The only thing we care for splitting is the sum_s/sum_c
                    agg_exp = {
                        attr: (attr, Aggregator.IDENTITY),
                        "object": ((g_col, h_col), Aggregator.DIV),
                        g_col: (g_col, Aggregator.IDENTITY),
                        h_col: (h_col, Aggregator.IDENTITY),
                    }
                    spja_data = SPJAData(
                        aggregate_expressions=agg_exp, from_tables=[absoprtion_view]
                    )
                    obj_view = self.cjt.exe.execute_spja_query(
                        spja_data, mode=ExecuteMode.NESTED_QUERY
                    )
                    agg_exp = cur_semi_ring.col_sum((g_col, h_col))
                    agg_exp[attr] = (attr, Aggregator.IDENTITY)
                    agg_exp["object"] = ("object", Aggregator.IDENTITY)
                    spja_data = SPJAData(
                        aggregate_expressions=agg_exp,
                        from_tables=[obj_view],
                        window_by=["object"],
                    )
                    view_to_max = self.cjt.exe.execute_spja_query(
                        spja_data, mode=ExecuteMode.NESTED_QUERY
                    )
                elif attr_type == "CAT":
                    view_to_max = absoprtion_view

                # check if executor is of type PandasExecutor or DuckdbExecutor
                # TODO: move this logic somewhere else
                if isinstance(self.cjt.exe, PandasExecutor):
                    func = (
                        lambda row: (row[f"{g_col}"] / row[f"{h_col}"])
                        * row[f"{g_col}"]
                        + ((g - row["s"]) / (h - row["c"])) * (g - row["s"])
                        if h > row["c"]
                        else 0
                    )
                else:
                    func = (
                        "CASE WHEN "
                        + str(h)
                        + f" > {h_col} THEN (({g_col}/{h_col})*{g_col} + ("
                        + str(g)
                        + f"-{g_col})/("
                        + str(h)
                        + f"-{h_col})*("
                        + str(g)
                        + f"-{g_col})) ELSE 0 END"
                    )

                l2_agg_exp = {
                    attr: (attr, Aggregator.IDENTITY),
                    "criteria": (func, Aggregator.IDENTITY_LAMBDA),
                    g_col: (g_col, Aggregator.IDENTITY),
                    h_col: (h_col, Aggregator.IDENTITY),
                }
                spja_data = SPJAData(
                    aggregate_expressions=l2_agg_exp,
                    from_tables=[view_to_max],
                    order_by=[("criteria", "DESC")],
                    limit=1,
                )
                results = self.cjt.exe.execute_spja_query(
                    spja_data, mode=ExecuteMode.EXECUTE
                )
                if not results:
                    continue
                cur_value, cur_criteria, left_g, left_h = results[0]
                # print((cur_value, cur_criteria, left_g, left_h))
                if cur_criteria > best_criteria:
                    best_criteria = cur_criteria
                    # relation name, split attribute, split value, left gradient, left hessian
                    best_criteria_ann = (r_name, attr, str(cur_value), left_g, left_h)
        self.split_candidates.put(
            (
                const_ - float(best_criteria),
                cjt_depth,
            )
            + best_criteria_ann
            + (cjt_id,)
        )

    # split the semi-ring according to current split
    def split_semi_ring(
        self, total_semi_ring: varSemiRing, left_semi_ring: varSemiRing
    ):
        return left_semi_ring, total_semi_ring - left_semi_ring

    # don't update error for single ecision tree
    def _update_error(self):
        pass

    def _get_split_cjt(
        self, expanding_cjt: CJT, l_semi_ring: varSemiRing, r_semi_ring: varSemiRing
    ):
        l_cjt, r_cjt = expanding_cjt.copy_cjt(l_semi_ring), expanding_cjt.copy_cjt(
            r_semi_ring
        )
        next_id = len(self.nodes)

        self.nodes[next_id] = l_cjt
        self.nodes[next_id + 1] = r_cjt

        return l_cjt, r_cjt, next_id, next_id + 1

    def _build_tree(self):
        self.cjt.calibration()
        self._get_best_split(0, 0)

        # while there are beneficial splits and doesn't read max leaves
        while (
            not self.split_candidates.empty()
            and self.split_candidates.queue[0][0] < 0
            and self.split_candidates.qsize() < self.max_leaves
        ):
            (
                criteria,
                cur_level,
                r_name,
                attr,
                cur_value,
                left_g,
                left_h,
                c_id,
            ) = self.split_candidates.get()
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
            # currently it has an ugly solution. find a better solution
            l_annotations, r_annotations = self._comp_annotations(
                r_name=r_name,
                attr=attr,
                cur_value=cur_value,
                obj=math.ceil(left_g / left_h * 100) / 100,
                expanding_cjt=expanding_cjt,
            )

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
            # print('level, right g, h')
            # print((cur_level, r_cjt.semi_ring.pair[0], r_cjt.semi_ring.pair[1]))
            self._get_best_split(r_id, cur_level + 1)

        self.leaf_nodes = [self.nodes[ele[-1]] for ele in self.split_candidates.queue]


class GradientBoosting(DecisionTree):
    # TODO: add some checks. E.g., paramters have to be positive
    def __init__(
        self,
        max_leaves: int = 31,
        learning_rate: float = 1,
        max_depth: int = 6,
        iteration: int = 1,
        debug: bool = False,
    ):
        super().__init__(max_leaves, learning_rate, max_depth, debug=debug)
        self.iteration = iteration

    def _fit(self, jg: JoinGraph):
        super()._fit(jg)

        for _ in range(self.iteration - 1):
            self.train_one()

    def _update_error(self):
        for cur_cjt in self.leaf_nodes:
            cur_cond = []
            target_relation = cur_cjt.target_relation
            g, h = cur_cjt.get_semi_ring().get_value()
            pred = g / h * self.learning_rate
            _, join_conds = cur_cjt._get_income_messages(
                cur_cjt.target_relation, condition=2
            )

            g_col, _ = self.semi_ring.get_columns_name()
            self.cjt.exe.update_query(
                f"{g_col}={g_col}-({pred})", target_relation, join_conds + cur_cjt.get_annotations(target_relation)
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
    ):
        super().__init__(max_leaves, learning_rate, max_depth, subsample, debug=debug)
        self.iteration = iteration
        self.learning_rate = 1 / iteration

    def _fit(self, jg: JoinGraph):

        for _ in range(self.iteration):
            super()._fit(jg)
