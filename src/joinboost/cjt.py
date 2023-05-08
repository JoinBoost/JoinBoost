import copy
from .semiring import SemiRing
from .joingraph import JoinGraph
from .executor import Executor, PandasExecutor, SPJAData, ExecuteMode
from .aggregator import *


class CJT(JoinGraph):
    def __init__(
        self, semi_ring: SemiRing, join_graph: JoinGraph, annotations: dict = {}
    ):
        self.message_id = 0
        self.semi_ring = semi_ring
        super().__init__(
            join_graph.exe,
            join_graph.joins,
            join_graph.relation_schema,
            join_graph.target_var,
            join_graph.target_relation,
        )
        # CJT get the join structure from this
        self.annotations = annotations

    def get_message(self, from_table: str, to_table: str):
        return self.joins[from_table][to_table]["message"]

    def get_annotations(self, table):
        # return self.annotations[table] if in the dict, otherwise return empty list
        return self.annotations.get(table, [])
        # return selections_to_sql(self.annotations[table])

    # not qualified by default becuase the join table is different from orginal table
    def get_all_parsed_annotations(self, qualified: bool = False):
        # self.annotations is a dict of selectionExpressions, and the key is the table name
        list_of_ann = []
        for _, value in self.annotations.items():
            list_of_ann.extend(value)

        return selections_to_sql(list_of_ann, qualified)

    def add_annotations(self, r_name: str, annotation: str):
        if r_name not in self.annotations:
            self.annotations[r_name] = [annotation]
        else:
            self.annotations[r_name].append(annotation)

    def clean_message(self):
        for from_table in self.joins:
            for to_table in self.joins[from_table]:
                if self.joins[from_table][to_table]["message_type"] != Message.IDENTITY:
                    m_name = self.joins[from_table][to_table]["message"]
                    self.exe.delete_table(m_name)

    def get_semi_ring(self):
        return self.semi_ring

    def copy_cjt(self, semi_ring: SemiRing):
        annotations = copy.deepcopy(self.annotations)
        c_cjt = CJT(semi_ring=semi_ring, join_graph=self,
                    annotations=annotations)
        return c_cjt

    def get_root_neighbors(self):
        joins, neighbors = self.joins, {}
        for table in joins[self.target_relation]:
            if (
                self.joins[table][self.target_relation]["message_type"]
                != Message.IDENTITY
            ):
                neighbors[table] = (
                    self.get_join_keys(self.target_relation, table),
                    self.get_message(table, self.target_relation),
                )
        return neighbors

    def calibration(self, rooto_table: str = None):
        if not rooto_table:
            rooto_table = self.target_relation
        self.upward_message_passing(rooto_table, m_type=Message.IDENTITY)
        self.downward_message_passing(rooto_table, m_type=Message.FULL)

    def downward_message_passing(
        self, rooto_table: str = None, m_type: Message = Message.UNDECIDED
    ):
        msgs = []
        if not rooto_table:
            rooto_table = self.target_relation
        self._pre_dfs(rooto_table, m_type=m_type)
        return msgs

    # TODO: this is not working if upward_message_passing from non-fact table
    def upward_message_passing(
        self, rooto_table: str = None, m_type: Message = Message.UNDECIDED
    ):
        if not rooto_table:
            rooto_table = self.target_relation
        self._post_dfs(rooto_table, m_type=m_type)

    def _post_dfs(
        self,
        currento_table: str,
        parent_table: str = None,
        m_type: Message = Message.UNDECIDED,
    ):
        jg = self.joins
        if currento_table not in jg:
            return
        for c_neighbor in jg[currento_table]:
            if c_neighbor != parent_table:
                self._post_dfs(c_neighbor, currento_table, m_type=m_type)
        if parent_table:
            self._send_message(
                from_table=currento_table, to_table=parent_table, m_type=m_type
            )

    def _pre_dfs(
        self,
        currento_table: str,
        parent_table: str = None,
        m_type: Message = Message.UNDECIDED,
    ):
        joins = self.joins
        if currento_table not in joins:
            return
        if currento_table == self.target_relation:
            m_type = Message.FULL
        for c_neighbor in joins[currento_table]:
            if c_neighbor != parent_table:
                self._send_message(currento_table, c_neighbor, m_type=m_type)
                self._pre_dfs(c_neighbor, currento_table, m_type=m_type)

    def absorption(self, table: str, group_by: list, mode=ExecuteMode.NESTED_QUERY):
        from_table_attrs = self.get_relation_features(table)
        incoming_messages, join_conds = self._get_income_messages(table)

        cols = self.semi_ring.get_columns_name()
        aggregate_expressions = self.semi_ring.col_sum(cols)
        for attr in group_by:
            aggregate_expressions[attr] = (
                table + "." + attr, Aggregator.IDENTITY)

        spja_data = SPJAData(
            aggregate_expressions=aggregate_expressions,
            from_tables=[m["message"] for m in incoming_messages] + [table],
            join_conds=join_conds,
            select_conds=self.get_annotations(table),
            group_by=[table + "." + attr for attr in group_by],
        )

        return self.exe.execute_spja_query(spja_data, mode=mode)

    # get the incoming message from one table to another
    # key function for message passing, Sec 3.3 of CJT paper
    # allow two types of join condition: 1 is for selection, 2 is for semi-join
    def _get_income_messages(
        self, table: str, excluded_table: str = "", condition=1, semi_join_opt=True
    ):
        incoming_messages, join_conds = [], []
        for neighbour_table in self.joins[table]:
            # if neighbour_table != excluded_table:
            incoming_message = self.joins[neighbour_table][table]
            if incoming_message["message_type"] == Message.IDENTITY:
                continue

            # semijoin optimization
            # Naively, the table is excluded
            # but we can use its message for semi-join to accelerate message passing
            if excluded_table == neighbour_table:
                if semi_join_opt:
                    incoming_message = copy.deepcopy(incoming_message)
                    incoming_message["message_type"] = Message.SELECTION
                else:
                    continue

            # get the join conditions between from_table and incoming_message
            l_join_keys, r_join_keys = self.get_join_keys(
                neighbour_table, table)
            incoming_messages.append(incoming_message)
            if condition == 1:
                join_conds += [
                    SelectionExpression(SELECTION.NOT_DISTINCT,
                                        (QualifiedAttribute(incoming_message["message"], l_join_keys[i]),
                                         QualifiedAttribute(table, r_join_keys[i])))
                    for i in range(len(l_join_keys))
                ]
            if condition == 2:
                join_conds += [
                    SelectionExpression(SELECTION.SEMI_JOIN,
                                        ([QualifiedAttribute(table, key) for key in r_join_keys],
                                         [QualifiedAttribute(incoming_message["message"], key) for key in l_join_keys]))
                ]
        return incoming_messages, join_conds

    # 3 message types: identity, selection, FULL
    def _send_message(
        self, from_table: str, to_table: str, m_type: Message = Message.UNDECIDED
    ):
        # print('--Sending Message from', from_table, 'to', to_table, 'm_type is', m_type)
        # identity message optimization
        if m_type == Message.IDENTITY:
            self.joins[from_table][to_table].update(
                {
                    "message_type": m_type,
                }
            )
            return

        if from_table not in self.joins and to_table not in self.joins[from_table]:
            raise Exception(
                "Table", from_table, "and table", to_table, "are not connected"
            )

        # join with incoming messages
        incoming_messages, join_conds = self._get_income_messages(
            from_table, to_table)

        # assume fact table. Relax it for many-to-many!!
        if m_type == Message.UNDECIDED:
            if from_table == self.target_relation:
                m_type = Message.FULL
            else:
                m_type = Message.SELECTION
                for message_type in [m["message_type"] for m in incoming_messages]:
                    if message_type == Message.FULL:
                        m_type = Message.FULL

        # get the group_by key for this message
        l_join_keys, _ = self.get_join_keys(from_table, to_table)

        # compute aggregation
        cols = self.semi_ring.get_columns_name()
        aggregation = self.semi_ring.col_sum(
            cols) if m_type == Message.FULL else {}
        for attr in l_join_keys:
            aggregation[attr] = (from_table + "." + attr, Aggregator.IDENTITY)

        spja_data = SPJAData(
            aggregate_expressions=aggregation,
            from_tables=[m["message"]
                         for m in incoming_messages] + [from_table],
            join_conds=join_conds,
            select_conds=self.get_annotations(from_table),
            group_by=[from_table + "." + attr for attr in l_join_keys],
        )
        message_name = self.exe.execute_spja_query(
            spja_data, mode=ExecuteMode.WRITE_TO_TABLE
        )

        self.joins[from_table][to_table].update(
            {"message": message_name, "message_type": m_type}
        )

    # by default, lift the target variable
    def lift(self, var=None):
        if var is None:
            var = self.target_var
        lift_exp = self.semi_ring.lift_exp(var)
        # TODO: remove hack
        if isinstance(self.exe, PandasExecutor):
            lift_exp["s"] = (var, Aggregator.IDENTITY_LAMBDA)
        # copy the rest attributes
        for attr in self.get_useful_attributes(self.target_relation):
            lift_exp[attr] = (attr, Aggregator.IDENTITY)

        spja_data = SPJAData(
            aggregate_expressions=lift_exp, from_tables=[self.target_relation]
        )
        new_fact_name = self.exe.execute_spja_query(
            spja_data, mode=ExecuteMode.WRITE_TO_TABLE
        )
        self.replace(self.target_relation, new_fact_name)
