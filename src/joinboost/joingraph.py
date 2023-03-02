import copy
import time

from .executor import ExecutorFactory
import pkgutil


class JoinGraphException(Exception):
    pass


class JoinGraph:
    def __init__(
        self,
        exe=None,
        joins=None,
        relation_schema=None,
        target_var=None,
        target_relation=None,
    ):
        joins = joins if joins else {}
        relation_schema = relation_schema if relation_schema else {}

        self.exe = ExecutorFactory(exe)
        # maps each from_relation => to_relation => {keys: (from_keys, to_keys)}
        self._joins = copy.deepcopy(joins)
        # maps each relation => feature => feature_type
        self._relation_schema = copy.deepcopy(relation_schema)
        self._target_var = target_var
        self._target_relation = target_relation
        # some magic/random number used for jupyter notebook display
        self.session_id = int(time.time())
        self.rep_template = pkgutil.get_data(__name__, "d3graph.html").decode(
            "utf-8"
        )

    @property
    def relations(self):
        return list(self.relation_schema.keys())

    @property
    def relation_schema(self):
        return self._relation_schema

    @property
    def target_var(self):
        return self._target_var

    @property
    def target_relation(self):
        return self._target_relation

    @property
    def joins(self):
        return self._joins

    def get_type(self, relation, feature):
        return self.relation_schema[relation][feature]

    def has_join(self, table1, table2):
        return table1 in self.joins[table2] and table2 in self.joins[table1]

    def check_acyclic(self):
        seen = set()

        def dfs(cur_table, parent=None):
            seen.add(cur_table)
            for neighbour in self.joins[cur_table]:
                if neighbour != parent:
                    if neighbour in seen:
                        return False
                    else:
                        return dfs(neighbour, cur_table)
            return True

        # check acyclic
        if not dfs(list(self.joins.keys())[0]):
            raise JoinGraphException("The join graph is cyclic!")

        # check not disjoint
        if len(seen) != len(self.joins):
            raise JoinGraphException("The join graph is disjoint!")

    # add relation, features and target variable to join graph
    # current assumption: Y is in the fact table
    def add_relation(
        self,
        relation: str,
        X: list = None,
        y: str = None,
        categorical_feature: list = None,
        relation_address=None,
    ):

        X = X if X else []
        categorical_feature = categorical_feature if categorical_feature else []

        self.exe.add_table(relation, relation_address)
        self.joins[relation] = dict()
        if relation not in self.relation_schema:
            self.relation_schema[relation] = {}

        self.check_features_exist(relation, X + ([y] if y is not None else []))

        for x in X:
            # by default, assume all features to be numerical
            self.relation_schema[relation][x] = "NUM"

        for x in categorical_feature:
            self.relation_schema[relation][x] = "LCAT"

        if y is not None:
            if self.target_var is not None:
                print("Warning: Y already exists and has been replaced")
            self.target_var = y
            self.target_relation = relation

        # MATT: Add check_acyclic?

    def add_join(
        self,
        table_name_left: str,
        table_name_right: str,
        left_keys: list,
        right_keys: list,
    ):
        # MATT: underscore variables
        if len(left_keys) != len(right_keys):
            raise JoinGraphException("Join keys have different lengths!")
        if table_name_left not in self.relation_schema:
            raise JoinGraphException(table_name_left + " doesn't exit!")
        if table_name_right not in self.relation_schema:
            raise JoinGraphException(table_name_right + " doesn't exit!")

        left_keys = [attr for attr in left_keys]
        right_keys = [attr for attr in right_keys]

        self.joins[table_name_left][table_name_right] = {
            "keys": (left_keys, right_keys)
        }
        self.joins[table_name_right][table_name_left] = {
            "keys": (right_keys, left_keys)
        }

    # get features for each table
    def get_relation_features(self, r_name):
        if r_name not in self.relation_schema:
            raise JoinGraphException("Attribute not in " + r_name)
        return list(self.relation_schema[r_name].keys())

    # get the join keys between two tables
    # all get all the join keys of one table
    # TODO: check if the join keys exist
    def get_join_keys(self, f_table: str, t_table: str = None):
        if f_table not in self.joins:
            return []
        if t_table:
            if t_table not in self.joins[f_table]:
                raise JoinGraphException(t_table, " not connected to ", f_table)
            return self.joins[f_table][t_table]["keys"]
        else:
            keys = set()
            for table in self.joins[f_table]:
                l_keys, _ = self.joins[f_table][table]["keys"]
                keys = keys.union(set(l_keys))
            return list(keys)

    # useful attributes are features + join keys
    def get_useful_attributes(self, table):
        useful_attributes = self.get_relation_features(table) + self.get_join_keys(
            table
        )
        return list(set(useful_attributes))

    def get_full_join_sql(self):
        """Return the sql statement of full join."""

        sql = []
        seen = set()

        def dfs(rel1, parent=None):
            seen.add(rel1)
            for rel2 in self.joins[rel1]:
                if rel2 != parent:
                    if rel2 in seen:
                        return
                    else:
                        keys1, keys2 = self.get_join_keys(rel1, rel2)
                        key_sql = self._format_join_sql(rel1, rel2, keys1, keys2)
                        if not sql:
                            sql.append(f"{rel1} JOIN {rel2} ON {key_sql} ")
                        else:
                            sql.append(f"JOIN {rel2} ON {key_sql} ")
                        dfs(rel2, rel1)
            return

        dfs(list(self.joins.keys())[0])
        return "".join(sql)

    def _format_join_sql(self, rel1, rel2, keys1, keys2):
        sql = " AND ".join(
            f"{rel1}.{key1}={rel2}.{key2}" for key1, key2 in zip(keys1, keys2)
        )
        return sql

    def replace(self, table_prev, table_after):
        if table_prev not in self.relation_schema:
            raise JoinGraphException(table_prev + " doesn't exit!")
        if table_after in self.relation_schema:
            raise JoinGraphException(table_after + " already exits!")
        self.relation_schema[table_after] = self.relation_schema[table_prev]
        del self.relation_schema[table_prev]
        if self.target_relation == table_prev:
            self.target_relation = table_after

        for relation in self.joins:
            if table_prev in self.joins[relation]:
                self.joins[relation][table_after] = self.joins[relation][table_prev]
                del self.joins[relation][table_prev]

        if table_prev in self.joins:
            self.joins[table_after] = self.joins[table_prev]
            del self.joins[table_prev]

    def _preprocess(self):
        # self.check_all_features_exist()
        self.check_acyclic()

    # MATT: get rid of this?

    def check_target_exist(self):
        if self.target_var is None:
            raise JoinGraphException("Target variable doesn't exist!")

        if self.target_relation is None:
            raise JoinGraphException("Target relation doesn't exist!")

    def check_all_features_exist(self):
        for table in self.relation_schema:
            features = self.relation_schema[table].keys()
            self.check_features_exist(table, features)

    def check_features_exist(self, table, features):
        attributes = self.exe.get_schema(table)
        if not set(features).issubset(set(attributes)):
            raise JoinGraphException(
                "Key error in "
                + str(features)
                + ". Attribute does not exist in table "
                + table
                + " with schema "
                + str(attributes)
            )

    # output html that displays the join graph. Taken from JoinExplorer notebook
    def _repr_html_(self):
        nodes = []
        links = []
        for table_name in self.relation_schema:
            attributes = set(self.exe.get_schema(table_name))
            if table_name == self.target_relation:
                attributes.add(self.target_var)
            nodes.append({"id": table_name, "attributes": list(attributes)})

        # avoid edge in opposite direction
        seen = set()
        for table_name_left in self.joins:
            for table_name_right in self.joins[table_name_left]:
                if (table_name_right, table_name_left) in seen:
                    continue
                keys = self.joins[table_name_left][table_name_right]["keys"]
                links.append(
                    {
                        "source": table_name_left,
                        "target": table_name_right,
                        "left_keys": keys[0],
                        "right_keys": keys[1],
                    }
                )
                seen.add((table_name_left, table_name_right))

        self.session_id += 1

        s = self.rep_template
        s = s.replace("{{session_id}}", str(self.session_id))
        s = s.replace("{{nodes}}", str(nodes))
        s = s.replace("{{links}}", str(links))
        return s


#     def decide_feature_type(self, table, attrs, attr_types, threshold, exe: Executor):
#         self.relations.append(table)
#         r_meta = {}
#         for i, attr in enumerate(attrs):
#             if attr_types[i] == 2:
#                 r_meta[attr] = 'NUM'
#             else:
#                 r_meta[attr] = 'CAT'
#                 view = exe.execute_spja_query(aggregate_expressions={attr: (attr, Aggregator.DISTINCT_COUNT)},
#                                               f_table=table)
#                 res = exe.select_all(view)
#                 if res[0][0] <= threshold:
#                     r_meta[attr] = 'LCAT'
#         self.meta_data[table] = r_meta
#         self.r_attrs[table] = list(r_meta.keys())

# TODO: Check fact table and missing join keys
# auto dictionary encoding
# naming could conflict with semi-ring
# for prediction, what if two attributes have the same name?
# semi-join reduction for message to pass to
# todo: remove s,c logic from app
# infer executor from class
# app -> models
# support classification
# support predict based on fact table
# benchmark predict performance
