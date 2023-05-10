import time

from .aggregator import *
import copy
import time

from .executor import ExecutorFactory, SPJAData, ExecuteMode
import pkgutil


class JoinGraphException(Exception):
    pass


class JoinGraph:
    def __init__(
        self,
        exe=None,
        joins={},
        relations={},
        target_var=None,
        target_relation=None,
    ):
        self.exe = ExecutorFactory(exe)

        # join graph edge information
        # joins maps each relation => {joined relation: {keys: (from_keys, to_keys), 
        #                                                message_type: "", message: name,
        #                                                multiplicity: x, missing_keys: x ...}}
        self.joins = copy.deepcopy(joins)

        # join graph node information
        # relation_schema maps each relation => {feature: feature_type}
        self.relations = copy.deepcopy(relations)
        
        self.target_var = target_var
        self.target_relation = target_relation

        # TODO: this is a hack, we should have a better way to handle this
        self._target_rowid_colname = ""

        # some magic/random number used for jupyter notebook display
        self.session_id = int(time.time())
        
        # template for jupyter notebook display
        self.rep_template = pkgutil.get_data(__name__,"static_files/dashboard.html").decode('utf-8')


    def copy(self):
        return JoinGraph(
            self.exe,
            self.joins,
            self.relations,
            self.target_var,
            self.target_relation,
        )
    
    # the @property decorator is used to define properties,
    # these properties are read-only
    # example usage:
    # join_graph = JoinGraph()
    # join_graph.relations
    # join_graph.relations = {}
    # the last line will raise an error
    # because relations is read-only
    @property
    def relations(self):
        return list(self.relations.keys())

    @property
    def target_rowid_colname(self):
        return self._target_rowid_colname

    @property
    def relations(self):
        return self._relations

    @property
    def target_var(self):
        return self._target_var

    @property
    def target_relation(self):
        return self._target_relation

    @property
    def joins(self):
        return self._joins
    
    # the @property_name.setter
    # is used to define the setter of a property
    # example usage to set the joins property:
    # join_graph = JoinGraph()
    # join_graph.joins = {}
    # the last line will call the setter of joins
    # and set the joins property to {}
    @joins.setter
    def joins(self, value):
        assert isinstance(value, dict)
        self._joins = value

    @relations.setter
    def relations(self, value):
        assert isinstance(value, dict)
        self._relations = value

    @target_var.setter
    def target_var(self, value):
        assert isinstance(value, str) or value is None
        self._target_var = value

    @target_relation.setter
    def target_relation(self, value):
        assert isinstance(value, str) or value is None
        self._target_relation = value

    def has_join(self, table1, table2):
        return table1 in self.joins[table2] and table2 in self.joins[table1]
    
    def has_relation(self, relation):
        return relation in self.relations
    
    def get_type(self, relation, feature):
        return self.relations[relation][feature]
    
    # useful attributes are features + join keys
    def get_useful_attributes(self, table):
        useful_attributes = self.get_relation_features(table) + self.get_join_keys(table)
        return list(set(useful_attributes))
    
    def check_graph_validity(self):
        # for each graph edge, check if the end node is in the graph
        for relation in self.joins:
            for relation2 in self.joins[relation]:
                if not self.has_relation(relation):
                    raise JoinGraphException(relation + " doesn't exist!", "the relations are " + str(self.relations))
                if not self.has_relation(relation2):
                    raise JoinGraphException(relation2 + " doesn't exist!", "the relations are " + str(self.relations))
    
    # replace a table from table_prev to table_after
    def replace(self, table_prev, table_after):
        if not self.has_relation(table_prev):
            raise JoinGraphException(table_prev + " doesn't exit!")
        
        if self.has_relation(table_after):
            raise JoinGraphException(table_after + " already exits!")

        # replace the table name in relations
        self.relations[table_after] = self.relations[table_prev]
        del self.relations[table_prev]

        # replace the target_relation if necessary
        if self.target_relation == table_prev:
            self._target_relation = table_after

        # replace the joins
        for relation in self.joins:
            if table_prev in self.joins[relation]:
                self.joins[relation][table_after] = self.joins[relation][table_prev]
                del self.joins[relation][table_prev]

        if table_prev in self.joins:
            self.joins[table_after] = self.joins[table_prev]
            del self.joins[table_prev]

    # replace a table's attrbute from before_attribute to after_attribute
    def replace_relation_attribute(self, relation, before_attribute, after_attribute):
        # check if the relation exists
        if not self.has_relation(relation):
            raise JoinGraphException(relation + " doesn't exist!")
        
        # check if the attribute exists
        if before_attribute not in self.relations[relation]:
            raise JoinGraphException(before_attribute + " doesn't exist in " + relation + "!")
        
        # replace the attribute in relation schema
        if before_attribute in self.relations[relation]:
            self.relations[relation][after_attribute] = self.relations[relation][before_attribute]
            del self.relations[relation][before_attribute]
        
        # replace the target_var if necessary
        if relation == self.target_relation:
            if self.target_var == before_attribute:
                self.target_var = after_attribute

        # replace the attribute in joins as join keys
        for relation2 in self.joins[relation]:
            left_join_key = self.joins[relation][relation2]["keys"][0]
            if before_attribute in left_join_key:
                # Find the index of the before_attribute in the list
                index = left_join_key.index(before_attribute)
                # Replace the old string with the new string
                left_join_key[index] = after_attribute
    # check if the join graph is acyclic
    def check_acyclic(self):
        
        seen = set()
        # dfs to check if the graph is acyclic
        # it uses the fact that if a node is visited twice, then the graph is cyclic
        def dfs(cur_table, parent=None):
            seen.add(cur_table)
            for neighbour in self.joins[cur_table]:
                if neighbour != parent:
                    if neighbour in seen:
                        return False
                    else:
                        acyclic = dfs(neighbour, parent=cur_table)
                        if acyclic is False:
                            return False
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
        X: list = [],
        y: str = None,
        categorical_feature: list = [],
        relation_address=None,
    ):      
        # add relation to the join graph if the address is not None
        if relation_address is not None:
            # check if the relation exists
            if relation in self.relations:
                raise JoinGraphException(relation + " already exists!")
            self.exe.add_table(relation, relation_address)
            
        self.joins[relation] = dict()
        if relation not in self.relations:
            self.relations[relation] = {}

        attributes = self.check_features_exist(
            relation, X + ([y] if y is not None else [])
        )

        for x in X:
            # by default, assume all features to be numerical
            self.relations[relation][x] = "NUM"

        for x in categorical_feature:
            self.relations[relation][x] = "LCAT"

        if y is not None:
            if self.target_var is not None:
                print("Warning: Y already exists and has been replaced")
            self.target_var = y
            self.target_relation = relation
            self._target_rowid_colname = self._get_target_rowid_colname(attributes)

    def _get_target_rowid_colname(self, attributes):
        """Get the temporary rowid column name(if exists) for the target relation. If not exists, set to empty string."""

        attr = set(attributes)
        tmp = "rowid"
        while tmp in attr:
            tmp = "joinboost_tmp_" + tmp
        return tmp if tmp != "rowid" else ""

    # get features for each table
    def get_relation_features(self, r_name):
        if r_name not in self.relations:
            raise JoinGraphException("Attribute not in " + r_name)
        return list(self.relations[r_name].keys())

    # get the join keys between two tables
    # all get all the join keys of one table
    # TODO: check if the join keys exist
    # if t_table is not None, get the join keys between f_table and t_table
    # if t_table is None, all get all the join keys of f_table
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

    def add_join(
        self,
        table_name_left: str,
        table_name_right: str,
        left_keys: list,
        right_keys: list,
    ):
        if len(left_keys) != len(right_keys):
            raise JoinGraphException("Join keys have different lengths!")
        if table_name_left not in self.relations:
            raise JoinGraphException(table_name_left + " doesn't exit!")
        if table_name_right not in self.relations:
            raise JoinGraphException(table_name_right + " doesn't exit!")

        left_keys = [attr for attr in left_keys]
        right_keys = [attr for attr in right_keys]

        self.joins[table_name_left][table_name_right] = {"keys": (left_keys, right_keys)}
        self.joins[table_name_right][table_name_left] = {"keys": (right_keys, left_keys)}
        
        self.determine_multiplicity_and_missing(
            table_name_left, left_keys, table_name_right, right_keys)
        
    def determine_multiplicity_and_missing(self,
                                           relation_left: str,
                                           leftKeys: list,
                                           relation_right: str,
                                           rightKeys: list):

        num_miss_left, num_miss_right = self.get_num_missing_join_keys(relation_left,
                                                                       leftKeys,
                                                                       relation_right,
                                                                       rightKeys)

        self.joins[relation_right][relation_left]["missing_keys"] = num_miss_left
        self.joins[relation_left][relation_right]["missing_keys"] = num_miss_right

        self.joins[relation_left][relation_right]["multiplicity"] = \
            self.get_max_multiplicity(relation_left, leftKeys)
        self.joins[relation_right][relation_left]["multiplicity"] = \
            self.get_max_multiplicity(relation_right, rightKeys)

    def get_num_missing_join_keys(self,
                                  relation_left: str,
                                  leftKeys: list,
                                  relation_right: str,
                                  rightKeys: list):
        # below two queries get the set of join keys
        spja_data = SPJAData(
            # aggregate_expressions is a dictionary: for ith leftKey, the key is "key_i", the value is the ith leftKey
            aggregate_expressions = {f"key_{i}": AggExpression(Aggregator.IDENTITY, leftKeys[i]) for i in range(len(leftKeys))},
            from_tables=[relation_left],
            group_by=[",".join(leftKeys)]
        )
        set_left = self.exe.execute_spja_query(spja_data, mode=ExecuteMode.NESTED_QUERY)
        spja_data = SPJAData(
            aggregate_expressions = {f"key_{i}": AggExpression(Aggregator.IDENTITY, rightKeys[i]) for i in range(len(rightKeys))},
            from_tables=[relation_right],
            group_by=[",".join(rightKeys)]
        )
        set_right = self.exe.execute_spja_query(spja_data, mode=ExecuteMode.NESTED_QUERY)

        # below two queries get the difference of join keys
        diff_left = self.exe.set_query("EXCEPT", set_left, set_right)
        diff_right = self.exe.set_query("EXCEPT", set_right, set_left)

        spja_data = SPJAData(
            aggregate_expressions={'count': AggExpression(Aggregator.COUNT, '*')},
            from_tables=[diff_left]
        )
        # get the count of the difference of join keys
        res = self.exe.execute_spja_query(spja_data, mode=ExecuteMode.EXECUTE)

        if len(res) == 0:
            num_miss_left = 0
        else:
            num_miss_left = res[0][0]

        spja_data = SPJAData(
            aggregate_expressions={'count': AggExpression(Aggregator.COUNT, '*')},
            from_tables=[diff_right]
        )
        res = self.exe.execute_spja_query(spja_data, mode=ExecuteMode.EXECUTE)
        if len(res) == 0:
            num_miss_right = 0
        else:
            num_miss_right = res[0][0]

        return num_miss_left, num_miss_right

    def get_max_multiplicity(self, table, keys):
        spja_data = SPJAData(
            # TODO： why the second argument can't be '*'? COUNT don't really care about the argument
            aggregate_expressions={'count': AggExpression(Aggregator.COUNT, ','.join(keys))},
            from_tables=[table],
            group_by=keys
        )
        multiplicity = self.exe.execute_spja_query(spja_data, mode=ExecuteMode.NESTED_QUERY)

        spja_data = SPJAData(
            aggregate_expressions={'max_count': AggExpression(Aggregator.MAX, 'count')},
            from_tables=[multiplicity],
            )

        res = self.exe.execute_spja_query(spja_data, mode=ExecuteMode.EXECUTE)
        if len(res) == 0:
            max_multiplicity = 0
        else:
            max_multiplicity = res[0][0]

        return max_multiplicity

    def get_multiplicity(self, from_table, to_table, simple=False):
        if not simple:
            return self.joins[from_table][to_table]["multiplicity"]
        else:
            return "M" if (self.joins[from_table][to_table]["multiplicity"] > 1) else "1"

    def get_missing_keys(self, from_table, to_table):
        return self.joins[from_table][to_table]["missing_keys"]


    # Return the sql statement of full join
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

    def check_target_relation_contains_rowid_col(self):
        return len(self.target_rowid_colname) != 0

    def _format_join_sql(self, rel1, rel2, keys1, keys2):
        sql = " AND ".join(
            f"{rel1}.{key1}={rel2}.{key2}" for key1, key2 in zip(keys1, keys2)
        )
        return sql

    def _preprocess(self):
        self.check_graph_validity()
        self.check_all_features_exist()
        self.check_acyclic()
        self.check_target_exist()
        self.check_target_is_fact()
        
    
         
    
    # Below maybe move to preprocess
    def check_target_is_fact(self):
        seen = set()

        def dfs(rel1, parent=None):
            seen.add(rel1)
            for rel2 in self.joins[rel1]:
                if rel2 != parent:
                    if rel2 in seen:
                        return
                    else:
                        multiplicity = self.get_multiplicity(rel2, rel1)
                        if multiplicity != 1:
                            raise JoinGraphException(f"""
The target table doesn't have many-to-one relationship with the rest.
Please check the multiplicity between relations {rel2} and {rel1}.
                            """)
                        missing_keys = self.get_missing_keys(rel2, rel1)
                        if missing_keys != 0:
                            raise JoinGraphException(f"""
The dimension table have missing key along the path to the target.
Please check the missing key between relations {rel2} and {rel1}.
                            """)
                        
            return

        dfs(self.target_relation)
    
        if self.target_var is None:
            raise JoinGraphException("Target variable doesn't exist!")

        if self.target_relation is None:
            raise JoinGraphException("Target relation doesn't exist!")
    
    # Below maybe move to preprocess
    def check_target_exist(self):
        if self.target_var is None:
            raise JoinGraphException("Target variable doesn't exist!")

        if self.target_relation is None:
            raise JoinGraphException("Target relation doesn't exist!")
            
    # Below maybe move to preprocess
    def check_target_exist(self):
        if self.target_var is None:
            raise JoinGraphException("Target variable doesn't exist!")

        if self.target_relation is None:
            raise JoinGraphException("Target relation doesn't exist!")

    def check_all_features_exist(self):
        for table in self.relations:
            features = self.relations[table].keys()
            self.check_features_exist(table, features)

    def check_features_exist(self, relation, features):
        """Check if all the features exist in the relation."""

        attributes = self.exe.get_schema(relation)
        if not set(features).issubset(set(attributes)):
            raise JoinGraphException(
                f"Key error in {features}."
                + f" Attribute does not exist in table {relation} with schema {attributes}"
            )
        return attributes
    
    '''
    node structure:
    nodes: [
        { 
            id: relation,
            attributes: [dim1, dim2, join_key1, join_key2, measurement1, measurement2],
            join_keys: [
                {
                    key: col1
                    multiplicity: many/one
                },
            ],
            measurements: [ 
            {
                name: AVG(A,..),
                relations: [
                {name: t1, should_highlight: True/False, color: None},
                {name: t2, should_highlight: True/False, color: None}],
                edges: [
                {left_rel: t1, right_rel: t2, should_highlight: True/False, color: None},
                ]
            }, 
            {..}
            ]
        }
    ]
    
    edge structure: [
        {
            source: node_id,
            left_keys: [key1, key2, ...],
            dest: node_id,
            right_keys: [key1, key2, ...],
        }
    ]
    
    Edge and relation also stores:
            highlight: true/false (control the opacity)
            color: black by default
    These two can be updated in js function through interaction
    '''
    def get_graph(self):
        nodes = []
        links = []

        # avoid edge in opposite direction
        seen = set()
        for relation_left in self.joins:
            for relation_right in self.joins[relation_left]:
                if (relation_right, relation_left) in seen: continue
                links.append({"source": relation_left,
                              "target": relation_right,
                              "left_keys": self.get_join_keys(relation_left, relation_right),
                              "right_keys": self.get_join_keys(relation_right, relation_left),
                              "multiplicity": [self.get_multiplicity(relation_left, relation_right, simple=True),
                                               self.get_multiplicity(relation_right, relation_left, simple=True)],
                              "missing":[self.get_missing_keys(relation_left, relation_right),
                                         self.get_missing_keys(relation_right, relation_left)],
                              })
                seen.add((relation_left, relation_right))

        for relation in self.relations:
            join_keys = set(self.get_join_keys(relation))
            attributes = set(self.get_useful_attributes(relation))
            measurements = []

            nodes.append({"id": relation,
                          "name": relation,
                          "measurements": measurements,
                          "attributes": list(attributes),
                          "join_keys": list(join_keys),
                          })
        return nodes, links

    def _repr_html_(self):
        nodes, links = self.get_graph()
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




