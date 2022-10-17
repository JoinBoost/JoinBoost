import math
import copy
import subprocess
from queue import PriorityQueue
import duckdb
import time
from types import MethodType
import pandas as pd
import numpy as np
import subprocess
import threading

class cjt:
    def __init__(self, cur_id, parent=None, tc = 0, ts = 0, depth = 1):
        self.id = cur_id
        self.parent = -1
        self.is_leaf = False
        self.left_child = None
        self.right_child = None
       
        if parent is not None:
            self.parent = parent.id
            self.depth = parent.depth+1
            self.messages = copy.deepcopy(parent.messages)
            self.annotations = copy.deepcopy(parent.annotations)
        else:
            self.depth = depth
            self.messages = {}
            self.annotations = {}
            
        self.tc = tc
        self.ts = ts
    
    def set_leaf(self, state=True):
        self.is_leaf = True
    
    def add_annotation(self, table, annotation):
        if table not in self.annotations:
            self.annotations[table] = []
        self.annotations[table].append(annotation)
        
    def get_annotations(self, table):
        if table not in self.annotations:
            return []
        return self.annotations[table]
    
    def get_all_annotations(self):
        annotations = []
        for table in self.annotations:
            annotations += self.annotations[table]
        return annotations
    
    # for each node, what is the split condition for child node? 
    # used to efficiently build the tree conditions.
    def set_split_condition(self, table, feature, feature_type, value):
        self.split = (table, feature, feature_type, value)
    
    def set_children(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child
        
        
class joinGraph:
    def __init__(self, name, con, max_leaves = 8, min_samples_split=2, 
                 learning_rate=0.1, target_variable = "Y", log = False, 
                 max_depth=100, fact_pandas=False, fact_pandas_join=False, ):
        # store the table name -> features 
        # store the table name -> feature types
        self.features = dict()
        # 1 for nominal, 2 for ordinal
        self.feature_types = dict()
        # store the join keys between two tables
        self.joins = dict()
        self.total_messages = 0 
        self.total_nodes = 0
        self.name = name
        self.fact = ""
        self.smallest = ""
        self.cjts = dict()
        self.leaves = PriorityQueue()
        # some special separator not in the database
        self.separator = chr(1)
        self.max_leaves = max_leaves
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.target_variable = target_variable
        self.sql_log = ""
        self.log = log
        self.totaltime = 0
        self.max_depth = max_depth
        self.base_value = 0
        self.tree_queries = []
        self.fact_pandas = fact_pandas
        # the fact table is the full join result
        self.fact_pandas_join = fact_pandas_join
        self.con = con
        
    # execute it after "initialize_model_table"
    def get_fact_sum(self):
        if not self.fact_pandas:
            query = "SELECT SUM(CAST(" + self.target_variable + " AS DOUBLE)) AS TS, COUNT(*) AS TC FROM " + self.fact + "\n"

            results = self.execute_query(query)
    #         TS, TC = results[2].split(self.separator)
            TS, TC = results[0]
            return float(TS), int(TC)
        
        if self.fact_pandas:
            TS = self.fact_table[self.target_variable].sum()
            TC = len(self.fact_table)
            return float(TS), int(TC)
    
        
    def execute_query(self, query):
        if self.log:
            self.sql_log += query + "\n"
        if self.log:
            print(query)
        start_time = time.time()
        self.con.execute(query)
        elapsed_time = time.time() - start_time
        self.totaltime += elapsed_time
        result = None
        try:
            result = self.con.fetchall()
            if self.log:
                print(result)
        except Exception as e: 
            print(e)
        if self.log:
            print(elapsed_time)
        return result
    
    def next_cjt_id(self):
        self.total_nodes += 1
        return self.total_nodes - 1
    
    
    def initialize_model_table(self, base_prediction):
        
        sql = "DROP TABLE IF EXISTS gbm_model_table;\n"
        sql += "CREATE TABLE gbm_model_table AS\n"
#         sql += "SELECT CAST('' AS varchar(max)) AS PREDICATE, " + str(base_prediction) + " AS PREDICTION_VALUE\n"
        sql += "SELECT CAST('' AS varchar) AS PREDICATE, " + str(base_prediction) + " AS PREDICTION_VALUE\n"
#         sql += "INTO gbm_model_table;\n"
        # sql += "SET IDENTITY_INSERT gbm_model_table ON;\n"
        
        return sql
    
    def create_dummy_model(self,replace=False):
        
        ts, tc = self.get_fact_sum()
        
        # dummy model use average
        prediction_value = ts/tc
        if not self.fact_pandas:
            query = self.preprocess_fact_table(prediction_value, replace)
            self.execute_query(query)
        if self.fact_pandas:
            start_time = time.time()
            self.fact_table["c"] = 1
            self.fact_table["s"] = self.fact_table[self.target_variable] - prediction_value
            elapsed_time = time.time() - start_time
            self.totaltime += elapsed_time
        
        
#         query = self.initialize_model_table(prediction_value)
#         self.execute_query(query)
        self.base_value = prediction_value
        # upated ts and tc after dummy model
        ts, tc = (ts - ts/tc * tc), tc
        self.ts, self.tc = 0, tc
        return ts, tc
    
    def set_ts_tc(self, ts, tc):
        self.ts, self.tc = ts, tc
    
    def create_base_node(self):
        
        base_cjt = cjt(self.next_cjt_id(), ts = self.ts, tc = self.tc)
        self.cjts[base_cjt.id] = base_cjt
        
        sqls = self.upward_message_passing(self.fact, base_cjt, identity=True)
        query = "\n".join(sqls)
#         Don't need to execute since all empty
#         self.execute_query(query)
        
        sqls = self.downward_message_passing(self.fact, base_cjt)
#         query = "\n".join(sqls)
#         self.execute_query(query)
        
        # find the best split
        self.best_split(base_cjt)
    
    def get_sum_semiring(self, cjt_id):
        query = self.absorption(self.smallest, self.cjts[cjt_id])
        results = self.execute_query(query)
        c,s = results[2].split(self.separator)
        self.cjts[cjt_id].set_sum_semiring(int(c),int(s))
    
    def check_target_variable_no_null(self, target_variable):
        sql = "IF 0 != (SELECT COUNT(CASE WHEN " + target_variable + " IS NULL THEN 1 END) FROM " + self.fact + " )\n"
        sql+= 'ELSE\n PRINT N\'ERROR: There are missing values in the target value of fact table!!!\';\n'
        return sql
    
    # we choose DECIMAL to avoid overflow
    # TODO: check data type
    def preprocess_fact_table(self, base_prediction = 0, replace=False):
        sql = ""
        if replace:
            sql += "ALTER TABLE " + self.fact + " DROP COLUMN c;\n"
            sql += "ALTER TABLE " + self.fact + " DROP COLUMN s;\n"
        sql += "ALTER TABLE " + self.fact + " ADD COLUMN c DOUBLE DEFAULT 1;\n"
        sql += "ALTER TABLE " + self.fact + " ADD COLUMN s DOUBLE;\n"
        sql += "UPDATE " + self.fact + " SET s = " + self.target_variable + " - " + str(base_prediction) + " ;\n"
        
        return sql
    
    # set a smallest table such that we can get the sum semi-ring efficiently
    def set_smallest_table(self, table):
        self.smallest = table
   
    def add_table(self, name: str, features: list, feature_types: list, replace = False, fact = False, fact_table=None):
        if name in self.features and replace == False:
            raise Exception('There has already been a table named', name, 'If you want to replace the table, set replace = True.')
        
        if len(features) != len(feature_types):
            raise Exception('There are', len(features), 'but only', len(feature_types), 'feature types.')
            
        self.features[name] = features
        self.feature_types[name] = feature_types
        
        if fact:
            self.fact = name
        
        if fact and self.fact_pandas:
            self.fact_table = fact_table
    
    def get_table(self, name: str):
        if name not in self.tables:
            raise Exception('There is no table named', name)
        return self.tables[name]
    
    def clean_leaves(self):
        sql = "CASE "
        leaf_queris = []
        
        for i in range(len(self.leaves.queue)):
            cjt = self.cjts[self.leaves.queue[i][1][0]]
            # leaf nodes will be used to build tree
            cjt.set_leaf()
            
            ann = cjt.get_all_annotations()
            pre = cjt.ts/cjt.tc * self.learning_rate
            if len(ann) == 0:
                print("annotations have size 0???")
                continue
            # for now do the split, because each annotation is: relation.attribute ...
            # in the future, store annotation with attribute and predicate separately
            sql += " WHEN " + " AND ".join([a.split(".",1)[-1] for a in ann]) + " THEN CAST("+ str(pre) + " AS DOUBLE)\n"
            
        sql += " ELSE 0 END\n"
        self.tree_queries.append(sql)
        self.leaves = PriorityQueue()
    
    # if all cjts removed, can't rebuild the model
    def clean_table(self):
        sql = ""
        for i in range(self.total_messages):
            sql += "DROP TABLE IF EXISTS " + self.name + "_m" + str(i) + ";\n"
            
        for i in range(self.max_leaves):
            sql += "DROP VIEW IF EXISTS " + self.name + "_v" + str(i) + ";\n"
            
        self.total_nodes = 0
        self.total_messages = 0
        self.cjts = dict()
        self.leaves = PriorityQueue()
        self.execute_query(sql)
        
    def join(self, table_name_left: str, table_name_right: str, left_keys: list, right_keys: list):
        if len(left_keys) != len(right_keys):
            raise Exception('Join keys have different lengths!')
        if table_name_left not in self.features:
            raise Exception(table_name_left, 'table doesn\'t exit!')
        if table_name_right not in self.features:
            raise Exception(table_name_right, 'table doesn\'t exit!')
        
        if len(left_keys) != 1:
            raise Exception('only allow single attribute join key for performance!')
        if len(right_keys) != 1:
            raise Exception('only allow single attribute join key for performance!')
        
        if table_name_left not in self.joins:
            self.joins[table_name_left] = dict()
        if table_name_right not in self.joins:
            self.joins[table_name_right] = dict()
        
        self.joins[table_name_left][table_name_right] = (left_keys, right_keys)
        self.joins[table_name_right][table_name_left] = (right_keys, left_keys)
    
    def check_acyclic(self):
        seen = set()
        
        # is this right?
        # counter example: A,B,C connects together
        # allows disconnected components?
        def dfs(table_name):
            seen.add(table_name)
            for neighbour in self.joins[table_name]:
                if neighbour in seen:
                    if table_name not in self.joins[neighbour]:
                        return False
                else:
                    dfs(neighbour)
            return True
        
        for table_name in self.joins:
            if table_name not in seen:
                if not dfs(table_name):
                    return False
        
        return True
    
    def check_many_to_one_sql(self, table_name, keys):
        keys_str = ",".join(keys)
        sql = '''IF EXISTS(select top 1 count(*)
from %(table_name)s
group by %(keys_str)s
having count(*)>1)
 BEGIN
   PRINT N'not many-to-one';
 END'''% { 'table_name': table_name, 'keys_str': keys_str}
        return sql
        
    def check_many_to_one(self):
        def dfs(cur, parent):
            for child in self.joins[cur]:
                if parent is None or child != parent:
                    print(self.check_many_to_one_sql(child, self.joins[cur][child][1]))
                    dfs(child, cur)
            return True
        dfs(self.fact, None)
    
    # BIGINT by default
    # TODO: performance? slow and don't use it. create a new table instead!
    def add_column_sql(self, table_name, column, default_value = 1):
        constraint_name = ('DF_' + table_name + '_' + column).replace(".", "")
        sql = '''IF COL_LENGTH('%(table_name)s', '%(column)s') IS NOT NULL
    ALTER TABLE %(table_name)s
    DROP CONSTRAINT %(constraint_name)s, COLUMN %(column)s;
ALTER TABLE %(table_name)s
ADD %(column)s BIGINT NOT NULL CONSTRAINT %(constraint_name)s  DEFAULT 1;'''% { 'table_name': table_name, 'column': column, 'constraint_name': constraint_name}
        return sql

    # only use it to send identity relation!!
    # don't need it for other relations
    def upward_message_passing(self, root, cjt, identity = False):
        if self.log:
            print("--upward message passing for cjt " + str(cjt.id))
        sql = []
        def post_dfs(table_name, parent, sql):
            if table_name not in self.joins:
                return sql
            for neighbour in self.joins[table_name]:
                if neighbour != parent:
                    post_dfs(neighbour, table_name, sql)
            if parent is not None:
                sql.append(self.send_message(table_name, parent, cjt, identity = identity))
        
        post_dfs(root, None, sql)
        
        return sql
    
    
    def downward_message_passing(self, root, cjt):
        if self.log:
            print("--downward message passing for cjt " + str(cjt.id))
        sql = []
        def pre_dfs(table_name, parent, sql, from_fact):
            if table_name == self.fact:
                from_fact = True
            if table_name not in self.joins:
                return sql
            for child in self.joins[table_name]:
                if parent is None or child != parent:
                    query = self.send_message(table_name, child, cjt, from_fact=from_fact)
                    self.execute_query(query)
                    pre_dfs(child, table_name, sql, from_fact)
        
        pre_dfs(root, None, sql, False)

        
    def aggregation_sql(self, groupby = [], variance=False, join_tables = [], select_conditions = [], left_join_tables = [], 
                        left_join_conditions = [], into = None, grouping_sets = False, count_as_1 = False):
        sql = ""
        if into is not None:
            sql += "DROP TABLE IF EXISTS " + into +";\n"
            sql += "CREATE TABLE " + into + " AS\n"
            
        sql += "SELECT "
        sql += ",".join(groupby)
        
        # if there is one non_identity relation, sum the semi-ring
        if variance:
            if len(groupby) > 0:
                sql += ", "
            sql += "CAST(SUM(" + ("1" if count_as_1 else "c") + ") AS DOUBLE) as c, CAST(SUM(s) AS DOUBLE) as s\n"
        else:
            sql += "\n"
        
#         if into is not None:
#             sql += "INTO " + into + "\n"
            
        left_join = []
        if len(left_join_tables) > 0:
            left_join = [left_join_tables[0] + "  JOIN " + left_join_tables[1] + " ON (" + " AND ".join(left_join_conditions) + ")"]
#             left_join = [left_join_tables[0] + " LEFT JOIN " + left_join_tables[1] + " ON (" + " AND ".join(left_join_conditions) + ")"]

        sql += "FROM " + ",".join(join_tables + left_join) + "\n"
        
        if len(select_conditions) > 0:
            sql += "WHERE " + " AND ".join(select_conditions) + "\n"
        
        if len(groupby) > 0:
            if grouping_sets:
                sql += "GROUP BY GROUPING SETS " + ",".join([ '(' + att + ')' for att in groupby]) + ";\n"
            else:
                sql += "GROUP BY " + ",".join(groupby) + "\n"
#         sql += ";"
        return sql
    
    def build_gradient_tree(self):
        if len(self.leaves.queue) == 0:
            raise Exception('Can\'t find the tree root!')
            
        # the condition for a split: 
        # 1. there exists a split that reduces the variance
        # 2. the total number of leaves don't exceed max_leaves
        while  len(self.leaves.queue) > 0  and self.leaves.queue[0][0] < 0 and len(self.leaves.queue) < self.max_leaves:
        # while self.leaves.queue[0][0] <= 0 and len(self.leaves.queue) < 2:
            # get the current best split
            cjt_id, table, feature, feature_type, value, s, c = self.leaves.get()[1]
            if self.log:
                print("Next best is ", cjt_id, table, feature, feature_type, value, s, c)

            # split a node
            parent_cjt = self.cjts[cjt_id]
            left_cjt = cjt(self.next_cjt_id(), parent= parent_cjt, ts = s, tc = c)
            right_cjt = cjt(self.next_cjt_id(), parent= parent_cjt, ts = parent_cjt.ts - s, tc = parent_cjt.tc - c)
            parent_cjt.set_split_condition(table, feature, feature_type, value)
            parent_cjt.set_children(left_cjt, right_cjt)
            
            if feature_type == 1:
                if value == "NULL":
                    left_cjt.add_annotation(table, table + "." + feature + " IS NULL")
                    right_cjt.add_annotation(table, table + "." + feature + " IS NOT NULL")
                else :
                    left_cjt.add_annotation(table, table + "." + feature + " IS NOT DISTINCT FROM " + "'"+ value + "'")
                    right_cjt.add_annotation(table,  table + "." + feature + " IS DISTINCT FROM " + "'"+ value + "'")
            elif feature_type == 2:
                left_cjt.add_annotation(table, table + "." + feature + " <= " + "'"+ value + "'")
                right_cjt.add_annotation(table,  table + "." + feature + " > " + "'"+ value + "'")
            else:
                raise Exception("ordinal feature is not supported!!!")

            self.cjts[left_cjt.id] = left_cjt
            self.cjts[right_cjt.id] = right_cjt
            
            # pass message only if it doesn't reach max depth
#             if left_cjt.depth < self.max_depth:
            # we still need to pass messages for fact table
            sqls = self.downward_message_passing(table, left_cjt)
            sqls = self.downward_message_passing(table, right_cjt)
            
            # get the next best split
            self.best_split(left_cjt)
            self.best_split(right_cjt)
    
    # find the best split for the given cjt
    def best_split(self, cjt):   
        if self.log:
            print("-- Finding the best split for cjt " + str(cjt.id))
        
        final_table, final_feature, final_feature_type, final_value, final_reduction_in_variance, final_s, final_c = "", "", 1, 0, 0, 0, 0
        
        if cjt.depth >= self.max_depth:
            if self.log:
                print("max depth reached!")
            self.leaves.put((-final_reduction_in_variance, (cjt.id, final_table, final_feature, 1,  final_value, final_s, final_c)))
            return
        
        for table in self.features:
            for i in range(len(self.features[table])):
                feature = self.features[table][i]
                feature_type = self.feature_types[table][i]
                # for now, feature type is always 1, because ordinal needs to deal with NULL
                query = self.best_split_of_feature(table, cjt, [feature],  feature_type)
                results = self.execute_query(query)
                if(len(results) == 0):
                    continue
                cur_value, cur_reduction_in_variance, s, c = results[0]

                cur_value = str(cur_value)
                cur_value = cur_value.strip()
                cur_reduction_in_variance = int(cur_reduction_in_variance)
#                 s = int(s)
                s = s
                c = int(c)
                if cur_reduction_in_variance > final_reduction_in_variance:
                    final_table, final_feature, final_feature_type, final_value, final_reduction_in_variance, final_s, final_c = table, feature, feature_type, cur_value, cur_reduction_in_variance, s, c
        
        if self.log:
            if final_reduction_in_variance > 0:
                print("-- The best split for cjt " + str(cjt.id) + " is table " + final_table + ", feature " + final_feature + ", feature_type " + str(final_feature_type) + ", value " + final_value + ", with reduction in variance " + str(final_reduction_in_variance))
            else:
                print("-- Can't find any split for cjt " + str(cjt.id))
        # negative final_reduction_in_variance to make it a max priority queue
        self.leaves.put((-final_reduction_in_variance, (cjt.id, final_table, final_feature, final_feature_type,  final_value, final_s, final_c)))
        
        # need to chec if final_reduction_in_variance = 0. This means split is not needed
    
    def best_split_of_feature(self, table, cjt, feature, feature_type = 1):
        # because float, need -1  to ensure stability!!!! maybe a bad fix. (e.g. cjt.tc = 5 but c = 5.001)
        sql = 'SELECT ' + ",".join(feature) + ', CASE WHEN ' + str(cjt.tc) + '> c THEN ( - (CAST(' + str(cjt.ts) + ' AS DOUBLE)/' + str(cjt.tc) + ')* ' + str(cjt.ts) + \
        '  + (s/c)*s + (' + str(cjt.ts) + '-s)/(' + str(cjt.tc) + '-c)*(' + str(cjt.ts) + '-s)) ELSE 0 END  as reduction_in_variance, s as s, c as c\n'
        
        sql += 'FROM (\n'
        
        if feature_type == 2:
            # compute the cumulative sum for numeric feature
            sql += 'SELECT ' + ",".join(feature) + ', SUM(c) OVER(ORDER BY ' + ",".join(feature) + ') as c, SUM(s) OVER(ORDER BY ' + ",".join(feature) + ') as s\n'
            sql += 'FROM (\n'
        
        sql += self.absorption(table, cjt, [table + "." + f for f in feature])
        sql += ') as tmp1\n'
        
        if feature_type == 2:
            sql += ') as tmp2\n'
        sql += 'ORDER BY reduction_in_variance DESC\n'
        
        sql += 'limit 1'
        sql += ";"
        
        return sql
    
    # get the join_tables, select_conditions, 
    def absorption(self, table, cjt, feature = []):
        
        # for f in feature:
        #     if f not in self.features[table]:
        #         raise Exception(table + ' doesn\'t have this feature!')
                
        join_tables, select_conditions, left_join_tables, left_join_conditions, variance_table = self.get_join_tables(table, cjt)
        
        groupby = feature
        
        return self.aggregation_sql(groupby = groupby, variance=True, join_tables = join_tables, 
                                    select_conditions = select_conditions, left_join_tables = left_join_tables, 
                                    left_join_conditions = left_join_conditions)
    
    # get the join_tables, select_conditions, left_join_tables and left_join_conditions, variance_table
    # all the messgaes to this table will be either in join_tables or left_join_tables
    # tables from_fact will be in left_join_tables
    # for join tables, the last one will be itself
    # variance_table is the table name that sends variance to current table (this is used in "send_message" to check whether the message is variance)
    def get_join_tables(self, table, cjt, excludes=[]):
        left_join_tables = []
        left_join_conditions = []
        
        variance_table = ""
        
        # get children of table1
        children = []
        
        if table in self.joins:
            for child in self.joins[table]:
                if cjt.messages[child][table]['type'] == "variance":
                        variance_table = child
                # identity table can be safely discarded
                if not cjt.messages[child][table]['type'] == "identity":
                    if cjt.messages[child][table]['from_fact']:
                        left_join_tables.append(cjt.messages[child][table]['name'])
                        left_join_tables.append(table)
                        child_left_keys, child_right_keys = self.joins[child][table]
                        for i in range(len(child_left_keys)):
                            left_join_conditions.append(cjt.messages[child][table]['name'] + "." + child_left_keys[i] + " = " + table + "." + child_right_keys[i])
                    else:
                        children.append(child)
        
        join_tables = []
        # shouldn't it be len(children) != 0?
        # oh should be fine for large fact, as others are identity
        if len(left_join_tables) == 0:
            join_tables = [cjt.messages[child][table]['name'] for child in children] + [table]
        else:
            join_tables = [cjt.messages[child][table]['name'] for child in children]
        
        
        select_conditions = []
        
        for child in children:
            child_left_keys, child_right_keys = self.joins[child][table]
            equal_conditions = []
            null_conditions = []
            for i in range(len(child_left_keys)):
                # NOTE THAT, BESIDES EQUALITY, ALSO JOIN ON NULL TO SUPPORT LEFT JOIN (CONSIDER NULL FROM STORE)
#                 select_conditions.append(cjt.messages[child][table]['name'] + "." + child_left_keys[i] + " = " + table + "." + child_right_keys[i])
                # this is not suffcient! think about sales absorption! one is null and one is join key that not matched!!
                select_conditions.append(cjt.messages[child][table]['name'] + "." + child_left_keys[i] + " IS NOT DISTINCT FROM " + table + "." + child_right_keys[i])
            
            
        select_conditions += cjt.get_annotations(table)
            
        return join_tables, select_conditions, left_join_tables, left_join_conditions, variance_table
    
    # update the model given current cjts
    def update_model(self):
        query = self.update_model_sql()
        self.execute_query(query)
    
    # return the sql to upate the model given current cjts
    def update_model_sql(self):
        sql = ""
        for i in range(len(self.leaves.queue)):
            cur_cjt = self.cjts[self.leaves.queue[i][1][0]]

            predicate = []

            for table in self.features:
                if table in cur_cjt.annotations:
                    predicate += cur_cjt.annotations[table]

            prediction = int(cur_cjt.ts/cur_cjt.tc * self.learning_rate)
            sql += "INSERT INTO gbm_model_table\n"
            sql += "SELECT \'" + ",".join(predicate) + "\' AS PREDICATE, " + str(prediction) + " AS PREDICTION_VALUE\n"
        return sql
    
    def update_error(self):
        if not self.fact_pandas_join:
            for i in range(len(self.leaves.queue)):
                cjt = self.cjts[self.leaves.queue[i][1][0]]
                if self.log:
                    print(cjt.id, cjt.ts, cjt.tc)
                prediction = cjt.ts/cjt.tc * self.learning_rate
                table = self.fact
                children = []
                if table in self.joins:
                    for child in self.joins[table]:
                        if not cjt.messages[child][table]['type'] == "identity":
                            children.append(child)

                select_conditions = []
                for child in children:
                    child_left_keys, child_right_keys = self.joins[child][table]
                    select_conditions.append("(" +",".join([table + "." + key for key in child_right_keys]) + ") in (SELECT " +\
                                              "(" + ",".join(child_left_keys) + ") FROM " + cjt.messages[child][table]['name'] +")")

                select_conditions += cjt.get_annotations(table)
                sql = "UPDATE " + table + " SET s=s-(" + str(prediction) + ") \n"
                sql += "WHERE " + " AND ".join(select_conditions) + "\n"
                self.execute_query(sql)
                
        if self.fact_pandas_join:
            cjts = []
            for i in range(len(self.leaves.queue)):
                cjts.append(self.cjts[self.leaves.queue[i][1][0]])

            # sort by descending total count
            cjts.sort(key=lambda x:-x.tc)

            sql = "SELECT\n"

            def dfs(cjt, sql):
                indent = ' '*cjt.depth
                if cjt.left_child is None:
                    pre = cjt.ts/cjt.tc * self.learning_rate
                    sql +=  indent + "CAST(" + str(pre) + " AS DOUBLE)\n"
                    return sql

                table, feature, feature_type, value = cjt.split 
                left_predicate = ""

                if feature_type == 1:
                    if value == "NULL":
                        left_predicate = feature + " IS NULL"
                    else :
                        left_predicate = feature + " IS NOT DISTINCT FROM " + "'"+ value + "'"
                if feature_type == 2:
                    left_predicate = feature + " <= " + "'"+ value + "'"

                sql += indent + "CASE WHEN " + left_predicate + " THEN\n"
                sql = dfs(cjt.left_child, sql)
                sql += indent + "ELSE\n"
                sql = dfs(cjt.right_child, sql)
                sql += indent + "END\n"
                return sql

            table = self.fact
            sql = dfs(self.cjts[0], sql)
            sql += "AS s FROM " + table + "\n"

            if self.log:
                print(sql)
            start_time = time.time()
            preds = con.execute(sql).fetchnumpy()
            self.fact_table["s"] = self.fact_table["s"] - preds["s"]
            elapsed_time = time.time() - start_time
            self.totaltime += elapsed_time
    
    # three types of tables: identity is table that is skipped, selected is not skipped but don't have semi-ring, variance is one with semi-ring
    # send message from table1 to table2
    def send_message(self, table1, table2, cjt, identity = False, from_fact = False):
        if table1 not in self.joins and table2 not in self.joins[table1]:
            raise Exception('Table', table1, 'and table', table2, 'are not connected')
        
        sql = "--sending message from " + table1 + " to " + table2 + "\n"
        
        if table1 not in cjt.messages:
            cjt.messages[table1] = dict()
        
        cjt.messages[table1][table2] = {'name': self.name + "_m" + str(self.total_messages), 'from_fact': from_fact}
        self.total_messages += 1
        
        # if it's identity relation, we don't need to generate any sql
        if identity:
            cjt.messages[table1][table2]["type"] = "identity"
            return ""
        
        view_name = cjt.messages[table1][table2] 
        # whether there exists an relation with variance?
        variance = False
        
        if self.fact == table1:
            variance = True
        
        join_tables, select_conditions, left_join_tables, left_join_conditions, variance_table = self.get_join_tables(table1, cjt, [table2])
        
        if variance_table != table2:
            variance = True
            
        left_keys, _ = self.joins[table1][table2]
        
        if variance:
            cjt.messages[table1][table2]["type"] = "variance"
        else:
            cjt.messages[table1][table2]["type"] = "selected"
            
        groupby = [table1 + "." + key for key in left_keys]
        into = cjt.messages[table1][table2]['name'] 
        
        sql += self.aggregation_sql(groupby = groupby, variance=variance, join_tables = join_tables, 
                                    select_conditions = select_conditions, left_join_tables = left_join_tables, 
                                    left_join_conditions = left_join_conditions, into = into, count_as_1 = (table1 == self.fact))
        sql += ";"
        return sql
    
    def print_tree(self):
        for cjt_id in self.cjts:
            cjt = self.cjts[cjt_id]
            print(str(cjt.id) + ": parent " + str(cjt.parent) + ", total sum: " + str(cjt.ts) + ", total count:" + str(cjt.tc))
            for table in cjt.annotations:
                anns = cjt.annotations[table]
                for ann in anns:
                    print(table + ": " + ann)
    def predict(self, table):
        sql = ""
        sql += "SELECT SQRT(AVG(POW(" + self.target_variable + " - prediction,2))) AS RMSE FROM (\n"
        sql += "SELECT " + self.target_variable + ", CAST("+ str(self.base_value) + " AS DOUBLE)\n"
        ann_pres = self.get_all_leaf_annotations_predictions()
        for ann_pre in ann_pres:
            ann, pre = ann_pre
            if len(ann) == 0:
                print("annotations have size 0???")
                continue
            # for now do the split, because each annotation is: relation.attribute ...
            # in the future, store annotation with attribute and predicate separately
            sql += " + CASE WHEN " + " AND ".join([a.split(".",1)[-1] for a in ann]) + " THEN " + str(pre) + " ELSE 0 END\n"
        sql += " AS prediction\n"
        sql += "FROM " + table + "\n"
        sql += ")"
        return sql
    
    # have to run single-thread for sample to be consistent
    def create_sample_fact(self, sample_percent = 10, sample_seed = 1, view = False):
        sample_table = self.fact + "_sample_" + str(sample_percent) + "_" + str(sample_seed)
        sql = "CREATE OR REPLACE " + ("VIEW " if view else "TABLE ") + sample_table + " AS\n"
        sql += "SELECT * FROM " + self.fact + "\n"
        sql += "USING SAMPLE (" + str(sample_percent) + " %) REPEATABLE (" + str(sample_seed) + ");"
        self.execute_query(sql)
        self.features[sample_table] = self.features[self.fact]
        self.feature_types[sample_table] = self.feature_types[self.fact]
        self.fact = sample_table
    
    # this version is better as, for each tree, it's in one case when end
    # previous version has case for each leaf
    def predict_succ(self, table):
        sql = ""
        sql += "SELECT SQRT(AVG(POW(" + self.target_variable + " - prediction,2))) AS RMSE FROM (\n"
        sql += "SELECT " + self.target_variable + ", CAST("+ str(self.base_value) + " AS DOUBLE) +\n"
        sql += " + \n".join(self.tree_queries)
        sql += " AS prediction\n"
        sql += "FROM " + table + "\n"
        sql += ")"
        return sql

    def get_all_leaf_annotations_predictions(self):
        ann_pre = []
        for cjt_id in self.cjts:
            cjt = self.cjts[cjt_id]
            if cjt.is_leaf:
                ann = cjt.get_all_annotations()
                pre = cjt.ts/cjt.tc * self.learning_rate
                ann_pre.append((ann, pre))
        return ann_pre
    
    def get_full_join(self):
        features = [k + "." + v for k in self.features for v in self.features[k]] + [self.target_variable]
        select_sql = "SELECT " +", ".join(features)
        from_sql = ["FROM " + self.fact]
        def pre_dfs(current, parent, sql):
            for child in self.joins[current]:
                if parent is None or child != parent:
                    left_keys, right_keys = self.joins[current][child]

                    join_conditions = []
                    for i in range(len(left_keys)):
                        join_conditions.append(current + "." + left_keys[i] + " IS NOT DISTINCT FROM " + child + "." + right_keys[i])

                    sql.append(" LEFT JOIN " + child + " ON (" + " AND ".join(join_conditions) + ")" )
                    pre_dfs(child, current, sql)
        pre_dfs(self.fact, None, from_sql)
        return select_sql + "\n" + "\n".join(from_sql)
