from .aggregator import Aggregator
import copy
from .executor import Executor

class JoinGraph:
    def __init__(self, 
                exe, 
                joins = {}, 
                relation_schema = {},
                target_var = None,
                target_relation = None):
        
        self.exe = exe
        # maps each from_relation => to_relation => {keys: (from_keys, to_keys)}
        self.joins = copy.deepcopy(joins)
        # maps each relation => feature => feature_type
        self.relation_schema = copy.deepcopy(relation_schema)
        self.target_var = target_var
        self.target_relation = target_relation
    
    def get_relations(self): 
        return list(self.relation_schema.keys())
    
    def get_relation_schema(self): 
        return self.relation_schema
    
    def get_type(self, relation, feature): 
        return self.relation_schema[relation][feature]

    def get_target_var(self): 
        return self.target_var

    def get_target_relation(self): 
        return self.target_relation
    
    def get_joins(self):
        return self.joins
    
    # add relation, features and target variable to join graph
    # current assumption: Y is in the fact table
    def add_relation(self, relation: str, X: list, y: str = None):
        self.joins[relation] = dict()
        if relation not in self.relation_schema:
                self.relation_schema[relation] = {}
                
        for x in X:
            # by default, assume all features to be numerical
            self.relation_schema[relation][x] = 'NUM'
            
        if y is not None:
            if self.target_var is not None:
                print("Warning: Y already exists and has been replaced")
            self.target_var = y
            self.target_relation = relation
            
    # get features for each table
    def get_relation_features(self, r_name):
        if r_name not in self.relation_schema:
            raise Exception('Attribute not in ' + r_name)
        return list(self.relation_schema[r_name].keys())
    
    # get the join keys between two tables
    # all get all the join keys of one table
    def get_join_keys(self, f_table: str, t_table: str = None):
        if f_table not in self.joins:
            return []
        if t_table:
            if t_table not in self.joins[f_table]:
                raise Exception(t_table, 'not connected to', f_table)
            return self.joins[f_table][t_table]["keys"]
        else:
            keys = set()
            for table in self.joins[f_table]:
                l_keys, _ = self.joins[f_table][table]["keys"]
                keys = keys.union(set(l_keys))
            return list(keys)
    
    # useful attributes are features + join keys
    def get_useful_attributes(self, table):
        useful_attributes = self.get_relation_features(table) + \
                            self.get_join_keys(table)
        return list(set(useful_attributes))
    

    def add_join(self, table_name_left: str, table_name_right: str, left_keys: list, right_keys: list):
        if len(left_keys) != len(right_keys):
            raise Exception('Join keys have different lengths!')
        if table_name_left not in self.relation_schema:
            raise Exception(table_name_left + ' doesn\'t exit!')
        if table_name_right not in self.relation_schema:
            raise Exception(table_name_right + ' doesn\'t exit!')

        left_keys = [attr for attr in left_keys]
        right_keys = [attr for attr in right_keys]

        self.joins[table_name_left][table_name_right] = {"keys": (left_keys, right_keys)}
        self.joins[table_name_right][table_name_left] = {"keys": (right_keys, left_keys)}
        
    def rename(self, table_prev, table_after):
        if table_prev not in self.relation_schema: 
            raise Exception(table_prev + ' doesn\'t exit!')
        if table_after in self.relation_schema: 
            raise Exception(table_after + ' already exits!')
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
        self.check_all_features_exist()
    
    def check_all_features_exist(self):
        for table in self.relation_schema:
            features = self.relation_schema[table].keys()
            self.check_features_exist(table, features)
        
    def check_features_exist(self, table, features):
        attributes = self.exe.get_schema(table)
        if not set(features).issubset(set(attributes)):
            Exception('Key error in ' + str(features) + '. Attribute does not exist in table' + table)

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