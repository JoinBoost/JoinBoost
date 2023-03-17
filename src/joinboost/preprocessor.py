from typing import List
from joinboost.joingraph import JoinGraph
import json 


# Preprocess the join graph and store modifications.
class Preprocessor:
    def __init__(self):
        self.view2table = {} # keeps track of rename changes
        self.table2view = {}
        self.jg = None
        self.history = None #table: old col: new col
        self._prefix = "joinboost_reserved_"
        self._original_target_var = None

    def run_preprocessing(self, jg: JoinGraph, reserved_words: List[str]):
        self.jg = jg
        self._original_target_var = self.jg.target_var
        self.rename(reserved_words)

    def rename(self, reserved_words: List[str]):
        self.prepare_rename_mapping(reserved_words)

        for view in self.view2table:
            relation = self.view2table[view]["relation_name"]
            self.jg.replace(relation, view)
            l = []
            for new_word, old_word in self.view2table[view]["cols"].items():
                self.replace_relation_attribute(view, old_word, new_word)
                _sql = (
                    f"{old_word} AS {new_word}"
                    if old_word != new_word
                    else f"{old_word}"
                )
                l.append(_sql)
            sql = (
                f"CREATE OR REPLACE VIEW {view} AS \n"
                + f"SELECT {','.join(l)} FROM {relation}"
            )
            self.jg.exe._execute_query(sql)
    
    def prepare_rename_mapping(self, reserved_words):
        """Find columns that have a conflict with any reserved_word.

        Iterate through each table's columns to check if reserved_word exist.
        If so, rename it to avoid conflict. Return view2table and table2view
        mappings.
        """

        self.view2table = self._prepare_view2table(reserved_words)
        self.table2view = self._prepare_table2view(self.view2table)
        self.history = self._prepare_history(self.table2view)
        
    def _prepare_view2table(self, reserved_words):
        view2table = dict()
        
        # for each relation in the join graph
        for relation in self.jg.relations:
            # get its schema as a list of column names
            schema = self.jg.exe.get_schema(relation)
            # check if reserved_word is in schema
            if any(word in schema for word in reserved_words):
                # first decide a new view name
                view_name = self._prefix + relation
                while view_name in self.jg.relations:
                    view_name = self._prefix + view_name

                if view_name not in view2table:
                    view2table[view_name] = {
                        "relation_name": relation,
                        "cols": dict(),
                    }
                # then decide a new colulmn name
                for col in schema:
                    if col in reserved_words:
                        new_word = self._prefix + col
                        while new_word in schema:
                            new_word = self._prefix + new_word
                    elif col in view2table[view_name]["cols"]:
                        # No changes needed
                        continue
                    else:
                        new_word = col
                    view2table[view_name]["cols"][new_word] = col

        return view2table
    
    # build a reverse mapping from table to view
    def _prepare_table2view(self, view2table):
        pass
#         table2view = dict()
#         for view in view2table:
#             table = view2table[view]["relation_name"]
#             table2view[table] = {
#                 "view_name": view,
#                 "cols": dict(),
#             }
#             for col in view2table[view]["cols"]:
#                 original_col = view2table[view]["cols"][col]
#                 table2view[table]["cols"][original_col] = col
#         return table2view
    
    def _prepare_history(self, table2view):
        pass
#         """Store the changed tables and columns in history."""
#         history = dict()
#         for table in table2view:
#             history[table] = dict()
#             for col in table2view[table]["cols"]:
#                 if col != table2view[table]["cols"][col]:
#                     history[table][col] = table2view[table]["cols"][col]
#         return history

    def replace_relation_attribute(self, relation, before_attribute, after_attribute):
        if relation == self.jg.target_relation:
            if self.jg.target_var == before_attribute:
                self.jg._target_var = after_attribute

        if before_attribute in self.jg.relation_schema[relation]:
            self.jg.relation_schema[relation][after_attribute] = self.jg.relation_schema[
                relation
            ][before_attribute]
            del self.jg.relation_schema[relation][before_attribute]

        for relation2 in self.jg.joins[relation]:
            left_join_key = self.jg.joins[relation][relation2]["keys"][0]
            if before_attribute in left_join_key:
                # Find the index of the before_attribute in the list
                index = left_join_key.index(before_attribute)
                # Replace the old string with the new string
                left_join_key[index] = after_attribute

    def reapply_preprocessing(self):
        # reapply the preprocess steps according to the history
        pass
    
    def get_history(self):
        # records the history of preprocess
        return json.dumps(self.history, indent=4)
    
#     def get_view2table(self):
#         return self.view2table
        
#     def get_table2view(self):
#         return self.table2view
    
    def get_join_graph(self):
        return self.jg
    
    def get_original_target_name(self):
        return self._original_target_var
    
#     def is_target_relation_a_view(self):
#         return self.jg.target_relation in self.view2table

