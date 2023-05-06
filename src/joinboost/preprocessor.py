from typing import List
from joinboost.joingraph import JoinGraph
from .executor import ExecuteMode, SPJAData
from .aggregator import *
import json
from abc import ABC, abstractmethod


# Preprocess the join graph and store modifications.
class Preprocessor:
    def __init__(self, steps=[]):
        self.histories = []  # table: old col: new col
        self.steps = []
        self._prefix = "joinboost_reserved_"
        self._original_target_var = None

    def add_step(self, step):
        self.steps.append(step)

    def run_preprocessing(self, joingraph: JoinGraph):
        # these are temporary. Remove in the future.
        self._original_target_var = joingraph.target_var
        self.jg = joingraph

        for step in self.steps:
            self.histories.append(step.apply(joingraph))

    def reapply_preprocessing(self):
        # reapply the preprocess steps according to the history
        for i in range(len(self.steps)):
            history = self.histories[i]
            step.reapply(joingraph, history)

    def get_history(self):
        # records the history of preprocess
        return json.dumps(self.histories, indent=4)

    def get_original_target_name(self):
        return self._original_target_var

    def get_join_graph(self):
        return self.jg


class Step(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, joingraph):
        # return a data structure for reapply
        return None

    @abstractmethod
    def reapply(self, joingraph, history):
        pass


class RenameStep(Step):
    def __init__(self, reserved_words: List[str], prefix="joinboost_reserved_"):
        super().__init__()
        self.reserved_words = reserved_words
        self.prefix = prefix

    def apply(self, joingraph):
        # get a plan of renaming
        rename_mapping = self.construct_rename_mapping(joingraph)
        self.reapply(joingraph, rename_mapping)
        return rename_mapping

    # rename the tables according to the  rename_mapping
    def reapply(self, joingraph, rename_mapping):
        for relation in rename_mapping:
            mapping = rename_mapping[relation]
            expression = {}

            for old_name in mapping:
                new_name = mapping[old_name]
                expression[new_name] = (old_name, Aggregator.IDENTITY)
                joingraph.replace_relation_attribute(relation, old_name, new_name)

            spja_data = SPJAData(
                aggregate_expressions=expression, from_tables=[relation]
            )

            new_relation_name = joingraph.exe.execute_spja_query(
                spja_data, mode=ExecuteMode.CREATE_VIEW
            )
            joingraph.replace(relation, new_relation_name)

    def construct_rename_mapping(self, joingraph):
        # for each relation => old col => new column
        rename_mapping = dict()

        # for each relation in the join graph
        for relation in joingraph.relations:
            # get its schema as a list of column names
            schema = joingraph.exe.get_schema(relation)

            # check if reserved_word is in schema
            if any(word in schema for word in self.reserved_words):
                rename_mapping[relation] = {}

                # decide a new column name if reserved
                for old_name in schema:
                    if old_name in self.reserved_words:
                        new_name = self.prefix + old_name
                        while new_name in schema:
                            new_name = self._prefix + new_name
                    else:
                        new_name = old_name

                    rename_mapping[relation][old_name] = new_name

        return rename_mapping
