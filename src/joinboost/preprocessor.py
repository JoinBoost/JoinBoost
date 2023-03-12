from joingraph import JoinGraph

class Preprocess:
    def __init__(self, join_graph: JoinGraph):
        self.join_graph = join_graph
        # records the history of  preprocess
        self.history = {}
        self.view2table = {}  # keeps track of rename changes

    def run_preprocessing(self):
        new_join_graph = self.rename(self.join_graph)
        return new_join_graph

    def rename(self, join_graph: JoinGraph):
        pass
    
    def reapply_preprocessing(self):
        # reapply the preprocess steps according to the history
        pass