from joingraph import JoinGraph

class Preprocess:
    def __init__(self, join_graph: JoinGraph):
        self.join_graph = join_graph
        self.view2table = {}  # keeps track of rename changes

    def run_preprocessing(self):
        self.rename()

    def rename(self):
        pass