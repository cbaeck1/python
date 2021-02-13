from manimlib.imports import *

class DrawSine(GraphScene):
    CONFIG = {
        "x_min" : -10,
        "x_max" : 10,
        "y_min" : -1.5,
        "y_max" : 1.5,
        "graph_origin": ORIGIN,
    }
    def construct(self):
        self.setup_axes()

        func = lambda x: np.sin(x)
        graph = self.get_graph(func, x_min=-10,x_max=10)

        self.play(ShowCreation(graph))
        self.wait()

# draw = DrawSine()
