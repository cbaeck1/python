import graphviz 
import mglearn

def save_graph_as_svg(dot_string, output_file_name):
    if type(dot_string) is str:
        g = graphviz.Source(dot_string)
    elif isinstance(dot_string, (graphviz.dot.Digraph, graphviz.dot.Graph)):
        g = dot_string
    g.format='svg'
    g.filename = output_file_name
    g.directory = './images/svg/'
    g.render(view=False)
    return g

dot_graph = """
graph graphname {
    rankdir=LR;
     a -- b -- c;
     b -- d;
}"""


g = save_graph_as_svg(dot_graph, 'simple_dot_example1')
# g.title = "abc"
# print(g)

# mglearn.plots.plot_tree_progressive()
