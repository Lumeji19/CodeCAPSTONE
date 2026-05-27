from graphviz import Digraph

tree = Digraph()

tree.node("A", "Study hours > 40?", shape="box")
tree.node("B", "Attendance > 80%?", shape="box")
tree.node("C", "Homework completed?", shape="box")
tree.node("D", "Student passes", shape="box")
tree.node("E", "Student fails", shape="box")
tree.node("F", "Student passes", shape="box")
tree.node("G", "Student fails", shape="box")

tree.edge("A", "B", label="Yes")
tree.edge("A", "C", label="No")
tree.edge("B", "D", label="Yes")
tree.edge("B", "E", label="No")
tree.edge("C", "F", label="Yes")
tree.edge("C", "G", label="No")

tree.render("simple_classification_tree", format="png", view=True)