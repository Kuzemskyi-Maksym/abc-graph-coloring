"""All algorithms compared in the experiments: name -> f(graph, time_limit, seed) -> Result."""
from .bees import vertex_bee, site_bee, habc, tabucol, own_abc
from .mods import own_a, own_b, own_c

ALGORITHMS = {
    # existing modifications from the lecture / literature
    "VertexBee(CM)": lambda g, T, s: vertex_bee(g, T, s, select="random", color_rule="first"),
    "SiteBee(lecture)": lambda g, T, s: site_bee(g, T, s),
    "HABC(Fister)": lambda g, T, s: habc(g, T, s),
    # non-bee reference
    "TabuCol": lambda g, T, s: tabucol(g, T, s),
    # our modifications
    "Own-A": lambda g, T, s: own_a(g, T, s),
    "Own-B": lambda g, T, s: own_b(g, T, s),
    "Own-C": lambda g, T, s: own_c(g, T, s),
    # ablations
    "A: no foragers": lambda g, T, s: own_a(g, T, s, foragers=False),
    "A: first colour": lambda g, T, s: own_a(g, T, s, color_rule="first"),
    "C: uniform ops": lambda g, T, s: own_c(g, T, s, adaptive=False),
    "VertexBee+mostused": lambda g, T, s: vertex_bee(g, T, s, select="random", color_rule="most_used"),
    "Draft Own-ABC (12.10)": lambda g, T, s: own_abc(g, T, s),
}
