"""
Known causal structure for transformer circuits.

These are documented dependencies from mechanistic interpretability
research. Auto-correlation can discover additional relationships,
but these are the established ones.

References:
  - IOI circuit: Wang et al. 2022, "Interpretability in the Wild"
  - Induction heads: Olsson et al. 2022
"""

from margin.causal import CausalGraph, CauseType, Health


# -----------------------------------------------------------------------
# IOI (Indirect Object Identification) circuit graph
# -----------------------------------------------------------------------

IOI_GRAPH = CausalGraph()

# Name-mover heads move the IO name into the output position.
# When NM degrades, IOI gap drops.
IOI_GRAPH.add_degrades(
    "NM", "IOI", strength=0.8,
    evidence="Name-mover heads are the primary contributors to IOI logit gap. "
             "NM ablation directly reduces IOI gap.",
)

# Induction heads form the attention pattern that NM copies.
# IH degradation weakens the NM signal.
IOI_GRAPH.add_degrades(
    "IH", "NM", strength=0.6,
    evidence="Induction heads create the previous-token pattern that "
             "name-mover heads read. IH failure weakens NM attention.",
)

# Suppressor heads suppress the S (subject) token.
# SH failure causes S-token interference with IO prediction.
IOI_GRAPH.add_degrades(
    "SH", "IOI", strength=0.4,
    evidence="Suppressor heads reduce S-token logit. SH ablation "
             "increases S-token probability, reducing IOI gap.",
)

# IH degradation also weakens SH (indirect path via attention patterns)
IOI_GRAPH.add_degrades(
    "IH", "SH", strength=0.3,
    evidence="IH patterns feed into SH attention computation. "
             "Weak induction weakens suppression.",
)


# -----------------------------------------------------------------------
# Generic multi-circuit template
# -----------------------------------------------------------------------

def make_circuit_graph(
    circuits: list[str],
    primary: str = "",
) -> CausalGraph:
    """
    Build a minimal causal graph for a set of circuits.

    If `primary` is specified, all other circuits are linked to it
    as CORRELATES (direction unknown until auto-correlation discovers it).

    For known architectures (IOI), use the pre-built graphs instead.
    """
    graph = CausalGraph()
    if primary and primary in circuits:
        for c in circuits:
            if c != primary:
                # Add CORRELATES link — auto-correlation will refine
                from margin.causal import CausalLink
                graph.add(CausalLink(
                    source=c, target=primary,
                    cause_type=CauseType.CORRELATES, strength=0.5,
                    evidence="assumed correlation — verify with auto_causal_graph()",
                ))
    return graph
