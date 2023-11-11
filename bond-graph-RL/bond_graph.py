import networkx as nx
from enum import Enum

class BondGraph(nx.DiGraph):
    def __init__(self):
        self.add
        self.BondG
        pass
    
    def
    

class BondGraphNode:
    def __init__(self, num_ports, params, imposed_causality):
        self.num_ports = num_ports
        self.params = params
        self.imposed_causality = imposed_causality # Either effort or flow (constrains type of edges)
        pass

    def 

class BondGraphEdge:
    def __init__():
        pass
    
class BondGraphCapacitance(BondGraphNode):
    def __init__(self):
        super().__init__(num_ports=1)
    
    
    def x_dot():
        x_dot = 
        pass

class BondGraphInertance(BondGraphNode):
    def __init__(self):
        super().__init__(num_ports=1)
        pass


class BondGraphGeneralizedVariables(Enum):
    EFFORT = 0
    FLOW = 1
    MOMENTUM = 2
    DISPLACEMENT = 3
    
        
    
# Bond graph node attributes
# of ports
# energy/flow relation function: 
# - p_dot
# - x_dot