import networkx as nx
import numpy as np
from enum import Enum
from sympy import *

# enums for tracking standard bond graph variable and node types
class BondGraphPortTypes(Enum):
    # Passive 1-ports
    CAPACITANCE = 0
    INERTANCE = 1
    RESISTANCE = 2
    
    # Active 1-ports
    EFFORT_SOURCE = 3
    FLOW_SOURCE = 4
    
    # Multiport/junctions
    ZERO_JUNCTION = 5
    ONE_JUNCTION = 6
    
    # Two-ports
    TRANSFORMER = 7
    GYRATOR = 8

class GeneralizedVariables(Enum):
    # Power
    EFFORT = 0
    FLOW = 1
    
    # Energy
    MOMENTUM = 2
    DISPLACEMENT = 3
    
class BondGraphNode:
    def __init__(self, port_type:BondGraphPortTypes, max_ports:int, preferred_causality:GeneralizedVariables=None):
        self.port_type = port_type
        self.max_ports = max_ports
        
        self.e, self.f, self.p, self.q, self.t = symbols('e f p q t')
            
        pass
    
    def get_effort_expr(self):
        raise NotImplementedError
    
    def get_flow_expr(self):
        raise NotImplementedError
    
    def get_state_var(self):
        raise NotImplementedError
            
# Passive 1-Ports
class Capacitance(BondGraphNode):
    def __init__(self, capacitance):
        super().__init__(port_type=BondGraphPortTypes.CAPACITANCE, max_ports=1, preferred_causality=GeneralizedVariables.FLOW)
        self.C = capacitance
        pass
    
    def get_effort_expr(self):
        self.e = self.q/self.C
        return self.e
    
    def get_flow_expr(self):
        return self.Derivative(self.q, self.t)
    
    def get_state_var(self):
        return self.q

class Inertance(BondGraphNode):
    def __init__(self, inertance):
        super().__init__(port_type=BondGraphPortTypes.INERTANCE, max_ports=1)
        self.I = inertance
        pass
    
    def get_effort_expr(self):
        return self.Derivative(self.p, self.t)
    
    def get_flow_expr(self):
        return self.p/self.I
    
    def get_state_var(self):
        return self.p

class Resistance(BondGraphNode):
    def __init__(self, resistance):
        super().__init__(port_type=BondGraphPortTypes.RESISTANCE, max_ports=1)
        self.R = resistance
        pass
    
    def get_effort_expr(self):
        return self.R*self.f
    
    def get_flow_expr(self):
        return self.e/self.R
    
    def get_state_var(self):
        return None
    
# Active 1-Ports
class EffortSource(BondGraphNode):
    def __init__(self, effort_src):
        """Specifies an effort source in the bond graph.

        Args:
            effort_src (_type_): The effort source input vector (elements correspond to each point in time).
        """
        super().__init__(port_type=BondGraphPortTypes.EFFORT_SOURCE, max_ports=1)
        self.Se = effort_src
        pass
    
    def get_effort_expr(self):
        return self.e
    
    def get_flow_expr(self):
        return None
    
    def get_state_var(self):
        return None

# Passive multiport/junctions
class OneJunction(BondGraphNode):
    def __init__(self):
        super().__init__(port_type=BondGraphPortTypes.ONE_JUNCTION, max_ports=None)
        pass
    
class ZeroJunction(BondGraphNode):
    def __init__(self):
        super().__init__(port_type=BondGraphPortTypes.ZERO_JUNCTION, max_ports=None)
        pass
    
# Passive 2-Ports
class Transformer(BondGraphNode):
    def __init__(self, transformer_ratio):
        super().__init__(port_type=BondGraphPortTypes.TRANSFORMER, max_ports=2)
        self.tf_ratio = transformer_ratio
        pass
    
class Gyrator(BondGraphNode):
    def __init__(self, gyrator_ratio):
        super().__init__(port_type=BondGraphPortTypes.GYRATOR, max_ports=2)
        self.gyrator_ratio = gyrator_ratio
        pass
    

