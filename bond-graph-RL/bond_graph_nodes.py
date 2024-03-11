import networkx as nx
import numpy as np
from enum import Enum
import sympy
from sympy import *
import json
import functools


# enums for tracking standard bond graph variable and node types
class BondGraphElementTypes(Enum):
    NONE = -1
    
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
    # TRANSFORMER = 7
    # GYRATOR = 8
    
class BondGraphElementTypesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BondGraphElementTypes):
            return str(obj)
        return super().default(obj)
json.dumps = functools.partial(json.dumps, cls=BondGraphElementTypesEncoder)
    
    
# def is_1port(port_type:BondGraphElementTypes):
#     return port_type == BondGraphElementTypes.CAPACITANCE \
#         or port_type == BondGraphElementTypes.INERTANCE \
#         or port_type == BondGraphElementTypes.RESISTANCE \
#         or port_type == BondGraphElementTypes.EFFORT_SOURCE \
#         or port_type == BondGraphElementTypes.FLOW_SOURCE

# def is_multiport(port_type:BondGraphElementTypes):
#     return port_type == BondGraphElementTypes.ZeroJunction \
#         or port_type == BondGraphElementTypes.OneJunction

# def is_2port(port_type:BondGraphElementTypes):
#     return port_type == BondGraphElementTypes.TRANSFORMER \
#         or port_type == BondGraphElementTypes.GYRATOR

def is_energy_storage_element(element_type:BondGraphElementTypes):
    return element_type == BondGraphElementTypes.CAPACITANCE \
        or element_type == BondGraphElementTypes.INERTANCE

def is_source_element(element_type:BondGraphElementTypes):
    return element_type == BondGraphElementTypes.EFFORT_SOURCE \
        or element_type == BondGraphElementTypes.FLOW_SOURCE
        
def is_passive_1port(element_type:BondGraphElementTypes):
    return element_type == BondGraphElementTypes.CAPACITANCE \
        or element_type == BondGraphElementTypes.INERTANCE \
        or element_type == BondGraphElementTypes.RESISTANCE
        
class GeneralizedVariables(Enum):
    # Power
    EFFORT = 0
    FLOW = 1
    
    # Energy
    MOMENTUM = 2
    DISPLACEMENT = 3
    
class CausalityTypes(Enum): # TODO: do we need this if DiGraph uses directivity for causality?
    INTEGRAL = 0
    DERIVATIVE = 1
    
    
class BondGraphNode:
    """_summary_
    """
    def __init__(self, element_type:BondGraphElementTypes, max_ports:int, causality:GeneralizedVariables=None, params:Dict={}):
        self.element_type = element_type
        self.max_ports = max_ports
        self.causality = causality
        self.params = params
        
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
    def __init__(self, capacitance, causality:CausalityTypes=CausalityTypes.INTEGRAL):
        params = {'C': capacitance}
        super().__init__(element_type=BondGraphElementTypes.CAPACITANCE, max_ports=1, causality=causality, params=params)
        self.C = capacitance
        pass
    
    def get_effort_expr(self):
        self.e = self.q/self.C
        return self.e
    
    def get_flow_expr(self):
        return Derivative(self.q, self.t)
    
    def get_state_var(self):
        return self.q

class Inertance(BondGraphNode):
    def __init__(self, inertance, causality:CausalityTypes=CausalityTypes.INTEGRAL):
        params = {'I': inertance}
        super().__init__(element_type=BondGraphElementTypes.INERTANCE, max_ports=1, causality=causality, params=params)
        self.I = inertance
        pass
    
    def get_effort_expr(self):
        return Derivative(self.p, self.t)
    
    def get_flow_expr(self):
        return self.p/self.I
    
    def get_state_var(self):
        return self.p

class Resistance(BondGraphNode):
    def __init__(self, resistance):
        params = {'R': resistance}
        super().__init__(element_type=BondGraphElementTypes.RESISTANCE, max_ports=1, params=params)
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
    def __init__(self): 
        """Specifies an effort source in the bond graph.
        Args:
            effort_src (_type_): The effort source input vector (elements correspond to each point in time).
        """
        super().__init__(element_type=BondGraphElementTypes.EFFORT_SOURCE, max_ports=1)

        pass
    
class FlowSource(BondGraphNode):
    def __init__(self):
        super().__init__(element_type=BondGraphElementTypes.FLOW_SOURCE, max_ports=1)


# Passive multiport/junctions
class OneJunction(BondGraphNode):
    def __init__(self):
        super().__init__(element_type=BondGraphElementTypes.ONE_JUNCTION, max_ports=None)
        pass
    
    def get_flow_expr(self):
        # some flag saying that you should get the source of flow causality from the edges attached to this node
        return 
    
    def get_effort_expr(self):
        # some flag saying that you should get the sum of efforts into the node
        return
    
class ZeroJunction(BondGraphNode):
    def __init__(self):
        super().__init__(element_type=BondGraphElementTypes.ZERO_JUNCTION, max_ports=None)
        pass
    
# Passive 2-Ports
class Transformer(BondGraphNode):
    def __init__(self, transformer_ratio):
        super().__init__(element_type=BondGraphElementTypes.TRANSFORMER, max_ports=2)
        self.tf_ratio = transformer_ratio
        pass
    
class Gyrator(BondGraphNode):
    def __init__(self, gyrator_ratio):
        super().__init__(element_type=BondGraphElementTypes.GYRATOR, max_ports=2)
        self.gyrator_ratio = gyrator_ratio
        pass
    

