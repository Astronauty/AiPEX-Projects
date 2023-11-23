import networkx as nx
import numpy as np
from enum import Enum
from sympy import *

from bond_graph_nodes import *
from bond_graph_edges import *



class BondGraph():
    def __init__(self, num_effort_sources:int=0, num_flow_sources:int=0, time=np.float):
        """
        Creates a BondGraph system with a specified number of flow and effort sources to start.

        Args:
            num_effort_sources (int): _description_
            num_flow_sources (int): _description_
        """
        self.num_effort_sources = num_effort_sources
        self.num_flow_sources = num_flow_sources
        self.i = 0 # index for incrementally labeling nodes
        
        self.graph = nx.DiGraph()
    
    # Restrictive addition of ports (only allows valid bond graph ports)
    def add_port(self, port_type:BondGraphPortTypes, params:dict=None):
        """
        Adds a port to the bond graph. The port is labeled according to its type and the index of the port.

        Args:
            port (BondGraphNode): Specify the type of port to add to the bond graph. Each of the standard bond graph ports inherits the BondGraphNode class.
        """
        port_label = ""
        match port_type:
            case BondGraphPortTypes.CAPACITANCE:
                port_label = f"C_{self.i}"
            case BondGraphPortTypes.INERTANCE:
                port_label = f"I_{self.i}"
            case BondGraphPortTypes.RESISTANCE:
                port_label = f"R_{self.i}"
            case BondGraphPortTypes.EFFORT_SOURCE:
                port_label = f"Se_{self.i}"
            case BondGraphPortTypes.FLOW_SOURCE:
                port_label = f"Sf_{self.i}"
            case BondGraphPortTypes.ZERO_JUNCTION:
                port_label = f"0_{self.i}"
            case BondGraphPortTypes.ONE_JUNCTION:
                port_label = f"1_{self.i}"
                
        self.graph.add_node(port_label, port_index=self.i port_type=port_type, param=params)
        self.i += 1
        return
    
    def add_bond(self, u, v, imposed_causality:GeneralizedVariables):
        """
        Creates a bond graph bond between nodes u and v. The directed edge corresponds to energy sign (i.e. power flows from the source node to the target node).
        Args:
            u (_type_): 
            v (_type_): _description_
            imposed_causality (GeneralizedVariables): _description_
        """
        self.graph.add_edge(u,v, imposed_causality=imposed_causality)
        
        # Check if the node is valid or not
        return
    
    def get_num_energy_storage_elements(self):
        """
        Returns the number of energy storage elements in the bond graph.
        """
        num_energy_storage_elements = 0
        for node in self.graph.nodes:
            if self.graph.nodes[node]['port_type'] == BondGraphPortTypes.CAPACITANCE or self.graph.nodes[node]['port_type'] == BondGraphPortTypes.INERTANCE or self.graph.nodes[node]['port_type'] == BondGraphPortTypes.RESISTANCE:
                num_energy_storage_elements += 1
        return num_energy_storage_elements
    
    def get_possible_bonds():
        return

