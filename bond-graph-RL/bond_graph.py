import networkx as nx
import numpy as np
from enum import Enum
from sympy import *

from bond_graph_nodes import *
from bond_graph_edges import *

class BondGraph():
    def __init__(self, max_nodes, num_states, num_effort_sources:int=0, num_flow_sources:int=0, time_array:np.ndarray=None): # TODO: Add time array
        """
        Creates a BondGraph system with a specified number of flow and effort sources to start.

        Args:
            num_effort_sources (int): _description_
            num_flow_sources (int): _description_
        """
        self.max_nodes = max_nodes
        self.num_states = num_states
        self.num_effort_sources = num_effort_sources
        self.num_flow_sources = num_flow_sources
        self.i = 0 # index for incrementally labeling nodes
        
        self.graph = nx.DiGraph()
    
    # Restrictive addition of ports (only allows valid bond graph ports)
    def add_element(self, node:BondGraphNode):
        """
        Adds an element to the bond graph. The element is labeled according to its type and the index of the element.

        Args:
            element (BondGraphNode): Specify the type of element to add to the bond graph. Each of the standard bond graph elements inherits the BondGraphNode class.
        """
        
        # TODO: temp parsing for including attributes from BondGraphNode into the graph, but this should be done in a more elegant way
        element_type = node.element_type
        max_ports = node.max_ports
        causality = node.causality
        params = node.params
        node_index = self.i
        

        element_addition_mask = self.get_element_addition_mask()
     
        if element_addition_mask[element_type.value] == 0:   
            raise ValueError(f"element type {element_type} is masked and cannot be added to the bond graph.")
        
        
        # Generate element labels in format {element_type}_{index}
        match element_type:
            case BondGraphElementTypes.CAPACITANCE:
                element_label = f"C_{self.i}"
            case BondGraphElementTypes.INERTANCE:
                element_label = f"I_{self.i}"
            case BondGraphElementTypes.RESISTANCE:
                element_label = f"R_{self.i}"
            case BondGraphElementTypes.EFFORT_SOURCE:
                element_label = f"Se_{self.i}"
            case BondGraphElementTypes.FLOW_SOURCE:
                element_label = f"Sf_{self.i}"
            case BondGraphElementTypes.ZERO_JUNCTION:
                element_label = f"0_{self.i}"
            case BondGraphElementTypes.ONE_JUNCTION:
                element_label = f"1_{self.i}"
            case _: # Default case
                raise ValueError("Invalid element type")

        self.graph.add_node(node_index, element_label=element_label, element_type=element_type, node_index=node_index, max_ports=max_ports, causality=causality, params=params, node=node) # TODO: entire node class is stored in networkx attributes, but this is redundant
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
        if u not in self.graph.nodes or v not in self.graph.nodes:
            raise ValueError("Bond graph element does not exist")
    
        # causal_adjacency_mask, power_flow_adjacency_mask = self.get_bond_addition_mask()
        self.graph.add_edge(u, v, power_sign = 1)
        
        # Check if the node is valid or not
        return
    
    def get_element_addition_mask(self):
        num_energy_storage_elements = len(self.get_energy_storage_elements())

        if self.graph.number_of_nodes() < self.max_nodes:
            allowable_element_types = np.full(len(BondGraphElementTypes), 1) # Filter the allowable element types based on the number of energy storage elements
        else:
            allowable_element_types = np.full(len(BondGraphElementTypes), 0) # Do not allow addition of any nodes if max number of elements is reached
            return allowable_element_types
        
         # Remove energy storage element types based on number of states
        if num_energy_storage_elements >= self.num_states:
            allowable_element_types[BondGraphElementTypes.CAPACITANCE.value] = 0 # Remove capacitance based on enum key
            allowable_element_types[BondGraphElementTypes.INERTANCE.value] = 0 # Remove interance based on enum key
            
        return allowable_element_types
    
    def get_energy_storage_elements(self):
        """
        Returns a list of energy storage elements in the bond graph.
        """
        energy_storage_elements = []
        # for node in self.graph.nodes:
        #     if self.graph.nodes[node]['element_type'] == BondGraphElementTypes.CAPACITANCE or self.graph.nodes[node]['element_type'] == BondGraphElementTypes.INERTANCE or self.graph.nodes[node]['element_type'] == BondGraphElementTypes.RESISTANCE:
        #         energy_storage_elements.append(node)
        
        for node in self.graph.nodes:
            if is_energy_storage_element(self.graph.nodes[node]['element_type']):
                energy_storage_elements.append(node)
        
        energy_storage_elements = [x for x, y in self.graph.nodes(data=True) if is_energy_storage_element(y['element_type'])]
        return energy_storage_elements
    
    def get_n_element_elements(self, n:int):
        """
        Returns a list of elements with n elements in the bond graph.
        """
        n_port_elements = [x for x, y in self.graph.nodes(data=True) if y['max_ports'] == n]
        # for node in self.graph.nodes:
        #     if is_1port(self.graph.nodes[node]['port_type']):
        #         one_ports.append(node)
        return n_port_elements
    
    def derive_state_space_equations(self):
        state_nodes = self.get_energy_storage_elements()
        state_expressions = []
        
        for state_node in state_nodes:
            if state_node['causality'] == CausalityTypes.INTEGRAL:
                if state_node['element_type'] == BondGraphElementTypes.CAPACITANCE:
                    state_node['node'].get_flow_expr()
                    
                elif state_node['element_type'] == BondGraphElementTypes.INERTANCE:
                    state_node['node'].get_effort_expr()
            else:
                raise ValueError("Non-integral causalities are not supported yet!") # TODO: deal with derivative causality
        pass
    
    def get_flow_expr(node):
        return
    
    def get_effort_expr(node):
        pass
    
    def is_permitted_bond(self, u, v, power_sign):
        """
        Checks if a bond is permitted between two nodes. u -> v indicates causality while node attribute  (Implemented in addition to get_bond_addition_mask for efficiency purposes)
        """
        # Prohibit adding bonds to elements that have max ports already
        if self.graph.nodes[u]['max_ports'] == len(self.graph[u]) or self.graph.nodes[v]['max_ports'] == len(self.graph[u]):
            return False
        
        # Prohibit power flowing into sources
        elif is_source_element(self.graph.nodes[v]['element_type']) and power_sign > 0:
            return False
        
        elif is_source_element(self.graph.nodes[u]['element_type']) and power_sign < 0:
            return False
        
        # Prohibit power flowing out of passive 1-ports
        elif is_passive_1port(self.graph.nodes[u]['element_type']) and power_sign < 0:
            return False
        
        elif is_passive_1port(self.graph.nodes[v]['element_type']) and power_sign > 0:
            return False
            
        return True
    
    def get_bond_addition_mask(self):
        # Adjacency matrix in standard graph.nodes order
        # u -> v indicates: u imposes flow causality on v, v imposes effort causality on u. v -> u indicates the opposite
        causal_adjacency_mask = np.full([self.graph.number_of_nodes(), self.graph.number_of_nodes()], 1)
        power_flow_adjacency_mask = np.full([self.graph.number_of_nodes(), self.graph.number_of_nodes()], 1)
        
        for node in self.graph.nodes:
            # Prohibit adding bonds to elements that have max ports already
            if self.graph.nodes[node]['max_ports'] == len(self.graph[node]):
                causal_adjacency_mask[node, :] = 0
                causal_adjacency_mask[:, node] = 0
        
            # Prohibit power flowing into sources
            if is_source_element(self.graph.nodes[node]['element_type']):
                power_flow_adjacency_mask[:, node] = 0

            # Prohibit power flowing out of passive 1-ports
            if is_passive_1port(self.graph.nodes[node]['element_type']):
                power_flow_adjacency_mask[node, :] = 0
                
                # Prohibit integral causality on states
                # TODO: add integral causality functionality in future
                if self.graph.nodes[node]['element_type'] == BondGraphElementTypes.CAPACITANCE:
                    causal_adjacency_mask[node, :] = 0 # Prohibit imposed effort causality into capacitance
                if self.graph.nodes[node]['element_type'] == BondGraphElementTypes.INERTANCE:
                    causal_adjacency_mask[:, node] = 0 # Prohibit imposed flow causality into inertance
                    
                    
            
        return causal_adjacency_mask, power_flow_adjacency_mask
        
        
        return

    def get_state_space_matrix(self):
        """
        Returns the state space matrix of the bond graph.
        """
        return