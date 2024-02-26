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
        
        self.flow_causal_graph = nx.DiGraph()
        self.effort_causal_graph = self.flow_causal_graph.reverse(copy=False)
    
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

        self.flow_causal_graph.add_node(node_index, element_type=element_type, node_index=node_index, max_ports=max_ports, causality=causality, params=params) # TODO: entire node class is stored in networkx attributes, but this is redundant
        # Generate element labels in format {element_type}_{index}
        match element_type:
            case BondGraphElementTypes.CAPACITANCE:
                q = symbols(f"q_{self.i}")
                q_dot = symbols(f"q_dot_{self.i}")
                self.flow_causal_graph.nodes[node_index]['q'] = q
                self.flow_causal_graph.nodes[node_index]['q_dot'] = q_dot
                
                element_label = f"C_{self.i}"
                
            case BondGraphElementTypes.INERTANCE:
                p = symbols(f"p_{self.i}")
                p_dot = symbols(f"q_dot_{self.i}")
                self.flow_causal_graph.nodes[node_index]['p'] = p
                self.flow_causal_graph.nodes[node_index]['p_dot'] = p_dot
                
                element_label = f"I_{self.i}"
            case BondGraphElementTypes.RESISTANCE:
                element_label = f"R_{self.i}"
                
                
            case BondGraphElementTypes.EFFORT_SOURCE:
                Se = symbols(f"Se_{self.i}")
                self.flow_causal_graph.nodes[node_index]['Se'] = Se
                
                element_label = f"Se_{self.i}"
                
            case BondGraphElementTypes.FLOW_SOURCE:
                Sf = symbols(f"Sf_{self.i}")
                self.flow_causal_graph.nodes[node_index]['Sf'] = Sf
                
                element_label = f"Sf_{self.i}"
                
            case BondGraphElementTypes.ZERO_JUNCTION:
                element_label = f"0_{self.i}"
                
            case BondGraphElementTypes.ONE_JUNCTION:
                element_label = f"1_{self.i}"
                
                
            case _: # Default case
                raise ValueError("Invalid element type")
        
        self.flow_causal_graph.nodes[node_index]['element_label'] = element_label
        
        self.i += 1
        
        pass
    
    def add_bond(self, u:int, v:int, power_sign:int):
        """
        Creates a bond graph bond between nodes u and v. Directivity of the nodes in DAG corresponds to causality. u -> v encodes u imposing flow causality on v. By power conservation, v also imposes effort causality on u.
        Args:
            u (int): node index of the source node
            v (int): node index of the destination node
            power_sign (int): Power directivity of the flow. 1 indicates power flowing into the destination node, -1 indicates power flowing out of the destination node.
        """
        # Check if both nodes exist
        if u not in self.flow_causal_graph.nodes or v not in self.flow_causal_graph.nodes:
            raise ValueError("One of the bond graph element indices does not exist.")

        ### Check if the node is valid or not with mask ###
        causal_adjacency_mask, power_adjancency_mask = self.get_bond_addition_mask()
        
        # Check if the direction of causality is valid or not with mask
        if causal_adjacency_mask[u, v] == 0:
            raise ValueError("Bond addition is masked due to causal adjacency.")
        
        # Check if the direction of power flow is valid or not with mask
        if power_sign == 1:
            if power_adjancency_mask[u, v] == 0:
                raise ValueError("Bond addition is masked due to power adjacency.")
        elif power_sign == -1:
            if power_adjancency_mask[v, u] == 0:
                raise ValueError("Bond addition is masked due to power adjacency.")
        
        e = Symbol(f"e_{u}:{v}")
        f = Symbol(f"f_{u}:{v}")
        self.flow_causal_graph.add_edge(u, v, power_sign=power_sign, e=e, f=f)
            
        pass
    
    def get_element_addition_mask(self):
        num_energy_storage_elements = len(self.get_energy_storage_elements())

        if self.flow_causal_graph.number_of_nodes() < self.max_nodes:
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
        
        for node in self.flow_causal_graph.nodes:
            if is_energy_storage_element(self.flow_causal_graph.nodes[node]['element_type']):
                energy_storage_elements.append(node)
        
        energy_storage_elements = [x for x, y in self.flow_causal_graph.nodes(data=True) if is_energy_storage_element(y['element_type'])]
        return energy_storage_elements
    
    def get_n_port_elements(self, n:int):
        """
        Returns a list of elements with n ports in the bond graph.
        """
        n_port_elements = [x for x, y in self.flow_causal_graph.nodes(data=True) if y['max_ports'] == n]
        # for node in self.graph.nodes:
        #     if is_1port(self.graph.nodes[node]['port_type']):
        #         one_ports.append(node)
        return n_port_elements
    
    # def derive_state_space_equations(self):
    #     state_nodes = self.get_energy_storage_elements()
    #     state_expressions = []
        
    #     for state_node in state_nodes:
    #         if state_node['causality'] == CausalityTypes.INTEGRAL:
    #             if state_node['element_type'] == BondGraphElementTypes.CAPACITANCE:
    #                 state_node['node'].get_flow_expr() = q_dot/C
                    
    #             elif state_node['element_type'] == BondGraphElementTypes.INERTANCE:
    #                 state_node['node'].get_effort_expr()
    #         else:
    #             raise ValueError("Non-integral causalities are not supported yet!") # TODO: deal with derivative causality
    #     pass
    

    # def is_permitted_bond(self, u, v, power_sign):
    #     """
    #     Checks if a bond is permitted between two nodes. u -> v indicates causality while node attribute  (Implemented in addition to get_bond_addition_mask for efficiency purposes)
    #     """
    #     # Prohibit adding bonds to elements that have max ports already
    #     if self.graph.nodes[u]['max_ports'] == len(self.graph[u]) or self.graph.nodes[v]['max_ports'] == len(self.graph[u]):
    #         return False
        
    #     # Prohibit power flowing into sources
    #     elif is_source_element(self.graph.nodes[v]['element_type']) and power_sign > 0:
    #         return False
        
    #     elif is_source_element(self.graph.nodes[u]['element_type']) and power_sign < 0:
    #         return False
        
    #     # Prohibit power flowing out of passive 1-ports
    #     elif is_passive_1port(self.graph.nodes[u]['element_type']) and power_sign < 0:
    #         return False
    #     elif is_passive_1port(self.graph.nodes[v]['element_type']) and power_sign > 0:
    #         return False
            
    #     return True
    
    def get_bond_addition_mask(self): # TODO: is recomputing the bond addition mask every time the graph updates too computationally expensive? maybe do it iteratively at each node/edge addition
        # Adjacency matrix in standard graph.nodes order
        # u -> v indicates: u imposes flow causality on v, v imposes effort causality on u. v -> u indicates the opposite
        causal_adjacency_mask = np.full([self.flow_causal_graph.number_of_nodes(), self.flow_causal_graph.number_of_nodes()], 1)
        power_flow_adjacency_mask = np.full([self.flow_causal_graph.number_of_nodes(), self.flow_causal_graph.number_of_nodes()], 1)
        
        # Prevent bonds between the same node
        np.fill_diagonal(causal_adjacency_mask, 0)
        np.fill_diagonal(power_flow_adjacency_mask, 0)
        
        for node in self.flow_causal_graph.nodes:
            # Prohibit adding bonds to elements that have max ports already
            if self.flow_causal_graph.nodes[node]['max_ports'] == len(self.flow_causal_graph[node]):
                causal_adjacency_mask[node, :] = 0
                causal_adjacency_mask[:, node] = 0
            
            # Enforce deterministic flow/effort causality on 1 and 0 junctions (a single causal source)
            if self.flow_causal_graph.nodes[node]['element_type'] == BondGraphElementTypes.ZERO_JUNCTION:
                if self.flow_causal_graph.out_degree(node) == 1:
                    causal_adjacency_mask[node, :] = 0
            
            if self.flow_causal_graph.nodes[node]['element_type'] == BondGraphElementTypes.ONE_JUNCTION:
                if self.flow_causal_graph.in_degree(node) == 1:
                    causal_adjacency_mask[:, node] = 0
        
            # Prohibit power flowing into sources
            if is_source_element(self.flow_causal_graph.nodes[node]['element_type']):
                power_flow_adjacency_mask[:, node] = 0

            # Prohibit power flowing out of passive 1-ports
            if is_passive_1port(self.flow_causal_graph.nodes[node]['element_type']):
                power_flow_adjacency_mask[node, :] = 0
                
                # Prohibit integral causality on states
                # TODO: add integral causality functionality in future
                if self.flow_causal_graph.nodes[node]['element_type'] == BondGraphElementTypes.CAPACITANCE:
                    causal_adjacency_mask[node, :] = 0 # Prohibit imposed effort causality into capacitance
                if self.flow_causal_graph.nodes[node]['element_type'] == BondGraphElementTypes.INERTANCE:
                    causal_adjacency_mask[:, node] = 0 # Prohibit imposed flow causality into inertance
                        
        return causal_adjacency_mask, power_flow_adjacency_mask
        
    
    def get_flow_expr(self, node_index:int, prev_node_index:int = None):
        """
        Return the flow expression of a given node
        """
        node = self.flow_causal_graph.nodes[node_index]
        
        match node['element_type']:
            case BondGraphElementTypes.CAPACITANCE:
                return Derivative(node['q']) # INTEGRAL CAUSLAITY
                
            case BondGraphElementTypes.INERTANCE:
                return node['p']/node['params']['I']
            
            case BondGraphElementTypes.RESISTANCE:
                predecessors = list(self.flow_causal_graph.predecessors(node_index))
                
                # if len(predecessors) == 1:
                #     raise ValueError("Resistance element has more than one bond.")
                
                return node['e']/node['params']['R']
                
            case BondGraphElementTypes.EFFORT_SOURCE:
                raise ValueError("Effort source queried for flow - incorrect causality assignment present.")
            
            case BondGraphElementTypes.FLOW_SOURCE:
                return node['Sf']
            
            case BondGraphElementTypes.ZERO_JUNCTION: 
                # Since directivity governs flow causality, the successor nodes are the ones that impose flow causality on the given node
                successors = list(self.flow_causal_graph.successors(node_index))
                
                flow_summation_terms = [self.get_flow_expr(successor, prev_node_index=node_index)*self.flow_causal_graph.edges[node_index, successor]['power_sign'] for successor in successors]
                sum(flow_summation_terms)
                return 
            
            case BondGraphElementTypes.ONE_JUNCTION:
                return
            
            case BondGraphElementTypes.TRANSFORMER:
                return
            
            case BondGraphElementTypes.GYRATOR:
                pass
            
            case _:
                return
            
    def get_effort_expr(self, node_index:int, prev_node_index:int = None):
            node = self.flow_causal_graph.nodes[node_index]
        
            match node['element_type']:
                case BondGraphElementTypes.CAPACITANCE:
                    return Derivative(node['q']) # INTEGRAL CAUSLAITY
                    
                case BondGraphElementTypes.INERTANCE:
                    return node['p']/node['params']['I']
                
                case BondGraphElementTypes.RESISTANCE:
                    predecessors = list(self.flow_causal_graph.predecessors(node_index))
                    
                    # if len(predecessors) == 1:
                    #     raise ValueError("Resistance element has more than one bond.")
                    
                    return node['e']/node['params']['R']
                    
                case BondGraphElementTypes.EFFORT_SOURCE:
                    pass
                
                case BondGraphElementTypes.FLOW_SOURCE:
                    return node['Sf']
                
                case BondGraphElementTypes.ZERO_JUNCTION:  
                    # Since directivity governs flow causality, the successor nodes are the ones that impose flow causality on the given node
                    successors = list(self.flow_causal_graph.successors(node_index))
                    flow_summation_terms = [self.get_flow_expr(successor)*self.flow_causal_graph.edges[node_index, successor]['power_sign'] for successor in successors]
                    return sum(flow_summation_terms) 
                
                case BondGraphElementTypes.ONE_JUNCTION:
                    
                    pass
                
                case BondGraphElementTypes.TRANSFORMER:
                    pass
                
                case BondGraphElementTypes.GYRATOR:
                    pass
                
                case _:
                    return
            
            return 
            # Terminal condition for recursion: is a state node or a source node
            
    def derive_constitutive_laws(self, node_index:int):
        EDGE_ATTRIBUTE_DICTIONARY_INDEX = 2
    
        node = self.flow_causal_graph.nodes[node_index]
        
        flow_in_bonds = list(self.flow_causal_graph.predecessors(node_index))
        num_flow_predecessors = len(flow_in_bonds)
        
        flow_out_bonds = list(self.flow_causal_graph.successors(node_index))
        num_flow_successors = len(flow_out_bonds)
        
        effort_predecessors = list(self.effort_causal_graph.predecessors(node_index))
        num_effort_predecessors = len(effort_predecessors)
        
        effort_successors = list(self.effort_causal_graph.successors(node_index))
        num_effort_successors = len(effort_successors)
        
        
        
        expr = []
        
        match node['element_type']:
            case BondGraphElementTypes.CAPACITANCE:
                assert num_flow_predecessors == 1
                
                attached_bond = list(self.flow_causal_graph.in_edges(node_index, data=True))[0]
                
                # Contitutive law (integral causality): e = q/C
                expr.append(Eq(attached_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['e'] , node['q']/node['params']['C']))
                
                # Co-energy equality: set the derivative of the displacement equal to the flow attribute of the connecting bond
                expr.append(Eq(node['q_dot'], attached_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['f']))

            case BondGraphElementTypes.INERTANCE:
                assert num_effort_predecessors == 1
                attached_bond = list(self.effort_causal_graph.in_edges(node_index, data=True))[0]
                
                # Constitutive law (integral causality): f = p/I
                expr.append(Eq(attached_bond[2]['f'] , node['p']/node['params']['I']))
                
                # Co-energy equality: set the derivative of the momentum equal to the effort attribute of the connecting bond
                expr.append(Eq(node['p_dot'], attached_bond[2]['e']))
            
            case BondGraphElementTypes.RESISTANCE:
                assert num_effort_predecessors==1 or num_flow_predecessors==1
                assert num_effort_predecessors != num_flow_predecessors
                
                # Contitutive law (flow causality): e = f*R
                if num_flow_predecessors == 1:
                    attached_bond = list(self.flow_causal_graph.in_edges(node_index, data=True))[0]
                    expr.append(Eq(attached_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['e'] , attached_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['f']*node['params']['R']))
                
                # Contitutive law (flow causality): f = e/R
                elif num_effort_predecessors == 1:
                    attached_bond = list(self.effort_causal_graph.in_edges(node_index, data=True))[0]
                    expr.append(Eq(attached_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['f'] , attached_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['e']/node['params']['R']))
                
            case BondGraphElementTypes.EFFORT_SOURCE:
                assert num_effort_successors == 1
                attached_bond = list(self.effort_causal_graph.out_edges(node_index, data=True))[0]
                expr.append(Eq(node['Se'], attached_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['e']))
            
            case BondGraphElementTypes.FLOW_SOURCE:
                assert num_flow_successors == 1
                attached_bond = list(self.flow_causal_graph.out_edges(node_index, data=True))[0]
                expr.append(Eq(node['Sf'], attached_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['f']))
            
            case BondGraphElementTypes.ZERO_JUNCTION:  
                effort_in_bonds = list(self.effort_causal_graph.in_edges(node_index, data=True))
                assert len(effort_in_bonds) == 1 # Make sure there is only one flow causality source
                
                effort_out_bonds = list(self.effort_causal_graph.out_edges(node_index, data=True)) 

                # Power conservation via sum of flows equaling zero
                expr.append(-Eq(effort_in_bonds[0][EDGE_ATTRIBUTE_DICTIONARY_INDEX]['f']*effort_in_bonds[0][EDGE_ATTRIBUTE_DICTIONARY_INDEX]['power_sign'], \
                    sum(effort_out_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['f']*effort_out_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['power_sign'] for effort_out_bond in flow_out_bonds)))
                
                # All efforts equal each other
                for effort_out_bond in effort_out_bonds:
                    expr.append(Eq(effort_in_bonds[0][EDGE_ATTRIBUTE_DICTIONARY_INDEX]['e'], effort_out_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['e']))
            
            case BondGraphElementTypes.ONE_JUNCTION:
                
                flow_in_bonds = list(self.flow_causal_graph.in_edges(node_index, data=True))
                assert len(flow_in_bonds) == 1 # Make sure there is only one flow causality source
                
                flow_out_bonds = list(self.flow_causal_graph.out_edges(node_index, data=True)) 

                # Power conservation via sum of efforts equaling zero
                expr.append(Eq(flow_in_bonds[0][EDGE_ATTRIBUTE_DICTIONARY_INDEX]['e']*flow_in_bonds[0][EDGE_ATTRIBUTE_DICTIONARY_INDEX]['power_sign'], \
                    sum(-flow_out_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['e']*flow_out_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['power_sign'] for flow_out_bond in flow_out_bonds)))
                
                # All flows equal each other
                for flow_out_bond in flow_out_bonds:
                    expr.append(Eq(flow_in_bonds[0][EDGE_ATTRIBUTE_DICTIONARY_INDEX]['f'], flow_out_bond[EDGE_ATTRIBUTE_DICTIONARY_INDEX]['f']))

            # case BondGraphElementTypes.TRANSFORMER:
            #     pass
            
            # case BondGraphElementTypes.GYRATOR:
            #     pass
            
            case _:
                return
        return expr
    
    def get_state_space_matrix(self):
        """
        Returns the state space matrices A, B of the bond graph where x_dot = Ax + Bu
        """
        # for each state variable
            # get expression for each state variable
            # construct the state space matrix
        state_vars = self.get_energy_storage_elements()
        # sys_eq = []
        
        # for n in self.flow_causal_graph():
        return NotImplementedError
            
        

