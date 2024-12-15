import networkx as nx


class Framework:
    def __init__(self, graph, flow, B, arrival_rates, service_rates):
        self.graph = graph
        self.flow = flow  # Should be a dictionary {source: target, flow_value}
        self.B = B
        self.arrival_rates = arrival_rates
        self.service_rates = service_rates
        self.nodes = list(graph.nodes)

    def L(self):
        Ni = {node: 0 for node in self.nodes if node in self.arrival_rates and node in self.service_rates}
        for node in Ni:
            lam = self.arrival_rates[node]
            mu = self.service_rates[node]
            if mu == 0:
                Ni[node] = float('inf')
                continue
            rho = lam / mu
            if rho >= 1:
                Ni[node] = float('inf')
            else:
                Ni[node] = rho / (1 - rho)
        return Ni

    def compute_H(self):
        H_values = {node: 0 for node in self.nodes if node not in ('S', 'D')}  # Correctly exclude S and D
        if not self.flow:  # Handle empty flow
            return H_values

        (source, target), flow_value = list(self.flow.items())[0]  # Correctly unpack flow

        flow_graph = self.graph.copy()
        for u, v, data in flow_graph.edges(data=True):
            data['capacity'] = data.get('weight', float('inf'))

        try:
            flow_dict = nx.maximum_flow(flow_graph, source, target, capacity='capacity')[1]
        except nx.NetworkXUnbounded:
            print(f"Flow between {source} and {target} is unbounded.")
            return H_values

        for node in H_values:
            in_flow = 0
            for neighbor in flow_graph.predecessors(node):
                if neighbor in flow_dict and node in flow_dict[
                    neighbor]:  # Check if neighbor and node exists in the flow dict
                    in_flow += flow_dict[neighbor][node]
            H_values[node] = in_flow
        return H_values

    def compute_C(self):
        C_values = {node: 0 for node in self.nodes}
        dominating_set = nx.dominating_set(self.graph)
        for node in dominating_set:  # Apply rules only to dominating set
            if node not in self.nodes:
                continue
            neighbors = set(self.graph.neighbors(node))
            # Check if all neighbors exist in the graph
            if not all(neighbor in self.graph for neighbor in neighbors):
                continue  # Skip if any neighbor is missing

            subgraph = self.graph.copy()
            if node in subgraph:
                subgraph.remove_node(node)
            else:
                continue

            for v in neighbors:
                if neighbors.issubset(set(self.graph.neighbors(v))):
                    C_values[node] = 0
                    break
            else:  # Only execute if the break statement was not executed
                if neighbors:  # Check if neighbors is not empty
                    if not all(nx.has_path(subgraph, list(neighbors)[0], neighbor) for neighbor in list(neighbors)[1:]):
                        C_values[node] = 1
                        continue

                dominator_neighbors = [n for n in neighbors if n in dominating_set]
                if all(nx.has_path(self.graph, u, v) for u in dominator_neighbors for v in dominator_neighbors):
                    ordinary_neighbors = neighbors - set(dominator_neighbors)
                    if all(any(d in dominating_set for d in self.graph.neighbors(o)) for o in ordinary_neighbors):
                        C_values[node] = 0
                        continue

                for v in neighbors:
                    v_neighbors = set(self.graph.neighbors(v))
                    if v_neighbors.issubset(neighbors.union({node})):
                        if node in subgraph:
                            subgraph.remove_node(node)
                        if neighbors - {v}:
                            if not nx.has_path(subgraph, v, list(neighbors - {v})[0]):
                                C_values[node] = 1
                                break
                else:
                    for v in neighbors:
                        v_dominators = [d for d in self.graph.neighbors(v) if d in dominating_set and d != node]
                        if len(v_dominators) == 1 and v_dominators[0] == node:
                            C_values[node] = 1
                            break
        return C_values

    def normalize(self, values):
        if not values:  # Handle empty dictionary
            return {}
        v_list = list(values.values())  # Extract values from the dictionary
        v_min = min(v_list)
        v_max = max(v_list)
        if v_max == v_min:
            return {n: 0.5 for n in values}
        return {n: (v - v_min) / (v_max - v_min) for n, v in values.items()}

    def final_score(self, w_H=0.3, w_C=0.3, w_L=0.4):
        # Compute H and C scores
        H_values = self.compute_H()
        C_values = self.compute_C()

        # Compute raw L values (JUST ONCE!)
        L_values = self.L()  # Corrected line

        # Normalize H and L
        H_norm = self.normalize(H_values)
        L_norm = self.normalize(L_values)

        # Final scores
        F = {}
        for n in self.nodes:
            F[n] = w_H * H_norm.get(n, 0) + w_C * C_values.get(n, 0) + w_L * L_norm.get(n,
                                                                                        0)  # Added .get for cases where a node doesnt exist
        return F

    def sort_and_filter(self, F_scores):
        """
        Sort nodes by descending F score and return top B.
        """
        sorted_nodes = sorted(F_scores.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_nodes[:self.B]]

    def run(self):
        """
        Run the full framework and return the critical node(s).
        Steps:
        - Compute final scores
        - Sort and filter top B nodes
        - Return them
        """
        F_scores = self.final_score()
        critical_nodes = self.sort_and_filter(F_scores)
        return critical_nodes


if __name__ == "__main__":
    arrival_rates = {'A': 2.5, 'B': 1.8, 'C': 3.1, 'E': 0.7}
    service_rates = {'A': 3.0, 'B': 2.5, 'C': 4.0, 'E': 1.2}
    edges = [
        ('S', 'A', 10),
        ('S', 'B', 5),
        ('A', 'B', 15),
        ('A', 'C', 10),
        ('B', 'D', 10),
        ('C', 'D', 10),
        ('C', 'E', 5),
        ('E', 'D', 5)
    ]
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    flow = {('S', 'D'): 10}
    B = 2
    framework = Framework(graph, flow, B, arrival_rates, service_rates)

    H_scores = framework.compute_H()
    C_scores = framework.compute_C()
    LoNo = framework.L()
    F_scores = framework.final_score()
    critical_nodes = framework.run()

    print("H Scores:", H_scores)
    print("C Scores:", C_scores)
    print("LoNo :", LoNo)
    print("F Scores:", F_scores)
    print("Critical Nodes:", critical_nodes)
