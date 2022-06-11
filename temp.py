import pandas as pd

#import matplotlib.pyplot as plt

import networkx as nx

df = pd.read_excel('data.xlsx')
g = nx.from_pandas_edgelist(df,'FromNodeId','ToNodeId')
#nx.draw(g)
#print(g)
#//////////////////////////////////////////////////////

           # Adjacency matrix
           
"""nx.draw(g,with_labels=True,node_color='#FFC0CB',node_size=1500)
print(nx.to_numpy_matrix(g))"""

#//////////////////////////////////////////////
                    

              # degree sequence 
              
"""from collections import Counter
import matplotlib.pyplot as plt

degree_sequence = [g.degree(n) for n in g.nodes]
degree_counts = Counter(degree_sequence)

min_degree, max_degree = min(degree_counts.keys()), max(degree_counts.keys())

plt.xlabel("Degree", fontsize=20)
plt.ylabel("Number of Nodes", fontsize=20)
plot_x = list(range(min_degree, max_degree + 1))
plot_y = [degree_counts.get(x, 0) for x in plot_x]
plt.bar(plot_x, plot_y)

#//////////////////////////////////////////////

         # degree distribution 


def plot_degree_dist(g):
    
    degrees = g.degree()
    degrees = dict(degrees)
    values = sorted(set(degrees.values()))
    histo = [list(degrees.values()).count(x) for x in values]
    P_k = [x / g.order() for x in histo]
    
    plt.figure()
    plt.bar(values, P_k)
    plt.xlabel("k",fontsize=20)
    plt.ylabel("p(k)", fontsize=20)
    plt.title("Degree Distribution", fontsize=20)
    
    plt.show()
    plt.figure()
    plt.grid(False)
    plt.loglog(values, P_k, "bo")
    plt.xlabel("k", fontsize=20)
    plt.ylabel("log p(k)", fontsize=20)
    plt.title("log Degree Distribution")
    plt.show()
    plt.show()
    
plot_degree_dist(g)"""

# //////////////////////////////////////////////

             # Simulation

""""def initial_state(g):
    state = {}
    for node in g.nodes:
        state[node] = 'asleep'
    return state
#print(initial_state(g))

import random

P_AWAKEN = 0.2
def state_transition(G, current_state):
    next_state = {}
    for node in G.nodes:
        if current_state[node] == 'asleep':
            if random.random() < P_AWAKEN:
                next_state[node] = 'awake'
    return next_state
test_state = initial_state(g)
#print(state_transition(g, test_state))

from simulation import Simulation

sim = Simulation(g, initial_state, state_transition, name='Simple Sim')


#print(sim.state())
#sim.draw()
#sim.run()
#print(sim.steps)
#sim.draw(with_labels=True)
#print(sim.state())

#sim.run(10)
#print(sim.steps)

#sim.draw(with_labels=True)

#sim.plot()          مش بترسم

#sim.draw(7)

sim.plot(min_step=2, max_step=8)
"""

