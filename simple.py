import networkx as nx
import dgl

import torch as th



def msg_fcn(edges):
    print('msg_fcn~~~~~~')
    print("\t edges.data['e']", edges.data['e'])
    print("\t edges.src['z']", edges.src['z'])
    print("\t edges.dst['z']", edges.dst['z'])
    print('\n')
    return {'zsrc': edges.src['z'], 'zdst': edges.dst['z'], 'e': edges.data['e']}

def reduce_fcn(nodes):
    print("nodes.mailbox['zsrc']", nodes.mailbox['zsrc'])
    print("nodes.mailbox['zdst']", nodes.mailbox['zdst'])
    print("nodes.mailbox['e']", nodes.mailbox['e'])
    sum_edges = th.sum(nodes.mailbox['e'])
    new_z = nodes.mailbox['zsrc'] + sum_edges
    # print("nodes.mailbox['zsrc']", nodes.mailbox['zsrc'])
    print('\n')

    return {'z': new_z}


def e_fcn(edges):
    print('e_fcn~~~~~~')
    print("\t edges.data['e']", edges.data['e'])
    print("\t edges.src['z']", edges.src['z'])
    print("\t edges.dst['z']", edges.dst['z'])
    return {'e': edges.src['z']+edges.data['e']}


def edge_weight(edges):
    print("edges.data['e']~~~~", edges.data['e'])

    e_filtered = g.filter_edges(has_dst_one)
    print('e_filtered: ', e_filtered)

    e = edges.data['e'].squeeze(0)
    print('e', e)
    e_sum = th.sum(edges.data['e'])
    return {'e_w': e/e_sum}


def has_dst_one(edges):
    dst_tensor = th.tensor([0.,0.])

    dst_nodes = edges.dst['z']
    print("dst_nodes (edges.dst['z'])", dst_nodes)

    diff = dst_nodes - dst_tensor
    print('diff', diff)

    diff_sum = abs(diff).sum(-1)
    print('diff_sum', diff_sum)

    loc = th.where(diff_sum==0)
    e_filtered = edges.data['e'][loc]
    print("~~ edges.data['e']", edges.data['e'])
    print('~~ e_filtered', e_filtered)

    return e_filtered

    # print("edges.dst['z']===", edges.dst['z'])
    # return (edges.dst['z'] == th.Tensor([ [1, -1]] ))
    # return (edges.dst['z'] == 1 )



g = dgl.DGLGraph(multigraph=True)
for i in range(0,3):
    g.add_nodes(1, data={'z': th.Tensor([ [i, -i]] )})
# g.add_nodes(5)
# A couple edges one-by-one
# for i in range(1, 5):
#     g.add_edge(i, 0, data={'e': th.Tensor([1])})

# A few more with a paired list
# src = list(range(1, 5))
# dst = [0]*len(src)
# g.add_edges(src, dst, data={'e': th.Tensor([1])})

# finish with a pair of tensors
# src = th.tensor([1, 4])
# dst = th.tensor([0, 0])
# g.add_edges(src, dst, data={'e': th.Tensor([1])})
g.add_edge(2, 1, data={'e': th.Tensor([ [3,-3] ])})
g.add_edge(1, 0, data={'e': th.Tensor([ [1,-1] ])})
g.add_edge(1, 0, data={'e': th.Tensor([ [2,-2] ])})
g.add_edge(2, 0, data={'e': th.Tensor([ [4,-4] ])})
g.add_edge(2, 0, data={'e': th.Tensor([ [5,-5] ])})
# g.add_edge(4, 0, data={'e': th.Tensor([2])})

nodes_num = g.number_of_nodes()
edges_num = g.number_of_edges()
print('nodes num', nodes_num)
print('edges num', edges_num)

# save nodeid and edgeid to each node and edge
g.ndata['nid'] = th.zeros(nodes_num)
g.edata['eid'] = th.zeros(edges_num)

for nid in range(nodes_num):
    g.ndata['nid'][nid] = th.tensor([nid])
for eid in range(edges_num):
    g.edata['eid'][eid] = th.tensor([eid])


print(g.edata['e'].shape)
# edges = g.edges()

print('\n*** Init val ***')
for i in range(nodes_num):
    for s in range(nodes_num):
        eids = g.edge_id(i, s)
        if len(eids) > 0:
            edatas = []
            for eid in eids:
                edatas.append(g.edata['e'][eid])
                print('eid', eid, 'edata[eid]', g.edata['eid'][eid])
            print(i, '->', s, ' | ', len(eids), 'edges: ', eids)
            print('\t', i, g.ndata['z'][i], '    ->    ', s, g.ndata['z'][s])
            print('\t edatas', edatas)
print('\n')

# g.edata['e_w'] = th.zeros((g.number_of_edges(), 1))
# g.apply_edges(e_fcn)
g.group_apply_edges(func=edge_weight, group_by='src') # Apply func to the first edge.


print('\n*** Update edge ***')
for i in range(nodes_num):
    for s in range(nodes_num):
        eids = g.edge_id(i, s)
        if len(eids) > 0:
            edatas = []
            edatas_w = []
            for eid in eids:
                edatas.append(g.edata['e'][eid])
                edatas_w.append(g.edata['e_w'][eid])
            print(i, '->', s, eids)
            print('\t', i, g.ndata['z'][i], '    ->    ', s, g.ndata['z'][s])
            print('\t edatas', edatas, '    |    edatas_w', edatas_w)
print('\n')



g.update_all(message_func=msg_fcn, reduce_func=reduce_fcn)

print('\n*** Update node ***')
for i in range(0,3):
    print(i, g.ndata['z'][i])
print('\n')


# Edge broadcasting will do star graph in one go!
# g.clear()
# g.add_nodes(3)

# src = th.tensor(list(range(1, 3)))
# g.add_edges(src, 0)


import networkx as nx
import matplotlib.pyplot as plt
nx.draw(g.to_networkx(), with_labels=True)
# plt.show()