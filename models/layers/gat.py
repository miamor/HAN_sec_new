import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import GNN_MSG_KEY, GNN_NODE_FEAT_IN_KEY, GNN_NODE_FEAT_OUT_KEY, GNN_EDGE_FEAT_KEY, GNN_AGG_MSG_KEY, GNN_EDGE_LABELS_KEY, GNN_EDGE_TYPES_KEY, GNN_NODE_TYPES_KEY, GNN_NODE_LABELS_KEY



class GATLayer(nn.Module):
    def __init__(self, g, node_dim, edge_ft_out_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g

        # print('self.g', self.g)

        edge_lbl_dim = self.g.edata[GNN_EDGE_LABELS_KEY].shape[1]
        # edge_ft_out_dim = edge_lbl_dim

        self.weighted_agg_edge = WeightedAggEdge(g, node_dim, edge_lbl_dim, edge_ft_out_dim)
        # self.node_lv = NodeLevelGAT(g, node_dim, edge_ft_out_dim, out_dim)
        self.node_lv = NodeLevelGAT(g, node_dim, edge_ft_out_dim, out_dim)
        # self.semantic_lv = SemLevelGAT(g, out_dim, out_dim)


    def forward(self, node_features, g):
        if g is not None:
            self.g = g

        # weighted-sum edges features
        self.weighted_agg_edge(node_features, self.g)
        # print('z_concat_edges', z_concat_edges.shape)

        # node-level
        z_node_lv = self.node_lv(node_features, self.g)
        # print('z_node_lv', z_node_lv.shape)


        # semantic level
        # z_final = self.semantic_lv(z_node_lv, self.g)
        # print('z_final', z_final.shape)

        z_final = z_node_lv

        return z_final




class WeightedAggEdge(nn.Module):
    """
    Weight by edge features (edge_lbl_ft)
    """
    def __init__(self, g, node_dim, edge_lbl_dim, edge_ft_out_dim):
        super(WeightedAggEdge, self).__init__()
        self.g = g

        self.fc = nn.Linear(edge_lbl_dim, edge_ft_out_dim, bias=False)

        input_concat_dim = node_dim + edge_ft_out_dim

        self.e_attn_fc = nn.Linear(input_concat_dim, 1, bias=False)

        # print('node_dim', node_dim)
        # print('edge_ft_out_dim', edge_ft_out_dim)
        # self.linear = nn.Linear(input_concat_dim, z_concat_edges_dim, bias=False)


    def filter_by_dst(self, edges, dst_tensor_to_filter):

        # self.dst_tensor_to_filter = torch.tensor([0.,0.])

        dst_nodes = edges.dst['node_feat']
        # print("dst_nodes (edges.dst['z'])", dst_nodes)

        diff = dst_nodes - dst_tensor_to_filter
        # print('diff', diff)

        diff_sum = abs(diff).sum(-1)
        # print('diff_sum', diff_sum)

        # print('diff', diff.shape)
        # print('diff_sum', diff_sum.shape)

        loc = torch.where(diff_sum==0)
        # print('loc[0].shape', loc[0].shape)
        # print("~~ edges.data['e']", edges.data['e'])
        # print('~~ e_filtered', e_filtered)

        # print("edges.dst['z']===", edges.dst['z'])
        # return (edges.dst['z'] == torch.Tensor([ [1, -1]] ))
        return loc


    # def agg_edge(self, nodes):
    #     edges_ft = torch.sum(self.e_w_from_node, dim=1)
    #     ft_concat = torch.cat((edges_ft, nodes.ndata['node_feat']))
    #     ft_concat__fc = self.linear(ft_concat)
    #     return {'z_concat_edges', ft_concat}

    def edge_weight_src(self, edges):
        # print('num edges', len(edges))
        # e_fc = self.fc(edges.data[GNN_EDGE_LABELS_KEY])

        e_ft = edges.data[GNN_EDGE_LABELS_KEY]
        src_ft = edges.src['node_feat']

        z2 = torch.cat([e_ft, src_ft], dim=1)
        a = self.e_attn_fc(z2)
        e = F.leaky_relu(a)
        gamma = F.softmax(e, dim=1)
        e_weighted = gamma * edata_filtered
        edges.data['e_weighted'][loc] = e_weighted

        return {'e_weighted': edges.data['e_weighted'], 'src_id': edges.src['nid'], 'dst_id': edges.dst['nid']}


    def edge_weight(self, edges):
        # print('num edges', len(edges))
        # e_fc = self.fc(edges.data[GNN_EDGE_LABELS_KEY])
        print('Begin edge_weight')
        for nid in range(self.g.number_of_nodes()):
            dst_tensor_to_filter = self.g.ndata['node_feat'][nid]
            loc = self.filter_by_dst(edges, dst_tensor_to_filter)

            # print('nid', nid, 'loc', loc)

            if loc[0].shape[0] > 0:
                # print('loc[0].shape[0]', loc[0].shape[0])
                edata_filtered = edges.data['e_weighted'][loc]
                esrc_filtered = edges.src['node_feat'][loc]

                z2 = torch.cat([edata_filtered, esrc_filtered], dim=1)
                a = self.e_attn_fc(z2)
                e = F.leaky_relu(a)
                gamma = F.softmax(e, dim=1)
                e_weighted = gamma * edata_filtered
                edges.data['e_weighted'][loc] = e_weighted

        # Apply func to source node of these edges
        # self.e_w_from_node = edges.data['e_weighted']
        # self.g.apply_nodes(func=self.agg_edge)

        print('End edge_weight')

        return {'e_weighted': edges.data['e_weighted'], 'src_id': edges.src['nid'], 'dst_id': edges.dst['nid']}


    def forward(self, h, g):
        if g is not None:
            self.g = g

        e = self.g.edata[GNN_EDGE_LABELS_KEY]
        # print('e (WeightedAggEdge)', e.shape)

        self.g.ndata['node_feat'] = h
        # self.g.edata['e_weighted'] = self.g.edata[GNN_EDGE_LABELS_KEY]
        self.g.edata['e_weighted'] = self.fc(self.g.edata[GNN_EDGE_LABELS_KEY])

        # self.g.group_apply_edges(func=self.edge_weight, group_by='src')
        
        self.g.group_apply_edges(func=self.edge_weight_src, group_by='src')
        print('Done calculating e_weighted!')

        return self.g.ndata.pop('node_feat')



class NodeLevelGAT(nn.Module):
    """
    Weight by neighbor nodes
    """
    def __init__(self, g, node_in_dim, edge_ft_dim, out_dim):
        super(NodeLevelGAT, self).__init__()
        self.g = g

        # equation (1)
        # self.fc = nn.Linear(node_in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(node_in_dim + edge_ft_dim  +  node_in_dim, 1, bias=False)

        self.fc2 = nn.Linear(node_in_dim + edge_ft_dim, out_dim, bias=False)


    def concat_sum_edge_node_ft(self, edges):
        num_edges = len(edges)
        # concat_ft = torch.cat((edges.data['e_weighted'] * len(edges), edges.src['z']))
        # return {'e_weighted_concat_node_ft': concat_ft}
        e_w = edges.data['e_weighted']
        # print('e_w', e_w)
        # print('e_w', e_w.shape)
        # print('src', edges.src['nid'])

        e_w_sum = torch.sum(edges.data['e_weighted'], dim=1).unsqueeze(0)
        # print('e_w_sum', e_w_sum)
        # print('e_w_sum', e_w_sum.shape)
        e_w_sum = e_w_sum.repeat(1,1,e_w.shape[1]).reshape(e_w.shape)

        # print("edges.src['z']", edges.src['z'])
        # print("edges.src['z']", edges.src['z'].shape)
        # print('e_w_sum~', e_w_sum.shape)
        # print('e_w', e_w.shape)

        concat_ft = torch.cat((e_w_sum, edges.src['z']), dim=2)
        return {'e_weighted_sum_concat_node_ft': concat_ft}


    def edge_attention_(self, edges):
        print('edges', edges)

        # src_nodes, dst_nodes = self.g.find_edges(edges)
        # print('src_nodes', src_nodes)

        src_nodes_ids = edges.data['src_id']
        dst_nodes_ids = edges.data['dst_id']
        print('src_nodes_ids', src_nodes_ids)

        # # edges that come out from src_nodes
        out_edges = self.g.out_edges(src_nodes_ids, form='eid')
        print('out_edges', out_edges)
        num_out_edges = len(out_edges)
        print('num out_edges', num_out_edges)
        out_edges_e_w = out_edges.data['e_weighted']
        print('out_edges_e_w', out_edges_e_w)
        e_w_ft = torch.sum(out_edges_e_w, dim=1)

        z_src_concat = torch.cat((edges.src['z'], e_w_ft), dim=1)


        # edge UDF for equation (2)
        z2 = torch.cat([z_src_concat, edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}


    def edge_att(self, edges):
        # en_ft = edges.data['e_weighted_sum_concat_node_ft']
        en_ft = edges.data['e_weighted']

        src_nodes_ids = edges.data['src_id']
        dst_nodes_ids = edges.data['dst_id']
        # print('src_nodes_ids', src_nodes_ids)

        # print('en_ft', en_ft)
        # print('unique_en_ft', unique_en_ft)
        # print('en_ft.shape', en_ft.shape)
        # print('unique_en_ft.shape', unique_en_ft.shape)
        # print("edges.dst['z']", edges.dst['z'].shape)

        print('en_ft',en_ft)
        print('en_ft.shape', en_ft.shape)

        z2 = torch.cat([edges.src['z'], en_ft, edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a), 'edge_type': edges.data[GNN_EDGE_TYPES_KEY]}


    def edge_attention(self, edges):
        print('edges', edges)

        # src_nodes, dst_nodes = self.g.find_edges(edges)
        # print('src_nodes', src_nodes)

        e_w = edges.data['e_weighted']
        print('e_w', e_w)
        e_w_sum = torch.sum(e_w, dim=1).unsqueeze(0).repeat(edges.src['z'].shape[0], edges.src['z'].shape[1], 1)
        print('e_w_sum', e_w_sum)

        print("edges.src['z']", edges.src['z'])
        print("edges.src['z']", edges.src['z'].shape)
        print('e_w_sum', e_w_sum.shape)

        z_src_edge_concat = torch.cat((edges.src['z'], e_w_sum), dim=2)
        # z_src_edge_concat = F.leaky_relu(z_src_edge_concat)
        print('z_src_edge_concat', z_src_edge_concat)

        print("edges.dst['z']", edges.dst['z'])

        print('z_src_edge_concat.shape', z_src_edge_concat.shape)
        print('\n')

        # edge UDF for equation (2)
        z2 = torch.cat([z_src_edge_concat, edges.dst['z']], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}


    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        # return {'z': edges.src['z'], 'e': edges.data['e']}
        return {'z': edges.data['e_weighted_sum_concat_node_ft'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # Equation (3)
        #   Normalize torche attention scores using softmax (equation (3)).
        # print("nodes.mailbox['e']", nodes.mailbox['e'])
        alpha = F.softmax(nodes.mailbox['e'], dim=1)

        # Equation (4)
        #   Aggregate neighbor embeddings weighted by torche attention scores (equation(4)).
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        print('h', h.shape)

        h = self.fc2(h)

        return {'z_node_lv': h}


    def forward(self, h, g):
        if g is not None:
            self.g = g

        # node-level
        # Equation (1)
        # z = self.fc(h)
        z = h
        self.g.ndata['z'] = z

        self.g.group_apply_edges(func=self.concat_sum_edge_node_ft, group_by="src")

        # Equation (2)
        # The un-normalized attention score eij is calculated using torche embeddings of adjacent nodes i and j. This suggests torchat torche attention scores can be viewed as edge data, which can be calculated by torche apply_edges API. The argument to torche apply_edges is an Edge UDF, which is defined as below:
        # self.g.apply_edges(self.edge_attention)
        self.g.apply_edges(func=self.edge_att)

        # Equation (3) & (4)
        # `update_all` API is used to trigger message passing on all torche nodes. The message function sends out two tensors: 
        #   - torche transformed z embedding of torche source node, and 
        #   - torche un-normalized attention score e on each edge. 
        # The reduce function torchen performs two tasks:
        #   - Normalize torche attention scores using softmax (equation (3)).
        #   - Aggregate neighbor embeddings weighted by torche attention scores (equation(4)).
        self.g.update_all(self.message_func, self.reduce_func)

        return self.g.ndata.pop('z_node_lv')


'''
class SemLevelGAT(nn.Module):
    """
    Weight by edge type
    """
    def __init__(self, g, z_node_lv_dim, out_dim):
        super(SemLevelGAT, self).__init__()
        self.g = g
        # equation (1)
        edge_type_dim = self.g.edata[GNN_EDGE_TYPES_KEY].shape[1]
        self.attn_fc = nn.Linear(edge_type_dim, 1, bias=False)
        self.linear = nn.Linear(z_node_lv_dim, out_dim, bias=False)

    def edge_weight(self, edges):
        # edge UDF for equation (2)
        a = self.attn_fc(self.g.edata[GNN_EDGE_TYPES_KEY])
        return {'e_type_tanh': torch.tanh(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z_node_lv': edges.src['z_node_lv'], 'e_type_tanh': edges.data['e_type_tanh']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # print("len nodes.mailbox['e_type_tanh']", len(nodes.mailbox['e_type_tanh']))
        # print("len nodes", len(nodes))
        sum_tanh = torch.sum(nodes.mailbox['e_type_tanh'], dim=1)
        e_type_w = sum_tanh/len(nodes)

        # print("nodes.mailbox['e_type_tanh']", nodes.mailbox['e_type_tanh'])
        # print('sum tanh', sum_tanh)
        # print('e_type_w', e_type_w)
        # Equation (3)
        #   Normalize torche attention scores using softmax (equation (3)).
        beta = F.softmax(e_type_w, dim=1)
        rows = beta.shape[0]
        # beta = torch.transpose(beta, 0, 1)
        beta = beta.unsqueeze(1)

        # print('beta', beta)
        # print("nodes.mailbox['z_node_lv']", nodes.mailbox['z_node_lv'])

        # print("nodes.mailbox['z_node_lv']", nodes.mailbox['z_node_lv'].shape)
        # print('beta', beta.shape)

        # Equation (4)
        #   Aggregate neighbor embeddings weighted by torche attention scores (equation(4)).
        h = torch.sum(beta * nodes.mailbox['z_node_lv'], dim=1)
        # print('~!~~~h.shape', h.shape)
        z_final = self.linear(h)

        return {'z_final': z_final}


    def forward(self, h, g): # h: z_noed_lv
        if g is not None:
            self.g = g

        self.g.ndata['z_node_lv'] = h

        self.g.apply_edges(self.edge_weight)

        self.g.update_all(self.message_func, self.reduce_func)

        return self.g.ndata.pop('z_final')

'''



class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, node_dim, edge_ft_out_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, node_dim, edge_ft_out_dim, out_dim))
        self.merge = merge

    def forward(self, node_features, g):
        # print('self.heads~~~', self.heads)
        # for attn_head in self.heads:
        #     head_out = attn_head(node_features, g)
        #     print('head_out~~~', head_out)
        
        # head_outs = [head_out]
        # return torch.cat(head_outs, dim=1)

        head_outs = [attn_head(node_features, g) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on torche output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))
