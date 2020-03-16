import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import GNN_MSG_KEY, GNN_NODE_FEAT_IN_KEY, GNN_NODE_FEAT_OUT_KEY, GNN_EDGE_FEAT_KEY, GNN_AGG_MSG_KEY, GNN_EDGE_LABELS_KEY, GNN_EDGE_TYPES_KEY, GNN_NODE_TYPES_KEY, GNN_NODE_LABELS_KEY



class GATLayer(nn.Module):
    def __init__(self, g, node_dim, edge_dim, node_ft_out_dim, edge_ft_out_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g

        self.weighted_agg_edge = WeightedAggEdge(g, node_dim, node_ft_out_dim, edge_dim, edge_ft_out_dim)
        self.node_lv = BaseGAT_Modified(g, node_ft_out_dim, edge_ft_out_dim, out_dim)


    def forward(self, node_features, edge_features, g):
        if g is not None:
            self.g = g

        # weighted-sum edges features
        z_, e_weighted = self.weighted_agg_edge(node_features, edge_features, self.g)

        z_final, e_weighted = self.node_lv(z_, e_weighted, self.g)

        return z_final, e_weighted




class WeightedAggEdge(nn.Module):
    """
    Weight by edge features (edge_lbl_ft)
    """
    def __init__(self, g, node_in_dim, node_ft_out_dim, edge_dim, edge_ft_out_dim):
        super(WeightedAggEdge, self).__init__()
        self.g = g

        self.fc_n = nn.Linear(node_in_dim, node_ft_out_dim, bias=False)
        self.fc_e = nn.Linear(edge_dim, edge_ft_out_dim, bias=False)

        input_concat_dim = node_ft_out_dim + edge_ft_out_dim
        self.e_attn_fc = nn.Linear(input_concat_dim, 1, bias=False)


    def filter_by_dst(self, edges, dst_tensor_to_filter):

        # self.dst_tensor_to_filter = torch.tensor([0.,0.])
        dst_nodes = edges.dst['node_feat']
        diff = dst_nodes - dst_tensor_to_filter
        diff_sum = abs(diff).sum(-1)

        loc = torch.where(diff_sum==0)
        return loc


    def edge_weight_src(self, edges):
        e_ft = edges.data['e_weighted']
        src_ft = edges.src['node_feat']

        z2 = torch.cat([e_ft, src_ft], dim=2)
        a = self.e_attn_fc(z2)
        e = F.leaky_relu(a)
        gamma = F.softmax(e, dim=1)
        e_weighted = gamma * e_ft

        return {'e_weighted': e_weighted, 'src_id': edges.src['nid'], 'dst_id': edges.dst['nid']}

    '''
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
    '''

    def forward(self, h, e, g):
        if g is not None:
            self.g = g

        self.g.ndata['node_feat'] = self.fc_n(h)
        self.g.edata['e_weighted'] = self.fc_e(e)

        # self.g.group_apply_edges(func=self.edge_weight_src, group_by='src')
        # print('~~~~~~~Done calculating e_weighted!\n------------------------')

        return self.g.ndata.pop('node_feat'), self.g.edata.pop('e_weighted')



class BaseGAT_Modified(nn.Module):
    """
    Weight by neighbor nodes
    """
    def __init__(self, g, node_dim, edge_ft_out_dim, out_dim):
        super(BaseGAT_Modified, self).__init__()
        self.g = g

        self.attn_fc = nn.Linear(node_dim + edge_ft_out_dim  +  node_dim, 1, bias=False)
        self.sem_attn = nn.Linear(node_dim + edge_ft_out_dim, 1, bias=False)
        self.fc2 = nn.Linear(node_dim + edge_ft_out_dim, out_dim, bias=False)


    def edge_att(self, edges):
        en_ft = edges.data['e_weighted']

        src_ft_by_type = edges.src['z']
        en_ft_by_type = en_ft
        dst_ft_by_type = edges.dst['z']

        src_cat_edge_ft = torch.cat((src_ft_by_type, en_ft_by_type), dim=1)
        z2 = torch.cat((src_cat_edge_ft, dst_ft_by_type), dim=1)

        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a), 'src_cat_edge_ft': src_cat_edge_ft}


    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.data['src_cat_edge_ft'], 'e': edges.data['e'], 'edge_type': edges.data[GNN_EDGE_TYPES_KEY], 'src_id': edges.src['nid']}


    def reduce_func(self, nodes):
        ''' Node-level attention '''
        # Equation (3)
        #   Normalize torche attention scores using softmax (equation (3)).
        alpha = F.softmax(nodes.mailbox['e'], dim=1)

        # Equation (4)
        #   Aggregate neighbor embeddings weighted by torche attention scores (equation(4)).
        weighted = alpha * nodes.mailbox['z']
        Z = torch.sum(weighted, dim=1)

        ''' Final embedding '''
        Z = self.fc2(Z)

        return {'z': Z}


    def reduce_func_w_sem(self, nodes):
        ''' Node-level attention '''

        z = nodes.mailbox['z']
        e = nodes.mailbox['e']
        e_type = nodes.mailbox['edge_type']

        e_type_shape = e_type.shape
        num_edge = e_type_shape[1]
        e_type_dim = e_type_shape[2]

        e_type_ = e_type.unsqueeze(0).view(e_type.shape[0], num_edge, e_type_dim, -1)


        # Equation (3)
        #   Normalize torche attention scores using softmax (equation (3)).
        # print('nodes.mailbox', nodes.mailbox.keys())
        alpha = F.softmax(e, dim=1)

        # Equation (4)
        #   Weigh neighbor nodes to sum later.
        weighted = alpha * z


        ''' 
        Represent features by type so we can calculate semantic-level attention later 
        Assuming data tensor for 3 edges:
            [[1,1,1],
             [2,2,2],
             [3,3,3],
             [4,4,4]]
        with types relatively:
            [[0,0,1],
             [0,1,0],
             [1,0,0],
             [0,0,1]]
        We transform original data tensor to:
            [[ [0,0,0], [0,0,0], [1,1,1] ],
             [ [0,0,0], [2,2,2], [0,0,0] ],
             [ [3,3,3], [0,0,0], [0,0,0] ],
             [ [0,0,0], [0,0,0], [4,4,4] ]]
        So that after multiply with weight and sum, we may end up with:
            [[ [3,3,3], [2,2,2], [5,5,5] ]]
        And pass that tensor to the semantic net
        '''
        weighted_rp = weighted.repeat(1, 1, e_type_dim).view(z.shape[0], z.shape[1], e_type_dim, -1)
        weighted_by_type = e_type_ * weighted_rp



        # Equation (4)
        #   Aggregate neighbor embeddings weighted by torche attention scores (equation(4)).
        h = torch.sum(weighted_by_type, dim=1)

        

        ''' Semantic-level attention '''
        w_phi = self.sem_attn(h)
        w_phi = F.leaky_relu(w_phi)
        beta = F.softmax(w_phi, dim=1)
        Z = torch.sum(beta * h, dim=1)


        ''' Final embedding '''
        Z = self.fc2(Z)

        return {'z': Z}


    def forward(self, h, e, g):
        if g is not None:
            self.g = g

        self.g.ndata['z'] = h
        self.g.edata['e_weighted'] = e

        # self.g.group_apply_edges(func=self.concat_sum_edge_node_ft, group_by="src")

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
        self.g.update_all(self.message_func, self.reduce_func_w_sem)
        # self.g.update_all(self.message_func, self.reduce_func)

        return self.g.ndata.pop('z'), self.g.edata.pop('e_weighted')



class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, node_dim, edge_dim, node_ft_out_dim, edge_ft_out_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, node_dim, edge_dim, node_ft_out_dim, edge_ft_out_dim, out_dim))
        self.merge = merge

    def forward(self, node_features, edge_features, g):
        # print('self.heads~~~', self.heads)

        n_head_outs = []
        e_head_outs = []
        for attn_head in self.heads:
            n_head_out, e_head_out = attn_head(node_features, edge_features, g)
            n_head_outs.append(n_head_out)
            e_head_outs.append(e_head_out)

        if self.merge == 'cat':
            # concat on torche output feature dimension (dim=1)
            return torch.cat(n_head_outs, dim=1), torch.cat(e_head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(n_head_outs)), torch.mean(torch.stack(e_head_outs))
