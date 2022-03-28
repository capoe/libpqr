import math

import torch
import torch.nn.functional as F

from typing import Optional

from torch import Tensor
from torch.nn import Linear, Parameter, GRUCell
from torch_scatter import scatter_softmax, scatter_sum

from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import (
    MessagePassing, 
    GATConv, 
    global_add_pool, 
    global_max_pool
)

from ..aux import tensor_info, Timer
from ..common import Featurizer


class LexFeaturizer(Featurizer):
    def __init__(self):
        Featurizer.__init__(self, connect_full=False, ignore_missing=True)


class LexData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "center_index":
            return self.x.shape[0]
        elif key == "env_node_index":
            return self.x.shape[0]
        elif key == "env_center_index":
            return self.env_node_index.shape[0]
        elif key == "env_hyperedge_index":
            return self.env_node_index.shape[0]
        elif key == "lig_center_index":
            return self.lig_x.shape[0]
        elif key == "lig_edge_index":
            return self.lig_x.shape[0]
        elif key == "motif_edge_index":
            return self.motif_x.shape[0]
        elif key == "motif_vectors":
            return self.motif_x.shape[0]
        elif key == "motif_index":
            return self.motif_vectors.shape[0]
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "center_index":
            return 0
        elif key == "env_node_index":
            return 0
        elif key == "env_center_index":
            return 0
        elif key == "env_hyperedge_index":
            return 1
        elif key == "lig_center_index":
            return 0
        elif key == "lig_edge_index":
            return 1
        elif key == "motif_edge_index":
            return 1
        elif key == "motif_vectors":
            return 0
        elif key == "motif_index":
            return 0
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


def lex_dot(x, y):
    return ((x*y).sum(dim=-1)/torch.sqrt(torch.tensor(x.shape[1]))).sigmoid_()


def lex_dot_dense(x, y):
    return (torch.mm(x, y.T)/torch.sqrt(torch.tensor(x.shape[1]))).sigmoid_()


def featurize_hyperedges(
        hyperedge_attr, 
        device,
        r_sigma,
        r_min,
        r_max,
        r_centers,
        r_eps,
        w_width,
        w_r_max_intra,
        w_r_max_inter,
        w_pivot,
        w_decay,
        w_const
):
        rij = hyperedge_attr[:,0]
        rik = hyperedge_attr[:,1]
        rjk = hyperedge_attr[:,2]

        # Cutoff
        wrm = torch.tensor([w_r_max_inter, w_r_max_intra], device=device)
        wrmij = wrm[hyperedge_attr[:,3].type(torch.long)]
        wrmik = wrm[hyperedge_attr[:,4].type(torch.long)]
        wij = 0.5*(1 + torch.cos(math.pi*(rij - wrmij + w_width) / w_width))
        wik = 0.5*(1 + torch.cos(math.pi*(rik - wrmik + w_width) / w_width))
        wij0 = torch.heaviside(wrmij - w_width - rij, torch.tensor(0., device=device))
        wik0 = torch.heaviside(wrmik - w_width - rik, torch.tensor(0., device=device))
        wij = torch.maximum(wij, wij0)
        wik = torch.maximum(wik, wik0)
        wijk = wij*wik

        # Distance discounting
        rij2 = rij**2 + r_eps
        rik2 = rik**2 + r_eps
        tij = torch.sigmoid(-w_decay*(rij2-w_pivot**2))
        tik = torch.sigmoid(-w_decay*(rik2-w_pivot**2))
        wij = (1. - tij)/rij2 + tij*w_const
        wik = (1. - tik)/rik2 + tik*w_const
        wijk = wijk*wij*wik

        # Line expansions
        r_alpha = 1./(2*r_sigma**2)
        r = torch.linspace(r_min, r_max, r_centers, device=device, dtype=torch.float).view(-1,1)

        cosi = (rij**2 + rik**2 - rjk**2)/(2*rij*rik + r_eps)
        cosj = (rij**2 + rjk**2 - rik**2)/(2*rij*rjk + r_eps)
        cosk = (rik**2 + rjk**2 - rij**2)/(2*rik*rjk + r_eps)

        sini = torch.sqrt((1. - cosi**2).relu_())
        sinj = torch.sqrt((1. - cosj**2).relu_())
        sink = torch.sqrt((1. - cosk**2).relu_())

        gij = torch.exp(-r_alpha*(rij - r)**2).transpose(0,1)
        gik = torch.exp(-r_alpha*(rik - r)**2).transpose(0,1)
        gjk = torch.exp(-r_alpha*(rjk - r)**2).transpose(0,1)

        fii = hyperedge_attr[:,3].view(-1,1)

        hyperedge_attr_out = torch.cat([ 
                fii, gij, gik, gjk, 
                rij.view(-1,1), rik.view(-1,1), rjk.view(-1,1),
                cosi.view(-1,1), cosj.view(-1,1), cosk.view(-1,1),
                sini.view(-1,1), sinj.view(-1,1), sink.view(-1,1)
            ], dim=-1)

        return hyperedge_attr_out, wijk


class LexComponents(torch.nn.Module):
    def __init__(self, 
            feat,
            settings,
            hidden_channels=256,
            depth_2d=4,
            dropout=0.
    ):
        torch.nn.Module.__init__(self)
        self.settings = settings
        self.encoder_2d = Encoder2d(
            in_channels=feat.dimNode(),
            hidden_channels=hidden_channels,
            edge_dim=feat.dimEdge(),
            depth=depth_2d,
            dropout=dropout
        )
        self.lex_2d = Lex2d(dim=hidden_channels)
        self.lex_3d = Lex3d(dim_edge=37) # TODO Read dim from settings

    def reset_parameters(self):
        self.encoder_2d.reset_parameters()
        self.lex_2d.reset_parameters()
        self.lex_3d.reset_parameters()

    def num_parameters(self):
        return (
            sum([ p.data.numel() for p in self.parameters() ]),
            self.encoder_2d.num_parameters(),
            self.lex_2d.num_parameters(),
            self.lex_3d.num_parameters()
        )

    def forward_complex(self, data, device, timer=None):
        env_hyperedge_attr, env_hyperedge_weight = featurize_hyperedges(
            data.env_hyperedge_attr, 
            device=device, 
            r_sigma=1.,
            r_min=0., 
            r_max=self.settings.cut_hyper_inter, 
            r_centers=9, #int(self.settings.cut_hyper_inter+0.5)+1, # TODO
            r_eps=1e-8,
            w_width=self.settings.cut_hyper_width,
            w_r_max_intra=self.settings.cut_hyper_intra,
            w_r_max_inter=self.settings.cut_hyper_inter,
            w_pivot=self.settings.radial_weight_pivot,
            w_decay=self.settings.radial_weight_decay,
            w_const=self.settings.radial_weight_const
        )
        x_2d = self.encoder_2d(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
        )
        x_2d = x_2d[data.env_node_index]
        x_3d = self.lex_3d(
            x=x_2d,
            hyperedge_index=data.env_hyperedge_index,
            hyperedge_attr=env_hyperedge_attr,
            hyperedge_weight=env_hyperedge_weight
        )
        x_3d = x_3d[data.env_center_index]
        return x_3d

    def forward_motif(self, data, device, timer=None):
        x_2d_global = self.encoder_2d(
            x=data.motif_x,
            edge_index=data.motif_edge_index,
            edge_attr=data.motif_edge_attr
        )
        x_2d = x_2d_global[data.motif_vectors]
        x_2d = self.lex_2d(x_2d, x_2d_global, data.motif_index)
        return x_2d


class Encoder2d(torch.nn.Module):
    def __init__(self,
            in_channels: int,
            hidden_channels: int,
            edge_dim: int,
            depth: int,
            dropout: float=0.0
    ):
        torch.nn.Module.__init__(self)
        self.dim_in = in_channels
        self.dim_h = hidden_channels
        self.dim_e = edge_dim
        self.dropout = dropout
        self.depth = depth
        # Atom convs and grus
        self.lin_enter = Linear(self.dim_in, self.dim_h)
        self.atom_convs = torch.nn.ModuleList([
            Edge2dConv(self.dim_h, self.dim_h, self.dim_e, self.dropout) 
        ])
        self.atom_grus = torch.nn.ModuleList([
            GRUCell(self.dim_h, self.dim_h) 
        ])
        for _ in range(self.depth - 1):
            self.atom_convs.append(
                GATConv(
                    self.dim_h, 
                    self.dim_h, 
                    dropout=self.dropout, 
                    add_self_loops=False, 
                    negative_slope=0.01) 
            )
            self.atom_grus.append(
                GRUCell(
                    hidden_channels, 
                    hidden_channels) 
            )

    def reset_parameters(self):
        self.lin_enter.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()

    def num_parameters(self):
        return sum([ p.data.numel() for p in self.parameters() ])

    def forward(self, 
            x, 
            edge_index, 
            edge_attr, 
    ):
        x = F.leaky_relu_(self.lin_enter(x))
        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.atom_grus[0](h, x).relu_()
        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()
        x = F.layer_norm(x, (x.shape[-1],))
        return x


class Edge2dConv(MessagePassing):
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            edge_dim: int,
            dropout: float = 0.0):
        super(Edge2dConv, self).__init__(aggr='add', node_dim=0)
        self.dropout = dropout
        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))
        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out += self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class Lex2d(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.trans_local = TransferBlock(dim, dim)
        self.trans_global = TransferBlock(dim, dim)
        self.reduce_global = ReduceAtomic(dim)
        self.lin_exit = Linear(dim, dim, False)

    def reset_parameters(self):
        self.trans_local.reset_parameters()
        self.trans_global.reset_parameters()
        self.reduce_global.reset_parameters()
        self.lin_exit.reset_parameters()

    def num_parameters(self):
        return sum([ p.data.numel() for p in self.parameters() ])

    def forward(self, x_centers, x_global, batch_global):
        x_centers = self.trans_local(x_centers)
        x_global = self.trans_global(x_global)
        x_global_red = self.reduce_global(x_global, batch_global)
        x_out = self.lin_exit(x_centers + x_global_red)
        return x_out


class Lex3d(torch.nn.Module):
    def __init__(self, dim_edge):
        torch.nn.Module.__init__(self)
        self.dim_in = 256
        self.dim_h = 256
        self.dim_tri = 128
        self.dim_edge = dim_edge

        self.outward_pass = TriangularAttention(
            dim_in=self.dim_h, 
            dim_h=self.dim_tri,
            dim_out=self.dim_h, 
            dim_edge=self.dim_edge
        )

        self.inward_pass = TriangularAttention(
            dim_in=self.dim_h, 
            dim_h=self.dim_tri, 
            dim_out=self.dim_h, 
            dim_edge=self.dim_edge
        )

        self.trans0 = TransferBlock()
        self.trans1 = TransferBlock()
        self.lin_exit = Linear(self.dim_h, self.dim_h, False)

    def reset_parameters(self):
        self.outward_pass.reset_parameters()
        self.inward_pass.reset_parameters()
        self.trans0.reset_parameters()
        self.trans1.reset_parameters()
        self.lin_exit.reset_parameters()

    def num_parameters(self):
        return sum([ p.data.numel() for p in self.parameters() ])

    def forward(self,
            x,
            hyperedge_index,
            hyperedge_attr,
            hyperedge_weight,
            verbose=False
    ):
        if verbose:
            tensor_info(x, "lex3d x_in")
            if hyperedge_attr.shape[1] > 0:
                tensor_info(hyperedge_attr, "lex3d ha")
        x = self.trans0(x)
        x = self.outward_pass(
            x=x,
            hyperedge_index=hyperedge_index,
            hyperedge_attr=hyperedge_attr,
            hyperedge_weight=hyperedge_weight,
            update_vertex=2,
            verbose=verbose
        )
        x = self.inward_pass(
            x=x,
            hyperedge_index=hyperedge_index,
            hyperedge_attr=hyperedge_attr,
            hyperedge_weight=hyperedge_weight,
            update_vertex=0,
            verbose=verbose
        )
        x = self.trans1(x)
        x = self.lin_exit(x)
        if verbose:
            tensor_info(x, "lex3d x_out")
        return x


class TriangularAttention(torch.nn.Module):
    def __init__(self,
            dim_in,
            dim_h,
            dim_out,
            dim_edge
    ):
        torch.nn.Module.__init__(self)
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.dim_edge = dim_edge

        self.lin_i = Linear(self.dim_in, self.dim_h)
        self.lin_j = Linear(self.dim_in, self.dim_h)
        self.lin_k = Linear(self.dim_in, self.dim_h)
        self.lin_e = Linear(self.dim_edge, self.dim_h)

        self.conv = Edge3dConv(
            in_channels_att_l=self.dim_h, 
            in_channels_att_r=self.dim_in,
            out_channels_att=self.dim_h,
            in_channels=self.dim_h,
            out_channels=self.dim_out
        )

    def reset_parameters(self):
        self.lin_i.reset_parameters()
        self.lin_j.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_e.reset_parameters()
        self.conv.reset_parameters()

    def forward(self,
            x,
            hyperedge_index,
            hyperedge_attr,
            hyperedge_weight,
            update_vertex, # 0 = i or 2 = k
            verbose=False
    ):
        # Embed triplets
        xi = self.lin_i(x)
        xj = self.lin_j(x)
        xk = self.lin_k(x)
        xi = xi[hyperedge_index[0]]
        xj = xj[hyperedge_index[1]]
        xk = xk[hyperedge_index[2]]
        eijk = self.lin_e(hyperedge_attr)
        xijk = F.elu_(xi + xj + xk + eijk) 

        # Triplet-to-node message passing
        hyperedge_index_src = torch.arange(0, hyperedge_index.shape[1], device=x.device)
        update_index = torch.stack([ hyperedge_index_src, hyperedge_index[update_vertex] ])
        out = self.conv(xl=xijk, xr=x, edge_index=update_index, edge_weight=hyperedge_weight)

        if verbose:
            tensor_info(x, "x")
            tensor_info(out_k, "out_k")
            tensor_info(out_i, "out_i")

        x = F.elu_(x + out)
        return x


class Edge3dConv(MessagePassing):
    def __init__(self, 
            in_channels_att_l: int, 
            in_channels_att_r: int,
            out_channels_att: int,
            in_channels: int,
            out_channels: int, 
            dropout: float = 0.0
    ):
        super().__init__(aggr='add', node_dim=0)
        self.dropout = dropout
        self.lin1_l = Linear(in_channels_att_l, out_channels_att, False)
        self.lin1_r = Linear(in_channels_att_r, out_channels_att, False)
        self.att = Parameter(torch.Tensor(1, out_channels_att))
        self.lin2 = Linear(in_channels, out_channels, False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin1_l.weight)
        glorot(self.lin1_r.weight)
        glorot(self.att)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, xl, xr, edge_index, edge_weight):
        out = self.propagate(edge_index, x=(xl, xr), edge_weight=edge_weight)
        out += self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_weight: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]
    ) -> Tensor:
        # Conditioning
        x_ji = F.leaky_relu_(self.lin1_l(x_j) + self.lin1_r(x_i))
        alpha = (x_ji * self.att).sum(dim=-1)
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        # Apply weight prior
        alpha = alpha*edge_weight
        sums = scatter_sum(alpha, index, dim_size=size_i)
        alpha = alpha/(sums[index] + 1e-5) # TODO Define epsilon
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # Weight messages
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class TransferBlock(torch.nn.Module):
    def __init__(self, dim_in=256, dim_out=256, bias=False):
        super().__init__()
        self.lin1 = Linear(dim_in, dim_out, bias)
        self.lin2 = Linear(dim_out, dim_out, bias)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        res = x
        out = self.lin1(x)
        out = F.elu_(out)
        out = F.layer_norm(out, (out.shape[-1],))
        out = self.lin2(out)
        out = out + res
        out = F.elu_(out)
        return out


class ReduceAtomic(torch.nn.Module):
    def __init__(self,
            dim: int=256,
            depth: int=2,
            dropout: float=0.0
    ):
        super().__init__()
        self.dim = dim
        self.dropout = dropout

        self.red_conv_loc_to_glob = ReduceConv(
            dim, dim, dim, dim, dropout=dropout
        )

    def reset_parameters(self):
        self.red_conv_loc_to_glob.reset_parameters()
    
    def forward(self, x, batch):
        edge_index_loc_to_glob = torch.stack([
            torch.arange(batch.size(0), device=batch.device),
            batch
        ])

        xm = global_add_pool(x, batch).relu_()
        xm_n = self.red_conv_loc_to_glob(
            xl=x, xr=xm, edge_index=edge_index_loc_to_glob
        )
        xm = F.elu_(xm + xm_n)
        xm = F.layer_norm(xm, (xm.shape[-1],))
        return xm


class ReduceConv(MessagePassing):
    def __init__(self, 
            in_channels_att_l: int=256, 
            in_channels_att_r: int=256,
            out_channels_att: int=256,
            out_channels: int=256, 
            dropout: float = 0.0
    ):
        super().__init__(aggr='add', node_dim=0)
        self.dropout = dropout
        self.lin1_l = Linear(in_channels_att_l, out_channels_att, False)
        self.lin1_r = Linear(in_channels_att_r, out_channels_att, False)
        self.att = Parameter(torch.Tensor(1, out_channels_att))
        self.lin2 = Linear(out_channels, out_channels, False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin1_l.weight)
        glorot(self.lin1_r.weight)
        glorot(self.att)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, xl, xr, edge_index):
        out = self.propagate(edge_index, x=(xl, xr))
        out += self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, 
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]
    ) -> Tensor:
        x_ji = F.leaky_relu_(self.lin1_l(x_j) + self.lin1_r(x_i)) # Maybe move to .forward for efficiency
        alpha = (x_ji * self.att).sum(dim=-1)
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1) # Note that lin2 is applied to original x_j. Change?


