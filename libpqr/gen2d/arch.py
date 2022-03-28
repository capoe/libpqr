import torch
import torch.nn.functional as F

from typing import Optional

from torch import Tensor
from torch.nn import Linear, Parameter, GRUCell
from torch_scatter import scatter_softmax
from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv, MessagePassing 
from torch_geometric.nn.inits import glorot, zeros

from ..common import Featurizer


class VlbData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "pair_link_index":
            return self.x.shape[0]
        elif key == "atom_vector":
            return self.x.shape[0]
        else:
            return super().__inc__(key, value, *args, **kwargs)
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "pair_link_index":
            return 1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class VlbFeaturizer(Featurizer):
    def __init__(self):
        super().__init__(connect_full=False, ignore_missing=True)


class VlbComponents(torch.nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.vector_model = VectorModel(feat.dimNode(), feat.dimEdge())
        self.linker_model = LinkerModel(feat.dimNode(), feat.dimEdge())
        self.bonding_model = BondingModel(feat.dimNode(), feat.dimEdge())

    def reset_parameters(self):
        self.vector_model.reset_parameters()
        self.linker_model.reset_parameters()
        self.bonding_model.reset_parameters()

    def prepare_recalibrate(self):
        self.linker_model.prepare_recalibrate()


class VectorModel(torch.nn.Module):
    def __init__(self,
            in_channels: int,
            edge_dim: int,
            hidden_channels: int=256,
            depth: int=4,
            out_channels: int=1,
            dropout: float=0.0
    ):
        torch.nn.Module.__init__(self)
        self.dim_in = in_channels
        self.dim_h = hidden_channels
        self.dim_e = edge_dim
        self.dim_out = out_channels
        self.dropout = dropout
        self.depth = depth

        self.lin_enter = Linear(self.dim_in, self.dim_h)
        self.lin_exit = Linear(self.dim_h, self.dim_out)

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
        self.lin_exit.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()

    def num_parameters(self):
        return sum([ p.data.numel() for p in self.parameters() ])

    def forward(self, 
            x, 
            edge_index, 
            edge_attr, 
            batch
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
        x = self.lin_exit(x)
        x = x.sigmoid_()
        return x


class LinkerModel(torch.nn.Module):
    def __init__(self,
            in_channels: int,
            edge_dim: int,
            hidden_channels: int=256,
            out_channels: int=256,
            depth: int=4,
            dropout: float=0.0
    ):
        torch.nn.Module.__init__(self)
        self.dim_in = in_channels
        self.dim_h = hidden_channels
        self.dim_e = edge_dim
        self.dim_out = out_channels
        self.dropout = dropout
        self.depth = depth

        self.lin_enter = Linear(self.dim_in, self.dim_h)
        self.lin_post_a = torch.nn.ModuleList([
                Linear(self.dim_h, self.dim_h),
                Linear(self.dim_h, self.dim_h),
                Linear(self.dim_h, self.dim_out)
            ])
        self.lin_post_b = torch.nn.ModuleList([
                Linear(self.dim_h, self.dim_h),
                Linear(self.dim_h, self.dim_h),
                Linear(self.dim_h, self.dim_out)
            ])

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
        for lin in self.lin_post_a:
            lin.reset_parameters()
        for lin in self.lin_post_b:
            lin.reset_parameters()

    def prepare_recalibrate(self):
        for trafo in [
            self.lin_enter,
            self.atom_convs[:-1],
            self.atom_grus[:-1]
        ]:
            for par in trafo.parameters():
                par.requires_grad = False

    def num_parameters(self):
        return sum([ p.data.numel() for p in self.parameters() ])

    def forward(self, 
            x, 
            edge_index, 
            edge_attr, 
            batch):
        x = F.leaky_relu_(self.lin_enter(x))
        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.atom_grus[0](h, x).relu_()
        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()
        x = F.layer_norm(x, (x.shape[-1],))
        xa = x
        xb = x
        for post, (lin_a, lin_b) in enumerate(
                zip(self.lin_post_a, self.lin_post_b)
        ):
            xa = lin_a(xa)
            xb = lin_b(xb)
            if post < len(self.lin_post_a) - 1:
                xa = F.softplus(xa)
                xb = F.softplus(xb)
        return xa, xb

    def final(self,
            xai,
            xbi,
            xaj,
            xbj):
        yij = (1./xai.shape[1]**0.5)*0.5*((xai*xbj).sum(dim=-1) + (xbi*xaj).sum(dim=-1))
        yij = yij.sigmoid_().view((-1,1))
        return yij

    def final_dense(self, xai, xbi, xaj, xbj):
        yij = (1./xai.shape[1]**0.5)*0.5*(torch.mm(xai, xbj.T) + torch.mm(xbi, xaj.T))
        yij = yij.sigmoid_()
        return yij


class BondingModel(torch.nn.Module):
    def __init__(self,
            in_channels: int,
            edge_dim: int,
            hidden_channels: int=256,
            depth: int=4,
            out_channels: int=3,
            dropout: float=0.0
    ):
        torch.nn.Module.__init__(self)
        self.dim_in = in_channels
        self.dim_h = hidden_channels
        self.dim_e = edge_dim
        self.dim_out = out_channels
        self.dropout = dropout
        self.depth = depth

        self.lin_enter = Linear(self.dim_in, self.dim_h)
        self.lin_post = torch.nn.ModuleList([
                Linear(self.dim_h, self.dim_h),
                Linear(self.dim_h, self.dim_h)
            ])
        self.lin_final_enter = Linear(2*self.dim_h, self.dim_h)
        self.lin_final = torch.nn.ModuleList([
                Linear(self.dim_h, self.dim_out) 
            ])

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
        for lin in self.lin_post:
            lin.reset_parameters()
        self.lin_final_enter.reset_parameters()
        for lin in self.lin_final:
            lin.reset_parameters()

    def num_parameters(self):
        return sum([ p.data.numel() for p in self.parameters() ])

    def forward(self, 
            x, 
            edge_index, 
            edge_attr, 
            batch):
        x = F.leaky_relu_(self.lin_enter(x))
        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.atom_grus[0](h, x).relu_()
        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()
        x = F.layer_norm(x, (x.shape[-1],))
        for post, lin in enumerate(self.lin_post):
            x = lin(x)
            if post < len(self.lin_post) - 1:
                x = F.softplus(x)
        return x

    def final(self, xa, xb):
        xab = torch.cat([xa, xb], dim=1)
        xba = torch.cat([xb, xa], dim=1)
        xab = self.lin_final_enter(xab)
        xba = self.lin_final_enter(xba)
        xab = F.elu_(xab)
        xba = F.elu_(xba)
        xsym = F.layer_norm(xab + xba, (xab.shape[-1],))
        for post, lin in enumerate(self.lin_final):
            xsym = lin(xsym)
            if post < len(self.lin_post) - 1:
                xsym = F.softplus(xsym)
        xsym = F.softmax(xsym, dim=-1)
        return xsym


class Edge2dConv(MessagePassing):
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            edge_dim: int,
            dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)
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
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1))) # this concat is unnecessary
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


