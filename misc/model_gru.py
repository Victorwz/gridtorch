import torch
from torch import nn
import numpy as np
def truncated_normal_(tensor, mean=0, std=1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def init_trunc_normal(t, size):
    std = 1. / np.sqrt(size)
    return truncated_normal_(t, 0, std)

def rearrange_tf_weights(weights):
    i, j, f, o = weights.chunk(4, 0)
    return torch.cat((i, f, j, o))



def load_tf_param(loc, T=True):
    if T:
        return nn.Parameter(torch.Tensor(np.load(loc).T))
    else:
        return nn.Parameter(torch.Tensor(np.load(loc)))

class GridTorch(nn.Module):

    def __init__(self,
               target_ensembles,
               input_size,
               init_conds_size=268,
               nh_lstm=128,
               nh_bottleneck=256,
               n_pcs = 256,
               n_hdcs = 12,
               dropoutrates_bottleneck=0.5,
               bottleneck_weight_decay=1e-5,
               bottleneck_has_bias=False,
               init_weight_disp=0.0,
               tf_weights_loc=None):

        super(GridTorch, self).__init__()
        self.target_ensembles = target_ensembles

        self.rnn = TFGRUCell(n_inputs=3,
                            n_units=nh_lstm,
                            #batch_first=True
                            )

        self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck,
                                                bias=bottleneck_has_bias)
        self.pc_logits = nn.Linear(nh_bottleneck, target_ensembles[0].n_cells)
        self.hd_logits = nn.Linear(nh_bottleneck, target_ensembles[1].n_cells)

        self.state_embed = nn.Linear(init_conds_size, nh_lstm, bias=True)

        self.dropout = nn.Dropout(dropoutrates_bottleneck)



        self.pc_logits.weight = init_trunc_normal(self.pc_logits.weight, 256)
        self.hd_logits.weight = init_trunc_normal(self.hd_logits.weight, 256)

        if tf_weights_loc:
            self.init_tf_weights(tf_weights_loc)

    @property
    def l2_loss(self,):
        return (self.bottleneck.weight.norm(2) +
                    self.pc_logits.weight.norm(2) +
                    self.hd_logits.weight.norm(2))


    def init_tf_weights(self, loc):

        self.pc_logits.bias = load_tf_param(loc + 'grid_cells_core_pc_logits_b:0.npy')
        self.pc_logits.weight = load_tf_param(loc + 'grid_cells_core_pc_logits_w:0.npy')

        self.hd_logits.bias = load_tf_param(loc + 'grid_cells_core_pc_logits_1_b:0.npy')
        self.hd_logits.weight = load_tf_param(loc + 'grid_cells_core_pc_logits_1_w:0.npy')

        self.bottleneck.weight = load_tf_param(loc + 'grid_cells_core_bottleneck_w:0.npy')

        self.state_embed.bias = load_tf_param(loc + "grid_cell_supervised_state_init_b:0.npy")
        self.state_embed.weight = load_tf_param(loc + "grid_cell_supervised_state_init_w:0.npy")

        self.rnn._Wz = load_tf_param(loc + 'grid_cells_core_gru_wz:0.npy', T=False)
        self.rnn._Wh = load_tf_param(loc + 'grid_cells_core_gru_wh:0.npy', T=False)
        self.rnn._Wr = load_tf_param(loc + 'grid_cells_core_gru_wr:0.npy', T=False)
        self.rnn._Uz = load_tf_param(loc + 'grid_cells_core_gru_uz:0.npy', T=False)
        self.rnn._Uh = load_tf_param(loc + 'grid_cells_core_gru_uh:0.npy', T=False)
        self.rnn._Ur = load_tf_param(loc + 'grid_cells_core_gru_ur:0.npy', T=False)
        self.rnn._bz = load_tf_param(loc + 'grid_cells_core_gru_bz:0.npy', T=False)
        self.rnn._bh = load_tf_param(loc + 'grid_cells_core_gru_bh:0.npy', T=False)
        self.rnn._br = load_tf_param(loc + 'grid_cells_core_gru_br:0.npy', T=False)



    def forward(self, x, initial_conds):
        init = torch.cat(initial_conds, dim=1)

        h_t = self.state_embed(init)

        logits_hd = []
        logits_pc = []
        bottleneck_acts = []
        rnn_states = []
        for t in x: # get rnn output predictions
            h_t = self.rnn(t, h_t)


            bottleneck_activations = self.dropout(self.bottleneck(h_t))

            pc_preds = self.pc_logits(bottleneck_activations)
            hd_preds = self.hd_logits(bottleneck_activations)

            logits_hd += [hd_preds]
            logits_pc += [pc_preds]
            bottleneck_acts += [bottleneck_activations]
            rnn_states += [h_t]

        final_state = h_t
        outs = (torch.stack(logits_hd),
                torch.stack(logits_pc),
                bottleneck_acts,
                rnn_states)
        return outs, final_state



class TFGRUCell(nn.Module):

    def __init__(self, n_inputs=3, n_units=128):
        super(TFGRUCell, self).__init__()
        self.n_units = n_units
        self.n_inputs = n_inputs

        self._Wz = nn.Parameter(torch.Tensor(n_inputs, n_units),requires_grad=True)
        self._Wh = nn.Parameter(torch.Tensor(n_inputs, n_units),requires_grad=True)
        self._Wr = nn.Parameter(torch.Tensor(n_inputs, n_units),requires_grad=True)

        self._Uz = nn.Parameter(torch.Tensor(n_units, n_units),requires_grad=True)
        self._Uh = nn.Parameter(torch.Tensor(n_units, n_units),requires_grad=True)
        self._Ur = nn.Parameter(torch.Tensor(n_units, n_units),requires_grad=True)

        self._bz = nn.Parameter(torch.Tensor(n_units,),requires_grad=True)
        self._bh = nn.Parameter(torch.Tensor(n_units,),requires_grad=True)
        self._br = nn.Parameter(torch.Tensor(n_units,),requires_grad=True)


    def forward(self, x_t, state):
        """Gated recurrent unit (GRU) with nunits cells."""


        z = nn.functional.sigmoid(torch.matmul(x_t, self._Wz) +
                       torch.matmul(state, self._Uz) + self._bz)

        r = nn.functional.sigmoid(torch.matmul(x_t, self._Wr) +
                   torch.matmul(state, self._Ur) + self._br)

        h_twiddle = torch.tanh(torch.matmul(x_t, self._Wh) +
                        torch.matmul(r * state, self._Uh) + self._bh)

        state = (1 - z) * state + z * h_twiddle
        return state
