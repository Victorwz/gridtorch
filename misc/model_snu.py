import torch
from torch import nn
import numpy as np
from all_or_nothing import AllOrNothing


aon = AllOrNothing.apply

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

def load_tf_param(loc):
    return nn.Parameter(torch.Tensor(np.load(loc).T))

class GridTorch(nn.Module):

    def __init__(self,
               target_ensembles,
               input_size,
               init_conds_size=268,
               nh_lstm=256,
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

        self.rnn = RecurrentSNUCell(n_inputs=3,
                            n_units=nh_lstm,
                            #batch_first=True
                            )

        self.pc_logits = nn.Linear(nh_lstm, target_ensembles[0].n_cells)
        self.hd_logits = nn.Linear(nh_lstm, target_ensembles[1].n_cells)

        self.state_embed = nn.Linear(init_conds_size, nh_lstm)
        self.cell_embed = nn.Linear(init_conds_size,  nh_lstm)


        self.dropout = nn.Dropout(dropoutrates_bottleneck)
        print("DROPOUT RATE", dropoutrates_bottleneck)

        self.tanh = torch.nn.Tanh()

        self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck)


        with torch.no_grad():
            self.state_embed.weight = init_trunc_normal(self.state_embed.weight, 128)
            self.cell_embed.weight = init_trunc_normal(self.cell_embed.weight, 128)
            self.bottleneck.weight = init_trunc_normal(self.bottleneck.weight, 256)
            self.pc_logits.weight = init_trunc_normal(self.pc_logits.weight, 256)
            self.hd_logits.weight = init_trunc_normal(self.hd_logits.weight, 12)

            nn.init.zeros_(self.state_embed.bias)
            nn.init.zeros_(self.cell_embed.bias)
            nn.init.zeros_(self.pc_logits.bias)
            nn.init.zeros_(self.hd_logits.bias)

        if tf_weights_loc:
            self.init_tf_weights(tf_weights_loc)

    @property
    def l2_loss(self,):
        return (self.rnn.iw.norm(2) +
                    self.pc_logits.weight.norm(2) +
                    self.hd_logits.weight.norm(2))



    def forward(self, x, initial_conds):

        cuda = next(self.parameters()).is_cuda

        init = torch.cat(initial_conds, dim=1)

        init_state = torch.sigmoid(self.state_embed(init))
        init_cell = torch.sigmoid(self.cell_embed(init))

        h_t, c_t = init_state , init_cell
        logits_hd = []
        logits_pc = []
        bottleneck_acts = []
        rnn_states = []
        cell_states = []

        for t in x: # get rnn output predictions


            h_t, c_t = self.rnn(t, (h_t, c_t), cuda)

            y = self.dropout(h_t)

            pc_preds = self.pc_logits(y)
            hd_preds = self.hd_logits(y)
            logits_hd += [hd_preds]
            logits_pc += [pc_preds]
            bottleneck_acts += [y]
            rnn_states += [h_t]
            cell_states += [c_t]

        outs = (torch.stack(logits_hd),
                torch.stack(logits_pc),
                torch.stack(bottleneck_acts),
                torch.stack(rnn_states),
                torch.stack(cell_states))
        return outs



class SNUCell(nn.Module):

    def __init__(self, n_inputs=3, n_units=128):
        super(SNUCell, self).__init__()
        self.n_units = n_units
        self.n_inputs = n_inputs
        self.iw = nn.Parameter(torch.ones(n_inputs, n_units), requires_grad=True)
        self.iw = nn.init.kaiming_uniform_(self.iw)
        self.b = nn.Parameter(torch.zeros((n_units,)), requires_grad=True)
        self.g = torch.nn.ReLU()
        #self looping weight
        self.l = torch.zeros((n_units,))
        self.l = nn.Parameter(init_trunc_normal(self.l, n_units), requires_grad=True)
        #self.s = torch.zeros(n_units).cuda()
        #self.y = torch.zeros(n_units).cuda()



    def forward(self, x_t, state, cuda=True):
        """
        forward pass of the mode
        """
        #y_t, s_t = self.y, self.s
        (y_t, s_t) = state

        s = self.g(torch.matmul(x_t, self.iw) +
                            torch.mul(torch.mul(self.l, s_t),
                                        (1 - y_t)))


        y = aon(s + self.b)
        #self.y, self.s = y, s
        #print(y.shape)
        return y, s

class RecurrentSNUCell(SNUCell):
    """
    Recurrent form of SNU
    """

    def forward(self, x_t, state, cuda=True):
        """
        forward pass of the mode
        """
        #y_t, s_t = self.y, self.s
        (y_t, s_t) = state
        s = self.g(torch.matmul(x_t, self.iw) + y_t +
                            torch.mul(torch.mul(self.l, s_t),
                                        (1 - y_t)))


        y = aon(s + self.b)
        #self.y, self.s = y, s
        #print(y.shape)
        return y, s

class EISNUCell(SNUCell):
    """
    Recurrent form of SNU
    """

    def __init__(self, n_inputs=3, n_units=256, ei_split=0.5):
        super(SNUCell, self).__init__()

        self.n_exc = int(ei_split * n_units)
        print(self.n_exc)
        self.n_inh = int((1 - ei_split) * n_units)

        self.n_inputs = n_inputs
        self.iw = nn.Parameter(torch.ones(n_inputs, n_units), requires_grad=True)
        self.eiw = nn.Parameter(torch.ones(self.n_exc, self.n_inh), requires_grad=True)
        self.iw = nn.init.kaiming_uniform_(self.iw)
        self.eiw = nn.init.kaiming_uniform_(self.eiw)

        self.b = nn.Parameter(torch.zeros((n_units,)), requires_grad=True)
        self.g = nn.ReLU()
        #self looping weight
        self.l = torch.zeros((n_units,))
        self.l = nn.Parameter(init_trunc_normal(self.l, n_units), requires_grad=True)




    def forward(self, x_t, exc_state, inh_sate, cuda=True):
        """
        forward pass of the mode
        """
        #y_t, s_t = self.y, self.s
        (y_exc_t, s_exc_t) = exc_state
        (y_inh_t, s_inh_t) = inh_sate



        eiw = self.eiw

        w_exc, w_inh = self.iw[:, :self.n_exc], self.iw[:, self.n_exc:]
        b_exc, b_inh = self.b[:self.n_exc], self.b[self.n_exc:]
        l_exc, l_inh = self.l[:self.n_exc], self.l[self.n_exc:]


        s_exc = self.g(torch.matmul(x_t, w_exc) + y_exc_t -
                        torch.matmul(y_inh_t, -eiw) +
                            torch.mul(torch.mul(l_exc, s_exc_t),
                                        (1 - y_exc_t)))


        y_exc = aon(s_exc + b_exc)


        s_inh = self.g(torch.matmul(x_t, w_inh) + y_inh_t -
                        torch.matmul(y_exc_t, eiw.transpose(1,0)) +
                            torch.mul(torch.mul(l_inh, s_inh_t),
                                        (1 - y_inh_t)))


        y_inh = aon(s_inh + b_inh)
        #self.y, self.s = y, s
        #print(y.shape)
        return y_exc, s_exc, y_inh, s_inh
