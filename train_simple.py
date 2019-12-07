import numpy as np
import utils

import model_snu as model
import model_lstm as model_lstm
import ensembles
from dataloading import Dataset
from model_utils import get_latest_model_file, get_model_epoch

import torch
from torch.utils import data
from torch import nn

N_EPOCHS = 1000
STEPS_PER_EPOCH = 150
ENV_SIZE = 2.2
BATCH_SIZE = 200
GRAD_CLIPPING = 0.01
SEED = 9101
N_PC = [256]
N_HDC = [12]
BOTTLENECK_DROPOUT = 0.5
WEIGHT_DECAY = 1e-5
LR = 0.01
MOMENTUM = 0.9
TIME = 50
PAUSE_TIME = None
SAVE_LOC = 'experiments/snu_r/'


torch.manual_seed(SEED)
np.random.seed(SEED)


# CUDA for PyTorch
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("USING DEVICE:", device)
# Parameters
data_params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6, # num cpus,
          }

test_params = {'batch_size': 100,
          'shuffle': True,
          'num_workers': 2, # num cpus,
          }

dataset = Dataset(batch_size=data_params['batch_size'])
data_generator = data.DataLoader(dataset, **data_params)
test_generator = data.DataLoader(dataset, **test_params)

# Create the ensembles that provide targets during training
place_cell_ensembles = utils.get_place_cell_ensembles(
        env_size=ENV_SIZE,
        neurons_seed=SEED,
        targets_type='softmax',
        lstm_init_type='softmax',
        n_pc=N_PC,
        pc_scale=[0.01])

head_direction_ensembles = utils.get_head_direction_ensembles(
        neurons_seed=SEED,
        targets_type='softmax',
        lstm_init_type='softmax',
        n_hdc=N_HDC,
        hdc_concentration=[20.])

target_ensembles = place_cell_ensembles + head_direction_ensembles





model = model.GridTorch(target_ensembles, (BATCH_SIZE, 100, 3)).cuda()
params = model.parameters()

model_lstm = model_lstm.GridTorch(target_ensembles, (BATCH_SIZE, 100, 3), tf_weights_loc='weights/').cuda()



saved_model_file = get_latest_model_file(SAVE_LOC)
start_epoch = 0

if saved_model_file:
    state_dict = torch.load(saved_model_file)
    if use_cuda:
        for k, v in state_dict.items():
            state_dict[k] = v.cuda()
    model.load_state_dict(state_dict)
    start_epoch = get_model_epoch(saved_model_file)
    print("RESTORING MODEL AT:", saved_model_file)
    print("STARTING AT EPOCH:", start_epoch)



#loss ops:
logsoftmax = nn.MSELoss()
def cross_entropy(pred, soft_targets):
    a = torch.sigmoid(pred)
    b = torch.sigmoid(soft_targets)
    return logsoftmax(a, b)

#Optimisation opts
optimiser = torch.optim.Adam(params,
                                    lr=LR,
                                    #momentum=MOMENTUM,
                                    #alpha=0.9,
                                    #eps=1e-10
                                    )


to_cuda = lambda x:x.cuda()

def encode_inputs(X, y, place_cell_ensembles, head_direction_ensembles, cuda=True):
    init_pos , init_hd, ego_vel = X
    target_pos, target_hd = y

    inputs = ego_vel.transpose(1,0)
    initial_conds = utils.encode_initial_conditions(init_pos ,
                                                    init_hd,
                                                    place_cell_ensembles,
                                                    head_direction_ensembles)

    ensembles_targets = utils.encode_targets(target_pos,
                                            target_hd,
                                            place_cell_ensembles,
                                            head_direction_ensembles)

    if cuda:

        init_pos = init_pos.cuda()
        init_hd = init_hd.cuda()
        inputs = inputs.cuda()
        target_pos = target_pos.cuda()
        target_hd = target_hd.cuda()
        initial_conds = tuple(map(to_cuda, initial_conds))

    return init_pos, init_hd, inputs, target_pos, target_hd, initial_conds, ensembles_targets

def decode_outputs(outs, ensembles_targets, cuda=True):
        if cuda:
            pc_targets = ensembles_targets[0].cuda()
            hd_targets = ensembles_targets[1].cuda()
        else:
            pc_targets = ensembles_targets[0]
            hd_targets = ensembles_targets[1]

        logits_hd, logits_pc, bottleneck_acts, lstm_states, _ = outs
        pc_targets, hd_targets = (pc_targets.transpose(1,0),
                                    hd_targets.transpose(1,0))


        logits_pc = logits_pc.view(-1, N_PC[0])
        logits_hd = logits_hd.view(-1, N_HDC[0])

        pc_targets = pc_targets.contiguous().view(-1, N_PC[0])
        hd_targets = hd_targets.contiguous().view(-1, N_HDC[0])

        return bottleneck_acts, logits_pc, logits_hd, pc_targets, hd_targets

def get_loss(logits_pc, logits_hd, pc_targets, hd_targets, bottleneck_acts):
    pc_loss = cross_entropy(logits_pc,pc_targets)
    hd_loss = cross_entropy(logits_hd, hd_targets)

    return torch.mean(pc_loss + hd_loss)


def eval_loop():
    pass

if __name__ == '__main__':
    torch.save(target_ensembles, SAVE_LOC + 'target_ensembles.pt')
    torch.save(model.state_dict(), SAVE_LOC + 'model_epoch_0.pt')
    for e in range(start_epoch, N_EPOCHS):


        model.eval()
        step = 0
        losses = []
        for X, y in data_generator:

            optimiser.zero_grad()

            (init_pos,
            init_hd,
            inputs,
            target_pos,
            target_hd,
            initial_conds,
            ensembles_targets) = encode_inputs(X, y, place_cell_ensembles, head_direction_ensembles)


            outs  = model.forward(inputs, initial_conds)
            outs_lstm  = model_lstm.forward(inputs, initial_conds)

            bottleneck_acts, logits_pc, logits_hd, pc_targets, hd_targets = decode_outputs(outs, ensembles_targets)
            lstm_acts, _, _, _, _ = decode_outputs(outs_lstm, ensembles_targets)



            loss = cross_entropy(lstm_acts, bottleneck_acts)
            loss.backward()
            torch.nn.utils.clip_grad_value_(params, GRAD_CLIPPING)
            optimiser.step()
            losses.append(loss.clone().item())
            if step > STEPS_PER_EPOCH:
                break

            step += 1
        print("EPOCH", e, 'LOSS :', torch.mean(torch.Tensor(losses)))
