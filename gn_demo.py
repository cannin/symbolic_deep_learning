# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="QTA_EZGpYwlx" colab_type="text"
# This notebook demonstrates our graph network inductive bias with a symbolic model extraction.

# %% [markdown] id="lEKUuuYfTjP5" colab_type="text"
# # Preamble and data generation

# %% [markdown] id="IXpsg0LUb2sT" colab_type="text"
# ## Make sure to turn on the GPU via Edit-> Notebook settings.

# %% id="9imqRxveZl-J" colab_type="code" colab={}
#Basic pre-reqs:
import numpy as np
import os
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from pathlib import Path
# %matplotlib inline

PLOTS_DIR = Path("plots")
try:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    print("Warning: Could not create plots directory; saving plots to current directory instead.")
    PLOTS_DIR = Path(".")


def save_plot(fig, basename, close=True, dpi=200):
    """Save figure to the plots directory with a sanitized filename."""
    safe_name = basename.replace(' ', '_').replace('/', '_')
    path = PLOTS_DIR / f"{safe_name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if close:
        plt.close(fig)
    print(f"Saved plot to {path}")


print("9imqRxveZl-J done")

# %% [markdown] id="9RTQhSJDbfLZ" colab_type="text"
# ## Download pre-reqs, and then code for simulations and model files:
#
# (Note: installing torch-geometric may take a long time.)

# %% id="7XHuxuzNg0Ko" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 173} outputId="844f96c1-520b-40a0-d6b7-8930ed7eecae"
# !pip install celluloid

print("7XHuxuzNg0Ko done")

# %% id="DQy02pANbx2e" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 836} outputId="c81fa951-cb3a-44f1-bda6-de617363c013"
version_nums = torch.__version__.split('.')
# Torch Geometric seems to always build for *.*.0 of torch :
version_nums[-1] = '0' + version_nums[-1][1:]
os.environ['TORCH'] = '.'.join(version_nums)

# !pip install --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}.html && pip install --upgrade torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}.html && pip install --upgrade torch-geometric
print("DQy02pANbx2e done")

# %% id="MbF0-MSZbhUI" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 391} outputId="9737287f-6fb0-4a24-be86-45b1000ea86c"
# !wget https://raw.githubusercontent.com/MilesCranmer/symbolic_deep_learning/master/models.py -O models.py
# !wget https://raw.githubusercontent.com/MilesCranmer/symbolic_deep_learning/master/simulate.py -O simulate.py

print("MbF0-MSZbhUI done")

# %% id="hVwSWJM-bl6Q" colab_type="code" colab={}
import models
import simulate

print("hVwSWJM-bl6Q done")

# %% [markdown] id="kdgdkJwFZl-V" colab_type="text"
# ## Assert we have a GPU:

# %% id="krahINlfZl-W" colab_type="code" colab={"base_uri": "https://localhost:8080/"} outputId="33d6e7e1-8fb5-47ef-c2fc-fc84316ffaff"
torch.ones(1).cuda()

print("krahINlfZl-W done")

# %% [markdown] id="Zw9W7SB-Zl-i" colab_type="text"
# ## Create the simulation:

# %% id="v8mC0I_lZl-j" colab_type="code" colab={"base_uri": "https://localhost:8080/"} outputId="dd430230-f1a9-4539-e612-467ad34f4e45"
# Number of simulations to run (it's fast, don't worry):
ns = 10000
# Potential (see below for options)
sim = 'spring'
# Number of nodes
n = 4
# Dimension
dim = 2
# Number of time steps
nt = 1000


#Standard simulation sets:
n_set = [4, 8]
sim_sets = [
 {'sim': 'r1', 'dt': [5e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
 {'sim': 'r2', 'dt': [1e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
 {'sim': 'spring', 'dt': [1e-2], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
 {'sim': 'string', 'dt': [1e-2], 'nt': [1000], 'n': [30], 'dim': [2]},
 {'sim': 'charge', 'dt': [1e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
 {'sim': 'superposition', 'dt': [1e-3], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
 {'sim': 'damped', 'dt': [2e-2], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
 {'sim': 'discontinuous', 'dt': [1e-2], 'nt': [1000], 'n': n_set, 'dim': [2, 3]},
]


#Select the hand-tuned dt value for a smooth simulation
# (since scales are different in each potential):
dt = [ss['dt'][0] for ss in sim_sets if ss['sim'] == sim][0]

# Default regularization/test configuration used throughout
test = '_l1_'

title = '{}_n={}_dim={}_nt={}_dt={}'.format(sim, n, dim, nt, dt)
print('Running on', title)

print("v8mC0I_lZl-j done")

# %% [markdown] id="ARWJg6SbZl-n" colab_type="text"
# ## Generate simulation data:

# %% id="38srsNZbZl-o" colab_type="code" colab={}
from simulate import SimulationDataset
s = SimulationDataset(sim, n=n, dim=dim, nt=nt//2, dt=dt)
# Update this to your own dataset, or regenerate:
base_str = './'
data_str = title
s.simulate(ns)

print("38srsNZbZl-o done")

# %% id="_iWH9BfUZl-r" colab_type="code" colab={"base_uri": "https://localhost:8080/"} outputId="4f70a812-4f4f-42bb-c066-23e92fb738c6"
data = s.data
s.data.shape

print("_iWH9BfUZl-r done")

# %% [markdown] id="8sCvhjgVpWIx" colab_type="text"
# ### Let's visualize an example simulation:

# %% id="y5hFgzapZl-u" colab_type="code" colab={"base_uri": "https://localhost:8080/"} outputId="49a5e715-e36d-473c-c77d-e5ccafbac24b"
simulation_preview = s.plot(0, animate=True, plot_size=False)
save_plot(plt.gcf(), f"{title}{test}simulation_traces")

print("y5hFgzapZl-u done")

# %% [markdown] id="1PRxdcqSZl-z" colab_type="text"
# ## We'll train on the accelerations, so let's generate the dataset:

# %% id="ZzHxTkcKZl-0" colab_type="code" colab={}
accel_data = s.get_acceleration()

print("ZzHxTkcKZl-0 done")

# %% id="j2j1M4coZl-3" colab_type="code" colab={}
X = torch.from_numpy(np.concatenate([s.data[:, i] for i in range(0, s.data.shape[1], 5)]))
y = torch.from_numpy(np.concatenate([accel_data[:, i] for i in range(0, s.data.shape[1], 5)]))

print("j2j1M4coZl-3 done")

# %% id="7e8JdFUWZl-5" colab_type="code" colab={}
from sklearn.model_selection import train_test_split

print("7e8JdFUWZl-5 done")

# %% id="skQDNVroZl-8" colab_type="code" colab={}
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

print("skQDNVroZl-8 done")

# %% [markdown] id="Fz05x2HLT5Kg" colab_type="text"
# # Set up the model

# %% id="XWcn4EEFZl-_" colab_type="code" colab={}
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing

print("XWcn4EEFZl-_ done")

# %% id="E-uVcXRcZl_D" colab_type="code" colab={}
from models import OGN, varOGN, make_packer, make_unpacker, get_edge_index

print("E-uVcXRcZl_D done")

# %% [markdown] id="kBtdfcvgZl_F" colab_type="text"
# ## Use the L1 regularization model:

# %% id="PyKcc5zbZl_H" colab_type="code" colab={}
aggr = 'add'
hidden = 300

#This test applies an explicit bottleneck:

msg_dim = 100
n_f = data.shape[3]

print("PyKcc5zbZl_H done")

# %% [markdown] id="oDiU99YHZl_K" colab_type="text"
# L1 loss: we simply add the loss to the batch number. I.e., * 32 for batch size 32.
#
# KL loss: model the messages as a distribution with the prior a Gaussian. The means add in the final Gaussian.
# Recall in the D_KL(p||q), the prior is q.  Then, for sigma_q = 1, mu_q = 0, we have ($p=1$):
#
# $$D_{KL}(p||q) = \frac{\sigma_p^2 + \mu_p^2}{2} -\log({\sigma_p}) - \frac{1}{2}$$

# %% [markdown] id="gTHRRgTuZl_K" colab_type="text"
# ## We use a custom data loader for the graphs for fast training:

# %% id="oL-1skODZl_L" colab_type="code" colab={}
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

print("oL-1skODZl_L done")

# %% id="JbkHAAPzZl_N" colab_type="code" colab={}
from models import get_edge_index

print("JbkHAAPzZl_N done")

# %% id="FOI75DW0Zl_Q" colab_type="code" colab={}
edge_index = get_edge_index(n, sim)

print("FOI75DW0Zl_Q done")

# %% [markdown] id="g46wRgIWZl_S" colab_type="text"
# ## Initiate the model:

# %% id="1bs7l6xYZl_T" colab_type="code" colab={}
if test == '_kl_':
    ogn = varOGN(n_f, msg_dim, dim, dt=0.1, hidden=hidden, edge_index=get_edge_index(n, sim), aggr=aggr).cuda()
else:
    ogn = OGN(n_f, msg_dim, dim, dt=0.1, hidden=hidden, edge_index=get_edge_index(n, sim), aggr=aggr).cuda()

messages_over_time = []
ogn = ogn.cuda()

print("1bs7l6xYZl_T done")

# %% [markdown] id="i1SJ012hZl_V" colab_type="text"
# ### Let's test it:

# %% id="eSea_DJ-Zl_W" colab_type="code" colab={"base_uri": "https://localhost:8080/"} outputId="65989f6e-0ae4-4590-dd9d-a39291375619"
_q = Data(
    x=X_train[0].cuda(),
    edge_index=edge_index.cuda(),
    y=y_train[0].cuda())
ogn(_q.x, _q.edge_index), ogn.just_derivative(_q).shape, _q.y.shape, ogn.loss(_q),

print("eSea_DJ-Zl_W done")

# %% [markdown] id="RszfVgfhZl_a" colab_type="text"
# # Set up training

# %% [markdown] id="WK5_u9F7UttT" colab_type="text"
# ## Organize into data loader:

# %% id="8JyKnPWnZl_b" colab_type="code" colab={}
batch = int(64 * (4 / n)**2)
trainloader = DataLoader(
    [Data(
        Variable(X_train[i]),
        edge_index=edge_index,
        y=Variable(y_train[i])) for i in range(len(y_train))],
    batch_size=batch,
    shuffle=True
)

testloader = DataLoader(
    [Data(
        X_test[i],
        edge_index=edge_index,
        y=y_test[i]) for i in range(len(y_test))],
    batch_size=1024,
    shuffle=True
)

print("8JyKnPWnZl_b done")

# %% [markdown] id="yyduWbndZl_e" colab_type="text"
# ## We'll use OneCycleLR for fast training:

# %% id="3dXp3rkuZl_f" colab_type="code" colab={}
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR


print("3dXp3rkuZl_f done")

# %% [markdown] id="hSqOrC1WZl_l" colab_type="text"
# ## Create the loss function
#
# This holds definition of our L1 and KL regularizations:

# %% id="U6EHSHiOZl_m" colab_type="code" colab={}
def new_loss(self, g, augment=True, square=False):
    if square:
        return torch.sum((g.y - self.just_derivative(g, augment=augment))**2)
    else:
        base_loss = torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment)))
        if test in ['_l1_', '_kl_']:
            s1 = g.x[self.edge_index[0]]
            s2 = g.x[self.edge_index[1]]
            if test == '_l1_':
                m12 = self.message(s1, s2)
                regularization = 1e-2
                #Want one loss value per row of g.y:
                normalized_l05 = torch.sum(torch.abs(m12))
                return base_loss, regularization * batch * normalized_l05 / n**2 * n
            elif test == '_kl_':
                regularization = 1
                #Want one loss value per row of g.y:
                tmp = torch.cat([s1, s2], dim=1)  # tmp has shape [E, 2 * in_channels]
                raw_msg = self.msg_fnc(tmp)
                mu = raw_msg[:, 0::2]
                logvar = raw_msg[:, 1::2]
                full_kl = torch.sum(torch.exp(logvar) + mu**2 - logvar)/2.0
                return base_loss, regularization * batch * full_kl / n**2 * n
        return base_loss


print("U6EHSHiOZl_m done")

# %% [markdown] id="w12Qg4t_em8w" colab_type="text"
# ## Set up optimizer and training parameters:
#
# **Use 200 epochs for full version; can use fewer for test.**

# %% id="y_KbxGDEZl_p" colab_type="code" colab={"base_uri": "https://localhost:8080/"} outputId="1c7f38d0-c33e-4c2b-ee92-480b36eed172"
init_lr = 1e-3

opt = torch.optim.Adam(ogn.parameters(), lr=init_lr, weight_decay=1e-8)

# total_epochs = 200
total_epochs = 30


batch_per_epoch = int(1000*10 / (batch/32.0))

sched = OneCycleLR(opt, max_lr=init_lr,
                   steps_per_epoch=batch_per_epoch,#len(trainloader),
                   epochs=total_epochs, final_div_factor=1e5)

batch_per_epoch

print("y_KbxGDEZl_p done")

# %% id="4AyhMeZ7Zl_r" colab_type="code" colab={}
epoch = 0

print("4AyhMeZ7Zl_r done")

# %% id="s6Q_XHHOZl_v" colab_type="code" colab={}
from tqdm import tqdm

print("s6Q_XHHOZl_v done")

# %% [markdown] id="Y4EzD79nU8DO" colab_type="text"
# ## Organize the recording of messages over time
#
# This is for fitting the forces, and extracting laws:

# %% id="J2jTb8r-Zl_z" colab_type="code" colab={}
import numpy as onp
onp.random.seed(0)
test_idxes = onp.random.randint(0, len(X_test), 1000)

#Record messages over test dataset here:
newtestloader = DataLoader(
    [Data(
        X_test[i],
        edge_index=edge_index,
        y=y_test[i]) for i in test_idxes],
    batch_size=len(X_test),
    shuffle=False
)

print("J2jTb8r-Zl_z done")

# %% [markdown] id="Uxtqu4E4erXd" colab_type="text"
# ### Function to record messages from model

# %% id="0k9hfsslZl_2" colab_type="code" colab={}
import numpy as onp
import pandas as pd

def get_messages(ogn):

    def get_message_info(tmp):
        ogn.cpu()

        s1 = tmp.x[tmp.edge_index[0]]
        s2 = tmp.x[tmp.edge_index[1]]
        tmp = torch.cat([s1, s2], dim=1)  # tmp has shape [E, 2 * in_channels]
        if test == '_kl_':
            raw_msg = ogn.msg_fnc(tmp)
            mu = raw_msg[:, 0::2]
            logvar = raw_msg[:, 1::2]

            m12 = mu
        else:
            m12 = ogn.msg_fnc(tmp)

        all_messages = torch.cat((
            s1,
            s2,
            m12), dim=1)
        if dim == 2:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d y%d vx%d vy%d q%d m%d'.split(' ')]
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        elif dim == 3:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d y%d z%d vx%d vy%d vz%d q%d m%d'.split(' ')]
            columns += ['e%d'%(k,) for k in range(msg_dim)]


        return pd.DataFrame(
            data=all_messages.cpu().detach().numpy(),
            columns=columns
        )

    msg_info = []
    for i, g in enumerate(newtestloader):
        msg_info.append(get_message_info(g))

    msg_info = pd.concat(msg_info)
    msg_info['dx'] = msg_info.x1 - msg_info.x2
    msg_info['dy'] = msg_info.y1 - msg_info.y2
    if dim == 2:
        msg_info['r'] = np.sqrt(
            (msg_info.dx)**2 + (msg_info.dy)**2
        )
    elif dim == 3:
        msg_info['dz'] = msg_info.z1 - msg_info.z2
        msg_info['r'] = np.sqrt(
            (msg_info.dx)**2 + (msg_info.dy)**2 + (msg_info.dz)**2
        )

    return msg_info

print("0k9hfsslZl_2 done")

# %% id="RsLYGJX8Zl_5" colab_type="code" colab={}
recorded_models = []

print("RsLYGJX8Zl_5 done")

# %% [markdown] id="lGJSTkWOZl_7" colab_type="text"
# # Train the model:

# %% [markdown] id="x-M9IDuiVrot" colab_type="text"
# ## Training loop:

# %% jupyter={"outputs_hidden": true, "source_hidden": true} id="NHxCTt_tZl_7" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 544} outputId="b7f9e679-f490-4f39-92cb-61d7e6563666"
for epoch in tqdm(range(epoch, total_epochs)):
    ogn.cuda()
    total_loss = 0.0
    i = 0
    num_items = 0
    while i < batch_per_epoch:
        for ginput in trainloader:
            if i >= batch_per_epoch:
                break
            opt.zero_grad()
            ginput.x = ginput.x.cuda()
            ginput.y = ginput.y.cuda()
            ginput.edge_index = ginput.edge_index.cuda()
            ginput.batch = ginput.batch.cuda()
            if test in ['_l1_', '_kl_']:
                loss, reg = new_loss(ogn, ginput, square=False)
                ((loss + reg)/int(ginput.batch[-1]+1)).backward()
            else:
                loss = ogn.loss(ginput, square=False)
                (loss/int(ginput.batch[-1]+1)).backward()
            opt.step()
            sched.step()

            total_loss += loss.item()
            i += 1
            num_items += int(ginput.batch[-1]+1)

    cur_loss = total_loss/num_items
    print(cur_loss)
    cur_msgs = get_messages(ogn)
    cur_msgs['epoch'] = epoch
    cur_msgs['loss'] = cur_loss
    messages_over_time.append(cur_msgs)

    ogn.cpu()
    from copy import deepcopy as copy
    recorded_models.append(ogn.state_dict())

print("NHxCTt_tZl_7 done")

# %% [markdown] id="NZx6ngcZjeif" colab_type="text"
# Normally you should run all the way to 200 epochs (set this as a parameter so that OneCycleLR is set correctly). This loop was cut off early to quickly test this notebook.

# %% [markdown] id="YAqyXbr9dpKE" colab_type="text"
# ## Save and Load models here to prevent re-training:

# %% id="7eQ9087RZmAA" colab_type="code" colab={}
import pickle as pkl
# pkl.dump(messages_over_time,
#     open('messages_over_time.pkl', 'wb'))
# messages_over_time = pkl.load(open('messages_over_time.pkl', 'rb'))

print("7eQ9087RZmAA done")

# %% id="i8i0qy2tZmAD" colab_type="code" colab={}
# pkl.dump(recorded_models,
#     open('models_over_time.pkl', 'wb'))

# recorded_models = pkl.load(open('models_over_time.pkl', 'rb'))

print("i8i0qy2tZmAD done")

# %% [markdown] id="iC7pYpZnVwFG" colab_type="text"
# # Analyze trained model:

# %% [markdown] id="c3R-CJtQZmAJ" colab_type="text"
# ## Plot a comparison of the force components with the messages:
#
# ### (Or plot the rotation or sparsity - turn these on and off with flags:)

# %% id="12_ldordZmAP" colab_type="code" colab={}
from celluloid import Camera
from copy import deepcopy as copy

print("12_ldordZmAP done")

# %% [markdown] id="NMVQTRFMV2ZL" colab_type="text"
# Options include:
#
# - plot_force_components: the scatter plots of true force versus message
# - plot_sparsity: the grayscale animation of the message components over time
# - plot_rotation: plot the vectors showing how the messages are rotations of the true vectors

# %% jupyter={"outputs_hidden": true} id="x4slGzWtZmAS" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="d82a2e68-5f63-4673-8748-9042974c134f"

#Only turn on one of these:
plot_force_components = True
plot_sparsity = False
plot_rotation = False
if plot_force_components:
    fig, ax = plt.subplots(1, dim, figsize=(4*dim, 4))
if plot_sparsity or plot_rotation:
    fig, ax = plt.subplots(1, 1)
cam = Camera(fig)


last_alpha_x1 = 0.0
last_alpha_y1 = 0.0
t = lambda _: _#tqdm
for i in t(range(0, len(messages_over_time), 1)):
    msgs = copy(messages_over_time[i])

    msgs['bd'] = msgs.r + 1e-2

    try:
        msg_columns = ['e%d'%(k) for k in range(1, msg_dim+1)]
        msg_array = np.array(msgs[msg_columns])
    except:
        msg_columns = ['e%d'%(k) for k in range(msg_dim)]
        msg_array = np.array(msgs[msg_columns])

    msg_importance = msg_array.std(axis=0)
    most_important = np.argsort(msg_importance)[-dim:]
    msgs_to_compare = msg_array[:, most_important]
    msgs_to_compare = (msgs_to_compare - np.average(msgs_to_compare, axis=0)) / np.std(msgs_to_compare, axis=0)

    if plot_sparsity:
        ax.pcolormesh(msg_importance[np.argsort(msg_importance)[::-1][None, :15]], cmap='gray_r', edgecolors='k')
        # plt.colorbar()
        plt.axis('off')
        plt.grid(True)
        ax.set_aspect('equal')
        plt.text(15.5, 0.5, '...', fontsize=30)
        # fig.suptitle(title + test + 'mse=%.3e'%(min_result.fun/len(msgs),))
        plt.tight_layout()

    if plot_force_components or plot_rotation:
        pos_cols = ['dx', 'dy']
        if dim == 3:
            pos_cols.append('dz')

        if sim != 'spring':
            raise NotImplementedError("The current force function is for a spring. You will need to change the force function below to that expected by your simulation.")

        def force_fnc(msg):
            bd = msg.bd.to_numpy()
            pos = np.asarray(msg[pos_cols])
            return -((bd - 1)[:, None] * pos) / bd[:, None]

        expected_forces = force_fnc(msgs)

        def percentile_sum(x):
            x = x.ravel()
            bot = x.min()
            top = np.percentile(x, 90)
            msk = (x>=bot) & (x<=top)
            frac_good = (msk).sum()/len(x)
            return x[msk].sum()/frac_good

        from scipy.optimize import minimize

        def linear_transformation_2d(alpha):

            lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1]) + alpha[2]
            lincomb2 = (alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1]) + alpha[5]

            score = (
                percentile_sum(np.square(msgs_to_compare[:, 0] - lincomb1)) +
                percentile_sum(np.square(msgs_to_compare[:, 1] - lincomb2))
            )/2.0

            return score

        def out_linear_transformation_2d(alpha):
            lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1]) + alpha[2]
            lincomb2 = (alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1]) + alpha[5]

            return lincomb1, lincomb2

        def linear_transformation_3d(alpha):

            lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1] + alpha[2] * expected_forces[:, 2]) + alpha[3]
            lincomb2 = (alpha[0+4] * expected_forces[:, 0] + alpha[1+4] * expected_forces[:, 1] + alpha[2+4] * expected_forces[:, 2]) + alpha[3+4]
            lincomb3 = (alpha[0+8] * expected_forces[:, 0] + alpha[1+8] * expected_forces[:, 1] + alpha[2+8] * expected_forces[:, 2]) + alpha[3+8]

            score = (
                percentile_sum(np.square(msgs_to_compare[:, 0] - lincomb1)) +
                percentile_sum(np.square(msgs_to_compare[:, 1] - lincomb2)) +
                percentile_sum(np.square(msgs_to_compare[:, 2] - lincomb3))
            )/3.0

            return score

        def out_linear_transformation_3d(alpha):

            lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1] + alpha[2] * expected_forces[:, 2]) + alpha[3]
            lincomb2 = (alpha[0+4] * expected_forces[:, 0] + alpha[1+4] * expected_forces[:, 1] + alpha[2+4] * expected_forces[:, 2]) + alpha[3+4]
            lincomb3 = (alpha[0+8] * expected_forces[:, 0] + alpha[1+8] * expected_forces[:, 1] + alpha[2+8] * expected_forces[:, 2]) + alpha[3+8]

            return lincomb1, lincomb2, lincomb3

        if dim == 2:
            min_result = minimize(linear_transformation_2d, np.ones(dim**2 + dim), method='Powell')
        if dim == 3:
            min_result = minimize(linear_transformation_3d, np.ones(dim**2 + dim), method='Powell')
        print(title, test, 'gets', min_result.fun/len(msgs))

        if plot_rotation:
            q = min_result.x
            alphax1, alphay1, offset1 = q[:3]
            alphax2, alphay2, offset2 = q[3:]

            s1 = alphax1**2 + alphay1**2
            s2 = alphax2**2 + alphay2**2

            if (
                    (alphax2 - last_alpha_x1)**2
                    + (alphay2 - last_alpha_y1)**2  <
                   (alphax1 - last_alpha_x1)**2
                    + (alphay1 - last_alpha_y1)**2):

                alphax1, alphay1, offset1 = q[3:]
                alphax2, alphay2, offset2 = q[:3]

            last_alpha_x1 = alphax1
            last_alpha_y1 = alphay1
            s1 = alphax1**2 + alphay1**2
            s2 = alphax2**2 + alphay2**2
            alphax1 /= s1**0.5 * 2
            alphay1 /= s1**0.5 * 2
            alphax2 /= s2**0.5 * 2
            alphay2 /= s2**0.5 * 2

            ax.arrow(0.5, 0.5, alphax1, alphay1, color='k', head_width=0.05, length_includes_head=True)
            ax.arrow(0.5, 0.5, alphax2, alphay2, color='k', head_width=0.05, length_includes_head=True)
            ax.axis('off')

        if plot_force_components:
            for i in range(dim):
                if dim == 3:
                    px = out_linear_transformation_3d(min_result.x)[i]
                else:
                    px = out_linear_transformation_2d(min_result.x)[i]

                py = msgs_to_compare[:, i]
                ax[i].scatter(px, py,
                              alpha=0.1, s=0.1, color='k')
                ax[i].set_xlabel('Linear combination of forces')
                ax[i].set_ylabel('Message Element %d'%(i+1))

                xlim = np.array([np.percentile(px, q) for q in [10, 90]])
                ylim = np.array([np.percentile(py, q) for q in [10, 90]])
                xlim[0], xlim[1] = xlim[0] - (xlim[1] - xlim[0])*0.05, xlim[1] + (xlim[1] - xlim[0])*0.05
                ylim[0], ylim[1] = ylim[0] - (ylim[1] - ylim[0])*0.05, ylim[1] + (ylim[1] - ylim[0])*0.05

                ax[i].set_xlim(xlim)
                ax[i].set_ylim(ylim)

        plt.tight_layout()

    cam.snap()

plot_kind = 'force_components'
if plot_sparsity:
    plot_kind = 'sparsity'
elif plot_rotation:
    plot_kind = 'rotation'

save_plot(fig, f"{title}{test}{plot_kind}_messages", close=False)
ani = cam.animate(interval=100, blit=True)
animation_path = PLOTS_DIR / f"{title}{test}{plot_kind}_messages.mp4"
try:
    ani.save(str(animation_path), writer="ffmpeg")
    print(f"Saved animation to {animation_path}")
except Exception as err:
    print(f"Warning: Could not save animation to {animation_path}: {err}")

from IPython.display import HTML
HTML(ani.to_jshtml())
plt.close(fig)

print("x4slGzWtZmAS done")

# %% [markdown] id="huBwFGpGZmAa" colab_type="text"
# ## Plot some predicted versus true trajectories:

# %% id="MCJkRYfNZmAb" colab_type="code" colab={}
from simulate import make_transparent_color

print("MCJkRYfNZmAb done")

# %% id="os18WuvhZmAc" colab_type="code" colab={}
from scipy.integrate import odeint

print("os18WuvhZmAc done")

# %% id="-9fKkNNeZmAi" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 662} outputId="d1cbb00f-484f-4896-9c34-84039ce3df61"
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
camera = Camera(fig)

for current_model in [-1] + [1, 34, 67, 100, 133, 166, 199]:
    i = 4 #Use this simulation
    if current_model > len(recorded_models):
        continue

    #Truth:
    cutoff_time = 300
    times = onp.array(s.times)[:cutoff_time]
    x_times = onp.array(data[i, :cutoff_time])
    masses = x_times[:, :, -1]
    length_of_tail = 75

    #Learned:
    e = edge_index.cuda()
    ogn.cpu()
    if current_model > -1:
        ogn.load_state_dict(recorded_models[current_model])
    else:
        # Random model!
        ogn = OGN(n_f, msg_dim, dim, dt=0.1, hidden=hidden, edge_index=get_edge_index(n, sim), aggr=aggr).cuda()
    ogn.cuda()

    def odefunc(y, t=None):
        y = y.reshape(4, 6).astype(np.float32)
        cur = Data(
            x=torch.from_numpy(y).cuda(),
            edge_index=e
        )
        dx = y[:, 2:4]
        dv = ogn.just_derivative(cur).cpu().detach().numpy()
        dother = np.zeros_like(dx)
        return np.concatenate((dx, dv, dother), axis=1).ravel()

    datai = odeint(odefunc, (onp.asarray(x_times[0]).ravel()), times).reshape(-1, 4, 6)
    x_times2 = onp.array(datai)

    d_idx = 10
    for t_idx in range(d_idx, cutoff_time, d_idx):
        start = max([0, t_idx-length_of_tail])
        ctimes = times[start:t_idx]
        cx_times = x_times[start:t_idx]
        cx_times2 = x_times2[start:t_idx]
        for j in range(n):
            rgba = make_transparent_color(len(ctimes), j/n)
            ax[0].scatter(cx_times[:, j, 0], cx_times[:, j, 1], color=rgba)
            ax[1].scatter(cx_times2[:, j, 0], cx_times2[:, j, 1], color=rgba)
            black_rgba = rgba
            black_rgba[:, :3] = 0.75
            ax[1].scatter(cx_times[:, j, 0], cx_times[:, j, 1], color=black_rgba, zorder=-1)

        for k in range(2):
            ax[k].set_xlim(-1, 3)
            ax[k].set_ylim(-3, 1)
        plt.tight_layout()
        camera.snap()

# camera.animate().save('multiple_animations_with_comparison.mp4')
from IPython.display import HTML
save_plot(fig, f"{title}{test}trajectory_comparison", close=False)
trajectory_animation = camera.animate(interval=100, blit=True)
trajectory_video = PLOTS_DIR / f"{title}{test}trajectory_comparison.mp4"
try:
    trajectory_animation.save(str(trajectory_video), writer="ffmpeg")
    print(f"Saved animation to {trajectory_video}")
except Exception as err:
    print(f"Warning: Could not save animation to {trajectory_video}: {err}")
HTML(trajectory_animation.to_jshtml())
plt.close(fig)

print("-9fKkNNeZmAi done")

# %% [markdown] id="hLJGeV4IXtxc" colab_type="text"
# # Symbolic regression

# %% [markdown] id="-LHESeG4XquY" colab_type="text"
# Extract the force laws with the following procedure:
# - The data in `messages_over_time` correspond to inputs to, and features of, $\phi^e$, recorded during each training epoch.
# - Select the last element of this list.
# - Find the most significant message feature. Each message feature corresponds to 'e1', 'e2', etc. Calculate the one with the largest standard deviation.
#
# Train [PySR](https://github.com/MilesCranmer/PySR) to fit this relationship.
# Thus, we have extracted a force law from the graph network without priors on the functional form.
#
# This is the same technique we used to extract the unknown dark matter overdensity equation from the Quijote simulations.
#

# %% [markdown] id="hwgQjt-_ic5k" colab_type="text"
# ## Here's the best message, which we will study:

# %% id="G1rhnbq-XvG5" colab_type="code" colab={}
best_message = np.argmax([np.std(messages_over_time[-1]['e%d'%(i,)]) for i in range(100)])

print("G1rhnbq-XvG5 done")

# %% [markdown] id="EAhN3sdzif0p" colab_type="text"
# ## Here's a pandas dataframe of the message data:

# %% id="-2lNO8p1h2pt" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 419} outputId="05648bd9-81b3-4d5c-a4c3-d453707b9751"
messages_over_time[-1][['e%d'%(best_message,), 'dx', 'dy', 'r', 'm1', 'm2']]

print("-2lNO8p1h2pt done")

# %% [markdown] id="0Ipu1gX0ijYv" colab_type="text"
# ## Now we just fit e4 as a function of dx, dy, r, m1, and m2, inside [PySR](https://github.com/MilesCranmer/PySR).
