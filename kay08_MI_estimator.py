#!/usr/bin/env python
# coding: utf-8

# # Mutual Information Neural Estimator for Visual Stimulus and fMRI response
# 
# This notebook demonstrates how a mutual information between stimulus presented in fMRI experiment and the resulting betas can be estimated using "Mutual Information Neural Estimation" (MINE) machinery from Belghazi et al. 2018 https://arxiv.org/abs/1801.04062 

# ## Environment
# This notebook runs in container with docker image: nvcr.io/nvidia/pytorch:22.08-py3

# In[1]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import types


# In[2]:


import pickle as pkl
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torch.optim.lr_scheduler import ConstantLR, SequentialLR


import torch
from torch.utils.data import Dataset, DataLoader


# In[3]:


from ae_builder import BuildAutoEncoder


# ## Load Kay 2008 dataset
# Here we load data from a single subject. There also exists another subject. We can later include his data as well.

# In[4]:


# @title Download the data

import requests, tarfile

fnames = ["kay_labels.npy", "kay_labels_val.npy", "kay_images.npz"]
urls = ["https://osf.io/r638s/download",
        "https://osf.io/yqb3e/download",
        "https://osf.io/ymnjv/download"]

for fname, url in zip(fnames, urls):
  if not os.path.isfile(fname):
    try:
      r = requests.get(url)
    except requests.ConnectionError:
      print("!!! Failed to download data !!!")
    else:
      if r.status_code != requests.codes.ok:
        print("!!! Failed to download data !!!")
      else:
        print(f"Downloading {fname}...")
        with open(fname, "wb") as fid:
          fid.write(r.content)
        print(f"Download {fname} completed!")


# In[5]:


with np.load(fname) as dobj:
  dat = dict(**dobj)
labels = np.load('kay_labels.npy')
val_labels = np.load('kay_labels_val.npy')


# 
# `labels` is a 4 by stim array of class names:  
# - row 3 has the labels predicted by a deep neural network (DNN) trained on Imagenet
# - rows 0-2 correspond to different levels of the wordnet hierarchy for the DNN predictions

# `dat` has the following fields:  
# - `stimuli`: stim x i x j array of grayscale stimulus images
# - `stimuli_test`: stim x i x j array of grayscale stimulus images in the test set  
# - `responses`: stim x voxel array of z-scored BOLD response amplitude
# - `responses_test`:  stim x voxel array of z-scored BOLD response amplitude in the test set  
# - `roi`: array of voxel labels
# - `roi_names`: array of names corresponding to voxel labels

# This is the number of voxels in each ROI. Note that `"Other"` voxels have been removed from this version of the dataset:

# In[6]:


dict(zip(dat["roi_names"], np.bincount(dat["roi"])))


# Each stimulus is a 128 x 128 grayscale array:

# In[7]:


fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
for ax, im, lbl in zip(axs.flat, dat["stimuli"], labels[-1, :]):
  ax.imshow(im, cmap="gray")
  ax.set_title(lbl)
fig.tight_layout()
fig.show()


# Each stimulus is associated with a pattern of BOLD response across voxels in visual cortex:

# In[8]:


fig, ax = plt.subplots(figsize=(12, 5))
ax.set(xlabel="Voxel", ylabel="Stimulus")
heatmap = ax.imshow(dat["responses"],
                    aspect="auto", vmin=-1, vmax=1, cmap="bwr")
fig.colorbar(heatmap, shrink=.5, label="Response amplitude (Z)")
fig.tight_layout()
fig.show()


# The training/validation splits from the original paper are retained, so the 120 test stimuli and responses are in separate data structures:

# In[9]:


fig, ax = plt.subplots(figsize=(12, 2.5))
ax.set(xlabel="Voxel", ylabel="Test Stimulus")
heatmap = ax.imshow(dat["responses_test"],
                    aspect="auto", vmin=-1, vmax=1, cmap="bwr")
fig.colorbar(heatmap, shrink=.75, label="Response amplitude (Z)")
fig.tight_layout()
fig.show()


# In[10]:


dat["stimuli"].shape, dat["stimuli_test"].shape


# In[11]:


dat["stimuli"].max(),dat["stimuli"].min()


# # Mutual Information Estimate and Loss
# Mutual information between random variables $X$ and $Y$ can be represented as a KL divergence between joint and marginal distribution of $X$ and $Y$:
# $I(X;Y) = KL(\mathbb{P}_{XY} || \mathbb{P}_{X} \otimes \mathbb{P}_{Y})$
# 
# The latter can be formulated as a maximization problem (Donsker-Varadhan representation):
# $KL(\mathbb{P} || \mathbb{Q}) = \sup_{T} \mathbb{E}_{P}T - \log{\mathbb{E}_{Q}\exp{T}}$.
# See: Belghazi et al. https://arxiv.org/abs/1801.04062 for details
# 
# The loss is calculated as a derivative of the MI estimate with a *debiasing* term added to compensate for baiased estimatin of marginal in logarithmic derivative. 

# In[12]:


def estimate_mi(joint_pred, marginal_pred):
    with torch.no_grad():  # Ensure no gradients are computed to save memory and computations
        t_joint = joint_pred.mean()
        et_marginal = torch.exp(marginal_pred).mean()
        mi_estimate = t_joint - torch.log(et_marginal)
    return mi_estimate.item()

def calculate_loss(joint_pred, marginal_pred, ma_et=None, ma_rate=0.02):
    t_joint = joint_pred.mean()
    et_marginal = torch.exp(marginal_pred).mean()
    #debiasing:
    if ma_et is not None:
        ma_et = (1-ma_rate)*ma_et + (ma_rate)*et_marginal
        loss = -t_joint + (1/ma_et.mean()).detach()*et_marginal
        return loss, ma_et
    else:
        #if no debiasing then the loss is just -Mutual Information
        loss = -(t_joint - torch.log(et_marginal))
        return loss



# ## Data Preprocessing 
# Preprocessing Kay 2008 grayscale images for a pytorch vgg16 based autoencoder

# In[13]:


def bias_and_expand_pytorch(x):
    x = x - x.min()
    x = np.expand_dims(x, axis=2)
    x = np.repeat(x, 3, axis=2)  # Duplicate channels
    return x




def preprocess_for_vgg16_pytorch(x, imresize=(224, 224), size=(224, 224)):
    # Assuming x is a single image with shape (H, W) or (H, W, C)
    if x.ndim == 2:  # Add a channel dimension if grayscale
        x = np.expand_dims(x, axis=-1)
    x = np.repeat(x, 3, axis=2)  # Repeat the channel to get 3-channel image

    # Convert to uint8
    x = ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(imresize, interpolation=Image.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    x = transform(x)
    return x


# # Networks and Models

# ## Load Autoencoder
# We use a pretrained encoder model to obtain a flat vector representation of image that can be easily used for our mutual informatin estimator.
# We will only use the encoder part. No use for the decoder in this code

# In[14]:


'''
We use an autoencoder from here as a preprocessing for the visual input: 
https://github.com/Horizon2333/imagenet-autoencoder/tree/main
'''
def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model


# In[15]:


args = types.SimpleNamespace()


args.arch = "vgg16"
args.parallel = False

'''
you can use the following bash command to download the ae weights:
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WwJiQ1kBcNCZ37F6PJ_0bIL0ZeU3_sV8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WwJiQ1kBcNCZ37F6PJ_0bIL0ZeU3_sV8" -O FILENAME && rm -rf /tmp/cookies.txt
'''
resume_path = '/home/arivkind/models/ae_models/imagenet-vgg16.pth'

ae_model = BuildAutoEncoder(args)
load_dict(resume_path, ae_model)
enc_model = ae_model.module.encoder


# ## Wrap the encoder

# In[16]:


class CoreNetWrapper(nn.Module):
    '''
    class that wraps an autorncoder network including:
    -flattening the output
    -passing it via MLP
    -optionally normalizing
    -optionally freezing the core weights
    '''
    def __init__(self, core_net, mlp_widths=[4096, 1024], apply_layer_norm=True, train_core=False):
        super(CoreNetWrapper, self).__init__()
        self.core_net = core_net
        # Freeze the weights of core_net
        for param in self.core_net.parameters():
            param.requires_grad = train_core

        # Calculate the flattened input size for the first MLP layer
        flattened_size = 512 * 7 * 7  # C*H*W where C=512, H=W=7
        mlp_layers = []
        mlp_layers.append(nn.LayerNorm(flattened_size))
        # Creating the MLP
        input_size = flattened_size
        for width in mlp_widths:
            mlp_layers.append(nn.Linear(input_size, width))
            mlp_layers.append(nn.ReLU())
            input_size = width
        
        # Optionally apply layer normalization at the top
        if apply_layer_norm:
            mlp_layers.append(nn.LayerNorm(input_size))
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        # Pass input through core_net
        x = self.core_net(x)
        # Flatten the output
        x = torch.flatten(x, 1)
        # Pass through the MLP
        x = self.mlp(x)
        return x


# ## Add MLP for voxel data

# In[17]:


class VanillaMLP(nn.Module):
    '''
    simple mlp network with layer normalization
    '''
    def __init__(self, layer_sizes, batchnrom_after=[]):
        super(VanillaMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for ii, (size_in,size_out) in enumerate(zip(layer_sizes[:-1],layer_sizes[1:])):
            self.layers.append(nn.Linear(size_in,size_out))
            self.norms.append(nn.LayerNorm(size_out))

        self.nl = nn.ReLU()

    def forward(self, x):
        for layer,norm in zip(self.layers[:-1],self.norms[:-1]):
            x = layer(x)
            x = self.nl(x)
            x= norm(x)
        x = self.layers[-1](x)
        x = self.norms[-1](x)
        return x
    


# # Concatenate processed image data with processed voxel data

# In[18]:


class MergeNet(nn.Module):
    def __init__(self,net1,net2):
        super(MergeNet,self).__init__()
        self.net1 = net1
        self.net2 = net2
    def forward(self,x1,x2):
        x1 = self.net1(x1)
        x2 = self.net2(x2)
        return torch.cat((x1,x2),dim=1)
    


# ## Calculate T for Donsker-Varadhan estimate of KL divergence

# In[19]:


class MINE(nn.Module):
    '''
    MINE network takes a concatenation of datasamples
    either from joint or from a marginal distribution and returns a scalar function T
    that is used in to estimate the Mutual Information between the two vari
    '''
    def __init__(self, n):
        super(MINE, self).__init__()
        # Define the network architecture
        # This should be adjusted depending on the problem. This is just a simple example.
        self.fc1 = nn.Linear(n, n//2) 
        self.norm1 = nn.LayerNorm(n//2)

        self.fc2 = nn.Linear(n//2, 256) 
        self.norm2 = nn.LayerNorm(256)

        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.fc3(x)
        return x


# ## Dataset
# Dataset includes images which are preprocessed using preprocess_func and the corresponding fMRI measurements.
# For the *joint* distribution the Dataset is used as is.
# Fro the *marginal* distribution measurements are shifted by one within the batch vs. images. Since batches are resampled randomly at each iteration, the marginal samples vary between epochs.

# In[20]:


class CustomDatasetWithMeasurements(Dataset):
    def __init__(self, images, measurements, preprocess_func):
        """
        images: A numpy array of images with shape (S, H, W)
        measurements: A numpy array of measurements with shape (S, N)
        preprocess_func: A function to preprocess images
        """
        assert len(images) == len(measurements), "Images and measurements must have the same length"
        self.images = images
        self.measurements = measurements
        self.preprocess_func = preprocess_func


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.preprocess_func(self.images[idx]).float()  # Ensure image is float
        measurement = torch.tensor(self.measurements[idx], dtype=torch.float)  # Ensure measurement is float

        
        
        # Return both corresponding and unrelated measurement along with the image
        return image, measurement
    

train_dataset = CustomDatasetWithMeasurements(dat["stimuli"], dat["responses"], preprocess_for_vgg16_pytorch)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)

val_dataset = CustomDatasetWithMeasurements(dat["stimuli_test"], dat["responses_test"], preprocess_for_vgg16_pytorch)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=True)


# ## Epoch runner
# Scroll the annotation for the main loop, for more details about bias and *ma_et* estimator

# In[21]:


def run_epoch(dataloader, 
              mode, 
              net_and_observations, 
              mine_net, 
              calculate_loss, 
              estimate_mi, 
              optimizer=None,
              iters = 1,
              device='cpu',
             ma_et = 1.):
    """
    Runs a single epoch in either train or validation mode.

    Args:
    - dataloader: The data loader for the current phase.
    - mode: A string, either 'train' or 'val', indicating the mode.
    - net_and_observations: Function to get joint and marginal samples.
    - mine_net: The neural network used for MI estimation.
    - calculate_loss: Function to calculate the loss between joint and marginal predictions.
    - estimate_mi: Function to estimate MI from joint and marginal predictions.
    - optimizer: The optimizer used for training. This is not used in validation mode.
    - device: The device to run the operations on.

    Returns:
    - epoch_loss: The average loss over the epoch.
    - mi_estimate: The average MI estimate over the epoch.
    """
    if mode == 'train' and optimizer is None:
        raise ValueError("Optimizer must be provided in train mode.")
    
    epoch_loss = 0.0
    mi_estimate = 0.0
    if mode == 'train':
        net_and_observations.train()  # Set the network to training mode
    else:
        net_and_observations.eval()  # Set the network to evaluation mode
    
    #we alow an option to evaluate the data few times 
    #(which leads to a more accurate estimation of MI, mainly due to improved accuracy in marginal)
    for i in range(iters): 
        for images, observations in dataloader:
            images = images.to(device)
            observations = observations.to(device)

            with torch.set_grad_enabled(mode == 'train'):
                joint = net_and_observations(images, observations)
                marginal = net_and_observations(images, torch.roll(observations, shifts=1, dims=0))

                joint_pred = mine_net(joint)
                marginal_pred = mine_net(marginal)
                loss, ma_et = calculate_loss(joint_pred, marginal_pred, ma_et=ma_et)

                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epoch_loss += loss.item()
            mi_estimate += estimate_mi(joint_pred, marginal_pred)

    epoch_loss /= len(dataloader)*iters
    mi_estimate /= len(dataloader)*iters
    
    return epoch_loss, mi_estimate, ma_et


# In[22]:


device = torch.device("cuda")


# In[23]:


visu_net = CoreNetWrapper(enc_model, train_core = False)
mlp_net = VanillaMLP([dat["responses"].shape[1], 2048,1024])
visu_net = visu_net.to(device)
mlp_net = mlp_net.to(device)

net_and_observations = MergeNet(visu_net,mlp_net).to(device)
mine_net = MINE(2048).to(device)


# ## Optimizer
# Optimizer is set to slow down the learning rate by a factor of *10* in the middle of the training. 
# This is something one may want to play with, depending on the specific dataset details.

# In[24]:


epochs = 50
optimizer = optim.Adam(list(mine_net.parameters())+list(net_and_observations.parameters()), lr=1e-4)

scheduler0 = ConstantLR(optimizer, factor=1.0)
scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=epochs)
scheduler = SequentialLR(optimizer, schedulers=[scheduler0,scheduler1], milestones=[int(epochs//2)])


#a hack for switching learning rate manually if needed
# for g in optimizer.param_groups:
#     g['lr'] = 1e-5


# ## Running training and validation
# Few things to pay attention for:
# 1. We are tracking the running empirical average of $\mathbb{E}_{marginal} \exp T$ using a variable *ma_et*. This average is used in the denominator of stochastic gradient estimator. If you take it over many batches using running average, the result is less biased than if you compute it for every individual batch.
# 2. Validation set is very small, just 120 data samples. To obtain a better estimation of validation loss we iterate over it 10 times so that we obatain 10 different estimates of the marginal term. This helps to partially mitigate the noise that the smallness of the validation set induces.

# In[25]:


train_mi_rec = []
val_mi_rec = []
ma_et = 1.0
for epoch in range(epochs):
    train_loss, train_mi, ma_et = run_epoch(train_dataloader, 'train', net_and_observations,
                                     mine_net, calculate_loss, estimate_mi, optimizer, device=device, ma_et=ma_et)
    val_loss, val_mi, _ = run_epoch(val_dataloader, 'val', net_and_observations, mine_net,
                                 calculate_loss, estimate_mi, None, device=device, iters=10)
    this_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Train MI Estimate: {train_mi:.5f}, Val Loss: {val_loss:.5f}, Val MI Estimate: {val_mi:.5f}, LR: {this_lr}')
    train_mi_rec.append(train_mi)
    val_mi_rec.append(val_mi)
    scheduler.step()


# ## Ploting results
# Note that we present the results in "natural basis (ln) the information is therefore measured in *nats*. 
# 
# Note: $1 nat \approx 1.44 bits$.

# In[26]:


plt.plot(train_mi_rec)
plt.plot(val_mi_rec)
plt.xlabel('epoch')
plt.ylabel('MI estimate [nats]')
plt.legend(['train', 'val'])
plt.grid()


# ## Discussion in a nutshell
# We observe that MINE estimator converges to slightly more than 1 nat, (approximately 1.5 bits) but overfits afterwise. It is reasonable to expect that for a much larger NSD dataset, the estimator will keep climbing up and will also be more stable.
# 

# In[ ]:




