from __future__ import print_function
from builtins import super
"""
Implementation of Block Neural Autoregressive Flow
http://arxiv.org/abs/1904.04676
"""
import torch.nn.functional as F
import random
import torch
torch.set_printoptions(profile="full")
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import warnings
import torchvision
print ("torch",torch.__version__)
print ("torchvision",torchvision.__version__)
warnings.filterwarnings("ignore")
import pandas as pd 
import sys
import ROOT
import array
import math
import os
import time
import argparse
import pprint
import scipy
from scipy import special
from functools import partial
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)
from sklearn.manifold import TSNE
import csv
from torch.utils import data
from bnaf import *
from tqdm import trange
from data.generate2d import sample2d, energy2d
from optim.adam import Adam
from optim.lr_scheduler import ReduceLROnPlateau
from sklearn.neighbors import KernelDensity

def rms(x, axis=None):
    return math.sqrt(np.mean((x-(np.mean(x)))**2, axis=axis))


ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--plot', action='store_true', help='Plot a flow and target density.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--cuda', type=int, default=0, help='Which GPU to run on.')
parser.add_argument('--ncol', type=int, default=1, help='')
parser.add_argument('--tosum',  action='store_true')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
parser.add_argument('--mseinpdf', action='store_true', help='Evaluate a flow.')
parser.add_argument('--mult', action='store_true', help='Evaluate a flow.')
parser.add_argument('--normmullt', action='store_true', help='Evaluate a flow.')
parser.add_argument('--simul', action='store_true', help='Evaluate a flow.')
parser.add_argument('--nofreeze', action='store_true', help='Evaluate a flow.')
parser.add_argument('--dataset', type=str, help='Which potential function to approximate.')
parser.add_argument('--extex', type=str, default='', help='')
parser.add_argument('--data_dim', type=int, default=2, help='Dimension of the data.')
parser.add_argument('--hidden_dim', type=int, default=100, help='Dimensions of hidden layers.')
parser.add_argument('--n_hidden', type=int, default=5, help='Number of hidden layers.')
parser.add_argument('--flows', type=int, default=1, help='Number of hidden layers.')
parser.add_argument('--fac_maf', type=float, default=1.0, help='')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--n_steps', type=int, default=1, help='Number of steps to train.')
parser.add_argument('--percentile', type=int, default=0, help='p')
parser.add_argument('--n_tot', type=int, default=100000, help='Number of steps to train.')
parser.add_argument('--n_val', type=int, default=15000, help='Number of steps to train.')
parser.add_argument('--batch_size', type=int, default=200, help='Training batch size.')
parser.add_argument('--lr0', type=float, default=1e-1, help='Initial learning rate.')
parser.add_argument('--lr1', type=float, default=1e-1, help='Initial learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay.')
parser.add_argument('--lr_patience', type=float, default=2, help='Number of steps before decaying learning rate.')
parser.add_argument('--log_interval', type=int, default=50, help='How often to save model and samples.')


def freezelayer(child):
  for param in child.parameters():
     param.requires_grad = False
def releaselayer(child):
  for param in child.parameters():
     param.requires_grad = True


def expand_array(images,ncol=1,tosum=False):
  Nimages=len(images)
  ncolors=ncol
  arrsize=1
  if not tosum:
    arrsize=ncolors
  expandedimages=np.zeros((Nimages,40,40,arrsize),dtype=np.float32)
  for i in range(Nimages):
    npart = len(images[i])
    for j in range(npart):
       if tosum:
               expandedimages[i,images[i][j][0][0],images[i][j][0][1]][0]=0.0
               for nn in range(ncolors):
                   #print(expandedimages[i,images[i][j][0][0],images[i][j][0][1]])
                   expandedimages[i,images[i][j][0][0],images[i][j][0][1]][0] += images[i][j][1][nn]
       else:
               for nn in range(ncolors):
                   expandedimages[i,images[i][j][0][0],images[i][j][0][1]][nn] = images[i][j][1][nn]
  expandedimages=expandedimages.reshape(Nimages,arrsize,40,40)
  return expandedimages

# --------------------
# Data
# --------------------

datamap=        {
                'qcd':"/cms/knash/EOS/decorrtritopv142017/QCDv142017allstd_shuftritopconstpt.dat",
                'qcdtest':"/cms/knash/EOS/decorrtritopv142017/QCDv142017allstd_shuftritopconstpt.dat",
                'ttbar':"/cms/knash/EOS/decorrtritopv142017/ZPv142017allstd_shuftritopconstpt.dat",
                'shihqcd':"/cms/daj111/autoencoder/datafiles/shihfiles/train_QCD_1800k.dat",
                'shihqcdtest':"/cms/daj111/autoencoder/datafiles/shihfiles/test_QCD_200k.dat",
                'shihttbar':"/cms/daj111/autoencoder/datafiles/shihfiles/train_TOPS_70k.dat",
                'shihttbartest':"/cms/daj111/autoencoder/datafiles/shihfiles/test_TOPS_15k.dat",
                'shihglu':"/cms/daj111/autoencoder/datafiles/shihfiles/test_GLUI_200k.dat"
                }
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) 

class sample_2d_data:
    def __init__(self,dataset, n_samples, ntot_samples,ncol=1,tosum=False,firstev=0):


        self.dataset=dataset
        self.nepoch=0
        self.glos=0
        self.jsonsamp=None
        self.ntot_samples=ntot_samples
        self.n_samples=n_samples


        self.fileb=dataset


        ebit=False
        self.jsonsamp=(json.loads(s) for s in open(self.fileb))
        x_val=[]
        print("Populating...")
        for isamp in range(firstev): 
            next(self.jsonsamp)

        for iii in range(ntot_samples): 
            if (iii%10000)==0:
              print("Event",iii,"of",ntot_samples)
            xy=next(self.jsonsamp)
            x_val.append(xy[0])

        x_val=np.array(x_val)

        exparr=expand_array(x_val,ncol,tosum)
        exparr=np.array(exparr,dtype=np.float32)
        self.jsonsamp=torch.from_numpy(exparr)
        self.totsamp=len(self.jsonsamp)
        
    def sreturn(self):

        if (self.glos+self.n_samples>self.totsamp):
          #print(torch.randperm(self.totsamp))
          self.jsonsamp = self.jsonsamp[torch.randperm(self.totsamp)]
          self.glos=0
          self.nepoch+=1
          ebit=True

        toreturn=self.jsonsamp[self.glos:self.glos+self.n_samples]
        self.glos+=self.n_samples
        
        #toreturn = toreturn.exp()-1.0
        #toreturn /= (toreturn.sum(axis=(1,2,3)).reshape(toreturn.shape[0],1,1,1))

        return toreturn



class Autoencoder(nn.Module):
    def __init__(self,ncol=1,data_dim=6,n_hidden=5,hidden_dim=40):
        super(Autoencoder, self).__init__()
        self.ncol=ncol
        self.data_dim=data_dim
        self.conv_dim=128
        self.conv_dim1=128
        self.conv_dim2=128
        if (self.data_dim==12 or self.data_dim==48):
                self.conv_dim1=64 #not sure why
                self.conv_dim2=128
        data_interdim=self.conv_dim2*100
        data_interdim1=int(32)
        self.n_hidden=n_hidden
        self.hidden_dim=hidden_dim
        self.conv1 = nn.Sequential( 
            nn.Conv2d(self.ncol, self.conv_dim, stride=1,kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim1, stride=1,kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential( 
            nn.Conv2d(self.conv_dim1, self.conv_dim2, stride=1,kernel_size=3, padding=1)
        )

        self.fc1 = nn.Linear(data_interdim, data_interdim1)
        self.latsp = nn.Sequential(
            nn.Linear(data_interdim1, self.data_dim)
        )

        self.lnorm = nn.Sequential(
            nn.LayerNorm(data_interdim1)
        )
        self.latsp2 = nn.Linear(self.data_dim,self.data_dim)


        self.fc2 = nn.Linear(self.data_dim,data_interdim1)
        self.fc3 = nn.Linear(data_interdim1,data_interdim)
        self.t_conv1 = nn.Sequential(
            nn.Conv2d(self.conv_dim2, self.conv_dim1, stride=1, kernel_size=3, padding=1)
        )
        self.t_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(self.conv_dim1, self.conv_dim, stride=1, kernel_size=3, padding=1)
        )
        self.t_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(self.conv_dim, self.ncol, stride=1, kernel_size=3, padding=1)
        )


    def forward(self, x):



        EN = F.relu(self.conv1(x))
        EN = F.relu(self.conv2(EN))
        EN = F.relu(self.conv3(EN))
        latent = EN.view(EN.size(0), -1) 
        latent = F.relu(self.fc1(latent))
        latent = self.latsp(latent)
        DE = F.relu(self.fc2(latent))
        DE = F.relu(self.fc3(DE))
        DE = DE.view(EN.shape)  
        DE = F.relu(self.t_conv1(DE))
        DE = F.relu(self.t_conv2(DE))
        DE = self.t_conv3(DE)
        DE = F.softmax(DE.reshape(DE.size(0), -1))
        DE = DE.view(x.shape)  
  
        return DE,latent




class ABNAF(nn.Module):
    def __init__(self,data_dim=6,n_hidden=5,hidden_dim=40,flows=1):
        super(ABNAF, self).__init__()

        self.flows=flows
        self.data_dim=data_dim

        self.n_hidden=n_hidden
        self.hidden_dim=hidden_dim

        flowsvec = []
        for f in range(self.flows):
                layers = []
                for _ in range(self.n_hidden - 1):
                        layers.append(MaskedWeight(self.data_dim * self.hidden_dim, self.data_dim * self.hidden_dim, dim=self.data_dim))
                        layers.append(Tanh())
                flowsvec.append(BNAF(*([MaskedWeight(self.data_dim, self.data_dim * self.hidden_dim, dim=self.data_dim), Tanh()] + layers + [MaskedWeight(self.data_dim * self.hidden_dim, self.data_dim, dim=self.data_dim)]),res='None' if f < self.flows - 1 else None))
                if f < self.flows - 1:
                         flowsvec.append(Permutation(self.data_dim, 'flip'))



        self.bnafv=nn.Sequential(*flowsvec)


    def forward(self, x):
        return self.bnafv(x)



def compute_log_p_x(mafs):
    y_mb, log_diag_j_mb = mafs
    log_p_y_mb = torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb)).log_prob(y_mb).sum(-1)
    return log_p_y_mb + log_diag_j_mb

def compute_pdf(mafs):
    prob = torch.exp(compute_log_p_x(mafs))
    return prob

def lossae_fndiv(outs, sample,criterion,weights,fac_maf):
    allmese=[]
    ALLMSE = criterion(outs,sample)
    elemwiseMSE=ALLMSE.view(ALLMSE.size(0), -1).mean(1)
    returnval= ((elemwiseMSE/(1.0+fac_maf*weights)))
    return returnval,ALLMSE

def lossae_normmult(outs, sample,criterion,weights,perc):
    allmese=[]
    ALLMSE = criterion(outs,sample)
    elemwiseMSE=ALLMSE.view(ALLMSE.size(0), -1).mean(1)
    cweights=weights.clone().detach()
    perc=np.percentile(cweights.detach().cpu().numpy(),perc)
    #print (perc,weights.mean())
    cweights-=perc
    #cwminmax=(cweights.max()-cweights.min())
    #cweights/=cwminmax
    divval=(-1.0*cweights)
    stdev=cweights.std()
    #print(stdev)
    if (divval==0).sum()!=0 or stdev==0:
        divval=100000.
    #else:
     #   #divval/=abs(divval)
      #  divval/=abs(divval)
    #print(weights)
    #print((divval==0).sum())

    #print (divval/abs(divval))
    returnval= (elemwiseMSE*divval)
    return returnval,ALLMSE

def lossae_fnmult(outs, sample,criterion,weights,fac_maf):
    allmese=[]
    ALLMSE = criterion(outs,sample)
    elemwiseMSE=ALLMSE.view(ALLMSE.size(0), -1).mean(1)
    returnval= ((elemwiseMSE*(1.0+fac_maf*weights)))
    return returnval,ALLMSE


def lossae_simul(outs, sample,criterion,weights,fac_maf):
    allmese=[]
    ALLMSE = criterion(outs,sample)
    elemwiseMSE=ALLMSE.view(ALLMSE.size(0), -1).mean(1)
    returnval= (elemwiseMSE+fac_maf*weights)
    return returnval,ALLMSE



# --------------------
# Training
# --------------------


def train_flow(model, fulltrain, fullval, fullsig, partialtrain, fullttbar,optimizer, scheduler, args):
    model[0].train()
    model[1].train()
    #criterion =   nn.KLDivLoss(reduction='none')
    criterion = nn.MSELoss(reduction='none')
    #criterion = mean_absolute_percentage_error
    #criterion = nn.L1Loss(reduction='none')
    nch=0
    for child in model[0].children():
        nch+=1

    lossarr=    {
                "epoch":[],
                "mse":[],
                "prob":[],
                "msetrain":[],
                "probtrain":[],
                "MSEttbar":[],
                "MAFttbar":[],
                "MSEsig":[],
                "MAFsig":[]
                }
    loss0=[]
    loss1=[]

    stepsperepoch=float(args.n_tot)/float(args.batch_size)
    vmean1=0.
    vrms1=0.
    with tqdm(total=args.n_steps, desc='Start step {}; Training for {} steps'.format(args.step, args.n_steps)) as pbar:
        for _ in range(args.n_steps):
            args.step += 1
            mfrac=0.0
            dfrac=float(args.step)/float(args.n_steps)

            startmaf=0.0
            domaf=True
            if (dfrac<startmaf and (not args.simul)):
                domaf=False

            endae=0.8
            doae=True
            if (dfrac>endae and (not args.simul)):
                doae=False
                
            #if (dfrac>startmaf):
            #    mfrac=1.0
            #else:
            #    mfrac=dfrac/startmaf



            sample = fulltrain.sreturn().to('cuda:'+str(args.cuda))

            #print((sample[:,:]*sample[:,:]!=sample[:,:].max()).shape)
            ####
            #_, temp = model[0](sample)
            #temp = - compute_log_p_x(model[1](temp))
            #perc=np.percentile(temp.detach().cpu().numpy(),50)
            #sample=sample[temp<perc]
            ####


            topass  = model[0](sample)           
            outs, lats = topass
            zerosupp=False
            
            #print(cond)
            #print(sample[0].numel())
            if zerosupp:
                cond=(sample>10e-10)
                condsum=cond.sum(axis=2).sum(axis=2).reshape(args.batch_size)
                nnonzero=condsum.float()/float(sample[0].numel())
            else:
                cond=1.0
                nnonzero=1.0
            #print(nnonzero)
            if not args.mseinpdf:
                mafs  = model[1](lats)
                lossmafall = - compute_log_p_x(mafs)
                normpdfall = compute_pdf(mafs)
                normpdfall -= normpdfall.mean().tolist()
            else:
                lossmafall = 1.0
                normpdfall = 1.0

            gap=0
            normmullt=args.normmullt
            doperc=((args.percentile>0) and (not normmullt)) 
            if (doperc):
                #pfracl=100.0-((dfrac)*(100.0-(args.percentile-gap)))
                #pfrach=100.0-((dfrac)*(100.0-(args.percentile+gap)))
                pfracl=(args.percentile-gap)
                pfrach=(args.percentile+gap)
                perc=np.percentile(lossmafall.detach().cpu().numpy(),pfracl)
                perchigh=np.percentile(lossmafall.detach().cpu().numpy(),pfrach)

            if (args.mult):
                bothaeloss = lossae_fnmult(outs*cond, sample, criterion,lossmafall,mfrac*args.fac_maf)
            elif(args.simul):
                bothaeloss = lossae_simul(outs*cond, sample, criterion,lossmafall,mfrac*args.fac_maf)
            elif(normmullt):
                bothaeloss = lossae_normmult(outs*cond, sample, criterion, lossmafall,args.percentile)
                #print (bothaeloss[0])
            else:
                bothaeloss = lossae_fndiv(outs*cond, sample, criterion,lossmafall,mfrac*args.fac_maf)

            if args.mseinpdf:
                tomerge=bothaeloss[0].reshape((bothaeloss[0].shape)[0],1)
                latplusmse = torch.cat((lats,tomerge),axis=1)
                mafs  = model[1](latplusmse)
                lossmafall=- compute_log_p_x(mafs)
            lossmafallnorm=lossmafall-lossmafall.mean()
            aefac=100000.0
            #print(bothaeloss[0].shape,lossmafall.shape)

            #print (perc)
            #print (lossmafall>perc)
            #lossmafall=lossmafall[lossmafall<perc]
            pdfweight=True
            if (doperc):
                 bothaelosslow = [(bothaeloss[0]/nnonzero)[lossmafall<perc].mean(),bothaeloss[1][lossmafall<perc].mean()]
                 bothaelosshigh = [(bothaeloss[0]/nnonzero)[lossmafall>perchigh].mean(),bothaeloss[1][lossmafall>perchigh].mean()]
            
            else:               
                 #print (bothaeloss[0].shape,bothaeloss[1].shape)
                 bothaelosslow = [(bothaeloss[0]/nnonzero).mean(),(bothaeloss[1]).mean()]
                 bothaelosshigh = [0,0]

            #print("TEST")
            #print(lossmafall[lossmafall<perc])
            #print(lossmafall[lossmafall>perc])
            #print(aefac*bothaeloss[0][lossmafall>perc].mean())
            #print(aefac*bothaeloss[0][lossmafall<perc].mean())
            lossaeonly = aefac*bothaelosslow[1]
            lossae = aefac*bothaelosslow[0]
            lossaehigh = aefac*bothaelosshigh[0]
            lossmafbare = lossmafall.mean()

            loss = lossae
            #print(loss)
            losshigh=0.0
            if (doperc):
                losshigh = 1.0/lossaehigh

            if (not args.nofreeze):  #AE does not affect MAF etc
                freezelayer(model[1])

            if doae:
                if(args.simul):
                        optimizer[1].zero_grad()
                optimizer[0].zero_grad()

                (loss+losshigh).backward(retain_graph=True)
                if(normmullt):
                        torch.nn.utils.clip_grad_norm_(model[0].parameters(), max_norm=20.0)
        
                optimizer[0].step()
                if(args.simul):
                        optimizer[1].step()
                        torch.nn.utils.clip_grad_norm_(model[1].parameters(), max_norm=0.1)


            if (not args.nofreeze):
                releaselayer(model[1])

            if(not args.simul and domaf): 
                    if (not args.nofreeze):
                        freezelayer(model[0])

                    optimizer[1].zero_grad()
                    lossmafbare.backward(retain_graph=False)
                    torch.nn.utils.clip_grad_norm_(model[1].parameters(), max_norm=0.1)
                    optimizer[1].step()

                    if (not args.nofreeze):
                         releaselayer(model[0])

                
            loss0.append(loss.tolist())
            loss1.append(lossmafbare.tolist())

            pbar.set_postfix(Lae='{:.4f}'.format(lossae),Lhigh='{:.4f}'.format(lossaehigh), LmafB='{:.4f}'.format(lossmafbare),L='{:.4f}'.format(loss))
            pbar.update()

            if (args.step%(int(stepsperepoch)+1))==0:

                print ("...")
                print ("Epoch",round(args.step/stepsperepoch),"end...")
                model[0].eval()
                model[1].eval()
                nvalsteps=fullval.ntot_samples/fullval.n_samples
                vlossarr0 = []
                vlossarr1 = []
                vlats = []
                trainlossarr0 = []
                trainlossarr1 = []
                slossarr0 = []
                slossarr1 = []
                tlossarr0 = []
                tlossarr1 = []

                for vv in range(int(nvalsteps)):    
                        valsample = fullval.sreturn().to('cuda:'+str(args.cuda))
                        valeval=evaluate(model, valsample, args)

                        vlossarr1+=np.array(valeval)[:,0].tolist()
                        vlossarr0+=(aefac*np.array(valeval)[:,1]).tolist()
                        
                        templat=np.array((np.array(valeval)[:,-1])[0])
                        #print(templat.shape)
                        templat=templat.reshape(1,templat.shape[0])
                        #print(templat.shape)
                        if (vv==0):
                                vlats=templat
                        else:
                                vlats=np.append(vlats,templat,axis=0)
                        #print(vlats.shape)
                        trainsample = partialtrain.sreturn().to('cuda:'+str(args.cuda))
                        traineval=evaluate(model, trainsample, args)

                        trainlossarr1+=np.array(traineval)[:,0].tolist()
                        trainlossarr0+=(aefac*np.array(traineval)[:,1]).tolist()

                        if sig_potential_or_sampling_fn!=None:
                                sigsample = fullsig.sreturn().to('cuda:'+str(args.cuda))
                                sigeval=evaluate(model, sigsample, args)

                                slossarr1+=np.array(sigeval)[:,0].tolist()
                                slossarr0+=(aefac*np.array(sigeval)[:,1]).tolist()
                        if ttbar_potential_or_sampling_fn!=None:
                                ttbarsample = fullttbar.sreturn().to('cuda:'+str(args.cuda))
                                ttbareval=evaluate(model, ttbarsample, args)

                                tlossarr1+=np.array(ttbareval)[:,0].tolist()
                                tlossarr0+=(aefac*np.array(ttbareval)[:,1]).tolist()

                validation_loss0 = sum(vlossarr0) / len(vlossarr0)   
                validation_loss1 = sum(vlossarr1) / len(vlossarr1) 
                #print ("Shapes")
                print (vlats.shape)

                latmean=np.mean(vlats,axis=1)
                latrms=np.std(vlats,axis=1)/latmean
                train_loss0 = sum(trainlossarr0) / len(trainlossarr0)   
                train_loss1 = sum(trainlossarr1) / len(trainlossarr1) 

                vmean0=np.mean(np.array(np.array(vlossarr0)))
                vmean1=np.mean(np.array(np.array(vlossarr1)))
                vrms0=np.std(vlossarr0)/abs(vmean0)
                vrms1=np.std(vlossarr1)/abs(vmean1)
                if sig_potential_or_sampling_fn!=None:

                        sig_loss0 = sum(slossarr0) / len(slossarr0)   
                        sig_loss1 = sum(slossarr1) / len(slossarr1)
                        print ("Sig loss, MSE",sig_loss0," , MAF",sig_loss1)   

                        srms0=np.std(slossarr0)/abs(sig_loss0) #math.sqrt(np.mean(np.array(np.array(slossarr0) - sig_loss0)**2))
                        srms1=np.std(slossarr1)/abs(sig_loss1)

                        #print (np.array(slossarr1),sig_loss1)
                        #print (np.array(slossarr0),sig_loss0)
                        #print (srms1**2,vrms1**2)
                        MSEks=scipy.stats.ks_2samp(slossarr0,vlossarr0)
                        MAFks=scipy.stats.ks_2samp(slossarr1,vlossarr1)
                        try:
                                MSEsig=(sig_loss0-validation_loss0)/math.sqrt(srms0**2+vrms0**2)
                                MAFsig=(sig_loss1-validation_loss1)/math.sqrt(srms1**2+vrms1**2)
                        except:
                                print("sig failed")
                                print(slossarr0)
                                print(slossarr1)
                                MSEsig=0.
                                MAFsig=0.


                        lossarr["MSEsig"].append(float(MSEks.statistic))
                        lossarr["MAFsig"].append(float(MAFks.statistic))
                        print ("Significance, MSE",MSEsig," , MAF",MAFsig)
                        print ("KS, MSE",MSEks.statistic," , MAF",MAFks.statistic)
                        print ("means, MSE",vmean0," , MAF",vmean1)
                        print ("rmss, MSE",vrms0," , MAF",vrms1)
                        print ("lat mean",np.mean(latmean),"rms",np.mean(latrms))



                if ttbar_potential_or_sampling_fn!=None:
                        ttbar_loss0 = sum(tlossarr0) / len(tlossarr0)   
                        ttbar_loss1 = sum(tlossarr1) / len(tlossarr1)
                        print ("ttbar loss, MSE",ttbar_loss0," , MAF",ttbar_loss1)   
                        trms0=np.std(tlossarr0)/abs(ttbar_loss0)
                        trms1=np.std(tlossarr1)/abs(ttbar_loss1)
                        try:
                                MSEttbar=(ttbar_loss0-validation_loss0)/math.sqrt(trms0**2+vrms0**2)
                                MAFttbar=(ttbar_loss1-validation_loss1)/math.sqrt(trms1**2+vrms1**2)
                        except:
                                print("sig (ttbar) failed")
                                print(tlossarr0)
                                print(tlossarr1)
                                MSEttbar=0.
                                MAFttbar=0.

                        MSEttks=scipy.stats.ks_2samp(tlossarr0,vlossarr0)
                        MAFttks=scipy.stats.ks_2samp(tlossarr1,vlossarr1)

                        lossarr["MSEttbar"].append(float(MSEttbar))
                        lossarr["MAFttbar"].append(float(MAFttbar))
                        print ("Significance (ttbar), MSE",MSEttbar," , MAF",MAFttbar)
                        print ("KS, MSE",MSEttks.statistic," , MAF",MAFttks.statistic)


             
                print ("Val loss, MSE",validation_loss0," , MAF",validation_loss1)
                print ("Val loss, logMSE",math.log(validation_loss0/aefac))
                print ("Train loss, MSE",train_loss0," , MAF",train_loss1)

                lossarr["mse"].append(float(validation_loss0))
                lossarr["prob"].append(float(validation_loss1))
                lossarr["msetrain"].append(float(train_loss0))
                lossarr["probtrain"].append(float(train_loss1))
                lossarr["epoch"].append(round(args.step/stepsperepoch))
                print ("...Updating scheduler 0...")
                scheduler[0].step(validation_loss0)
                for param_group in optimizer[0].param_groups:
                        print(param_group["lr"])
                if(not args.simul): 
                    print ("...Updating scheduler 1...")
                    scheduler[1].step(100.+validation_loss1)
                    for param_group in optimizer[1].param_groups:
                        print(param_group["lr"])
                loss0=[]
                loss1=[]
                model[0].train()
                model[1].train()

            if args.step % args.log_interval == 0:
                print("SAVE",args.output_dir+"/checkpoint.pt")
                torch.save({'step': args.step,
                            'state_dict': model[0].state_dict()},
                           './latest0'+args.dataset+args.extex+'.pt')
                torch.save({'step': args.step,
                            'state_dict': model[1].state_dict()},
                           './latest1'+args.dataset+args.extex+'.pt')
                print("ratio",float(lossae.item()/lossmafbare.item()))
    return lossarr


def evaluate(model, zz, args):
    allreturn=[]
    arrsize=1
    if not args.tosum:
      arrsize=args.ncol
    topass  = model[0](zz)
          
    outs, lats = topass
    criterion = nn.MSELoss(reduction='none')
    #criterion = nn.KLDivLoss(reduction='none')
    #criterion = nn.CrossEntropyLoss(reduction='none')
    


    if args.mseinpdf:
                mafs=0.0
    else:
                modelout = model[1](lats)
                mafs = -1*compute_log_p_x(modelout)



    if (args.mult):
                bothaeloss = lossae_fnmult(outs, zz, criterion, mafs, args.fac_maf)
    elif(args.simul):
                bothaeloss = lossae_simul(outs, zz, criterion, mafs, args.fac_maf)
    else:
                bothaeloss = lossae_fndiv(outs, zz, criterion, mafs, args.fac_maf)




    if args.mseinpdf:
                tomerge=bothaeloss[0].reshape((bothaeloss[0].shape)[0],1)
                latplusmse = torch.cat((lats,tomerge),axis=1)
                modelout = model[1](latplusmse)
                mafs  = -1*compute_log_p_x(modelout)
                pdfs = compute_pdf(modelout)
    else:
                pdfs = compute_pdf(modelout)


    aefac=100000.0
    mses = bothaeloss[0]


    allreturn=[]
    mselist=[]

    for imaf,maf in enumerate(mafs):  
               allreturn.append([maf.tolist()])
               allreturn[imaf].append(abs(mses[imaf]).mean().tolist())
               allreturn[imaf].append(0.0)
               allreturn[imaf].append(0.0)
               allreturn[imaf].append(pdfs[imaf].tolist())
               allreturn[imaf].append(np.array(lats[imaf].tolist(),dtype=np.float32))
    return allreturn


if __name__ == '__main__':
    args = parser.parse_args()
    args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else os.path.join(args.output_dir, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    if args.train:
        if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir)

    print (torch.cuda.is_available())
    torch.manual_seed(args.seed)
    AEsize=1
    if not args.tosum:
      AEsize=args.ncol
    model = Autoencoder(ncol=AEsize,data_dim=args.data_dim,n_hidden=args.n_hidden,hidden_dim=args.hidden_dim).to('cuda:'+str(args.cuda))
    totdim=args.data_dim
    if (args.mseinpdf):
        totdim+=1
    modelB = ABNAF(data_dim=totdim,n_hidden=args.n_hidden,hidden_dim=args.hidden_dim,flows=args.flows).to('cuda:'+str(args.cuda))
    if args.restore_file:
            FRsplit=(args.restore_file).split(",")
            model0_checkpoint = torch.load((FRsplit)[0], map_location='cuda:'+str(args.cuda))
            model.load_state_dict(model0_checkpoint['state_dict'],strict=True)
            model1_checkpoint = torch.load((FRsplit)[1], map_location='cuda:'+str(args.cuda))
            modelB.load_state_dict(model1_checkpoint['state_dict'],strict=True)


    # save settings
    config = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__)) + \
             'Num trainable params: {:,.0f}\n\n'.format(sum(p.numel() for p in model.parameters())) + \
             'Model:\n{}'.format(model)
    config_path = os.path.join(args.output_dir, 'config.txt')
    if not os.path.exists(config_path):
        with open(config_path, 'a') as f:
            print(config, file=f)

    # setup data -- density to estimate/match
    args.samples = not (args.dataset.startswith('u') and len(args.dataset) == 2)
    print ("args.tosum",args.tosum)

    if args.train:

        potential_or_sampling_fn = sample_2d_data(datamap[args.dataset],args.batch_size,args.n_tot,ncol=args.ncol,tosum=args.tosum,firstev=args.n_val)
        partialpotential_or_sampling_fn = sample_2d_data( datamap[args.dataset],args.batch_size,args.n_val,ncol=args.ncol,tosum=args.tosum,firstev=args.n_val)
        val_potential_or_sampling_fn = sample_2d_data( datamap[args.dataset],int(args.batch_size/8),args.n_val,ncol=args.ncol,tosum=args.tosum,firstev=0)
        sig_potential_or_sampling_fn = sample_2d_data( datamap["shihglu"],args.batch_size,args.n_val,ncol=args.ncol,tosum=args.tosum,firstev=0)
        spec = "shihttbar"
        if args.dataset=="shihttbar":
                spec = "shihqcd"
        ttbar_potential_or_sampling_fn = sample_2d_data( datamap[spec],args.batch_size,args.n_val,ncol=args.ncol,tosum=args.tosum,firstev=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr0, weight_decay=0.0)
        #optimizerB = torch.optim.Adam(modelB.parameters(), lr=args.lr1, weight_decay=0)
        optimizerB = Adam(modelB.parameters(), lr=args.lr1, amsgrad=True, polyak=0.998)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.lr_patience, cooldown=4, verbose=True)
        #schedulerB = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerB, factor=args.lr_decay, patience=args.lr_patience, cooldown=10, verbose=True)
        schedulerB = ReduceLROnPlateau(optimizerB, factor=args.lr_decay, patience=args.lr_patience, cooldown=4,min_lr=0.0, early_stopping=100,threshold_mode='abs', verbose=True)



        print("Train!")

        lossarr=train_flow([model,modelB], potential_or_sampling_fn, val_potential_or_sampling_fn, sig_potential_or_sampling_fn, partialpotential_or_sampling_fn,ttbar_potential_or_sampling_fn,[optimizer,optimizerB], [scheduler,schedulerB], args)
        print (lossarr)
        trainfile = ROOT.TFile("trainfile"+args.dataset+args.extex+".root", "RECREATE")
        mg=ROOT.TMultiGraph("mg","mg")
        mgtrain=ROOT.TMultiGraph("mgtrain","mgtrain")
        grapharr={}
        print(len(lossarr["mse"]),array.array('d',lossarr["epoch"]),array.array('d',lossarr["mse"]))
        gprob=ROOT.TGraph(len(lossarr["prob"]),array.array('d',lossarr["epoch"]),array.array('d',lossarr["prob"]))
        gmse=ROOT.TGraph(len(lossarr["mse"]),array.array('d',lossarr["epoch"]),array.array('d',lossarr["mse"]))

        gprob.SetMarkerStyle(21)
        gprob.SetMarkerColor(3)

        gmse.SetMarkerStyle(21)
        gmse.SetMarkerColor(2)


        gprobtrain=ROOT.TGraph(len(lossarr["probtrain"]),array.array('d',lossarr["epoch"]),array.array('d',lossarr["probtrain"]))
        gmsetrain=ROOT.TGraph(len(lossarr["msetrain"]),array.array('d',lossarr["epoch"]),array.array('d',lossarr["msetrain"]))

        gprobtrain.SetMarkerStyle(21)
        gprobtrain.SetMarkerColor(3)

        gmsetrain.SetMarkerStyle(21)
        gmsetrain.SetMarkerColor(2)

        if (sig_potential_or_sampling_fn!=None):
                mg1=ROOT.TMultiGraph()

                print(len(lossarr["MSEsig"]),array.array('d',lossarr["epoch"]),array.array('d',lossarr["MSEsig"]))
                gMSEsig=ROOT.TGraph(len(lossarr["MSEsig"]),array.array('d',lossarr["epoch"]),array.array('d',lossarr["MSEsig"]))
                gMAFsig=ROOT.TGraph(len(lossarr["MAFsig"]),array.array('d',lossarr["epoch"]),array.array('d',lossarr["MAFsig"]))
                
                gMSEsig.SetMarkerStyle(21)
                gMSEsig.SetMarkerColor(3)

                gMAFsig.SetMarkerStyle(21)
                gMAFsig.SetMarkerColor(2)

                mg1.Add(gMAFsig)
                mg1.Add(gMSEsig)

                gMAFsig.Write("gMAFsig")
                gMSEsig.Write("gMSEsig")
                mg1.Write("mg1")

        mg.Add(gprob)
        mg.Add(gmse)
        gprob.Write("grprob")
        gmse.Write("gmse")
        mg.Write("mg")

        mgtrain.Add(gprob)
        mgtrain.Add(gmse)
        gprobtrain.Write("grprobtrain")
        gmsetrain.Write("gmsetrain")
        mgtrain.Write("mgtrain")
    if args.plot:
        plot(model, potential_or_sampling_fn, args)



    if args.evaluate:

        model.eval()
        modelB.eval()
        spldata=(args.dataset).split(",")
        dobdt=True
        probs = {}
        probstrain = {}
        probsval = {}
        alleff = {}
        alleffAE = {}
        alleffPR = {}
        alleffPRkde = {}
        alleffCOMB = {}
        alleffBDT = {}
        alleffBDTkde = {}
        for spl in spldata:
                probs[spl]=[]
                probstrain[spl]=[]
                probsval[spl]=[]
        ntots=[args.n_tot,2*args.n_tot,args.n_tot]
        nbatchs=[int(ntots[0]/args.batch_size),int(ntots[1]/args.batch_size),int(ntots[2]/args.batch_size)]
        totot=np.sum(ntots)
        curspl=[]
        
        with torch.no_grad():
                for spl in spldata:

                        curspl.append(sample_2d_data(datamap[spl],args.batch_size,totot,ncol=AEsize,tosum=args.tosum))
                        print("Testset")
                        print("")
                        for nb in range(nbatchs[0]):
                                print("\t",nb,"/",nbatchs[0], end='\r')
                                probs[spl].extend(copy.deepcopy(evaluate([model,modelB], curspl[-1].sreturn().to('cuda:'+str(args.cuda)), args)))
                                

                        print("Trainset")
                        print("")
                        for nb in range(nbatchs[1]):
                                print("\t",nb,"/",int(nbatchs[1]), end='\r')
                                probstrain[spl].extend(copy.deepcopy(evaluate([model,modelB], curspl[-1].sreturn().to('cuda:'+str(args.cuda)), args)))


                        print("Valset")
                        print("")
                        for nb in range(nbatchs[2]):
                                print("\t",nb,"/",int(nbatchs[2]), end='\r')
                                probsval[spl].extend(copy.deepcopy(evaluate([model,modelB], curspl[-1].sreturn().to('cuda:'+str(args.cuda)), args)))
        for spl in spldata: 
 
                print (spl,(sum(np.array(probs[spl])[:,0])/len(np.array(probs[spl])[:,0]),(sum(np.array(probs[spl])[:,1])/len(np.array(probs[spl])[:,1]))))
                print (spl,(sum(np.array(probstrain[spl])[:,0])/len(np.array(probstrain[spl])[:,0]),(sum(np.array(probstrain[spl])[:,1])/len(np.array(probstrain[spl])[:,1]))))
        ROOT.TMVA.Tools.Instance()


        vlossarr0 = []
        vlossarr1 = []
        slossarr0 = []
        slossarr1 = []
        aefac=100000.0
        vlossarr1+=np.array(probstrain[spldata[1]])[:,0].tolist()
        vlossarr0+=(aefac*np.array(probstrain[spldata[1]])[:,1]).tolist()


        vmean0=np.mean(np.array(np.array(vlossarr0)))
        vmean1=np.mean(np.array(np.array(vlossarr1)))


        validation_loss0 = sum(vlossarr0) / len(vlossarr0)   
        validation_loss1 = sum(vlossarr1) / len(vlossarr1) 

        vrms0=np.std(vlossarr0)/abs(vmean0)
        vrms1=np.std(vlossarr1)/abs(vmean1)


        slossarr1+=np.array(probstrain[spldata[0]])[:,0].tolist()
        slossarr0+=(aefac*np.array(probstrain[spldata[0]])[:,1]).tolist()


        smean0=np.mean(np.array(np.array(slossarr0)))
        smean1=np.mean(np.array(np.array(slossarr1)))


        sig_loss0 = sum(slossarr0) / len(slossarr0)   
        sig_loss1 = sum(slossarr1) / len(slossarr1)
        #print (slossarr0)
        #print (np.std(slossarr0))
        srms0=np.std(slossarr0)/abs(smean0)
        srms1=np.std(slossarr1)/abs(smean1)
        try:
                MSEsig=(sig_loss0-validation_loss0)/math.sqrt(srms0**2+vrms0**2)
                MAFsig=(sig_loss1-validation_loss1)/math.sqrt(srms1**2+vrms1**2)
        except:
                MSEsig=0.
                MAFsig=0.

        print ("Significance, MSE",MSEsig," , MAF",MAFsig) #Is same as training?
        print ("QCD Means, MSE",validation_loss0," , MAF",validation_loss1) #Is same as training?
        print ("SIG Means, MSE",sig_loss0," , MAF",sig_loss1) #Is same as training?
        trainlats=[]
        for curprob in probsval[spldata[1]]:
          trainlats.append(np.array(curprob[5],dtype=np.float32))

        testlats=[]
        for curprob in probs[spldata[1]]:
          testlats.append(np.array(curprob[5],dtype=np.float32))

        trainlats =np.array(trainlats)
        trainlatsT =(trainlats).T

        testlats =np.array(testlats)
        testlatsT =(testlats).T

        print("KDE train")
        kdesamps = int(args.n_tot)
        gauskde = scipy.stats.gaussian_kde(trainlatsT)
        kdecov = gauskde.covariance_factor()
        bw = kdecov * trainlats.std()
        print("bw",bw)
        sklearnkde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(trainlatsT)

        print("KDE sample")
        kderersamp = gauskde.resample(size=(int(kdesamps)))
        print(kderersamp.shape)
        print("KDE pdf")
        kdepdfs = gauskde.pdf(kderersamp)

        print (kderersamp.shape)
        print (testlats.shape)
        kdelogpdfs = -1*gauskde.logpdf(kderersamp)

        kdelogpdfstest = -1*gauskde.logpdf(testlatsT)
        sklearnkdelogpdfstest = -1*sklearnkde.score_samples(testlats.T)

        kdepdfs = kdepdfs.T
        kderersamp = kderersamp.T

        sklearnkderesamp=sklearnkde.sample(100)
        print ("sklearnkderesampshape",sklearnkderesamp.shape)
        print (kdelogpdfstest.shape)
        print (sklearnkdelogpdfstest.shape)
        print (kdelogpdfstest.mean())
        print (sklearnkdelogpdfstest.mean())
        print("KDE done")

        root_file = ROOT.TFile("testttreefile.root", "RECREATE")

        tree = ROOT.TTree("tree", "tree")

        mse = np.empty((1), dtype="float32")
        pdf = np.empty((1), dtype="float32")
        kdepdf = np.empty((1), dtype="float32")
        sigv = np.empty((1), dtype="float32")

        tree.Branch("mse", mse, "mse/F")
        tree.Branch("pdf", pdf, "pdf/F")
        tree.Branch("kdepdf", kdepdf, "kdepdf/F")
        tree.Branch("sigv", sigv, "sigv/F")



        ent = 0
        with open("Output_"+args.extex+"_L"+str(args.data_dim)+"_"+spldata[0]+".csv","w") as tempf:
                 for curprob in probstrain[spldata[0]]:
                         tempf.write(str(curprob[1]))
                         for cp in curprob[5]:
                                tempf.write(","+str(cp))
                         tempf.write("\n")
                         mse[0] = curprob[1]
                         pdf[0] = curprob[0]
                         temp=gauskde.pdf(np.array(curprob[5]).T)
                         kdepdf[0] = -1*gauskde.logpdf(np.array(curprob[5]).T)
                         

                         sigv[0] = 1.0
                         tree.Fill()
                         ent += 1
    
        with open("Output_"+args.extex+"_L"+str(args.data_dim)+"_"+spldata[1]+".csv","w") as tempf:
                  for curprob in probstrain[spldata[1]]:

                        tempf.write(str(curprob[1]))
                        for cp in curprob[5]:
                                tempf.write(","+str(cp))
                        tempf.write("\n")
                        mse[0] = curprob[1]
                        pdf[0] = curprob[0]
                        temp=gauskde.pdf(np.array(curprob[5]).T)
                        kdepdf[0] = -1*gauskde.logpdf(np.array(curprob[5]).T)

                        sigv[0] = 0.0
                        tree.Fill()
                        ent += 1

        print ("treeentries",ent)
        tree.Write()
    
        tfile=ROOT.TFile("curfeval"+args.dataset.replace(",","")+args.extex+".root","recreate")

        X={}



        if(dobdt):
                reader={}
                factory={}
                for pdftype in ["pdf","kdepdf"]:
                        print()
                        print()                   
                        print(pdftype)
                        print()
                        print()
                        fout = ROOT.TFile("test.root","RECREATE")
                        tmva = ROOT.TFile("tmva.root","RECREATE")
                
                        exfold="dataset"+args.dataset.replace(",","")+args.extex+pdftype
                        dataloader = ROOT.TMVA.DataLoader(exfold)

                        factory[pdftype] = ROOT.TMVA.Factory("TMVAClassification", fout,
                            ":".join([
                                "!V",
                                "!Silent",
                                "Color",
                                "DrawProgressBar",
                                "Transformations=I;D;P;G,D",
                                "AnalysisType=Classification"]
                                     ))


                        dataloader.AddVariable("mse","F")
                        dataloader.AddVariable(pdftype,"F") 
                        print (tree,pdftype)
                        dataloader.AddSignalTree(tree)
                        dataloader.AddBackgroundTree(tree)

                        sigCut = ROOT.TCut("sigv > 0.5")
                        bgCut = ROOT.TCut("sigv <= 0.5")

                        dataloader.PrepareTrainingAndTestTree(sigCut,   # signal events
                                                   bgCut,    # background events
                                                   ":".join([
                                                   "nTrain_Signal=0",
                                                   "nTrain_Background=0",
                                                   "SplitMode=Random",
                                                   "NormMode=NumEvents",
                                                   "!V"
                                                   ]))




                        method = factory[pdftype].BookMethod(dataloader,ROOT.TMVA.Types.kBDT, "BDT"+pdftype,
                                        ":".join([
                                        "!H",
                                        "!V",
                                        "NTrees=250",
                                        "nEventsMin=150",
                                        "MaxDepth=3",
                                        "BoostType=AdaBoost",
                                        "AdaBoostBeta=0.5",
                                        "SeparationType=GiniIndex",
                                        "nCuts=20",
                                        "PruneMethod=NoPruning",
                                        ]))

                        factory[pdftype].TrainAllMethods()
                        factory[pdftype].TestAllMethods()
                        factory[pdftype].EvaluateAllMethods()


                        reader[pdftype] = ROOT.TMVA.Reader()
                        import array

                        reader[pdftype].AddVariable("mse",mse)
                        reader[pdftype].AddVariable(pdftype,pdf)
                        if(dobdt):
                                reader[pdftype].BookMVA("BDT"+pdftype,exfold+"/weights/TMVAClassification_BDT"+pdftype+".weights.xml")
                        root_file1 = ROOT.TFile("testttreefile"+pdftype+".root", "RECREATE")
                        treepost = ROOT.TTree("treepost", "treepost")

                        treepost.Branch("mse", mse, "mse/F")

                        treepost.Branch(pdftype, pdf, pdftype+"/F")
                        treepost.Branch("sigv", sigv, "sigv/F")
                        bdt = np.empty((1), dtype="float32")
                        treepost.Branch("bdt", bdt, "bdt/F")
                        bdtkde = np.empty((1), dtype="float32")
                        treepost.Branch("bdtkde", bdtkde, "bdtkde/F")


        nlat=len(probs[spldata[0]][1][5])
        print("nlat",nlat)
        latarr={}
        
        for ip,curprob in enumerate(probs[spldata[0]]):

          mse[0] = curprob[1]
          pdf[0] = curprob[0]
          kdepdf[0] = -1*gauskde.logpdf(np.array(curprob[5]).T)
          if(dobdt):
                bdtOutput = reader["pdf"].EvaluateMVA("BDTpdf")
                bdtOutputkde = reader["kdepdf"].EvaluateMVA("BDTkdepdf")
          else:
                bdtOutput = 1.0
                bdtOutputkde = 1.0

          probs[spldata[0]][ip][2] = (bdtOutput)
          probs[spldata[0]][ip][3] = (bdtOutputkde)
          sigv[0] = 1.0
          bdt[0] = bdtOutput
          bdtkde[0] = bdtOutputkde

          probs[spldata[0]][ip].append(kdepdf[0])

          treepost.Fill()
        for ip,curprob in enumerate(probs[spldata[1]]):

          mse[0] = curprob[1]
          pdf[0] = curprob[0]
          kdepdf[0] = -1*gauskde.logpdf(np.array(curprob[5]).T)

          if(dobdt):
                bdtOutput = reader["pdf"].EvaluateMVA("BDTpdf")
                bdtOutputkde = reader["kdepdf"].EvaluateMVA("BDTkdepdf")
          else:
                bdtOutput = 1.0
                bdtOutputkde = 1.0
          probs[spldata[1]][ip][2] = (bdtOutput)    
          probs[spldata[1]][ip][3] = (bdtOutputkde)   

          sigv[0] = 0.0
          bdt[0] = bdtOutput
          bdtkde[0] = bdtOutputkde

          probs[spldata[1]][ip].append(kdepdf[0])

          treepost.Fill()
        treepost.Write()

        nspl=0
        allplots=[]
        for spl in spldata:
                nlat=len(probs[spldata[0]][1][5])
                canv=ROOT.TCanvas("canv_lat"+spl,"canv_lat"+spl)
                if spl==spldata[0]:

                        rmsvalhist=(np.array(probs[spl])[:,0]).std()
                        meanvalhist=np.mean(np.array(probs[spl])[:,0])
                        rmsvalkdehist=(np.array(probs[spl])[:,6]).std()
                        meanvalkdehist=np.mean(np.array(probs[spl])[:,6])

                        minmse=min(np.array(probs[spl])[:,1])
                        maxmse=max(np.array(probs[spl])[:,1])
                
                plottingtonpdf=ROOT.TH1F("pdf"+spl,"pdf"+spl,200,-2.0,2.0)
                plottingtonprob=ROOT.TH1F("prob"+spl,"prob"+spl,200,meanvalhist-15*rmsvalhist,meanvalhist+15*rmsvalhist)
                plottingtonprobkde=ROOT.TH1F("probkde"+spl,"probkde"+spl,200,meanvalkdehist-15*rmsvalkdehist,meanvalkdehist+15*rmsvalkdehist)
                plottingtonlogmse=ROOT.TH1F("logmse"+spl,"logmse"+spl,200,math.log(minmse*0.8),math.log(maxmse*1.2))
                plottingtonmse=ROOT.TH1F("mse"+spl,"mse"+spl,200,minmse*0.8,maxmse*1.2)
                msevprob=ROOT.TH2F("msevprob"+spl,"msevprob"+spl,50,min(np.array(probs[spl])[:,1])*0.8,max(np.array(probs[spl])[:,1])*1.2,50,meanvalhist-10*rmsvalhist,meanvalhist+10*rmsvalhist)
                latarr=[]
                latarrpdf=[]
                latarrkdepdf=[]
                latarr1d=[]
                for nn1 in range(nlat):

                        for nn2 in range(nlat):
                                if nn1!=nn2:
                                        latvals=np.array(probs[spl])[:,5]
                                        minmaxnn1=[999.,-999.]
                                        minmaxnn2=[999.,-999.]
                                        for clat in latvals:
                                                minmaxnn1[0]=min(minmaxnn1[0],clat[nn1])
                                                minmaxnn1[1]=max(minmaxnn1[1],clat[nn1])
                                                minmaxnn2[0]=min(minmaxnn2[0],clat[nn2])
                                                minmaxnn2[1]=max(minmaxnn2[1],clat[nn2])

                                        latarr.append(ROOT.TH2F(spl+"lat"+str(nn1)+"vslat"+str(nn2),"lat"+str(nn1)+"vslat"+str(nn2),50,minmaxnn1[0],minmaxnn1[1],50,minmaxnn2[0],minmaxnn2[1]))
                        latarrpdf.append(ROOT.TH1F(spl+"pdflat"+str(nn1),"pdflat"+str(nn1),20,minmaxnn1[0],minmaxnn1[1]))
                        latarrkdepdf.append(ROOT.TH1F(spl+"kdepdflat"+str(nn1),"kdepdflat"+str(nn1),20,minmaxnn1[0],minmaxnn1[1]))
                        latarr1d.append(ROOT.TH1F(spl+"lat"+str(nn1),"lat"+str(nn1),20,minmaxnn1[0],minmaxnn1[1]))
                        latarrkdepdf[-1].Sumw2()
                        latarrpdf[-1].Sumw2()
                        latarr1d[-1].Sumw2()

                alllats=[]
                means=[]
                rmss=[]
                minmaxs=[]
                minmaxdist=[]
                means=[]
                #covs=[]
                curlats=np.array(probs[spl])[:,5]
                for ilat in range(nlat): 
                        nlatsplit=[]    
                        for cc in curlats:
                                nlatsplit.append(cc[ilat])

                        rmsval=rms(np.array(nlatsplit),-1)
                        meanval=np.mean(np.array(nlatsplit))
                        means.append(meanval)
                        rmss.append(rmsval)
                        minmaxs.append([min(nlatsplit),max(nlatsplit)])

                        minmaxdist.append([meanval-1.0*rmsval,meanval+1.0*rmsval])
                print (minmaxs)
                kdestd=np.std(kdepdfs)
                kdemean=np.mean(kdepdfs)
                print("kdemean,kdestd",kdemean,kdestd)
                kdeslice=[2*kdemean,math.inf] #Avoid infinite weights, incur bias
                for lp in probs[spl]:

                      plottingtonprob.Fill(lp[0])
                      plottingtonprobkde.Fill(lp[6])

                      plottingtonmse.Fill(lp[1])
                      plottingtonlogmse.Fill(math.log(lp[1]))
                      plottingtonpdf.Fill(lp[3])

                      msevprob.Fill(lp[1],lp[0])
                      alllats.append(lp[5])


                      ihist=0
                      for nn1 in range(nlat):
                            for nn2 in range(nlat):
                                   if nn1!=nn2:

                                                latarr[ihist].Fill(lp[5][nn1],lp[5][nn2])
                                                ihist+=1

                      ihist=0
                      curpdf=gauskde.pdf(np.array(lp[5]).T)[0]
                      if(curpdf>kdeslice[0]):
                                for nn1 in range(nlat):
                                        latarr1d[ihist].Fill(lp[5][nn1])
                                        ihist+=1
                doresample=False
                print("Resample")
                if doresample and spl=="shihqcdtest":
                        fromtest=False
                        fromkde=True
                        if (fromtest):
                                for nn1 in range(nlat):
                                        for xbin in range(latarrpdf[nn1].GetNbinsX()):
                                                testlats=np.append(np.array(probstrain[spl])[:,5],np.array(probs[spl])[:,5])
                                                resamppoints=[]
                                                for ilat,lat in enumerate(testlats):
                                                       resamppoints.append(copy.deepcopy(lat))
                                                       resamppoints[-1][nn1]=latarrpdf[nn1].GetBinCenter(xbin)
                                                resamppoints=torch.from_numpy(np.array(resamppoints)).to('cuda:'+str(args.cuda))
                                                evalpdfs=compute_pdf(modelB(resamppoints.float()))
                                             
                                                for evp in evalpdfs:
                                                       evp.tolist()
                                                                
                                                       latarrpdf[nn1].Fill(latarrpdf[nn1].GetBinCenter(xbin),evp)

                        elif fromkde:
                               
                                toprint=1000
                                totev=20000
                                kdeevperbatch=10000
                                for igb in range(int(totev/kdeevperbatch)):
                                        print (igb*int(kdeevperbatch))
                                      
                                        tempkderersamp = gauskde.resample(size=(int(kdeevperbatch)))
                                        tempkdepdfs = gauskde.pdf(tempkderersamp)

                                        torchresamppoints=torch.from_numpy(tempkderersamp.T)
                                        evallist=compute_pdf(modelB((torchresamppoints).to('cuda:'+str(args.cuda)).float())).tolist()
                                        #print (tempkdepdfs[0])
                                        for iv,ev in enumerate(evallist):
                                                for iresamp in range(nlat):
                                                        if (tempkdepdfs[iv]>kdeslice[0]):
                                                                latarrpdf[iresamp].Fill(torchresamppoints[iv][iresamp],evallist[iv]/tempkdepdfs[iv])
                                                                latarrkdepdf[iresamp].Fill(torchresamppoints[iv][iresamp])
                                        print (latarrpdf[0].GetEntries())
                        else:

                                latpoint=[]
                                resamplebatch=100000
                                nresamp=resamplebatch
                                nrebatch=int(nresamp/resamplebatch)

                                latpdf=[]
                                ihist=0
                                reiter=0
                                evallisttoplot=[]
                                toprint=1000
                                while(len(evallisttoplot)<100000):
                                        if (len(evallisttoplot)>toprint):
                                                toprint+=1000
                                                print(toprint) 

                                        resamppoints=[]
                                        for irebatch in range(nrebatch):
                                                for iresamp in range(resamplebatch):
                                                        resamppoints.append([])
                                                        for nn1 in range(nlat):
                                                               resamppoints[iresamp].append( random.uniform(minmaxs[nn1][0],minmaxs[nn1][1]))

                                        resamppoints=np.array(resamppoints)
                                        resamppoints=torch.from_numpy(resamppoints).to('cuda:'+str(args.cuda))
                                        evallist=compute_pdf(modelB(resamppoints.float()))
                                        torchfilter = torch.where(evallist > 0.1)


                                        if reiter==0:
                                                evallisttoplot=(evallist[torchfilter]).cpu().detach().numpy()
                                                resamptoplot=(resamppoints[torchfilter]).cpu().detach().numpy()
                                        else:
                                                evallisttoplot=np.append(evallisttoplot, (evallist[torchfilter]).cpu().detach().numpy())
                                                resamptoplot=np.append(resamptoplot,(resamppoints[torchfilter]).cpu().detach().numpy(),0)
                 
                                        reiter+=1
                                   
                                for iv,ev in enumerate(evallisttoplot):
                                        if (iv%10000==0):
                                                print (iv)
                                        for iresamp in range(nlat):
                                                latarrpdf[iresamp].Fill(resamptoplot[iv][iresamp],ev)
                                ihist+=1
                print("ROC")

                tfile.cd()
                X[spl] = np.array(alllats)


                for ll in latarr:
                    ll.Write()
                canvs=[]
                for il,ll in enumerate(latarr1d):
                    canvs.append(ROOT.TCanvas("canvs"+str(il),"canvs"+str(il)))
                    canvs[-1].cd()
                    

           

                    if (latarr1d[il].Integral()>0.):
                        latarr1d[il].Scale(1.0/latarr1d[il].Integral())
                    if (latarrpdf[il].Integral()>0.):
                        latarrpdf[il].Scale(1.0/latarrpdf[il].Integral())
                    if (latarrkdepdf[il].Integral()>0.):
                        latarrkdepdf[il].Scale(1.0/latarrkdepdf[il].Integral())

                    maxmax=max([latarr1d[il].GetMaximum(),latarrpdf[il].GetMaximum(),latarrkdepdf[il].GetMaximum()])

                    latarr1d[il].SetLineColor(1)
                    latarrpdf[il].SetLineColor(2)
                    latarrkdepdf[il].SetLineColor(3)
                    latarr1d[il].SetMaximum(maxmax*1.1)
    
       
                    latarr1d[il].Draw("e")
                    latarrpdf[il].Draw("samee")
                    latarrkdepdf[il].Draw("samee")

                    latarr1d[il].Write()
                    latarrpdf[il].Write()
                    canvs[-1].Write()
                exstr=""
                if nspl==0:
                        disccanvs={}
                        disccanvs["PROB"]=ROOT.TCanvas("PROB"+spl,"PROB"+spl)
                        disccanvs["PROBkde"]=ROOT.TCanvas("PROBkde"+spl,"PROBkde"+spl)
                        disccanvs["MSE"]=ROOT.TCanvas("MSE"+spl,"MSE"+spl)
                        disccanvs["logMSE"]=ROOT.TCanvas("logMSE"+spl,"logMSE"+spl)
                        disccanvs["PDF"]=ROOT.TCanvas("PDF"+spl,"PDF"+spl)
                        disccanvs["PROB"].SetLogy()
                        disccanvs["MSE"].SetLogy()
                        disccanvs["logMSE"].SetLogy()
                        disccanvs["PROBkde"].SetLogy()
                        disccanvs["PDF"].SetLogy()

                if nspl>0:
                      exstr="same"
                plottingtonmse.SetLineColor(nspl+1)
                plottingtonlogmse.SetLineColor(nspl+1)
                plottingtonpdf.SetLineColor(nspl+1)
                plottingtonprob.SetLineColor(nspl+1)
                plottingtonprobkde.SetLineColor(nspl+1)

                print ("hist"+exstr)

                disccanvs["PROB"].cd()
                allplots.append(copy.deepcopy(plottingtonprob))
                allplots[-1].Draw("hist"+exstr)
                plottingtonprob.Write()

                disccanvs["PROBkde"].cd()
                allplots.append(copy.deepcopy(plottingtonprobkde))
                allplots[-1].Draw("hist"+exstr)
                plottingtonprobkde.Write()

                disccanvs["MSE"].cd()
                allplots.append(copy.deepcopy(plottingtonmse))
                allplots[-1].Draw("hist"+exstr)
                plottingtonmse.Write()

                disccanvs["logMSE"].cd()
                allplots.append(copy.deepcopy(plottingtonlogmse))
                allplots[-1].Draw("hist"+exstr)
                plottingtonlogmse.Write()

                disccanvs["PDF"].cd()
                allplots.append(copy.deepcopy(plottingtonpdf))
                allplots[-1].Draw("hist"+exstr)
                plottingtonpdf.Write()

                if nspl>0:
                        disccanvs["PROB"].Write("PROB")
                        disccanvs["PDF"].Write("PDF")
                        disccanvs["logMSE"].Write("logMSE")
                        disccanvs["MSE"].Write("MSE")
                        disccanvs["PROBkde"].Write("PROBkde")

                msevprob.Write()





                max0=np.max(np.append(np.array(probs[spldata[0]])[:,0],np.array(probs[spldata[1]])[:,0]))+0.000001
                max1=np.max(np.append(np.array(probs[spldata[0]])[:,1],np.array(probs[spldata[1]])[:,1]))+0.000001
                max2=np.max(np.append(np.array(probs[spldata[0]])[:,2],np.array(probs[spldata[1]])[:,2]))+0.000001
                max3=np.max(np.append(np.array(probs[spldata[0]])[:,3],np.array(probs[spldata[1]])[:,3]))+0.000001
                max4=np.max(np.append(np.array(probs[spldata[0]])[:,6],np.array(probs[spldata[1]])[:,6]))+0.000001

                min0=np.min(np.append(np.array(probs[spldata[0]])[:,0],np.array(probs[spldata[1]])[:,0]))-0.000001
                min1=np.min(np.append(np.array(probs[spldata[0]])[:,1],np.array(probs[spldata[1]])[:,1]))-0.000001
                min2=np.min(np.append(np.array(probs[spldata[0]])[:,2],np.array(probs[spldata[1]])[:,2]))-0.000001
                min3=np.min(np.append(np.array(probs[spldata[0]])[:,3],np.array(probs[spldata[1]])[:,3]))-0.000001
                min4=np.min(np.append(np.array(probs[spldata[0]])[:,6],np.array(probs[spldata[1]])[:,6]))-0.000001
                range0=max0-min0
                #min1=0.0
                #max1=0.002
                range1=max1-min1
                range2=max2-min2
                range3=max3-min3
                range4=max4-min4
                #range1=max1
                #optimval=0.000405812765966
                optimfrac0=1.0
                optimfrac1=1.0 
                grapharr=[]
                npos=60
                effs=[]
                effskde=[]
                effsAE=[]
                effsPR=[]
                effsPRkde=[]
                effsBDT=[]
                effsBDTkde=[]
                effsCOMB=[]
                print ("effcalc")
                prevmival0 = 0.0
                prevmival1 = 999.0
                temp0=np.array(probs[spl])[:,0]
                temp1=np.array(probs[spl])[:,1]
                temp2=np.array(probs[spl])[:,2]
                temp3=np.array(probs[spl])[:,3]
                temp4=np.array(probs[spl])[:,6]
                for q0 in range(npos+1):
                  for q1 in range(npos+1):  
                        newmival0 = min0+float(range0)*(float(q0)/float(npos))
                        newmival1 = min1+float(range1)*((float(q1)/float(npos)*(float(q1)/float(npos))))
                        newmival2 = min2+float(range2)*(float(q0)/float(npos))
                        newmival3 = min3+float(range3)*(float(q0)/float(npos))
                        newmival4 = min4+float(range4)*(float(q0)/float(npos))
                        combfrac=optimfrac0*newmival0+optimfrac1*newmival1 
                        effsCOMB.append(0.0)
                        effsBDT.append(0.0)
                        effsBDTkde.append(0.0)
                        effs.append(0.0)
                        effskde.append(0.0)


                        effs[-1]=np.count_nonzero(np.bitwise_and(temp0>newmival0,temp1>newmival1))

                        effskde[-1]=np.count_nonzero(np.bitwise_and(temp4>newmival4,temp1>newmival1))
                        effsBDT[-1]=np.count_nonzero(temp2>newmival2)
                        effsBDTkde[-1]=np.count_nonzero(temp3>newmival3)
                        effsCOMB[-1]=0

                        #for qind in range(len(probs[spl])):
                         #     if ((probs[spl][qind][0]>=newmival0) and (probs[spl][qind][1]>=newmival1)):
                          #      effs[-1]+=1.0
                           #   if ((probs[spl][qind][6]>=newmival4) and (probs[spl][qind][1]>=newmival1)):
                            #    effskde[-1]+=1.0
                         #     if (probs[spl][qind][2]>=newmival2):
                          #      effsBDT[-1]+=1.0
                           #   if (probs[spl][qind][3]>=newmival3):
                            #    effsBDTkde[-1]+=1.0
                             # if ((optimfrac0*probs[spl][qind][0]+optimfrac1*probs[spl][qind][1])>=combfrac):
                              #  effsCOMB[-1]+=1.0

                        effs[-1]/=float(len(probs[spl]))
                        effs[-1]=max(effs[-1],0.000000000001)

                        effskde[-1]/=float(len(probs[spl]))
                        effskde[-1]=max(effskde[-1],0.000000000001)


                        effsCOMB[-1]/=float(len(probs[spl]))
                        effsCOMB[-1]=max(effsCOMB[-1],0.000000000001)


                        effsBDT[-1]/=float(len(probs[spl]))
                        effsBDT[-1]=max(effsBDT[-1],0.000000000001)

                        effsBDTkde[-1]/=float(len(probs[spl]))
                        effsBDTkde[-1]=max(effsBDTkde[-1],0.000000000001)
                        if q0==0:
                                effsAE.append(effs[-1])
                        if q1==0:
                                #print ("effsPR",effs[-1],newmival0)
                                #print ("effskde",effskde[-1],newmival4)
                                effsPR.append(effs[-1])
                                effsPRkde.append(effskde[-1])
                       
                        prevmival0 = newmival0
                        prevmival1 = newmival1

                #print (effs)
                alleff[spl]=copy.deepcopy(effs)
                alleffAE[spl]=copy.deepcopy(effsAE)
                alleffPR[spl]=copy.deepcopy(effsPR)
                alleffPRkde[spl]=copy.deepcopy(effsPRkde)
                alleffCOMB[spl]=copy.deepcopy(effsCOMB)
                alleffBDT[spl]=copy.deepcopy(effsBDT)
                alleffBDTkde[spl]=copy.deepcopy(effsBDTkde)
                nspl+=1
        print("TSNE")
        toTSNE=True
        lens=[]
        if (toTSNE):
                Xtot=np.vstack([X[spldata[0]],X[spldata[1]]])
                lens=[len(X[spldata[0]]),len(X[spldata[1]])]
                X_embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000).fit_transform(Xtot)
              
                grapharr.append(ROOT.TGraph(len(X_embedded[:lens[0]]),array.array('d',X_embedded[:lens[0],0]),array.array('d',X_embedded[:lens[0],1])))
                grapharr.append(ROOT.TGraph(len(X_embedded[lens[0]:]),array.array('d',X_embedded[lens[0]:,0]),array.array('d',X_embedded[lens[0]:,1])))
                grapharr[0].SetMarkerColor(2)
                grapharr[0].SetMarkerStyle(24)
                grapharr[1].SetMarkerColor(3)
                grapharr[1].SetMarkerStyle(24)

        plottingtonprob=ROOT.TGraph(len(alleff[spldata[0]]),array.array('d',alleff[spldata[0]]),array.array('d',np.divide(1.0,alleff[spldata[1]])))
        plottingtonprob.SetMarkerStyle(21)
        plottingtonprob.SetMaximum(1000)
        plottingtonprob.Draw("AP")
        plottingtonprob.Write("roc")


        gcanv1=ROOT.TCanvas("gcanv1","gcanv1")
        plottingtonprobAE=ROOT.TGraph(len(alleffAE[spldata[0]]),array.array('d',alleffAE[spldata[0]]),array.array('d',np.divide(1.0,alleffAE[spldata[1]])))
        plottingtonprobAE.SetMarkerStyle(21)
        plottingtonprobAE.SetMaximum(1000)
        plottingtonprobAE.SetMarkerColor(2)
        plottingtonprobAE.Write("rocAE")


        plottingtonprobPR=ROOT.TGraph(len(alleffPR[spldata[0]]),array.array('d',alleffPR[spldata[0]]),array.array('d',np.divide(1.0,alleffPR[spldata[1]])))
        plottingtonprobPR.SetMarkerStyle(21)
        plottingtonprobPR.SetMarkerColor(3)
        plottingtonprobPR.Write("rocPR")



        plottingtonprobPRkde=ROOT.TGraph(len(alleffPRkde[spldata[0]]),array.array('d',alleffPRkde[spldata[0]]),array.array('d',np.divide(1.0,alleffPRkde[spldata[1]])))
        plottingtonprobPRkde.SetMarkerStyle(21)
        plottingtonprobPRkde.SetMarkerColor(7)
        plottingtonprobPRkde.Write("rocPRkde")

        plottingtonprobCOMB=ROOT.TGraph(len(alleffCOMB[spldata[0]]),array.array('d',alleffCOMB[spldata[0]]),array.array('d',np.divide(1.0,alleffCOMB[spldata[1]])))
        plottingtonprobCOMB.SetMarkerStyle(21)
        plottingtonprobCOMB.SetMarkerColor(4)
        plottingtonprobCOMB.Write("rocCOMB")



        plottingtonprobBDT=ROOT.TGraph(len(alleffBDT[spldata[0]]),array.array('d',alleffBDT[spldata[0]]),array.array('d',np.divide(1.0,alleffBDT[spldata[1]])))
        plottingtonprobBDT.SetMarkerStyle(21)
        plottingtonprobBDT.SetMarkerColor(5)
        plottingtonprobBDT.Write("rocBDT")



        plottingtonprobBDTkde=ROOT.TGraph(len(alleffBDTkde[spldata[0]]),array.array('d',alleffBDTkde[spldata[0]]),array.array('d',np.divide(1.0,alleffBDTkde[spldata[1]])))
        plottingtonprobBDTkde.SetMarkerStyle(21)
        plottingtonprobBDTkde.SetMarkerColor(6)
        plottingtonprobBDTkde.Write("rocBDTkde")



        gcanv1.SetLogy()
        mgtsne=ROOT.TMultiGraph()
        mgtsne.Add(plottingtonprobPR)
        mgtsne.Add(plottingtonprobPRkde)
        mgtsne.Add(plottingtonprobAE)
        mgtsne.Add(plottingtonprobBDT)
        mgtsne.Add(plottingtonprobBDTkde)
        mgtsne.SetTitle(";Eff("+spldata[0]+");1.0/Eff("+spldata[1]+")")
        mgtsne.Draw("AP")
        mgtsne.GetXaxis().SetRangeUser(0.,1.)
        mgtsne.GetYaxis().SetRangeUser(1.,10000.)
        gcanv1.Write("ROCsingle")
        if (toTSNE):
                gcanv2=ROOT.TCanvas("gcanv2","gcanv2")

                mgtsne=ROOT.TMultiGraph()
                mgtsne.Add(grapharr[0])
                mgtsne.Add(grapharr[1])
                mgtsne.SetTitle(";TSNE(1);TSNE(2)")
                mgtsne.Draw("AP")
                gcanv2.Write("TSNE")

          


