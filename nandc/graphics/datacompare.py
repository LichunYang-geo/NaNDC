import os
import numpy as np
from matplotlib import pyplot as plt

# 
plt.rc('font',size=20) 
def plot_data(savepath: str, filename: str, rawdata, syn_amp, syn_ratio=None):

    width = 0.3
    x = np.arange(np.size(syn_amp,1))
    if syn_ratio is not None:
        pamp = rawdata.pamp
        spratio = rawdata.spratio
        fig = plt.figure(figsize=(16., 8.))  # size in inch
        ax0 = fig.add_subplot(211)
        ax0.bar(x - width/2, pamp, width=width, label='obs', color='#f07c82')
        ax0.bar(x + width/2, syn_amp.flatten(), width=width, label='syn', color='#3b818c')
        ax0.legend(loc='upper right')
        ax0.set_ylabel('P-wave amplitude')
        ax1 = fig.add_subplot(212)
        ax1.bar(x - width/2, spratio, width=width, label='obs', color='#f07c82')
        ax1.bar(x + width/2, syn_ratio.flatten(), width=width, label='syn', color='#3b818c')
        ax1.set_ylabel('S/P ratio (log)')
        ax1.legend(loc='upper right')
        ax1.set_xlabel('Station number')

    else:
        pamp = rawdata.pamp
        fig = plt.figure(figsize=(16., 4.))  # size in inch
        ax = fig.add_subplot(111)
        ax.bar(x - width/2, pamp, width=width, label='obs', color='#f07c82')
        ax.bar(x + width/2, syn_amp.flatten(), width=width, label='syn', color='#3b818c')
        ax.legend(loc='upper right')
        ax.set_xlabel('Station number')
        ax.set_ylabel('P-wave amplitude')

    fig.savefig(os.path.join(savepath,filename), dpi=600, bbox_inches='tight', facecolor='white', transparent=False)

def plot_confidence_curve(savepath: str, filename: str, curve: np.ndarray):

    fig = plt.figure(figsize=(14, 4))
    grid = plt.GridSpec(1, 3, wspace=0.5, hspace=0.5)
    main_ax = plt.subplot(grid[0:1,0:2])
    main_ax.plot(np.linspace(0,180,181),curve[:,0])
    main_ax.plot(np.linspace(0,180,181),curve[:,1])
    main_ax.set_aspect('auto')
    sec_ax = plt.subplot(grid[0:1,2:3],sharey=main_ax)
    sec_ax.plot(curve[:,0],curve[:,1])
    sec_ax.set_aspect(1)
    fig.savefig(os.path.join(savepath,filename), dpi=600, bbox_inches='tight', facecolor='white', transparent=False)
