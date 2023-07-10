import numpy as np 
from ezc3d import c3d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import resample
import pandas as pd


# 'J:\\Projet_RAC\\DATA\\RAW\\P12\\Nexus\\noRAC_slowWalk.c3d'
path_folder='J:\\Projet_RAC\\DATA\\RAW'
subjects=['P01','P02','P03','P04','P05','P06','P07','P08','P09','P10','P11','P12']
conditions=['noRAC_slowWalk','noRAC_preferredWalk','noRAC_fastWalk']
subject=['P01']
condition=['noRAC_slowWalk']

COMfigpath='D:\\KIN6838\\EEGdecodingproject\\Kinetic\\COMfigures\\'
vCOMfigpath='D:\\KIN6838\\EEGdecodingproject\\Kinetic\\vCOMfigures\\'

for sub in subject:
    for cond in condition:
        print(f'Importing {sub}_{cond} ...')
        path_c3d=path_folder + '\\' + sub + '\\'+ 'Nexus' +'\\' + cond + '.c3d'
        try: 
            c=c3d(path_c3d)
        except:
            continue
        analog_data = c['data']['analogs']
        analog_rate=c['parameters']['ANALOG']['RATE']['value'][0]
        # define trigger points
        temp=np.array(c['parameters']['ANALOG']['LABELS']['value'])
        isynch=np.where(temp=='Synchronization.1')[0][0]
        x=analog_data[isynch,:]
        peaks, _ = find_peaks(x, height=2)
        trigger_start,trigger_end=peaks[0], peaks[0]+ analog_rate*5*60 # select 5 mins of data
