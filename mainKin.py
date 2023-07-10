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
subject=['P07']
condition=['noRAC_slowWalk']

COMfigpath='D:\\KIN6838\\EEGdecodingproject\\Kinetic\\COMfigures\\'
vCOMfigpath='D:\\KIN6838\\EEGdecodingproject\\Kinetic\\vCOMfigures\\'

for sub in subject:
    for cond in conditions:
        print(f'Importing {sub}_{cond} ...')
        path_c3d=path_folder + '\\' + sub + '\\'+ 'Nexus' +'\\' + cond + '.c3d'
        try: 
            c=c3d(path_c3d)
        except:
            continue
        # print(c['parameters']['POINT']['USED']['value'][0]);  # Print the number of points used
        point_data = c['data']['points']
        points_residuals = c['data']['meta_points']['residuals']
        analog_data = c['data']['analogs'].reshape(46,-1)
        point_rate=c['parameters']['POINT']['RATE']['value'][0]
        analog_rate=c['parameters']['ANALOG']['RATE']['value'][0]
        # define trigger points
        temp=np.array(c['parameters']['ANALOG']['LABELS']['value'])
        isynch=np.where(temp=='Synchronization.1')[0][0]
        x=analog_data[isynch,:]
        peaks, _ = find_peaks(x, height=2)
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.plot(np.zeros_like(x), "--", color="gray")

        trigger_start,trigger_end=peaks[0], peaks[0]+ analog_rate*5*60 # select 5 mins of data
        trigger_start,trigger_end=round(trigger_start/(analog_rate/point_rate)), round(trigger_end/(analog_rate/point_rate))  #downsample to 200 Hz

        # Reading COM data from the c3d files
        obj=np.array(c['parameters']['POINT']['LABELS']['value']) 
        #iCOM=np.where(obj=='CentreOfMass')[0][0]
        # iCOM=np.where(obj=='RAnkleAngles')[0][0]
        iCOM=np.where(obj=='RKneeAngles')[0][0]

        COMx, COMy, COMz=point_data[0,iCOM,trigger_start:trigger_end], point_data[1,iCOM,trigger_start:trigger_end], point_data[2,iCOM,trigger_start:trigger_end]
        COMx[np.isnan(COMx)] = np.nanmean(COMx)
        COMy[np.isnan(COMy)] = np.nanmean(COMy)
        COMz[np.isnan(COMz)] = np.nanmean(COMz)
        COM=[COMx, COMy,COMz]
        # plot the values 
        fig, axs= plt.subplots(len(COM),1)
        ylabel=['COM LAT', 'COM AP', 'COM Z']
        figcolor=['red','blue','black']
        for i in range(len(COM)):
            axs[i].plot(COM[i],color=figcolor[i])
            axs[i].set_xlim([0,3000])
            axs[i].set_ylabel(ylabel[i])
        axs[0].set_title('COM')
        plt.savefig(COMfigpath+f'{sub}_{cond}_COM.png',
             format="png", dpi=500,bbox_inches='tight')    
        # upsample to 200Hz     
        COM_200=[]
        len200=len(COMx)*2
        for c in COM:
            COM_200.append(resample(c,len200))

        vCOM=[]
        dt=1/200
        for c in COM_200:
            vCOM.append(np.diff(c)/dt)  
        # plot the values 
        fig, axs= plt.subplots(len(vCOM),1)
        ylabel=['VCOM LAT', 'VCOM AP', 'VCOM Z']
        figcolor=['red','blue','black']
        for i in range(len(vCOM)):
            axs[i].plot(vCOM[i][trigger_start:trigger_end+1], color=figcolor[i])
            axs[i].set_xlim([0,6000])
            axs[i].set_ylabel(ylabel[i])
        axs[0].set_title('COM velocity')
        plt.savefig(vCOMfigpath+f'{sub}_{cond}_COM.png',
        format="png", dpi=500,bbox_inches='tight')    
        dataset=pd.DataFrame({'vCOMx':vCOM[0], 
                              'vCOMy':vCOM[1], 
                              'vCOMz': vCOM[2]})
        dataset2=pd.DataFrame({'vCOMx':COM[0], 
                        'vCOMy':COM[1], 
                        'vCOMz': COM[2]})
        path_csv='D:\\KIN6838\\EEGdecodingproject\\data\\raw\\COM velocity\\'
        path_csv2='D:\\KIN6838\\EEGdecodingproject\\data\\raw\\COM\\'
        filename=path_csv+sub+'_'+cond+'_vCOM'+'.csv'
        filename2=path_csv2+sub+'_'+cond+'_COM'+'.csv'
        dataset.to_csv(filename, index=False,header=False)
        dataset2.to_csv(filename2, index=False,header=False)

            
