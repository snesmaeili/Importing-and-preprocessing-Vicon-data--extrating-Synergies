import numpy as np 
from ezc3d import c3d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import resample
import pandas as pd
from pathlib import Path  

def kinmaker(kinNames, c ,sub,cond, kinSelect,kin_velocity):
 
    print(f'Importing {sub}_{cond} ...')
    DIR_CURRENT = Path(__file__).parent
    point_data = c['data']['points']
    # points_residuals = c['data']['meta_points']['residuals']
    analog_data = c['data']['analogs'].reshape(46,-1)   ###### need to change the 46 base on the number of Analog channels
    point_rate=c['parameters']['POINT']['RATE']['value'][0]
    analog_rate=c['parameters']['ANALOG']['RATE']['value'][0]
    # define trigger points
    temp=np.array(c['parameters']['ANALOG']['LABELS']['value'])
    isynch=np.where(temp=='Synchronization.1')[0][0]
    x=analog_data[isynch,:]
    peaks, _ = find_peaks(x, height=2)
    # plt.plot(x)
    # plt.plot(peaks, x[peaks], "x")
    # plt.plot(np.zeros_like(x), "--", color="gray")
    trigger_start,trigger_end=peaks[0], peaks[0]+ analog_rate*5*60 # select 5 mins of data
    trigger_start,trigger_end=round(trigger_start/(analog_rate/point_rate)), round(trigger_end/(analog_rate/point_rate))  #downsample to 200 Hz
    # Reading COM data from the c3d files
    obj=np.array(c['parameters']['POINT']['LABELS']['value']) 
    #iCOM=np.where(obj=='CentreOfMass')[0][0]
    # iCOM=np.where(obj=='RAnkleAngles')[0][0]
    kin={}
    kin2={}
    for  kinname in  kinNames:   
        try:
            ikin=np.where(obj==kinname)[0][0]
        except IndexError:
            print(f'There is no {kinname} in {sub}_{cond}')
            pass
        kinx, kiny, kinz=point_data[0,ikin,trigger_start:trigger_end], point_data[1,ikin,trigger_start:trigger_end], point_data[2,ikin,trigger_start:trigger_end]
        kinx[np.isnan(kinx)] = np.nanmean(kinx)
        kiny[np.isnan(kiny)] = np.nanmean(kiny)
        kinz[np.isnan(kinz)] = np.nanmean(kinz)
        kin[kinname]=[kinx, kiny,kinz]
        
        
        # plot the values 
        
        kinFigpath= DIR_CURRENT/ 'figures' / kinname
        filename=sub+'_'+cond+'_'+ kinname +'.png'
        kinFigname=Path( kinFigpath / filename )
        kinFigname.parent.mkdir(parents=True,exist_ok=True)
        fig, axs= plt.subplots(len(kin[kinname]),1)
        ylabel=[ 'x',   'y',   'z']
        figcolor=['red','blue','black']
        for i in range(len(kin[kinname])):
            axs[i].plot(kin[kinname][i],color=figcolor[i])
            axs[i].set_xlim([0,3000])
            axs[i].set_ylabel(ylabel[i])
        axs[0].set_title(kinname)
        plt.savefig ( kinFigname,format="png", dpi=500,bbox_inches='tight')    
    
        # upsample to 200Hz     
        kin_200=[]
        len200=len(kinx)*2
        for c in kin[kinname]:
            kin_200.append(resample(c,len200))
            
        # kin2[kinname]=kin_200   
        kinDataset=pd.DataFrame({'x':kin_200[0], 
                    'y':kin_200[1], 
                    'z': kin_200[2]})
        kin2[kinname]={'x':kin_200[0], 
                    'y':kin_200[1], 
                    'z': kin_200[2]} 
           
        pathKin_csv=DIR_CURRENT /  'kinVariables' /  kinname
        filename= sub+ cond.replace('noRAC','') +'_'+kinname+'_gonio.csv'
        kinFilename=Path(pathKin_csv / filename) 
        kinFilename.parent.mkdir(parents=True, exist_ok=True)
        kinDataset.to_csv(kinFilename, index=False,header=False)
        
        
        
        if kin_velocity==1:
            vkin=[]
            dt=1/200 
            for c in kin_200:
                vkin.append(np.diff(c,append=c[0])/dt)  
            # plot the values 
            # fig, axs= plt.subplots(len(vkin),1)
            # ylabel=['VCOM LAT', 'VCOM AP', 'VCOM Z']
            # figcolor=['red','blue','black']
            # for i in range(len(vkin)):
            #     axs[i].plot(vkin[i][trigger_start:trigger_end+1], color=figcolor[i])
            #     axs[i].set_xlim([0,3000])
            #     axs[i].set_ylabel(ylabel[i])
            # axs[0].set_title('COM velocity')
            # plt.savefig(vCOMfigpath+f'{sub}_{cond}_COM.png',
            # format="png", dpi=500,bbox_inches='tight')    
            # vCOMfigpath='D:\\KIN6838\\EEGdecodingproject\\Kinetic\\vCOMfigures\\'    
            path_csv='D:\\KIN6838\\EEGdecodingproject\\data\\raw\\COM velocity\\'
            filename=path_csv+sub+'_'+cond+'_vCOM_gonio'+'.csv'    
            dataset=pd.DataFrame({'vCOMx':vkin[0], 
                                'vCOMy':vkin[1], 
                                'vCOMz': vkin[2]})
            dataset.to_csv(filename, index=False,header=False)
            plt.close('all')
    data={}
    for i, key in enumerate(kin2.keys()):
        data[key]=kin2[key][kinSelect]
        
    kinDataset=pd.DataFrame.from_dict(data)    
    pathKin_csv=DIR_CURRENT /  'kinVariables' /  'mixed'
    filename= sub+ cond.replace('noRAC','') +'_gonio.csv'
    kinFilename=Path(pathKin_csv / filename) 
    kinFilename.parent.mkdir(parents=True, exist_ok=True)
    kinDataset.to_csv(kinFilename, index=False,header=False)


    
