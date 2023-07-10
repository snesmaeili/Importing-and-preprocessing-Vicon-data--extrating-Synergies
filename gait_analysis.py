import os
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt,sosfiltfilt
from ezc3d import c3d
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')  # Set backend to Agg to save figures without displaying them
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d
from muscle_synergy import NNMFSynergy ,MuscleSynergyAutoencoder
from scipy.signal import savgol_filter
from scipy import io
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import medfilt
from scipy.signal import resample

# from scipy.signal import total_variation
class GaitAnalysis:
    def __init__(self, path,name):
        self.path = path
        self.name = name
        self.calculated_methods = {}
    def force_event (self, side, data_type, min_distance=500, plot=True):
        self.events = {'Right': {}, 'Left': {},'all':{}}
        assert side in ['Left', 'Right'], "Invalid side. Must be 'Left' or 'Right'"
        
        # Ensure the kinetic data is present
        assert self.kinetic is not None, "Kinetic data is not present. Make sure to import data first."

        # Select the force data of the appropriate side
        if side == 'Left':
            force_data = self.kinetic['kineticDataL']
        elif side == 'Right':
            force_data = self.kinetic['kineticDataR']

        # Savitzky-Golay Filter
        sg_force_data = savgol_filter(force_data, window_length=50, polyorder=1)
        
        force_filtered = sg_force_data

        # Find the indices where force crosses the threshold
        heel_strike_indices = []
        toe_off_indices = []
        prev_force = force_filtered[0]
        last_event_index = 0
        threshold=force_filtered.min()+0.1
        # Assuming that the first event is a heel strike
        current_state = "heel_strike"  
        for i in range(1, len(force_filtered)):
            if prev_force < threshold and force_filtered[i] >= threshold:
                if current_state == "heel_strike" and i - last_event_index >= min_distance:
                    heel_strike_indices.append(i)
                    current_state = "toe_off"
                    last_event_index = i
            elif prev_force > threshold and force_filtered[i] <= threshold:
                if current_state == "toe_off" and i - last_event_index >= min_distance:
                    toe_off_indices.append(i)
                    current_state = "heel_strike"
                    last_event_index = i
            prev_force = force_filtered[i]

        self.events[side]['heel_strike'] = np.array(heel_strike_indices)
        self.events[side]['toe_off'] = np.array(toe_off_indices)

        setattr(self, f'{data_type}_force_event_calculated', True)
        
        # Optional plot
        if plot:
            plt.figure()
            plt.plot(force_data, label='Force data')
            plt.plot(force_filtered, label='Filtered force data')
            plt.plot([0, len(force_data)], [threshold, threshold], 'g--', label='Threshold')  # Threshold line
            plt.plot(heel_strike_indices, force_data[heel_strike_indices], 'ro', label='Heel strikes')
            plt.plot(toe_off_indices, force_data[toe_off_indices], 'bo', label='Toe offs')
            plt.legend()
            # plt.show()
        
        return heel_strike_indices, toe_off_indices




    
    def import_data(self,data_type):
        
        self.emg = {'Right': {}, 'Left': {},'Label':{}}
        
        c3d_object = c3d(self.path)
        # Find trigger data in c3d file
        trigger_index = [i for i, s in enumerate(c3d_object['parameters']['ANALOG']['LABELS']['value']) if 'Synchronization' in s]
        trigger_data = c3d_object['data']['analogs'][0,trigger_index,:].squeeze()  # Trigger data

        # Minimum height for peaks
        min_height = 0.8
        trigger_locs,_ = find_peaks(trigger_data, height=min_height)
        analog_rate = int(c3d_object['parameters']['ANALOG']['RATE']['value'])
        self.analog_rate=analog_rate
        # Check triggers (must be 5 minutes elapsed between triggerStart and triggerEnd)
        trigger_start,trigger_end=trigger_locs[0], trigger_locs[0]+ analog_rate*5*60 # select 5 mins of dat

        # Storing the trigger data
        self.trigger = [trigger_start,trigger_end]

        # Import EMG, Force, and Kinematic data here
        # Translate the Import EMG data section
        
        subject = self.name.split('_')[0]
        
        if subject in ['P01', 'P02' ,'P03']:
            musname_vicon = ['Tibialis.IM', 'Gluteus_Max.IM', 'RectusFemoris.IM', 'Iliocostalis.IM',
                            'Longissimus.IM', 'Gastroc_Med.IM', 'Soleus.IM', 'Peroneus.IM', 'Adductor.IM',
                            'Vastus_Lat.IM', 'Vastus_Med.IM', 'Semitendinosus.IM', 'Gluteus_Med.IM']
        elif subject=='P05':    
            musname_vicon=['Tibialis.IM','Gluteus_Max.IM','RectusFemoris.IM','Iliocostalis.IM', 
                'Longissimus.IM','Gastroc_med.IM','Soleus.IM','Peroneus.IM','Adductor.IM',
                'Vastus_Lat.IM','Vastus_Med.IM','Semitendinosus.IM','Gluteus_Med.IM']
        else:
            musname_vicon=['TA.IM','GMAX.IM','RF.IM','IC.IM', 
                            'LG.IM','MGAS.IM','SOL.IM','PL.IM','ADD.IM',
                                'VL.IM','VM.IM','SEMT.IM','GMED.IM']
            
        musname_mat = ['TA', 'GMAX', 'RF', 'IC', 'LG', 'MGAS', 'SOL', 'PL', 'ADD', 'VL', 'VM', 'SEMT', 'GMED']


        for mus_vicon, mus_mat in zip(musname_vicon, musname_mat):
            mus_index = [i for i, s in enumerate(c3d_object['parameters']['ANALOG']['LABELS']['value']) if mus_vicon in s]
            self.emg['Right'][mus_mat] = c3d_object['data']['analogs'][0][mus_index].squeeze()
        # Import Force data
        self.kinetic = {}
        right_index = [i for i, s in enumerate(c3d_object['parameters']['ANALOG']['LABELS']['value']) if 'Right_foot.Fz' in s]
        self.kinetic['kineticDataR'] = c3d_object['data']['analogs'][0][right_index].squeeze()[trigger_start:trigger_end]


        left_index = [i for i, s in enumerate(c3d_object['parameters']['ANALOG']['LABELS']['value']) if 'Left_foot.Fz' in s]
        self.kinetic['kineticDataL'] = c3d_object['data']['analogs'][0][left_index].squeeze()[trigger_start:trigger_end]

        #import Kinematic data , i want to write the code for the kinematic data
        self.kinematic = {'Right': {}, 'Left': {},'Label':{}}
        point_rate = c3d_object['parameters']['POINT']['RATE']['value'][0] 
        self.point_rate=point_rate      
        #list of kinematic data to be imported
        kinematic_data_to_import = ['RAnkleAngles', 'RKneeAngles', 'RHipAngles', 'RSpineAngles']
        # Convert the trigger points to the kinematic (point) frame rate
        kin_trigger_start = round(self.trigger[0] * (point_rate / analog_rate))
        kin_trigger_end = round(self.trigger[1] * (point_rate / analog_rate))
        
        for kin_data in kinematic_data_to_import:
            kin_index = [i for i, s in enumerate(c3d_object['parameters']['POINT']['LABELS']['value']) if kin_data in s]
            # 'points' data shape is (4, number of points, number of frames), where the 4 corresponds to (x, y, z, residual)
            kinematic_values = c3d_object['data']['points'][0:3, kin_index, kin_trigger_start:kin_trigger_end].squeeze()
            self.kinematic['Right'][f"{kin_data}_x"] = kinematic_values[0]
            # if kin_data is not  'RKneeAngles':
            self.kinematic['Right'][f"{kin_data}_y"] = kinematic_values[1]
            # self.kinematic['Right'][f"{kin_data}_z"] = kinematic_values[2]            
        self.kinematic['Label'] = list(self.kinematic['Right'].keys())    
        setattr(self, f'{data_type}_import_data_calculated', True)
        # Add additional markers as needed
        # Import COM data  i want to write the code for the kinematic data

    def emg_preprocessing(self, data_type,hp1=35, lp1=0, lp2=5, timeBin=1, max_amp=1):
        

        def muscle_no(muscle_name):  # Define muscle_no function
            # Define muscle names and corresponding numbers
            all_names = ['TA', 'PL', 'SOL', 'MGAS', 'GC_M',
                        'RF', 'SEMT', 'BFLH', 'VL', 'VM', 'GMED', 'G_M', 'GMAX', 'GLUT_MAX',
                        'IP', 'IPS', 'ADD', 'TFL',
                        'IC', 'LG', 'RG', 'RA', 'RECT_AB', 'EO']
            all_no = [1, 2, 3, 4, 4,
                    5, 6, 7, 8, 9, 10, 10, 11, 11,
                    12, 12, 13, 14,
                    15, 16, 16, 17, 17, 18]

            # Create a dictionary for muscle names and numbers
            muscle_dict = dict(zip(all_names, all_no))

            # Return the muscle number or 0 if not found
            return muscle_dict.get(muscle_name, 0)

        
        def filter_emg(emg_raw_mat, analog_rate, hp1, lp2, max_amp=0, plot=False):

            sf = analog_rate  # Sampling frequency
            # Pad the signal
            order_of_filter=2
            # Design filters
            sos_hp = butter(order_of_filter, 2*hp1/sf, 'high',output='sos')
            sos_lp = butter(order_of_filter, 2*lp2/sf,output='sos')

            # Apply high-pass filter
            emg_filtered  = sosfiltfilt(sos_hp, emg_raw_mat)

            # Detrend
            emg_filtered  = detrend(emg_filtered ,axis=0)

            # Take absolute value
            np.abs(emg_filtered , out=emg_filtered )

            # Apply low-pass filter
            emg_filtered = sosfiltfilt(sos_lp, emg_filtered )
                # Take absolute value
            np.abs(emg_filtered, out=emg_filtered)
            # Normalize by maximum value across each channel
            if max_amp:
                max_val = np.max(emg_filtered, axis=1).reshape(-1, 1)
                emg_filtered /= max_val

                # if plot:
                #     n_muscles = emg_raw_mat.shape[0]
                #     # fig, axs = plt.subplots(n_muscles//2 + n_muscles%2, 2, figsize=(10, n_muscles*2.5))

                #     for i in range(n_muscles):
                #         row = i // 2
                #         col = i % 2
                #         axs[row, col].plot(emg_raw_mat[i, :], label='Raw')
                #         axs[row, col].plot(emg_filtered[i,:], label='Filtered')
                #         axs[row, col].legend()

                #     # If n_muscles is odd, remove the last (empty) subplot
                #     if n_muscles % 2:
                #         fig.delaxes(axs[-1, -1])

                #     plt.tight_layout()
                #     # Save the figure if a save path is provided
                #     plt.savefig(f'figures\\muscles\\{self.name}_emg_preprocessing.png',dpi=500)
            return emg_filtered

        
        # Collect all EMG data into a matrix
        side = 'Right'
        self.emg_filtered={'Right':{},'Left':{}}
        muscle_order = sorted(self.emg[side], key=muscle_no)
        self.emg['Label']=muscle_order
        emg_data = np.concatenate([np.atleast_2d(self.emg[side][muscle]) for muscle in muscle_order], axis=0)        
        # Filter EMG data

        emg_data_filtered = filter_emg(emg_data, self.analog_rate,hp1, lp2, max_amp)
        trigger_start,trigger_end=self.trigger

        emg_data_normalized = emg_data_filtered
        
        # Update emg_data with preprocessed data
        self.emg_filtered[side] = {muscle: emg_data_normalized[i, trigger_start:trigger_end] for i, muscle in enumerate(muscle_order)}
        setattr(self, f'{data_type}_emg_preprocessing_calculated', True)

    

    def kin_preprocessing(self, data_type ,side='Right', cutoff_freq=0.1, filter_order=3, median_kernel_size=51):
        # If filtered data attribute does not exist, create it
        if not hasattr(self, 'kin_filtered'):
            self.kin_filtered = {'Right': {}, 'Left': {}}

        # Get the kinematic data
        kin_data = self.kinematic[side]

        # Initialize a dictionary to hold the filtered data
        filtered_data = {}

        # Design a high-pass Butterworth filter
        Nyquist_freq = 0.5 * self.point_rate
        cutoff_normal = cutoff_freq / Nyquist_freq
        b, a = butter(filter_order, cutoff_normal, btype='high')

        # Detrend and filter each kinematic signal
        for kin_point in kin_data:
            # Detrending            
            # Median Filtering to remove spike noise
            # median_filtered_data = medfilt(detrended_data, median_kernel_size)
            
            # High-pass filtering
            filtered_data[kin_point] = filtfilt(b, a, kin_data[kin_point])
            # filtered_data[kin_point] = detrend(filtered_data[kin_point],type='linear')
            # filtfilt(b, a, median_filtered_data)

        # Store the filtered data
        self.kin_filtered[side] = filtered_data
        # a=self.kin_filtered[side]['RAnkleAngles_x']
    
        # Set the flag to indicate that kinematic data preprocessing is complete
        setattr(self, 'kin_preprocessing_calculated', True)


    def force_event_new(self, side,data_type, height_factor=0.5, distance=500, add_indices=None, remove_indices=None, plot=False):
        self.events = {'Right': {}, 'Left': {},'all':{}}
        assert side in ['Left', 'Right'], "Invalid side. Must be 'Left' or 'Right'"
        
        # Ensure the kinetic data is present
        assert self.kinetic is not None, "Kinetic data is not present. Make sure to import data first."

        # Select the force data of the appropriate side
        if side == 'Left':
            force_data = self.kinetic['kineticDataL']
        elif side == 'Right':
            force_data = self.kinetic['kineticDataR']
        mat_contents = io.loadmat('stepData.mat')
        # Extract step data
        step_data = mat_contents['stepDataP']

        subject = self.name.split('_')[0]
        condition= self.name.split('_')[2][0:4]
        
        # Now you can access the data using subject and condition as keys
        heel_strike_indices= (step_data[0, 0][subject][0][condition][0]/10).astype('int')*10
        self.events[side]['heel_strike']=heel_strike_indices.squeeze()
        setattr(self, f'{data_type}_force_event_calculated', True)
        # Optional plot
        if plot:
            plt.figure()
            # plt.plot(force_data, label='Slope ')
            plt.plot(force_data, label='Slope of force data')
            plt.plot(heel_strike_indices, force_data[heel_strike_indices], 'ro', label='Heel strikes')
            # plt.plot(toe_off_indices, force_filtered[toe_off_indices], 'bo', label='Toe offs')
            plt.legend()
            plt.show()
        return heel_strike_indices


    def separate_events(self,data_type):
        # data_type=self.data_type
        side='Right'
        # If 'emg_separated' attribute does not exist, initialize it
        if not hasattr(self, data_type + '_separated'):
            setattr(self, data_type + '_separated', {'Right': {}, 'Left': {}})
            
        # if side not in getattr(self, data_type + '_separated'):
        #     getattr(self, data_type + '_separated')[side] = {}
        # Pre-allocate an array to hold all the gait cycles
        n_cycles = len(self.events[side]['heel_strike']) - 1
        if data_type == 'emg':
            data = self.emg_filtered
            rate_ratio = 1  # event indices match the EMG data frame rate
        elif data_type == 'kin':
            data = self.kin_filtered
            rate_ratio = self.point_rate / self.analog_rate  # adjust event indices to match the kinematic data frame rate
        else:
            raise ValueError('data_type must be either "emg" or "kin"')
        n_datapoints = len(data[side])
        gait_cycles_all = np.zeros((n_cycles, n_datapoints, 100))
        # For each data point
        for data_idx, datapoint in enumerate(data[side]):
            # Get the heel strike indices for the current side, adjusted for the current data type frame rate
            heel_strikes = [int(event * rate_ratio) for event in self.events[side]['heel_strike']]
            # Get the data for the current datapoint
            datapoint_data = data[side][datapoint]
            # For each gait cycle
            for i in range(n_cycles):
                # Get the data for the current gait cycle
                cycle_data = datapoint_data[heel_strikes[i]:heel_strikes[i + 1]]
                # Create a function to interpolate the cycle_data to 100 points
                interp_func = interp1d(np.linspace(0, 1, len(cycle_data)), cycle_data)
                # Apply the function to an array of 100 points between 0 and 1
                gait_cycles_all[i, data_idx] = interp_func(np.linspace(0, 1, 100))
            # Store the gait cycles for the current datapoint in dictionary
            getattr(self, data_type + '_separated')[side][datapoint] = gait_cycles_all[:, data_idx]
        setattr(self, f'{data_type}_separate_events_calculated', True)

    def calculate_avg_max_peak(self,data_type):
        # data_type=self.data_type
        if data_type not in ['emg', 'kin']:
            raise ValueError('data_type must be either "emg" or "kin"')
        if not hasattr(self, data_type + '_final'):
            setattr(self, data_type + '_avg_max_peak', {'Right': {}, 'Left': {}})

        side = 'Right'

        # For each datapoint
        for datapoint in getattr(self, data_type + '_separated')[side]:
            # Get the separated cycles for the current datapoint
            separated_cycles = getattr(self, data_type + '_separated')[side][datapoint]

            # Define a list to store max peak values for each cycle
            max_peaks = []

            # For each cycle
            for cycle_data in separated_cycles:
                # Find the max peak in the cycle_data
                max_peak = np.max(cycle_data)

                # Append the max peak to the list
                max_peaks.append(max_peak)
            
            # Calculate the average of the max peaks
            if data_type == 'emg':
                self.emg_avg_max_peak[side][datapoint] = np.mean(max_peaks)

            else:
                self.kin_avg_max_peak[side][datapoint] = np.mean(max_peaks)
        setattr(self, f'{data_type}_calculate_avg_max_peak_calculated', True)    
        
    def normalize_data(self ,data_type,side='Right'):
        # data_type=self.data_type
        if data_type not in ['emg', 'kin']:
            raise ValueError('data_type must be either "emg" or "kin"')
        if not hasattr(self, data_type + '_final'):
            setattr(self, data_type + '_final', {'Right': {}, 'Left': {}})
        # Initialize the final dictionary
        final_data = {}

        # Normalize each data's
        if data_type == 'emg':
            avg_max_peak=self.emg_avg_max_peak
        elif data_type == 'kin':
            avg_max_peak=self.kin_avg_max_peak
        for datapoint in getattr(self, data_type + '_separated')[side]:
            # if data_type=='emg':
            final_data[datapoint] = getattr(self, data_type + '_filtered')[side][datapoint] / avg_max_peak[side][datapoint]
            # elif data_type=='kin':
            #     final_data[datapoint] = self.kinematic[side][datapoint] / avg_max_peak[side][datapoint]   
            # Store the final data in the appropriate attribute
        if data_type == 'emg':
            self.emg_final[side] = final_data
        else:
            self.kin_final[side] = final_data
        setattr(self, f'{data_type}_normalize_data_calculated', True)
        
    def plot_data(self, data_type,side='Right'):
        # data_type=self.data_type
        if data_type == 'emg':
            data_label = self.emg['Label']
            separated_data = self.emg_separated
            avg_max_peak = self.emg_avg_max_peak
        elif data_type == 'kin':
            data_label = self.kinematic['Label']
            separated_data = self.kin_separated
            avg_max_peak = self.kin_avg_max_peak
            
        n_data = len(data_label)

        fig, axs = plt.subplots(n_data//3 + n_data%3, 3, figsize=(10, n_data*1.5))
        fig.suptitle(f'Mean {data_type.upper()} - {self.name}', fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=0.4)

        axs = axs.ravel()

        for i, label in enumerate(data_label):
            cycles = np.stack(separated_data[side][label])
            mean_data = np.mean(cycles, axis=0)
            std_data = np.std(cycles, axis=0)

            mean_data_normalized = mean_data / avg_max_peak[side][label]
            std_data_normalized = std_data / avg_max_peak[side][label]

            axs[i].plot(mean_data_normalized, color='tab:blue')
            axs[i].fill_between(range(len(mean_data_normalized)), 
                                mean_data_normalized - std_data_normalized, 
                                mean_data_normalized + std_data_normalized, 
                                color='lightgray', alpha=0.5)
            axs[i].set_title(label)
            axs[i].set_xlabel('Gait cycle %')
            # axs[i].set_ylim([0,1])
            # axs[i].set_yticks([0,1])
            # axs[i].set_yticklabels([])

        if n_data % 3 != 0:
            for i in range(n_data, len(axs)):
                fig.delaxes(axs[i])

        plt.savefig(f'figures\\{data_type}\\{self.name}_{data_type}_preprocessing.png',dpi=500)

        setattr(self, f'{data_type}_plot_data_calculated', True)



    def calculate_synergies(self, data_type, n_synergies=5, method='nmf', num_epochs=100):
        def separate_angles(kin_final,side='Right'):
            kin_final_label = {'Right': [], 'Left': []}
            # Get the kinematic data
            kin_data = kin_final[side]

            # Initialize dictionaries to hold the separated data
            separated_data_positive = {}
            separated_data_negative = {}

            # List of the new labels
            new_labels = ['A/DF', 'A/PF', 'A/Inv', 'A/Add', 'K/Flex', 'K/Ext', 'K/Abd', 'K/Add', 'H/Flex', 'H/Ext', 'H/Add', 'H/Abd', 'S/FT', 'S/BT', 'S/RT', 'S/LT']

            # First resample the data
            resampled_data = {}
            for kin_point in kin_data:
                data = kin_data[kin_point]
                n_old = len(data)
                n_new = int(n_old * self.analog_rate / self.point_rate)
                resampled_data[kin_point] = resample(data, n_new)

            # Then separate the positive and negative parts of each kinematic signal
            for i, kin_point in enumerate(resampled_data):
                # Get the data for the kinematic point
                data = resampled_data[kin_point]

                # Separate and take the absolute values of the negative and positive parts
                separated_data_positive[new_labels[i*2]] = np.abs(data * (data > 0))  # index i*2 corresponds to the positive parts
                separated_data_negative[new_labels[i*2+1]] = np.abs(data * (data < 0))  # index i*2+1 corresponds to the negative parts

            # Update the kinematic data with the separated data
            # kin_final = {**separated_data_positive, **separated_data_negative}
            kin_final = {k: v for pair in zip(separated_data_positive.items(), separated_data_negative.items()) for k, v in pair}
            kin_final_label = new_labels
            return kin_final_label,kin_final        
        no_dof=[0,1,2,3,4,8,9,10,11,12,13,14,15]
        side='Right'
        if data_type not in ['emg', 'kin']:
            raise ValueError('data_type must be either "emg" or "kin"')
        
        if data_type == 'emg':
            data_final = self.emg_final[side]
            data_label = self.emg['Label']
        elif data_type == 'kin':
            data_label,data_final = separate_angles(self.kin_final,side)
        # Normalize and concatenate data
        data = []
        for item in data_final:
            item_data = data_final[item]
            data.append(item_data.reshape(1,-1))
        data = np.concatenate(data, axis=0)
        if data_type == 'emg':
            data=data
        elif data_type == 'kin':
            data=data[no_dof,:]
            data_label=[data_label[i] for i in no_dof]
            self.kinematic['new_labels']=data_label
        # Perform synergy analysis
        if method == 'nmf':
            # Define the parameter values to explore
            # parameter_values = [
            #     {'alpha_W': 0.01, 'alpha_H': 0.01, 'l1_ratio': 0.01},
            #     {'alpha_W': 0.1, 'alpha_H': 0.1, 'l1_ratio': 0.1},
            #     {'alpha_W': 1.0, 'alpha_H': 1.0, 'l1_ratio': 0.5},
            #     {'alpha_W': 0.001, 'alpha_H': 0.001, 'l1_ratio': 0.001},
            #     {'alpha_W': 0.5, 'alpha_H': 0.5, 'l1_ratio': 0.2}
            # ]

                # Add more parameter combinations as needed
                    
            model = NNMFSynergy(n_synergies, self.name, data_label)
            # best_params = model.tune_parameters(data, self.events['Right']['heel_strike'], parameter_values)        
            # model.model.set_params(**best_params)
            model.fit(data, self.events['Right']['heel_strike'])
            model.organize_synergies_by_activation()
            model.plot_synergies(data_type)
        elif method == 'autoencoder':
            data = data.T
            input_dim = data.shape[1]
            encoding_dim = n_synergies

            model = MuscleSynergyAutoencoder(input_dim, encoding_dim, data_label, self.events['Right']['heel_strike'])

            # Convert the data to PyTorch tensors
            data_tensor = torch.from_numpy(data).float().requires_grad_(False)

            # Train the autoencoder
            model = model.fit(data_tensor)

        else:
            raise ValueError(f"Unsupported method: {method}")

        # Save results in gait_analysis
        model_name=f'{data_type}_synergy_model'
        setattr(self,model_name,model)
        setattr(self, f'{data_type}_calculate_synergies_calculated', True)  
   
           
    def is_calculated(self, method_name, data_type):
        # Assume that method_name corresponds to an attribute in the class
        # with the suffix '_calculated' and prefixed by data_type
        if hasattr(self, f'{data_type}_{method_name}_calculated') and \
        getattr(self, f'{data_type}_{method_name}_calculated'):
            return True
        else:
            return False

