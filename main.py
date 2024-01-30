import os
import pickle
from gait_analysis import GaitAnalysis
from synergy_clustering import SynergyClustering

def execute_and_save(function_name, gait_analysis, filename, data_type, function_sequence, **kwargs):
    if function_name in function_sequence and not gait_analysis.is_calculated(function_name, data_type):
        getattr(gait_analysis, function_name)(data_type=data_type, **kwargs)
        with open(filename, "wb") as f:
            pickle.dump(gait_analysis, f)
    return gait_analysis

def perform_gait_analysis():
    subjects = ['P12']
    conditions = ['noRAC_slowWalk']
    # conditions = ['noRAC_slowWalk', 'noRAC_preferredWalk', 'noRAC_fastWalk']

    pathData = 'J:\\Projet_RAC\\DATA\\'
    data_types = ['emg', 'kin']
    function_sequence_emg = ['import_data', 'emg_preprocessing', 'force_event', 'separate_events', 
                             'calculate_avg_max_peak', 'plot_data', 'normalize_data', 'calculate_synergies']

    function_sequence_kinematic = ['kin_preprocessing','separate_events', 'calculate_avg_max_peak', 'plot_data', 'normalize_data','calculate_synergies']

    for subject in subjects:
        for condition in conditions:
            filename = f"Data\\gait_analysis\\{subject}_{condition}_gait_analysis.pkl"
            
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    gait_analysis = pickle.load(f)
            else:
                gait_analysis = GaitAnalysis(pathData + 'RAW\\' + subject + '\\Nexus\\' + condition + '.c3d',f'{subject}_{condition}')
            
            for data_type in data_types:
                function_sequence = function_sequence_emg if data_type == 'emg' else function_sequence_kinematic
                for function_name in function_sequence:
                    if function_name == 'force_event':
                        gait_analysis = execute_and_save(function_name, gait_analysis, filename, data_type, function_sequence, side='Right')
                    elif function_name == 'calculate_synergies':
                        gait_analysis = execute_and_save(function_name, gait_analysis, filename, data_type, function_sequence, method='nmf')
                    else:
                        gait_analysis = execute_and_save(function_name, gait_analysis, filename, data_type, function_sequence)
                        # Plot and save figure
                synergy_model = getattr(gait_analysis, f"{data_type}_synergy_model")
                synergy_model.plot_synergies(data_type)
                # gait_analysis.synergy_model.plot_synergies(data_type)
            print(f"Finished {subject} {condition}")
            # Save the final analysis
            with open(filename, "wb") as f:
                pickle.dump(gait_analysis, f)
                


def main():
    ########Perform gait analysis
    perform_gait_analysis()

    # Clustering synergies
    synergy_cluster = SynergyClustering(n_clusters=5)
    subjects = ['P12','P11']
    conditions = ['noRAC_slowWalk']#, 'noRAC_preferredWalk', 'noRAC_fastWalk']
    synergy_cluster.load_synergies(subjects, conditions)

    synergy_cluster.cluster_synergies()
    # synergy_cluster.cluster_activations()
    synergy_cluster.plot_weight_clusters()
    synergy_cluster.plot_activation_clusters()
    synergy_cluster.plot_cluster_means()
    # print("Cluster labels:", cluster_labels)

if __name__ == '__main__':
    main()
