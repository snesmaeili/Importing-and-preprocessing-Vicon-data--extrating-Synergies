from sklearn.cluster import KMeans
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from matplotlib import cm
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

class SynergyClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.synergies = {'emg': {}, 'kin': {}}
        self.activations={'emg': {}, 'kin': {}}
        self.labels = {'kin':[],'emg':[]}
        self.events = {'heel_strike':{},'toe_off':{}}
        


    def load_synergies(self, subjects, conditions):
        
        for subject in subjects:
            for condition in conditions:
                filename = f"Data\\gait_analysis\\{subject}_{condition}_gait_analysis.pkl"
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        gait_analysis = pickle.load(f)
                    self.synergies['emg'][f'{subject}_{condition}'] = gait_analysis.emg_synergy_model.W
                    self.synergies['kin'][f'{subject}_{condition}'] = gait_analysis.kin_synergy_model.W
                    self.activations['emg'][f'{subject}_{condition}'] = gait_analysis.emg_synergy_model.H
                    self.activations['kin'][f'{subject}_{condition}'] = gait_analysis.kin_synergy_model.H
                    self.labels['kin']=gait_analysis.kin_synergy_model.labels
                    self.labels['emg']=gait_analysis.emg_synergy_model.labels
                    self.events['heel_strike'][f'{subject}_{condition}']=gait_analysis.events['Right']['heel_strike']


    def cluster_synergies(self):
        self.predicted_clusters = {'emg': {}, 'kin': {}}
        for data_type in ['emg', 'kin']:
            synergies_flat = []
            for synergy in self.synergies[data_type].values():
                # Check if synergy is a 2D array
                if len(synergy.shape) == 2:
                    # Add each flattened synergy to the list
                    synergies_flat.extend([np.ndarray.flatten(single_synergy) for single_synergy in synergy.T])
                else:  # if the synergy is already a 1D array, no need to loop through it
                    synergies_flat.append(np.ndarray.flatten(synergy))

            # Feature scaling
            scaler = StandardScaler()
            synergies_flat_scaled = scaler.fit_transform(synergies_flat)

            # Perform clustering using K-means++
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=0).fit(synergies_flat_scaled)
            cluster_labels = kmeans.labels_

            # Store cluster labels for each trial separately
            trial_names = list(self.synergies[data_type].keys())
            num_synergies = self.synergies[data_type][trial_names[0]].shape[1]
            trial_cluster_labels = []
            for i, trial_name in enumerate(trial_names):
                trial_cluster_labels.append(cluster_labels[i * num_synergies: (i + 1) * num_synergies])
            trial_cluster_labels = np.array(trial_cluster_labels)

            # Check if trials have all clusters equal to the number of synergies
            for i, trial_name in enumerate(trial_names):
                unique_labels = np.unique(trial_cluster_labels[i])
                if len(set(range(num_synergies)))!=len (set(unique_labels)):
                    missing_clusters = set(range(num_synergies)) - set(unique_labels)

                    # Find all clusters with the same ID in other trials
                    for missing_cluster in missing_clusters:
                        cluster_weights = []
                        for j, other_trial_name in enumerate(trial_names):
                            if j != i:
                                other_cluster_indices = np.where(trial_cluster_labels[j] == missing_cluster)[0]
                                other_cluster_weights = self.synergies[data_type][other_trial_name][:, other_cluster_indices]
                                cluster_weights.append(other_cluster_weights)

                        cluster_weights=np.array(cluster_weights)
                        cluster_weights = np.squeeze(cluster_weights, axis=2)
                        cluster_mean_weights=np.mean(cluster_weights,axis=0)
                        # Calculate correlation between mean weights and repetitive weights
                        # repeated_number=np.where(np.bincount(trial_cluster_labels[i])!=1)[0]
                        repeated_number = [num for num in np.unique(trial_cluster_labels[i]) if np.count_nonzero(trial_cluster_labels[i] == num) > 1]

                        repeat_cluster_indices=np.where(trial_cluster_labels[i] == repeated_number)[0]
                        trial_cluster_weights = self.synergies[data_type][trial_name][:, repeat_cluster_indices]
        
                        correlation_scores = []
                        for trial_cluster_weight in trial_cluster_weights.T:
                            correlation, _ = pearsonr(trial_cluster_weight, cluster_mean_weights)
                            correlation_scores.append(correlation)

                        # Select the cluster with the highest correlation as the true cluster
                        closest_cluster_index = np.argmax(correlation_scores)
                        trial_cluster_labels[i][repeat_cluster_indices[closest_cluster_index]] = missing_cluster

            for i, trial_name in enumerate(trial_names):
                self.predicted_clusters[data_type][trial_name] = trial_cluster_labels[i]


            # kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(synergies_flat)
            # self.predicted_clusters[data_type] = kmeans.labels_


    def plot_weight_clusters(self):
        for data_type in ['emg', 'kin']:
            synergy_example = next(iter(self.synergies[data_type].values()))
            num_synergies = synergy_example.shape[1]  # Get the number of synergies from the shape
            num_trials = len(self.synergies[data_type])
            fig, axes = plt.subplots(num_synergies, num_trials, figsize=(15, 15))

            # use a colormap that looks good in publications
            cmap = cm.get_cmap('tab10')

            for i, (trial_name, synergy) in enumerate(self.synergies[data_type].items()):
                subject, condition = trial_name.split('_', 1)
                cluster_labels = self.predicted_clusters[data_type][trial_name]

                # Sort the synergies based on cluster labels
                sorted_indices = np.argsort(cluster_labels)
                synergy_sorted = synergy[:, sorted_indices]

                for j in range(num_synergies):
                    # Adjusted to consider synergies in columns
                    color = cmap(cluster_labels[sorted_indices][j] / self.n_clusters)
                    axes[j, i].bar(np.arange(synergy.shape[0]), synergy_sorted[:, j], color=color)
                    if j == 0:
                        axes[j, i].set_title(f'{subject}\n{condition}')
                    if i == 0:
                        axes[j, i].set_ylabel(f'Synergy {j + 1}')
                    # remove y axis labels
                    axes[j, i].set_yticklabels([])
                    # remove x axis labels for all but the last row
                    if j < num_synergies - 1:
                        axes[j, i].set_xticklabels([])
                    # Add xtick labels for the last row
                    if j == num_synergies - 1:
                        axes[j, i].set_xticks(np.arange(synergy.shape[0]))
                        axes[j, i].set_xticklabels(self.labels[data_type],rotation=45,ha='right')

            fig.tight_layout()
            plt.savefig(f"clusters/{data_type}_weight_clusters.png")
            plt.close()

    def plot_activation_clusters(self):
        for data_type in ['emg', 'kin']:
            activations = self.activations[data_type]
            activation_example = next(iter(activations.values()))
            num_synergies = activation_example.shape[0]  # Get the number of synergies from the shape
            num_trials = len(activations)
            fig, axes = plt.subplots(num_synergies, num_trials, figsize=(15, 15))

            # use a colormap that looks good in publications
            cmap = cm.get_cmap('tab10')

            for i, (trial_name, activation) in enumerate(activations.items()):
                subject, condition = trial_name.split('_', 1)
                heel_strikes = self.events['heel_strike'][f'{subject}_{condition}']
                cluster_labels = self.predicted_clusters[data_type][trial_name]

                # Sort the activations based on cluster labels
                sorted_indices = np.argsort(cluster_labels)
                activation_sorted = activation[sorted_indices, :]

                for j in range(num_synergies):
                    # Calculate the mean activation for each cycle
                    resampled_cycle_activations = []
                    for cycle_start, cycle_end in zip(heel_strikes[:-1], heel_strikes[1:]):
                        original_cycle_activation = activation_sorted[j, cycle_start:cycle_end]
                        interpolator = interp1d(np.linspace(0, 1, original_cycle_activation.shape[0]), original_cycle_activation, kind='linear')
                        resampled_cycle_activation = interpolator(np.linspace(0, 1, 100))
                        resampled_cycle_activations.append(resampled_cycle_activation)

                    resampled_cycle_activations = np.array(resampled_cycle_activations)
                    mean_activation = np.mean(resampled_cycle_activations, axis=0)
                    std_activation = np.std(resampled_cycle_activations, axis=0)

                    time_points = np.arange(len(mean_activation))

                    # Use color based on cluster number
                    color = cmap(cluster_labels[sorted_indices][j] / self.n_clusters)

                    axes[j, i].plot(time_points, mean_activation, color=color)
                    axes[j, i].fill_between(time_points, mean_activation - std_activation, mean_activation + std_activation,
                                            color=color, alpha=0.2)
                    if j == 0:
                        axes[j, i].set_title(f'{subject}\n{condition}')
                    if i == 0:
                        axes[j, i].set_ylabel(f'Synergy {j + 1}')
                    # remove y axis labels
                    axes[j, i].set_yticklabels([])
                    # remove x axis labels for all but the last row
                    if j < num_synergies - 1:
                        axes[j, i].set_xticklabels([])

            fig.tight_layout()
            plt.show()
            plt.savefig(f"clusters/{data_type}_activation_clusters.png")
            plt.close()






 
    def plot_cluster_means(self):
        for data_type in ['emg', 'kin']:
            fig, axes = plt.subplots(1, 2, figsize=(20,10))  # Create two subplots side by side

            # Calculate means and plot for weights
            weights_flat = [np.ndarray.flatten(weight) for weight in self.synergies[data_type].values()]
            for i in range(self.n_clusters):
                cluster_indices = np.where(self.predicted_clusters[data_type] == i)
                cluster_data = [weights_flat[idx] for idx in cluster_indices[0]]
                cluster_mean = np.mean(cluster_data, axis=0)
                cluster_std = np.std(cluster_data, axis=0)
                bars = axes[0].bar(range(len(cluster_mean)), cluster_mean, yerr=cluster_std, alpha=0.5, ecolor='black', capsize=5)
            axes[0].set_title("Weights")
            axes[0].set_xlabel("Synergy Weight")
            axes[0].set_ylabel("Mean Value")

            # Calculate means and plot for activations
            activations_flat = [segment_by_gait_events(activation, gait_event) for activation, gait_event in zip(self.activations[data_type].values(), self.gait_events.values())]
            for i in range(self.n_clusters):
                cluster_indices = np.where(self.predicted_clusters[data_type] == i)
                cluster_cycles = [activations_flat[idx] for idx in cluster_indices[0]]
                # Compute mean across cycles, then across trials
                cluster_mean = np.mean([np.mean(cycle, axis=0) for cycle in cluster_cycles], axis=0)
                cluster_std = np.std([np.std(cycle, axis=0) for cycle in cluster_cycles], axis=0)
                bars = axes[1].bar(range(len(cluster_mean)), cluster_mean, yerr=cluster_std, alpha=0.5, ecolor='black', capsize=5)
            axes[1].set_title("Activations")
            axes[1].set_xlabel("Synergy Activation")
            axes[1].set_ylabel("Mean Value")

            fig.suptitle(f"{data_type.capitalize()} Cluster Means")
            plt.savefig(f"clusters/{data_type}_cluster_means.png")
            plt.close()

