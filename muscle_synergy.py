from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg to save figures without displaying them
from sklearn.utils import resample
from torch import nn
import os
from sklearn.metrics import r2_score

class NNMFSynergy :
    def __init__(self, n_synergies, name, labels, bootstrap_iterations=100):
        self.n_synergies = n_synergies
        self.model = NMF(n_components=n_synergies, init='nndsvd', random_state=0, max_iter=5000)
        self.bootstrap_iterations = bootstrap_iterations
        self.name = name
        self.labels = labels

    def normalize_variance(self, data):
        self.std = np.std(data, axis=1, keepdims=True)
        return data / self.std

    def fit(self, data, events):
        normalized_data = self.normalize_variance(data)
        # self.bootstrap(normalized_data)
        self.W = self.model.fit_transform(normalized_data)*self.std
        self.H = self.model.components_
        self.events = events  # Store gait events
        return self.W, self.H  
    

    def tune_parameters(self, data, events, parameter_values):
        vaf_scores = []
        r2_scores = []
        parameter_strings = []
        
        for params in parameter_values:
            parameter_strings.append(', '.join(f'{k}={v}' for k, v in params.items()))
            
            self.model.set_params(**dict(params))
            self.fit(data, events)
            self.organize_synergies_by_activation()
            reconstructed_data = self.reconstruct()
            
            vaf = 1 - np.var(data - reconstructed_data) / np.var(data)
            r2 = r2_score(data, reconstructed_data)
            
            vaf_scores.append(vaf)
            r2_scores.append(r2)

        best_params = parameter_strings[np.argmax(vaf_scores)]

        # Plot and save the VAF values
        fig, ax = plt.subplots()
        ax.plot(parameter_strings, vaf_scores, label='VAF Score')
        ax.plot(parameter_strings, r2_scores, label='R-squared')
        ax.set_xlabel('Parameter Settings')
        ax.set_ylabel('Score')
        ax.set_title('VAF Scores and R-squared for Different Parameter Settings')
        ax.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()
        save_dir = os.path.join('figures', 'tuning')
        save_path = os.path.join(save_dir, 'tuning_plot.png')
        plt.savefig(save_path)
        plt.close(fig)

        return best_params, save_path



      
    def organize_synergies_by_activation(self):
        peak_times = []
        standard_len = 100
        heel_strikes = self.events  # assuming self.events is already a list of heel_strike frame indices
        for i in range(len(heel_strikes) - 1):
            cycle_activations = self.H[:, heel_strikes[i]:heel_strikes[i+1]]
            resampled_activations = np.empty((cycle_activations.shape[0], standard_len))
            for j in range(cycle_activations.shape[0]):
                x_old = np.linspace(0, 1, len(cycle_activations[j]))
                x_new = np.linspace(0, 1, standard_len)
                resampled_activations[j] = np.interp(x_new, x_old, cycle_activations[j])
            cycle_peak_times = np.argmax(resampled_activations, axis=1)
            peak_times.append(cycle_peak_times)
        avg_peak_times = np.mean(peak_times, axis=0)
        sorting_indices = np.argsort(avg_peak_times)
        self.W = self.W[:, sorting_indices]
        self.H = self.H[sorting_indices, :]

    def bootstrap(self, data):
        # Initialize consensus matrix
        self.consensus_matrix = np.zeros((data.shape[0], data.shape[0]))

        for _ in range(self.bootstrap_iterations):
            # Generate bootstrap sample
            bootstrap_sample = resample(data, replace=True)
            # Fit model to bootstrap sample
            W_bootstrap = self.model.fit_transform(bootstrap_sample)
            # Update consensus matrix
            self.consensus_matrix += np.matmul(W_bootstrap, W_bootstrap.T)

        # Normalize consensus matrix
        self.consensus_matrix /= self.bootstrap_iterations

    def reconstruct(self, W=None, H=None):
        if W is None:
            W = self.W
        if H is None:
            H = self.H
        reconstructed_data = np.matmul(W, H)
        return reconstructed_data   # reverse the normalization



    def plot_activation(self, axs, colors):
        for i in range(self.n_synergies):
            # Calculate mean and std of activation signals across cycles
            activations = []
            for start, end in zip(self.events[:-1], self.events[1:]):
                activation = self.H[i, start:end]

                # Interpolate activation to 100 points
                x_old = np.linspace(0, 1, len(activation))
                x_new = np.linspace(0, 1, 100)
                activation_interp = np.interp(x_new, x_old, activation)
                
                activations.append(activation_interp)

            activations = np.array(activations)

            mean_activation = np.mean(activations, axis=0)
            std_activation = np.std(activations, axis=0)
            time = np.linspace(0, 1, 100)  # Change time to percent

            # Plot mean activation with std as shaded area
            axs[i, 1].plot(time, mean_activation, color=colors[i])
            axs[i, 1].fill_between(time, mean_activation - std_activation, mean_activation + std_activation, alpha=0.2, color=colors[i])
            axs[i, 1].set_yticks([])  # remove y ticks
            if i == 0:
                axs[i, 1].set_title(f'Activation', fontsize=14)
            if i == self.n_synergies - 1:
                axs[i, 1].set_xlabel('Gait cycle %', fontsize=14)
                axs[i, 1].set_xticks([0, 0.5, 1])
                axs[i, 1].set_xticklabels(['0', '50', '100'])
            else:
                axs[i, 1].set_xticks([])

    def plot_synergies(self,data_type):
        # Define colormap
        cmap = cm.get_cmap('tab10', self.n_synergies)
        colors = cmap(range(self.n_synergies))

        fig, axs = plt.subplots(self.n_synergies, 2, figsize=(10, self.n_synergies*2))

        # Title for the whole figure
        if data_type == 'emg':
            segment_title='Muscle synergies'
        elif data_type == 'kin':
            segment_title='Kinematic synergies'
        fig.suptitle(f'{segment_title} - {self.name}', fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # To provide space for global title

        for i in range(self.n_synergies):
            axs[i, 0].bar(range(self.W.shape[0]), self.W[:, i], color=colors[i])
            axs[i, 0].set_ylabel(f'Synergy #{i+1}', fontsize=14)
            axs[i, 0].set_yticks([])  # remove y ticks
            if i == 0:
                axs[i, 0].set_title('Weight', fontsize=14)
            if i == self.n_synergies - 1:
                axs[i, 0].set_xticks(range(self.W.shape[0]))
                axs[i, 0].set_xticklabels(self.labels, rotation=45, ha='right')
            else:
                axs[i, 0].set_xticks([])

        self.plot_activation(axs, colors)
        
        plt.tight_layout()
        plt.savefig(f'figures\\synergies\\{self.name}_synergies_{data_type}.png', dpi=500)  # Save the figure
        # plt.show()
        
    def plot_consensus_matrix(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.consensus_matrix, cmap='viridis')
        plt.colorbar(label='Consensus')
        plt.title(f'Consensus Matrix - {self.name}', fontsize=16, fontweight='bold')
        plt.xlabel('Muscle', fontsize=14)
        plt.ylabel('Muscle', fontsize=14)
        plt.savefig(f'figures\\consensus\\{self.name}_consensus.png', dpi=500)
        plt.show()
        
from torch import nn
from torch.optim import Adam
from matplotlib import cm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

class MuscleSynergyAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, labels, events,num_epochs=1000, learning_rate=1e-3, weight_decay=1e-3):
        super(MuscleSynergyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim), 
            nn.Sigmoid(), 
        )
        self.n_synergies = hidden_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.labels = labels
        self.events = events
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, X):
        criterion = nn.MSELoss()
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in range(self.num_epochs):
            output = self(X)  # Use the model instance directly
            loss = criterion(output, X)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, self.num_epochs, loss.item()))
        output = self.encoder(X)  # Use input X instead of data
        self.H = output.detach().numpy()  # save the activations as a numpy array
        self.W = self.encoder[0].weight.data.numpy()  # save the weights as a numpy array
        return self
    
    def reconstruct(self):
        reconstructed_data = np.matmul(self.W, self.H.T)
        return reconstructed_data


    def plot_activation(self, axs, colors, n_synergies):
        for i in range(n_synergies):
            # Calculate mean and std of activation signals across cycles
            activations = []
            for start, end in zip(self.events[:-1], self.events[1:]):
                activation = self.H[start:end, i]
                activation=activation.T
                # Interpolate activation to 100 points
                x_old = np.linspace(0, 1, len(activation))
                x_new = np.linspace(0, 1, 100)
                activation_interp = np.interp(x_new, x_old, activation)
                
                activations.append(activation_interp)

            activations = np.array(activations)

            mean_activation = np.mean(activations, axis=0)
            std_activation = np.std(activations, axis=0)
            time = np.linspace(0, 1, 100)  # Change time to percent

            # Plot mean activation with std as shaded area
            axs[i, 1].plot(time, mean_activation, color=colors[i])
            axs[i, 1].fill_between(time, mean_activation - std_activation, mean_activation + std_activation, alpha=0.2, color=colors[i])
            axs[i, 1].set_yticks([])  # remove y ticks
            if i == 0:
                axs[i, 1].set_title(f'Activation', fontsize=14)
            if i == n_synergies - 1:
                axs[i, 1].set_xlabel('Gait cycle %', fontsize=14)
                axs[i, 1].set_xticks([0, 0.5, 1])
                axs[i, 1].set_xticklabels(['0', '50', '100'])
            else:
                axs[i, 1].set_xticks([])

    def plot_synergies(self):
        n_synergies=self.n_synergies
        labels=self.labels
        # Define colormap
        cmap = cm.get_cmap('tab10', n_synergies)
        colors = cmap(range(n_synergies))

        fig, axs = plt.subplots(n_synergies, 2, figsize=(10, n_synergies*2))

        # Title for the whole figure
        fig.suptitle(f'Muscle Synergy - Autoencoder', fontsize=16, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # To provide space for global title

        for i in range(n_synergies):
            axs[i, 0].bar(range(self.W.shape[1]), self.W[i, :], color=colors[i])
            axs[i, 0].set_ylabel(f'Synergy #{i+1}', fontsize=14)
            axs[i, 0].set_yticks([])  # remove y ticks
            if i == 0:
                axs[i, 0].set_title('Weight', fontsize=14)
            if i == n_synergies - 1:
                axs[i, 0].set_xticks(range(self.W.shape[1]))
                axs[i, 0].set_xticklabels(labels, rotation=45, ha='right')
            else:
                axs[i, 0].set_xticks([])

        self.plot_activation(axs, colors, n_synergies)
        
        plt.tight_layout()
        plt.savefig(f'figures\\synergies\\autoencoder_synergies.png', dpi=500)  # Save the figure
        plt.show()
