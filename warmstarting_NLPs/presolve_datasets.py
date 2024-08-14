import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ast import literal_eval
from torch.utils.data import Dataset


class NLPDataset(Dataset):
    def __init__(self, path, train):
        self.df = pd.read_csv(path)

        # Process columns of the dataframe that contain arrays that are formatted as strings in the csv
        array_elements = ['params', 'X', 'U']
        for elem in array_elements:
            self.df[elem] = self.df[elem].apply(literal_eval) # Convert strings into lists 
            self.df[elem] = self.df[elem].apply(lambda x: np.array(x, dtype=np.float32)) # Convert lists into np arrays of type np.float64

        # Split the data into training and testing sets
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=1) # Rand state so that when we create train-test split datasets they dont overlap

        if train:
            self.df = train_df
        else:
            self.df = test_df

        # Store the number of params (NN inputs) and length of trajectory (NN outputs)
        self.n_params = len(self.df['params'].iloc[0])
        N = self.df['X'].iloc[0].shape[0] # number of timesteps
        n_states = len(self.df['X'].iloc[0][0])
        n_controls = len(self.df['U'].iloc[0][0])

        # print(n_states)
        # print(n_controls)
        # print(N)
        self.n_traj = n_states*N + n_controls*(N-1)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        params = self.df.iloc[idx].params
        X = self.df.iloc[idx].X
        U = self.df.iloc[idx].U
        
        Z = np.array([np.hstack((X[i], U[i])) for i in range(len(U))], dtype=np.float32) # create the trajectory vector where element z_i = [x_i, u_i]
        Z = Z.flatten()
        Z = np.concatenate((Z, X[-1])) # add the final state to the end of the trajectory

        return params, Z
    
    # def getitem(self, idx):
    #     params = self.df.iloc[idx].params
    #     X = self.df.iloc[idx].X
    #     U = self.df.iloc[idx].U
        
    #     Z = np.array([np.hstack((X[i], U[i])) for i in range(len(U))]) # create the trajectory vector where element z_i = [x_i, u_i]
        
    #     return params, Z
