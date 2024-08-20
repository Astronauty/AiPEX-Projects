import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from presolve_datasets import NLPDataset
from trajectory_utils import create_idx

class CartpoleNN(nn.Module):
    def __init__(self, n_params, n_traj):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_params, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_traj)
        )

        
    def forward(self, params):
        # params = self.flatten(params)
        traj = self.linear_relu_stack(params)
        return traj

class POCPSolver():
    """
    Loads pre-solved NLP data and trains a neural network to solve the NLP. Computes statistics and visualizations regarding the solutions.
    """
    def __init__(self, path, nlp_params, verbose=False, eq_constraint_fn=None, ineq_constraint_fn=None):
        sns.set_theme()
        
        # Load in equality and inequality constraint functions if PINNs style regularization is desired
        self.eq_constraint_fn = eq_constraint_fn
        self.ineq_constraint_fn = ineq_constraint_fn
        
        # Load the parameters that governed the NLP solve
        self.nlp_params = nlp_params
        
        # Set up data loaders for machine learning regression
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_data = NLPDataset(path, train=True)
        self.test_data = NLPDataset(path, train=False)

        self.train_dataloader = DataLoader(self.train_data, batch_size=1, shuffle=True)        
        self.test_dataloader = DataLoader(self.test_data, batch_size=1, shuffle=True)
        
        # Trajectory info
        self.t_vec = np.linspace(0, nlp_params.tf, nlp_params.N)
        self.idx = nlp_params.idx
            
        # Create NN
        self.n_params = self.train_data.n_params
        self.n_traj = self.train_data.n_traj
        
        self.model = CartpoleNN(self.train_data.n_params, self.train_data.n_traj).to(self.device)
        if verbose:
            print(f"Model structure: {self.model}\n\n")



    def _train_loop(self, loss_fn, optimizer):
        size = len(self.train_dataloader.dataset)
        self.model.train()

        train_loss = 0
        dynamics_eq_loss = 0


        for batch, (params, z_actual) in enumerate(self.train_dataloader):
            params = params.to(self.device).float() # parameters 
            z_actual = z_actual.to(self.device) # actual trajectories

            z_pred = self.model(params)
            loss = loss_fn(z_pred, z_actual)

            if self.eq_constraint_fn is not None:
                eq_loss = self.eq_constraint_fn(self.nlp_params, z_pred)
                # eq_loss = torch.tensor(eq_loss, device=self.device)
                loss += 10*eq_loss

                dynamics_eq_loss += eq_loss.item() 
                
            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(self.train_dataloader)
        dynamics_eq_loss /= len(self.train_dataloader)
        # loss, current = loss.item(), batch * len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return train_loss, dynamics_eq_loss
            

    def _test_loop(self, loss_fn):
        self.model.eval()
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches # average loss per batch
        # print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


        return test_loss
    
    def train(self, epochs, path=None):
        learning_rate = 5e-3
        batch_size = 64
        epochs = epochs
        
        self.writer = SummaryWriter(f'runs/cartpole-{time.strftime("%Y%m%d-%H%M%S")}')

        # loss_fn = nn.MSELoss()

        loss_fn = nn.MSELoss()

        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        print("Fitting NLP data with neural network...")
        for t in tqdm(range(epochs)):
            # print(f"Epoch {t+1}\n-------------------------------")
            train_loss, dynamics_eq_loss = self._train_loop(loss_fn, optimizer)
            scheduler.step()
            self.writer.add_scalar('train loss x epoch', train_loss, t)
            self.writer.add_scalar('dynamics eq loss x epoch', dynamics_eq_loss, t)

            test_loss = self._test_loop(loss_fn)
            self.writer.add_scalar('test loss x epoch', test_loss, t)

        self.writer.flush()
        self.writer.close()

        
    def save_model(self, path):
        try:
            torch.save(self.model.state_dict(), path)
            print("Model saved successfully!")
        except Exception as e:
            print(f"Error saving model: {e}")
            
    def load_model(self, path):
        try:
            # self.model = torch.load(path)
            self.model.load_state_dict(torch.load(path))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

    def compare_trajectories(self, idx):
        """
        Compares the neural network warmstart trajectory with the trajectory obtained from the NLP solution test set.
        """
        assert idx < len(self.test_data)

        plt.figure(figsize=(10,6))
        test_param  = torch.from_numpy(self.test_data.df.iloc[idx].params)
        test_param = test_param.to(self.device)

        self.model.eval()

        with torch.no_grad():
            Z = self.model(test_param)
        Z = Z.detach().cpu().numpy()

        plt.plot(self.t_vec, Z[self.idx.X][:,:2], linewidth=3.0)
        plt.plot(self.t_vec, self.test_data.df.iloc[idx].X[:,0], '--', linewidth=3.0, color='tab:blue')
        plt.plot(self.t_vec, self.test_data.df.iloc[idx].X[:,1], '--', linewidth=3.0, color='tab:orange')
        # font = {'family' : 'sans-serif',
        # 'weight' : 'normal',
        # 'size'   : 14}

        # plt.grid(True)
        # mpl.rc('font', **font)

        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.legend(['x', r'$\theta$', 'x (NLP)', r'$\theta$ (NLP)'])


    def compare_controls(self, idx):
        assert idx < len(self.test_data)

        plt.figure(figsize=(10,6))
        test_param  = torch.from_numpy(self.test_data.df.iloc[idx].params)
        test_param = test_param.to(self.device)

        self.model.eval()

        with torch.no_grad():
            Z = self.model(test_param)
        Z = Z.detach().cpu().numpy()

        plt.plot(self.t_vec, Z[self.idx.X][:,:2], linewidth=3.0)
        plt.plot(self.t_vec, self.test_data.df.iloc[idx].X[:,0], '--', linewidth=3.0, color='tab:blue')
        plt.plot(self.t_vec, self.test_data.df.iloc[idx].X[:,1], '--', linewidth=3.0, color='tab:orange')
        # font = {'family' : 'sans-serif',
        # 'weight' : 'normal',
        # 'size'   : 14}

        # plt.grid(True)
        # mpl.rc('font', **font)

        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.legend(['x', r'$\theta$', 'x (NLP)', r'$\theta$ (NLP)'])
        return None
    
    def store_warmstart_trajectories(self, path):
        warmstart_df = self.test_data.df.copy()
        warmstart_df.shape

        warmstart_df['X_warmstart'] = None
        warmstart_df['U_warmstart'] = None
        warmstart_df['Z_warmstart'] = None
        warmstart_df['inference_time_sec'] = None

        for index, row in warmstart_df.iterrows():
            test_param  = torch.from_numpy(row.params)
            test_param = test_param.to(self.device)
            
            with torch.no_grad():
                start = time.process_time()
                Z = self.model(test_param)
                end = time.process_time()
            Z = Z.detach().cpu().numpy()

            warmstart_df.at[index, 'inference_time_sec'] = end - start
            warmstart_df.at[index, 'X_warmstart'] = Z[self.idx.X]
            warmstart_df.at[index, 'U_warmstart'] = Z[self.idx.U]
            warmstart_df.at[index, 'Z_warmstart'] = Z

        warmstart_df.to_csv(path)

    def compare_solve_time(self):
        warmstart_df = self.test_data.df
        plt.figure(figsize=(10, 6))
        sns.histplot(warmstart_df['solve_time_sec'], bins=30, kde=True)
        plt.xlabel('Solve Time (seconds)')
        plt.ylabel('Count')
        plt.title('NLP Presolve Time Distribution')
        plt.show()






