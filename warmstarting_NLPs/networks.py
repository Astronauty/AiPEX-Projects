import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import time

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
            nn.Linear(n_params, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_traj)
        )
        self.linear_relu_stack = self.linear_relu_stack.double()

        
    def forward(self, params):
        # params = self.flatten(params)
        traj = self.linear_relu_stack(params)
        return traj

class POCPSolver():
    """
    Loads pre-solved NLP data and trains a neural network to solve the NLP. Computes statistics and visualizations regarding the solutions.
    """
    def __init__(self, path, t_vec, nx, nu, N, verbose=False, eq_constraint_loss=None, ineq_constraint_loss=None):
        sns.set_theme()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_data = NLPDataset(path, train=True)
        self.test_data = NLPDataset(path, train=False)

        self.train_dataloader = DataLoader(self.train_data, batch_size=16, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=16, shuffle=True)
        # Trajectory info
        self.t_vec = t_vec
        self.idx = create_idx(nx, nu, N)

        # Create NN
        self.n_params = self.train_data.n_params
        self.n_traj = self.train_data.n_traj
        
        self.model = CartpoleNN(self.train_data.n_params, self.train_data.n_traj).to(self.device)
        if verbose:
            print(f"Model structure: {self.model}\n\n")

        self.writer = SummaryWriter(f'runs/cartpole-{time.strftime("%Y%m%d-%H%M%S")}')


    def _train_loop(self, loss_fn, optimizer):
        size = len(self.train_dataloader.dataset)
        self.model.train()

        train_loss = 0
        for batch, (X,y) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)

            pred = self.model(X)
            loss = loss_fn(pred, y)

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(self.train_dataloader)
        # loss, current = loss.item(), batch * len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return train_loss
            

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
    
    def train(self):
        learning_rate = 5e-3
        batch_size = 64
        epochs = 100

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        for t in tqdm(range(epochs)):
            # print(f"Epoch {t+1}\n-------------------------------")
            train_loss = self._train_loop(loss_fn, optimizer)
            scheduler.step()
            self.writer.add_scalar('train loss x epoch', train_loss, t)

            test_loss = self._test_loop(loss_fn)
            self.writer.add_scalar('test loss x epoch', test_loss, t)

        self.writer.flush()
        self.writer.close()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def compare_trajectories(self, idx):
        assert idx < len(self.test_data)

        plt.figure(figsize=(10,6))
        test_param  = torch.from_numpy(self.test_data.df.iloc[idx].params)
        test_param = test_param.to(self.device)

        self.model.eval()

        with torch.no_grad():
            Z = self.model(test_param)

        plt.plot(self.t_vec, Z[self.idx.X][:,:2], linewidth=3.0)
        plt.plot(self.t_vec, self.test_data.df.iloc[idx].X[:,0], '--', linewidth=3.0, color='tab:blue')
        plt.plot(self.t_vec, self.test_data.df.iloc[idx].X[:,1], '--', linewidth=3.0, color='tab:orange')

    def compare_controls(self, idx):
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







