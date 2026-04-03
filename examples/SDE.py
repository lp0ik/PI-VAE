import sys
sys.path.append(r'../')

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import scipy.linalg as la
import time

from lib.models import PIVAE_SDE, MMD_loss
from lib.data_loader import trainingset_construct_SDE
from lib.visualization import *
import matplotlib.pyplot as plt

# convey the parameters from command line
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--case', type=str, default = None)
parser.add_argument('--data_size', type=int, default = 2000)
parser.add_argument('--u_sensor', type=int, default = None)
parser.add_argument('--k_sensor', type=int, default = None)
parser.add_argument('--f_sensor', type=int, default = None)
parser.add_argument('--latent_dim', type=int, default = 4)
parser.add_argument('--batch_val', type=int, default = 1000)
parser.add_argument('--epoch', type=int, default = 100)
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--mesh_size', type=int, default = 400)
args = parser.parse_args()

if torch.cuda.is_available():
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)

# define loss function
def criterion(u, k, f, u_recon, k_recon, f_recon, z, device):
    MMD = MMD_loss()
    recon = torch.cat((u_recon, k_recon, f_recon), dim = 1)
    ref = torch.cat((u, k, f), dim = 1)
    loss_recon = MMD(recon,ref) 
    zc = torch.randn_like(z).to(device)
    KLD = MMD(z, zc)
    
    return KLD + loss_recon 

# training function
def train(epoch,train_loader,model,optimize_operator,criterion,device):
    train_loss = 0
    for batch_idx, (u, k, f, u_coor, k_coor, f_coor) in enumerate(train_loader):
        u = u.to(device)
        k = k.to(device)
        f = f.to(device)
        u_coor = u_coor.to(device)
        k_coor = k_coor.to(device)
        f_coor = f_coor.to(device)

        optimize_operator.zero_grad()
        u_recon, k_recon, f_recon, Z = model.forward(u, k, f, u_coor, k_coor, f_coor)
        loss = criterion(u, k, f, u_recon, k_recon, f_recon, Z, device)
        loss.backward()
        train_loss += loss.item()
        optimize_operator.step()
    
    return model, loss
    
# load the data from database
u_data = np.load(file=r'../database/SDE/u_{}.npy'.format(args.case))[0:args.data_size]
k_data = np.load(file=r'../database/SDE/k_{}.npy'.format(args.case))[0:args.data_size]
f_data = np.load(file=r'../database/SDE/f_{}.npy'.format(args.case))[0:args.data_size]


# calculate ground true for comparison
if args.case == args.case:
    n_validate = 201    # number of validation points
    test_coor = np.floor(np.linspace(0,1,n_validate) * args.mesh_size).astype(int)
    u_test = u_data[:,test_coor]
    k_test = k_data[:,test_coor]
    f_test = f_data[:,test_coor]
    true_mean_u = torch.mean(torch.from_numpy(u_test),axis=0).type(torch.float).to(device)
    true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
    true_mean_k = torch.mean(torch.from_numpy(k_test),axis=0).type(torch.float).to(device)
    true_std_k = std_cal(torch.from_numpy(k_test)).type(torch.float).to(device)
    # calculate u reference solution
    std = np.std(u_data, axis=0)
    mean = np.mean(u_data, axis=0)
    low = mean - std
    up = mean + std
    u_ref = np.vstack((low,mean,up))
    # calculate ukreference solution
    std = np.std(k_data, axis=0)
    mean = np.mean(k_data, axis=0)
    low = mean - std
    up = mean + std
    k_ref = np.vstack((low,mean,up))

# define models
nblock = 3   # 3 blocks = 4 hidden layers
width = 128
model = PIVAE_SDE(args.latent_dim, args.u_sensor, args.k_sensor, args.f_sensor, nblock, width, device).to(device) 
optimize_operator = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5,0.9))

# define training data loader
u_coor = np.linspace(-1,1,args.u_sensor) * np.ones([len(u_data),args.u_sensor])
k_coor = np.linspace(-1,1,args.k_sensor) * np.ones([len(k_data),args.k_sensor])
if args.k_sensor == 1:
    k_coor = - np.ones([len(k_data), args.k_sensor])
f_coor = np.linspace(-1,1,args.f_sensor) * np.ones([len(f_data),args.f_sensor])
x_u_coor = np.floor(np.linspace(0,1,args.u_sensor) * args.mesh_size).astype(int)
x_k_coor = np.floor(np.linspace(0,1,args.k_sensor) * args.mesh_size).astype(int)
if args.k_sensor == 1:
    x_k_coor = [0]
x_f_coor = np.floor(np.linspace(0,1,args.f_sensor) * args.mesh_size).astype(int)
k_training_data = k_data[0:args.data_size, x_k_coor]
u_training_data = u_data[0:args.data_size, x_u_coor]
f_training_data = f_data[0:args.data_size, x_f_coor]   
VAE_train_loader = trainingset_construct_SDE(u_data=u_training_data, k_data=k_training_data, f_data=f_training_data, 
                                         x_u=u_coor, x_k=k_coor, x_f=f_coor, batch_val=args.batch_val)

# train the network
u_mean_error = []
u_std_error = []
k_mean_error = []
k_std_error = []
time_history = []
loss_history = []
if __name__ == "__main__":
    for epoch in range(args.epoch):

        if epoch % 100 == 0:
            print('epoch:', epoch)

            with torch.no_grad():
                z = torch.randn(1000, args.latent_dim).to(device)
                coordinate = (torch.linspace(-1,1,steps=n_validate) * torch.ones((1000,n_validate))).to(device)
                u_recon = model.u_decoder(model.combine_xz(coordinate, z)).view(-1,n_validate)
                k_recon = model.k_decoder(model.combine_xz(coordinate, z)).view(-1,n_validate)
                
                mean_u = torch.mean(u_recon,axis=0)
                std_u = std_cal(u_recon)
                mean_k = torch.mean(k_recon,axis=0)
                std_k = std_cal(k_recon)
                
                # mean_error_forward.append(torch.norm(mean-true_mean))
                # std_error_forward.append(torch.norm(std-true_std))
                mean_L2_error_u = (torch.norm(mean_u-true_mean_u)/torch.norm(true_mean_u)).cpu().numpy()
                std_L2_error_u = (torch.norm(std_u-true_std_u)/torch.norm(true_std_u)).cpu().numpy()
                u_mean_error.append(mean_L2_error_u)
                u_std_error.append(std_L2_error_u)
                print('u mean error:', mean_L2_error_u, 'u std error:', std_L2_error_u)
                
                mean_L2_error_k = (torch.norm(mean_k-true_mean_k)/torch.norm(true_mean_k)).cpu().numpy()
                std_L2_error_k = (torch.norm(std_k-true_std_k)/torch.norm(true_std_k)).cpu().numpy()
                k_mean_error.append(mean_L2_error_k)
                k_std_error.append(std_L2_error_k)
                print('k mean error:', mean_L2_error_k, 'k std error:', std_L2_error_k)

        time_start = time.time()
        model, L = train(epoch, VAE_train_loader, model, optimize_operator, criterion, device_name)
        time_stop = time.time()
        loss_history.append(L.detach().cpu().numpy())
        time_history.append(time_stop-time_start)
    
    # torch.save(model, './results/special_case/f_high/model_fhigh.pkl')
    torch.save(model, r'./kaggle/working/PI-VAE/examples/trained_model/model_{}_u={}_k={}_f={}_datasize={}_z={}.pkl'.format(args.case, args.u_sensor, args.k_sensor, args.f_sensor, args.data_size, args.latent_dim))

'''
    print("\nГенерация предсказаний k_pred, u_pred, f_pred...")
    model.eval()
    with torch.no_grad():
        # Генерируем выборку размером с исходные данные (или максимум 2000 для экономии памяти)
        num_pred = min(2000, args.data_size)
        z_sample = torch.randn(num_pred, args.latent_dim).to(device)
        
        # Получаем размер сетки (количество точек по оси X) из загруженных данных
        mesh_points = u_data.shape[1]
        eval_coordinate = (torch.linspace(-1, 1, steps=mesh_points) * torch.ones((num_pred, mesh_points))).to(device)
        
        # Пропускаем координаты и сэмплированные Z через декодеры
        xz_input = model.combine_xz(eval_coordinate, z_sample)
        
        u_pred = model.u_decoder(xz_input).view(-1, mesh_points).cpu().numpy()
        k_pred = model.k_decoder(xz_input).view(-1, mesh_points).cpu().numpy()
        f_pred = model.f_decoder(xz_input).view(-1, mesh_points).cpu().numpy()

    print(f"Форма предсказанных массивов: u={u_pred.shape}, k={k_pred.shape}, f={f_pred.shape}")


    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 2.1 График Loss
    axes[0].plot(loss_history, label='Total Loss', color='purple', alpha=0.7)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2.2 График ошибок для u(x)
    epochs_val = range(0, args.epoch, 100)
    axes[1].plot(epochs_val, u_mean_error, 'b-o', label='u Mean L2 Error')
    axes[1].plot(epochs_val, u_std_error, 'c-s', label='u Std L2 Error')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Relative L2 Error')
    axes[1].set_title('u(x) Validation Errors')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 2.3 График ошибок для k(x)
    axes[2].plot(epochs_val, k_mean_error, 'r-o', label='k Mean L2 Error')
    axes[2].plot(epochs_val, k_std_error, 'm-s', label='k Std L2 Error')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Relative L2 Error')
    axes[2].set_title('k(x) Validation Errors')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


    def plot_k_u_f_samples(k_data, u_data, f_data, k_sensor, u_sensor, f_sensor, num_samples=50):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        x_axis = np.linspace(-1, 1, k_data.shape[1])
        
        def plot_single(ax, data, n_sensor, title):
            for i in range(min(num_samples, data.shape[0])):
                ax.plot(x_axis, data[i, :])
            if n_sensor is not None and n_sensor > 0:
                if n_sensor == 1: 
                    sensor_position = [-1] 
                else:
                    sensor_position = np.linspace(-1, 1, n_sensor)
                lower_bound, upper_bound = np.min(data), np.max(data)
                for s in sensor_position:
                    ax.vlines(s, lower_bound, upper_bound, colors="k", linestyles="dashed")
            ax.set_title(title)
            ax.set_xlim([-1.05, 1.05])

        plot_single(axes[0], k_data, k_sensor, "sample paths of k(x)")
        plot_single(axes[1], u_data, u_sensor, "sample paths of u(x)")
        plot_single(axes[2], f_data, f_sensor, "sample paths of f(x)")
        plt.tight_layout()
        plt.show()

    def SDE_visual(pred_samples, true_samples, title_prefix="Variable"):
        x_axis = np.linspace(-1, 1, pred_samples.shape[1])
        pred_mean, pred_std = np.mean(pred_samples, axis=0), np.std(pred_samples, axis=0)
        true_mean, true_std = np.mean(true_samples, axis=0), np.std(true_samples, axis=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Среднее значение + Доверительный интервал
        axes[0].plot(x_axis, true_mean, 'k-', label='True Mean', linewidth=2)
        axes[0].plot(x_axis, pred_mean, 'r--', label='Pred Mean', linewidth=2)
        axes[0].fill_between(x_axis, pred_mean - 2*pred_std, pred_mean + 2*pred_std, color='red', alpha=0.2, label='Pred $\pm 2\sigma$')
        axes[0].fill_between(x_axis, true_mean - 2*true_std, true_mean + 2*true_std, color='gray', alpha=0.2, label='True $\pm 2\sigma$')
        axes[0].set_title(f"{title_prefix} - Mean & Confidence Interval")
        axes[0].legend()
        
        # Стандартное отклонение
        axes[1].plot(x_axis, true_std, 'k-', label='True Std', linewidth=2)
        axes[1].plot(x_axis, pred_std, 'r--', label='Pred Std', linewidth=2)
        axes[1].set_title(f"{title_prefix} - Standard Deviation")
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

    print("\nВизуализация Ground Truth Сэмплов (Рис. 9):")
    plot_k_u_f_samples(k_data, u_data, f_data, args.k_sensor, args.u_sensor, args.f_sensor)

    print("\nСравнение статистик предсказаний VAE и Истинных данных:")
    SDE_visual(k_pred, k_data, title_prefix="Coefficient k(x)")
    SDE_visual(u_pred, u_data, title_prefix="Solution u(x)")
    SDE_visual(f_pred, f_data, title_prefix="Forcing term f(x)")
'''






















