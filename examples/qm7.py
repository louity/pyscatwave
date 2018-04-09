import numpy as np
import torch
from tqdm import tqdm
from cheml.datasets import load_qm7
from cheml.utils import get_valence
from sklearn import linear_model, model_selection
from scipy.spatial.distance import pdist
from scatwave.scattering import SolidHarmonicScattering


def renormalize(positions, charges, sigma, overlapping_precision=1e-1):
    x_min, x_max = positions[:,:,0].min(), positions[:,:,0].max()
    y_min, y_max = positions[:,:,1].min(), positions[:,:,1].max()
    z_min, z_max = positions[:,:,2].min(), positions[:,:,2].max()

    positions[:,:,0] -= 0.5*(x_max + x_min)
    positions[:,:,1] -= 0.5*(y_max + y_min)
    positions[:,:,2] -= 0.5*(z_max + z_min)

    min_dist = np.inf
    for i in range(positions.shape[0]):
        n_atoms = np.sum(charges[i] != 0)
        pos = positions[i, :n_atoms, :]
        min_dist = min(min_dist, pdist(pos).min())

    delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))

    return positions * delta / min_dist


def get_qm7_positions_energies_and_charges(M, N, O, J, L, sigma):
    qm7 = load_qm7(align=True)
    positions = qm7.R
    charges = qm7.Z
    energies = qm7.T.transpose()
    # valence_charges = get_valence(charges)
    # core_charges = charges - valence_charges

    positions = renormalize(positions, charges, sigma)

    # return torch.from_numpy(positions), torch.from_numpy(energies), torch.from_numpy(charges), torch.from_numpy(valence_charges), torch.from_numpy(core_charges)
    return torch.from_numpy(positions), torch.from_numpy(energies), torch.from_numpy(charges) #, torch.from_numpy(valence_charges), torch.from_numpy(core_charges)


def generate_weighted_sum_of_gaussians(grid, positions, weights, sigma, cuda=False):
    _, M, N, O = grid.size()
    if cuda:
        signals = torch.cuda.FloatTensor(positions.size(0), M, N, O).fill_(0)
    else:
        signals = torch.FloatTensor(positions.size(0), M, N, O).fill_(0)

    for i_signal in range(positions.size(0)):
        n_points = positions[i_signal].size(0)
        for i_point in range(n_points):
            if weights[i_signal, i_point] == 0:
                break
            weight = weights[i_signal, i_point]
            center = positions[i_signal, i_point]
            signals[i_signal] += weight * torch.exp(
                -0.5 * ((grid[0]-center[0])**2 + (grid[1]-center[1])**2 + (grid[2]-center[2]))**2/ sigma**2)
    return signals / ((2 * np.pi)**1.5 * sigma**3)


def evaluate_linear_regression(scat_0, scat_1, scat_2, target):
    x_1 = np.concatenate([scat_0.numpy(), scat_1.numpy()], axis=1)
    x_1_2 = np.concatenate([x_1, scat_2.numpy()], axis=1)
    lin_regressor = linear_model.LinearRegression()
    scat_1_prediction = model_selection.cross_val_predict(lin_regressor, x_1, target)
    scat_1_MAE = np.mean(np.abs(scat_1_prediction - target))
    scat_1_RMSE = np.sqrt(np.mean((scat_1_prediction - target)**2))
    print('''scattering order 1, linear regression,
          MAE: {}, RMSE: {} (kcal.mol-1)'''.format(scat_1_MAE, scat_1_RMSE))

    scat_1_2_prediction = model_selection.cross_val_predict(lin_regressor, x_1_2, target)
    scat_1_2_MAE = np.mean(np.abs(scat_1_2_prediction - target))
    scat_1_2_RMSE = np.sqrt(np.mean((scat_1_2_prediction - target)**2))
    print('''scattering order 1 and 2, linear regression,
          MAE: {}, RMSE: {} (kcal.mol-1)'''.format(scat_1_2_MAE, scat_1_2_RMSE))


def main():
    """Trains a simple linear regression model with solid harmonic
    scattering coefficients on the atomisation energies of the QM7
    database.

    Achieves a MAE of ... kcal.mol-1
    """
    cuda = torch.cuda.is_available()
    batch_size = 32
    M, N, O = 192, 128, 96
    grid = torch.from_numpy(
            np.fft.ifftshift(np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'), axes=(1,2,3)))
    if cuda:
        grid = grid.cuda()
    sigma = 3.
    J, L = 4, 3
    integral_powers = [1.5, 1, 2, 3]
    args = {'integral_powers': integral_powers}
    # pos, energies, full_chrg, val_chrg, core_chrg = get_qm8_positions_energies_and_charges(M, N, O, J, L, sigma)
    pos, energies, full_chrg = get_qm7_positions_energies_and_charges(M, N, O, J, L, sigma)
    n_molecules = pos.size(0)
    n_batches = np.ceil(n_molecules / batch_size).astype(int)

    scat = SolidHarmonicScattering(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma)

    scat_0, scat_1, scat_2 = [], [], []
    print('Computing solid harmonic scattering coefficients of molecules \
          of QM7 database on {}'.format('GPU' if cuda else 'CPU'))
    print('L: {}, J: {}, integral powers: {}'.format(L, J, integral_powers))
    for i in tqdm(range(n_batches)):
        start, end = i*batch_size, min((i+1)*batch_size, n_molecules)
        pos_batch = pos[start:end]
        full_batch = full_chrg[start:end]
        full_density_batch = generate_weighted_sum_of_gaussians(grid, pos_batch, full_batch, sigma, cuda=cuda)
        full_scat_1, full_scat_2 = scat(full_density_batch, True, 'integral', args)
        scat_1.append(full_scat_1)
        scat_2.append(full_scat_2)

    scat_0 = full_chrg.sum(1, keepdim=True)
    scat_1 = torch.cat(scat_1, dim=0)
    scat_2 = torch.cat(scat_2, dim=0)

    n = scat_1.size(0)
    scat_1 = scat_1.contiguous().view(n, -1)
    scat_2 = scat_2.contiguous().view(n, -1)
    print('order 1 : {} coef, order 2 : {} coefs'.format(scat_1.size(1), scat_2.size(1)))

    evaluate_linear_regression(scat_0, scat_1, scat_2, energies.numpy())



if __name__ == '__main__':
    main()
