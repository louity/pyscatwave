import numpy as np
import torch
from tqdm import tqdm
from cheml.datasets import load_qm7
from cheml.utils import get_valence
from sklearn import linear_model, model_selection
from scatwave.scattering import SolidHarmonicScattering


def get_qm7_positions_energies_and_charges(M, N, O, J, L):
    # TODO: normalized unit system
    qm7 = load_qm7(align=True)
    positions = qm7.R
    charges = qm7.Z
    energies = qm7.T
    valence_charges = get_valence(charges)
    core_charges = charges - valence_charges

    return positions, energies, charges, valence_charges, core_charges


def generate_weighted_sum_of_gaussians(positions, weights, cuda=False):
    return None


def evaluate_linear_regression(scat_1, scat_2, target):
    x_1 = None # TODO
    x_1_2 = None # TODO
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
    batch_size = 20
    M, N, O = 256, 192, 128
    sigma = 6.
    J, L = 4, 3
    integral_powers = [0.5, 1, 2, 3]
    args = {'integral_powers': integral_powers}
    pos, energies, full_chrg, val_chrg, core_chrg = get_qm7_positions_energies_and_charges(M, N, O, J, L)
    n_molecules = pos.shape[0]
    scat = SolidHarmonicScattering(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma)

    scat_1, scat_2 = [], []
    print('''Computing solid harmonic scattering coefficients of molecules
          of QM7 database on {}'''.format('GPU' if cuda else 'CPU'))
    for i in tqdm(range(n_molecules // batch_size)):
        pos_batch = pos[i*batch_size: (i+1)*batch_size]
        full_batch = full_chrg[i*batch_size: (i+1)*batch_size]
        full_density_batch = generate_weighted_sum_of_gaussians(pos_batch, full_batch, cuda=cuda)
        full_scat_1, full_scat_2 = scat(full_density_batch, True, 'integral', args)
        scat_1.append(full_scat_1)
        scat_2.append(full_scat_2)

    evaluate_linear_regression(scat_1, scat_2, energies)


if __name__ == '__main__':
    main()
