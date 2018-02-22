"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['filters_bank']

import torch
import numpy as np
import scipy.fftpack as fft
from scipy.special import sph_harm, factorial
from .utils import get_3d_angles, double_factorial


def filters_bank(M, N, J, L=8):
    filters = {}
    filters['psi'] = []


    offset_unpad = 0
    for j in range(J):
        for theta in range(L):
            psi = {}
            psi['j'] = j
            psi['theta'] = theta
            psi_signal = morlet_2d(M, N, 0.8 * 2**j, (int(L-L/2-1)-theta) * np.pi / L, 3.0 / 4.0 * np.pi /2**j,offset=offset_unpad) # The 5 is here just to match the LUA implementation :)
            psi_signal_fourier = fft.fft2(psi_signal)
            for res in range(j + 1):
                psi_signal_fourier_res = crop_freq(psi_signal_fourier, res)
                psi[res]=torch.FloatTensor(np.stack((np.real(psi_signal_fourier_res), np.imag(psi_signal_fourier_res)), axis=2))
                # Normalization to avoid doing it with the FFT!
                psi[res].div_(M*N// 2**(2*j))
            filters['psi'].append(psi)

    filters['phi'] = {}
    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0, offset=offset_unpad)
    phi_signal_fourier = fft.fft2(phi_signal)
    filters['phi']['j'] = J
    for res in range(J):
        phi_signal_fourier_res = crop_freq(phi_signal_fourier, res)
        filters['phi'][res]=torch.FloatTensor(np.stack((np.real(phi_signal_fourier_res), np.imag(phi_signal_fourier_res)), axis=2))
        filters['phi'][res].div_(M*N // 2 ** (2 * J))

    return filters


def solid_harmonic_filters_bank(M, N, O, J, L, sigma_0, fourier=True):
    filters = []
    for l in range(1, L+1):
        filters_l = np.zeros((J+1, 2*l+1, M, N, O), np.complex64)
        for j in range(J+1):
            sigma = sigma_0 * 2**j
            filters_l[j] = solid_harmonic_3d(M, N, O, sigma, l, fourier=fourier)
        filters.append(filters_l)
    return filters


def crop_freq(x, res):
    M = x.shape[0]
    N = x.shape[1]

    crop = np.zeros((M // 2 ** res, N // 2 ** res), np.complex64)

    mask = np.ones(x.shape, np.float32)
    len_x = int(M * (1 - 2 ** (-res)))
    start_x = int(M * 2 ** (-res - 1))
    len_y = int(N * (1 - 2 ** (-res)))
    start_y = int(N * 2 ** (-res - 1))
    mask[start_x:start_x + len_x,:] = 0
    mask[:, start_y:start_y + len_y] = 0
    x = np.multiply(x,mask)

    for k in range(int(M / 2 ** res)):
        for l in range(int(N / 2 ** res)):
            for i in range(int(2 ** res)):
                for j in range(int(2 ** res)):
                    crop[k, l] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res)]

    return crop


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=None):
    """ This function generated a morlet"""
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset, fft_shift)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset, fft_shift)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=None):
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab = gab + np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab = gab / norm_factor

    if (fft_shift):
        gab = np.fft.fftshift(gab, axes=(0, 1))
    return gab


def solid_harmonic_3d(M, N, O, sigma, l, fourier=True):
    """Computes solid harmonic wavelets in Fourier or signal space.

    Input args:
        M, N, O: integers, shape of the grid
        sigma: float, width of the wavelets
        l: integer, degree of the harmonic
        fourier: boolean, compute wavelet in fourier space
                 or in signal space

    Returns:
        solid_harm: 4D tensors of shape (2l+1, M, N, O). The
                    tensor is ifftshifted such that the point 0 in
                    signal space or in Fourier space is at
                    [m, 0, 0, 0] for m = 0 ... 2*l+1
    """
    solid_harm = np.zeros((2*l+1, M, N, O), np.complex64)
    grid = np.mgrid[-M//2:M//2, -N//2:N//2, -O//2:O//2].astype('float32')

    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        grid[2] *= 2 * np.pi / O
        sigma = 1. / sigma

    r_square = (grid**2).sum(0)
    polynomial_gaussian = r_square**(0.5*l) / sigma**l * np.exp(-0.5 * r_square / sigma**2)

    polar, azimuthal = get_3d_angles(grid)

    for i_m, m in enumerate(range(-l, l+1)):
        solid_harm[i_m] = sph_harm(m, l, azimuthal, polar) * polynomial_gaussian

    if l % 2 == 0:
        norm_factor = 1. / (2 * np.pi * np.sqrt(l+0.5) * double_factorial(l+1))
    else :
        norm_factor = 1. / (2**(0.5*(l+3)) * np.sqrt(np.pi*(2*l+1)) * factorial((l+1)/2))

    if fourier:
        norm_factor *= (2 * np.pi)**1.5 * (-1j)**l
    else:
        norm_factor /= sigma**3

    solid_harm *= norm_factor

    return np.fft.ifftshift(solid_harm)
