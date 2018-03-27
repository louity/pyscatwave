""" This script will test the submodules used by the scattering module"""

import torch
import unittest
import numpy as np
from scatwave.scattering import SolidHarmonicScattering
from scatwave import utils as sl

def linfnorm(x,y):
    return torch.max(torch.abs(x-y))

class TestScattering(unittest.TestCase):
    def testFFT3dCentralFreqBatch(self):
        # Checked the 0 frequency for the 3D FFT
        for gpu in [False, True]:
            x = torch.FloatTensor(1, 32, 32, 32, 2).fill_(0)
            if gpu:
                x = x.cuda()

            a = x.sum()
            fft3d = sl.Fft3d()
            y = fft3d(x)
            c = y[:,0,0,0].sum()
            self.assertAlmostEqual(a, c, places=6)


    def testSolidHarmonicScattering(self):
        # Compare value to analytical formula in the case of a single Gaussian
        centers = np.zeros((1, 1, 3))
        sigma_gaussian = 6.
        sigma_wavelet = 8.
        M, N, O, J, L = 128, 128, 128, 0, 3
        x = sl.generate_sum_of_gaussians(centers, sigma_gaussian, M, N, O)
        scat = SolidHarmonicScattering(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma_wavelet)
        integral_powers = [1]
        s = scat(x, integral_powers)

        k = sigma_wavelet / np.sqrt(sigma_wavelet**2 + sigma_gaussian**2)
        for l in range(L):
            self.assertAlmostEqual(s[0, l, 0, 0], k**(l+1), places=2)


if __name__ == '__main__':
    unittest.main()
