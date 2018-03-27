"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['SolidHarmonicScattering']

import torch
from .utils import cdgmm3d, Fft3d, compute_integrals
from .filters_bank import solid_harmonic_filters_bank, gaussian_filters_bank


class SolidHarmonicScattering(object):
    """Scattering module.

    Runs solid scattering on an input 3D image

    Input args:
        M, N, O: input 3D image size
        J: number of scales
        L: number of l values
    """
    def __init__(self, M, N, O, J, L, sigma_0):
        super(SolidHarmonicScattering, self).__init__()
        self.M, self.N, self.O, self.J, self.L, self.sigma_0 = M, N, O, J, L, sigma_0
        self.filters = solid_harmonic_filters_bank(self.M, self.N, self.O, self.J, self.L, sigma_0)
        self.gaussian_filters = gaussian_filters_bank(self.M, self.N, self.O, self.J+1, sigma_0)
        self.fft = Fft3d()

    def _pad(self, input):
        output = input.new(input.size() + (2,)).fill_(0)
        output[..., 0] = input
        return output

    def _unpad(self, in_):
        return in_[..., 1:-1, 1:-1]

    def _compute_local_scattering_coefs(self, padded_input, points, j):
        local_coefs = torch.zeros(padded_input.size(0), points.size(1))
        convolved_input = self.fft(
            cdgmm3d(self.fft(padded_input, inverse=False), self.gaussian_filters[j+1])
            , inverse=True)
        for i_signal in range(padded_input.size(0)):
            for i_point in range(points[i_signal].size(0)):
                x, y, z = points[i_signal, i_point, 0], points[i_signal, i_point, 1], points[i_signal, i_point, 2]
                local_coefs[i_signal,i_point] = convolved_input[i_signal, int(x), int(y), int(z), 0]
        return local_coefs


    def _convolution_and_modulus(self, padded_input, l, j):
        cuda = isinstance(padded_input, torch.cuda.FloatTensor)
        filters_l_j = self.filters[l-1][j].type(torch.cuda.FloatTensor) if cuda else self.filters[l-1][j]
        fft = self.fft
        convolution_modulus = padded_input.new(padded_input.size()[:-1]).fill_(0)
        for m in range(filters_l_j.size(0)):
            convolution_modulus += (
                fft(cdgmm3d(fft(padded_input, inverse=False), filters_l_j[m]), inverse=True)**2
            ).sum(-1)
        return torch.sqrt(convolution_modulus)

    def forward(self, input, integral_powers=[1, 2], order_2=True, local_computation_points=None):
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            input = input.contiguous()

        if((input.size(-1)!=self.O or input.size(-2)!=self.N or input.size(-3)!=self.M)):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i,%i)!'%(self.M,self.N,self.O)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        pad = self._pad
        convolution_and_modulus = self._convolution_and_modulus

        Q = len(integral_powers)
        n_inputs = input.size(0)
        n_j_coefs = self.J+1
        if order_2:
            n_j_coefs += self.J*(self.J+1) // 2
        scat_coefs = torch.zeros(n_inputs, self.L,  n_j_coefs, Q)
        if local_computation_points is not None:
            n_points = local_computation_points.size(1) #FIXME: =! signals can have =! num points
            local_scat_coefs = torch.zeros(n_inputs, n_points, self.L,  n_j_coefs)

        padded_input = pad(input)
        for l in range(1, self.L + 1):
            i_coef = self.J+1
            for j_1 in range(self.J+1):
                conv_modulus = convolution_and_modulus(padded_input, l, j_1)
                scat_coefs[:, l-1, j_1] = compute_integrals(conv_modulus, integral_powers)
                padded_conv_modulus = pad(conv_modulus)
                if local_computation_points is not None:
                    local_scat_coefs[:, :, l-1, j_1] = self._compute_local_scattering_coefs(
                        padded_conv_modulus, local_computation_points, j_1)
                if not order_2:
                    continue
                for j_2 in range(j_1+1, self.J+1):
                    conv_modulus_2 = convolution_and_modulus(padded_conv_modulus, l, j_2)
                    scat_coefs[:, l-1, i_coef] = compute_integrals(conv_modulus_2, integral_powers)
                    i_coef += 1
        if local_computation_points is not None:
            return scat_coefs, local_scat_coefs
        return scat_coefs

    def __call__(self, input, integral_powers, order_2=True, local_computation_points=None):
        return self.forward(input, integral_powers, order_2=order_2,
                            local_computation_points=local_computation_points)

