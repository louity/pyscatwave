"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['Scattering']

import warnings
import torch
from .utils import cdgmm, Modulus, Periodize, Fft, Fft3d, SolidHarmonicModulus
from .filters_bank import filters_bank, solid_harmonic_filters_bank
from torch.legacy.nn import SpatialReflectionPadding as pad_function

class SolidHarmonicScattering(object):
    """Scattering module.

    Runs solid scattering on an input 3D image

    Input args:
        M, N, O: input 3D image size
        J: number of scales
        L: number of l values
    """
    def __init__(self, M, N, O, J, L):
        super(SolidHarmonicScattering, self).__init__()
        self.M, self.N, self.O, self.J, self.L = M, N, O, J, L
        self.fft = Fft3d()
        self.modulus = SolidHarmonicModulus()
        self.filters = solid_harmonic_filters_bank(self.M, self.N, self.O, self.J, self.L)

    def forward(self, input, integral_powers):
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if((input.size(-1)!=self.O or input.size(-2)!=self.N or input.size(-3)!=self.M)):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i,%i)!'%(self.M,self.N,self.O)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        Q = len(integral_powers)
        n_inputs = 100
        s_order_1 = np.zeros((n_inputs, self.L, self.J+1, Q))
        s_order_2= np.zeros((n_inputs, self.L, self.J*(self.J+1)/2, Q))

        batch_size = 10
        for l in range(1, self.L + 1):
            filters_l = self.filters[l-1]
            for i_batch, input_batch in []:
                i_coef = 0
                for j_1 in range(self.J+1):
                    # perform the convolution + molulus of the input batch with psi_{l,j_1}
                    for i_q, q in enumerate(integral_powers):
                        s_order_1[(i_batch)*batch_size:(i_batch+1)*batch_size, l-1, j_1, i_q] = 0 # compute the scattering coefficients l j1
                    for j_2 in range(j_1+1, self.J+1):
                        # perform the second order convolutions + modulus with psi_{l,j_2}
                        for i_q, q in enumerate(integral_powers):
                            s_order_2[i_batch*batch_size:(i_batch+1)*batch_size, l-1, i_coef, i_q] = 0 # compute the second order coefficients
                        i_coef += 1

        return s_order_1, s_order_2


    def __call__(self, input):
        return self.forward(input)

class Scattering(object):
    """Scattering module.

    Runs scattering on an input image in NCHW format

    Input args:
        M, N: input image size
        J: number of layers
        pre_pad: if set to True, module expect pre-padded images
        jit: compile kernels on the fly for speed
    """
    def __init__(self, M, N, J, pre_pad=False, jit=True):
        super(Scattering, self).__init__()
        self.M, self.N, self.J = M, N, J
        self.pre_pad = pre_pad
        self.jit = jit
        self.fft = Fft()
        self.modulus = Modulus(jit=jit)
        self.periodize = Periodize(jit=jit)

        self._prepare_padding_size([1, 1, M, N])

        self.padding_module = pad_function(2**J)

        # Create the filters
        filters = filters_bank(self.M_padded, self.N_padded, J)

        self.Psi = filters['psi']
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].iteritems():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.padding_module.type(str(_type).split('\'')[1])
        return self

    def cuda(self):
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        return self._type(torch.FloatTensor)

    def _prepare_padding_size(self, s):
        M = s[-2]
        N = s[-1]

        self.M_padded = ((M + 2 ** (self.J))//2**self.J+1)*2**self.J
        self.N_padded = ((N + 2 ** (self.J))//2**self.J+1)*2**self.J

        if self.pre_pad:
            warnings.warn('Make sure you padded the input before to feed it!', RuntimeWarning, stacklevel=2)

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = torch.Size([a for a in s])

    # This function copies and view the real to complex
    def _pad(self, input):
        if(self.pre_pad):
            output = input.new(input.size(0), input.size(1), input.size(2), input.size(3), 2).fill_(0)
            output.narrow(output.ndimension()-1, 0, 1).copy_(input)
        else:
            out_ = self.padding_module.updateOutput(input)
            output = input.new(out_.size(0), out_.size(1), out_.size(2), out_.size(3), 2).fill_(0)
            output.narrow(4, 0, 1).copy_(out_)
        return output

    def _unpad(self, in_):
        return in_[..., 1:-1, 1:-1]

    def forward(self, input):
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensor must be contiguous!'))

        if((input.size(-1)!=self.N or input.size(-2)!=self.M) and not self.pre_pad):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i)!'%(self.M,self.N)))

        if ((input.size(-1) != self.N_padded or input.size(-2) != self.M_padded) and self.pre_pad):
            raise (RuntimeError('Padded tensor must be of spatial size (%i,%i)!' % (self.M_padded, self.N_padded)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        J = self.J
        phi = self.Phi
        psi = self.Psi
        n = 0

        fft = self.fft
        periodize = self.periodize
        modulus = self.modulus
        pad = self._pad
        unpad = self._unpad

        S = input.new(input.size(0),
                      input.size(1),
                      1 + 8*J + 8*8*J*(J - 1) // 2,
                      self.M_padded//(2**J)-2,
                      self.N_padded//(2**J)-2)
        U_r = pad(input)
        U_0_c = fft(U_r, 'C2C')  # We trick here with U_r and U_2_c

        # First low pass filter
        U_1_c = periodize(cdgmm(U_0_c, phi[0], jit=self.jit), k=2**J)

        U_J_r = fft(U_1_c, 'C2R')

        S[..., n, :, :].copy_(unpad(U_J_r))
        n = n + 1

        for n1 in range(len(psi)):
            j1 = psi[n1]['j']
            U_1_c = cdgmm(U_0_c, psi[n1][0], jit=self.jit)
            if(j1 > 0):
                U_1_c = periodize(U_1_c, k=2 ** j1)
            fft(U_1_c, 'C2C', inverse=True, inplace=True)
            U_1_c = fft(modulus(U_1_c), 'C2C')

            # Second low pass filter
            U_2_c = periodize(cdgmm(U_1_c, phi[j1], jit=self.jit), k=2**(J-j1))
            U_J_r = fft(U_2_c, 'C2R')
            S[..., n, :, :].copy_(unpad(U_J_r))
            n = n + 1

            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                if(j1 < j2):
                    U_2_c = periodize(cdgmm(U_1_c, psi[n2][j1], jit=self.jit), k=2 ** (j2-j1))
                    fft(U_2_c, 'C2C', inverse=True, inplace=True)
                    U_2_c = fft(modulus(U_2_c), 'C2C')

                    # Third low pass filter
                    U_2_c = periodize(cdgmm(U_2_c, phi[j2], jit=self.jit), k=2 ** (J-j2))
                    U_J_r = fft(U_2_c, 'C2R')

                    S[..., n, :, :].copy_(unpad(U_J_r))
                    n = n + 1

        return S

    def __call__(self, input):
        return self.forward(input)
