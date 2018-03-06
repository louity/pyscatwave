"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""
from collections import defaultdict, namedtuple

import torch
from skcuda import cublas, cufft
from pynvrtc.compiler import Program
import numpy as np
import pyfftw
from cupy.cuda.function import Module
from cupy.cuda import device
from string import Template


Stream = namedtuple('Stream', ['ptr'])


def generate_sum_of_gaussians(centers, sigma, M, N, O):
    grid = np.mgrid[-M//2:M//2, -N//2:N//2, -O//2:O//2]
    n_signals = centers.shape[0]
    signals = torch.zeros((n_signals, M, N, O))

    for i_signal in range(n_signals):
        sum_of_gaussian = np.zeros((M, N, O), dtype='float32')
        n_centers = centers[i_signal].shape[0]
        for i_center in range(n_centers):
            center = centers[i_signal, i_center].reshape((3, 1, 1, 1))
            sum_of_gaussian += np.exp(-0.5 * ((grid - center)**2).sum(0) / sigma**2)
        sum_of_gaussian /= (2 * np.pi)**1.5 * sigma**3
        signals[i_signal] = torch.from_numpy(sum_of_gaussian)

    return signals


def compute_integrals(input, integral_powers):
    """Computes integrals of the input to the given powers."""
    integrals = torch.zeros(input.size(0), len(integral_powers))
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q] = torch.from_numpy((input.numpy()**q).sum(axis=(1,2,3)))
    return integrals


def get_3d_angles(cartesian_grid):
    """Given a cartisian grid, computes the spherical coord angles (theta, phi).
    Input args:
        cartesian_grid: 4D tensor of shape (3, M, N, O)
    Returns:
        polar, azimutal: 3D tensors of shape (M, N, O).
    """
    z, y, x = cartesian_grid
    azimuthal = np.arctan2(y, x)
    rxy = np.sqrt(x**2 + y**2)
    polar = np.arctan2(z, rxy) + np.pi / 2
    return polar, azimuthal


def double_factorial(l):
    return 1 if (l < 1) else np.prod(np.arange(l, 0, -2))


def getDtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


def get_compute_arch(t):
    return 'compute_%s' % device.Device().compute_capability


def iscomplex(input):
    return input.size(-1) == 2


class Periodize(object):
    """This class builds a wrapper to the periodiziation kernels and cache them.
        """
    def __init__(self, jit=True):
        self.periodize_cache = defaultdict(lambda: None)
        self.block = (32, 32, 1)
        self.jit = jit

    def GET_BLOCKS(self, N, threads):
        return (N + threads - 1) // threads

    def __call__(self, input, k):
        out = input.new(input.size(0), input.size(1), input.size(2) // k, input.size(3) // k, 2)

        if not self.jit or isinstance(input, (torch.FloatTensor, torch.DoubleTensor)):
            y = input.view(input.size(0), input.size(1),
                           input.size(2)//out.size(2), out.size(2),
                           input.size(3)//out.size(3), out.size(3),
                           2)

            out = y.mean(4).squeeze(4).mean(2).squeeze(2)
            return out

        if not iscomplex(input):
            raise (TypeError('The input and outputs should be complex'))

        input = input.contiguous()

        if (self.periodize_cache[(input.size(), out.size(), input.get_device())] is None):
            kernel = '''
            #define NW ${W} / ${k}
            #define NH ${H} / ${k}
            extern "C"
            __global__ void periodize(const ${Dtype}2 *input, ${Dtype}2 *output)
            {
              int tx = blockIdx.x * blockDim.x + threadIdx.x;
              int ty = blockIdx.y * blockDim.y + threadIdx.y;
              int tz = blockIdx.z * blockDim.z + threadIdx.z;
              if(tx >= NW || ty >= NH || tz >= ${B})
                return;
              input += tz * ${H} * ${W} + ty * ${W} + tx;
              ${Dtype}2 res = make_${Dtype}2(0.f, 0.f);
              for (int j=0; j<${k}; ++j)
                for (int i=0; i<${k}; ++i)
                {
                  const ${Dtype}2 &c = input[j * NH * ${W} + i * NW];
                  res.x += c.x;
                  res.y += c.y;
                }
              res.x /= ${k} * ${k};
              res.y /= ${k} * ${k};
              output[tz * NH * NW + ty * NW + tx] = res;
            }
            '''
            B = input.nelement() // (2*input.size(-2) * input.size(-3))
            W = input.size(-2)
            H = input.size(-3)
            k = input.size(-2) // out.size(-2)
            kernel = Template(kernel).substitute(B=B, H=H, W=W, k=k, Dtype=getDtype(input))
            name = str(input.get_device())+'-'+str(B)+'-'+str(k)+'-'+str(H)+'-'+str(W)+'-periodize.cu'
            print(name)
            prog = Program(kernel, name.encode())
            ptx = prog.compile(['-arch='+get_compute_arch(input)])
            module = Module()
            module.load(bytes(ptx.encode()))
            self.periodize_cache[(input.size(), out.size(), input.get_device())] = module
        grid = (self.GET_BLOCKS(out.size(-3), self.block[0]),
                self.GET_BLOCKS(out.size(-2), self.block[1]),
                self.GET_BLOCKS(out.nelement() // (2*out.size(-2) * out.size(-3)), self.block[2]))
        periodize = self.periodize_cache[(input.size(), out.size(), input.get_device())].get_function('periodize')
        periodize(grid=grid, block=self.block, args=[input.data_ptr(), out.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out


class Modulus(object):
    """This class builds a wrapper to the moduli kernels and cache them.
        """
    def __init__(self, jit=True):
        self.modulus_cache = defaultdict(lambda: None)
        self.CUDA_NUM_THREADS = 1024
        self.jit = jit

    def GET_BLOCKS(self, N):
        return (N + self.CUDA_NUM_THREADS - 1) // self.CUDA_NUM_THREADS

    def __call__(self, input):
        if not self.jit or not isinstance(input, torch.cuda.FloatTensor):
            norm = input.norm(2, input.dim() - 1)
            return torch.cat([norm, norm.new(norm.size()).zero_()], input.dim() - 1)

        out = input.new(input.size())
        input = input.contiguous()

        if not iscomplex(input):
            raise TypeError('The input and outputs should be complex')

        if (self.modulus_cache[input.get_device()] is None):
            kernel = b"""
            extern "C"
            __global__ void abs_complex_value(const float * x, float2 * z, int n)
            {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            z[i] = make_float2(normf(2, x + 2*i), 0);

            }
            """
            print('modulus.cu')
            prog = Program(kernel, b'modulus.cu')
            ptx = prog.compile(['-arch='+get_compute_arch(input)])
            module = Module()
            module.load(bytes(ptx.encode()))
            self.modulus_cache[input.get_device()] = module
        fabs = self.modulus_cache[input.get_device()].get_function('abs_complex_value')
        fabs(grid=(self.GET_BLOCKS(int(out.nelement())//2), 1, 1),
             block=(self.CUDA_NUM_THREADS, 1, 1),
             args=[input.data_ptr(), out.data_ptr(), out.numel() // 2],
             stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return out


class Fft3d(object):
    """This class builds a wrapper to 3D FFTW on CPU / cuFFT on nvidia GPU."""

    def __init__(self):
        self.fftw_cache = defaultdict(lambda: None)
        self.cufft_cache = defaultdict(lambda: None)

    def buildCufftCache(self, input, type):
        raise NotImplementedError('cuFFT cache not implemented yet')
        # self.cufft_cache = None

    def buildFftwCache(self, input, inverse):
        direction = 'FFTW_BACKWARD' if inverse else 'FFTW_FORWARD'
        batch_size, M, N, O, _ = input.size()
        fftw_input_array = pyfftw.empty_aligned((batch_size, M, N, O), dtype='complex64')
        fftw_output_array = pyfftw.empty_aligned((batch_size, M, N, O), dtype='complex64')
        fftw_object = pyfftw.FFTW(fftw_input_array, fftw_output_array, axes=(1, 2, 3), direction=direction, threads=1)
        self.fftw_cache[(input.size(), inverse)] = (fftw_input_array, fftw_output_array, fftw_object)

    def __call__(self, input, inverse=False):
        if not isinstance(input, torch.cuda.FloatTensor):
            if not isinstance(input, (torch.FloatTensor, torch.DoubleTensor)):
                raise(TypeError('The input should be a torch.cuda.FloatTensor, \
                                torch.FloatTensor or a torch.DoubleTensor'))
            else:
                f = lambda x: np.stack((np.real(x), np.imag(x)), axis=len(x.shape))
                if(self.fftw_cache[(input.size(), inverse)] is None):
                    self.buildFftwCache(input, inverse)
                input_arr, output_arr, fftw_obj = self.fftw_cache[(input.size(), inverse)]

                input_arr[:] = input[..., 0].numpy() + 1.0j * input[..., 1].numpy()
                fftw_obj()

                return torch.from_numpy(f(output_arr).astype(input.numpy().dtype))

        raise NotImplementedError("3d cuFFT wrapper not implemented yet.")


def cdgmm3d(A, B):
    """Pointwise multiplication of 3d matrices in CPU or GPU."""
    A, B = A.contiguous(), B.contiguous()

    if A.size()[-4:] != B.size():
        raise RuntimeError('The filters are not compatible for multiplication!')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    if B.ndimension() != 4:
        raise RuntimeError('The filters must be simply a complex array!')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')

    if isinstance(A, (torch.FloatTensor, torch.DoubleTensor)):
        C = A.new(A.size())

        A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3)*A.size(-4))
        A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3)*A.size(-4))

        B_r = B[..., 0].contiguous().view(B.size(-2)*B.size(-3)*B.size(-4)).unsqueeze(0).expand_as(A_i)
        B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)*B.size(-4)).unsqueeze(0).expand_as(A_r)

        C[..., 0].copy_(A_r * B_r - A_i * B_i)
        C[..., 1].copy_(A_r * B_i + A_i * B_r)

        return C
    else:
        raise NotImplementedError("Cuda cdgmm not implemented")


class Fft(object):
    """This class builds a wrapper to the FFTs kernels and cache them.

    As a try, the library will purely work with complex data. The FFTS are UNORMALIZED.
        """

    def __init__(self):
        self.fft_cache = defaultdict(lambda: None)

    def buildCache(self, input, type):
        k = input.ndimension() - 3
        n = np.asarray([input.size(k), input.size(k+1)], np.int32)
        batch = input.nelement() // (2*input.size(k) * input.size(k + 1))
        idist = input.size(k) * input.size(k + 1)
        istride = 1
        ostride = istride
        odist = idist
        rank = 2
        plan = cufft.cufftPlanMany(rank, n.ctypes.data, n.ctypes.data, istride,
                                   idist, n.ctypes.data, ostride, odist, type, batch)
        self.fft_cache[(input.size(), type, input.get_device())] = plan

    def __del__(self):
        for keys in self.fft_cache:
            try:
                cufft.cufftDestroy(self.fft_cache[keys])
            except:
                pass

    def __call__(self, input, direction='C2C', inplace=False, inverse=False):
        if direction == 'C2R':
            inverse = True

        if not isinstance(input, torch.cuda.FloatTensor):
            if not isinstance(input, (torch.FloatTensor, torch.DoubleTensor)):
                raise(TypeError('The input should be a torch.cuda.FloatTensor, \
                                torch.FloatTensor or a torch.DoubleTensor'))
            else:
                input_np = input[..., 0].numpy() + 1.0j * input[..., 1].numpy()
                f = lambda x: np.stack((np.real(x), np.imag(x)), axis=len(x.shape))
                out_type = input.numpy().dtype

                if direction == 'C2R':
                    out = np.real(np.fft.ifft2(input_np)).astype(out_type)*input.size(-2)*input.size(-3)
                    return torch.from_numpy(out)

                if inplace:
                    if inverse:
                        out = f(np.fft.ifft2(input_np)).astype(out_type)*input.size(-2)*input.size(-3)
                    else:
                        out = f(np.fft.fft2(input_np)).astype(out_type)
                    input.copy_(torch.from_numpy(out))
                    return
                else:
                    if inverse:
                        out = f(np.fft.ifft2(input_np)).astype(out_type)*input.size(-2)*input.size(-3)
                    else:
                        out = f(np.fft.fft2(input_np)).astype(out_type)
                    return torch.from_numpy(out)

        if not iscomplex(input):
            raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensors must be contiguous!'))

        if direction == 'C2R':
            output = input.new(input.size()[:-1])
            if(self.fft_cache[(input.size(), cufft.CUFFT_C2R, input.get_device())] is None):
                self.buildCache(input, cufft.CUFFT_C2R)
            cufft.cufftExecC2R(self.fft_cache[(input.size(), cufft.CUFFT_C2R, input.get_device())],
                               input.data_ptr(), output.data_ptr())
            return output
        elif direction == 'C2C':
            output = input.new(input.size()) if not inplace else input
            flag = cufft.CUFFT_INVERSE if inverse else cufft.CUFFT_FORWARD
            if (self.fft_cache[(input.size(), cufft.CUFFT_C2C, input.get_device())] is None):
                self.buildCache(input, cufft.CUFFT_C2C)
            cufft.cufftExecC2C(self.fft_cache[(input.size(), cufft.CUFFT_C2C, input.get_device())],
                               input.data_ptr(), output.data_ptr(), flag)
            return output


def cdgmm(A, B, jit=True, inplace=False):
    """This function uses the C-wrapper to use cuBLAS.
        """
    A, B = A.contiguous(), B.contiguous()

    if A.size()[-3:] != B.size():
        raise RuntimeError('The filters are not compatible for multiplication!')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    if B.ndimension() != 3:
        raise RuntimeError('The filters must be simply a complex array!')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')

    if not jit or isinstance(A, (torch.FloatTensor, torch.DoubleTensor)):
        C = A.new(A.size())

        A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3))
        A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3))

        B_r = B[...,0].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_i)
        B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_r)

        C[..., 0].copy_(A_r * B_r - A_i * B_i)
        C[..., 1].copy_(A_r * B_i + A_i * B_r)

        # faster if B is actually real
        #B[...,1] = B[...,0]
        #C = A * B.unsqueeze(0).expand_as(A)
        return C if not inplace else A.copy_(C)
    else:
        C = A.new(A.size()) if not inplace else A
        m, n = B.nelement() // 2, A.nelement() // B.nelement()
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
        return C
