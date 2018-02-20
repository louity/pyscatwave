PyScatWave
==========

CuPy/PyTorch Scattering implementation

A scattering network is a Convolutional Network with filters predefined to be wavelets that are not learned and it can be used in vision task such as classification of images. The scattering transform can drastically reduce the spatial resolution of the input (e.g. 224x224->14x14) with demonstrably neglible loss in dicriminative power.   

The software uses PyTorch + NumPy FFT on CPU, and PyTorch + CuPy + CuFFT on GPU.



Previous (lua-based) versions of the code can be found at <https://github.com/edouardoyallon/scatwave>

If using this code for your research please cite our paper:

E. Oyallon, E. Belilovsky, S. Zagoruyko [*Scaling the Scattering Transform: Deep Hybrid Networks*](https://arxiv.org/abs/1703.08961)

You can find experiments from the paper in the following repository:
https://github.com/edouardoyallon/scalingscattering/

We used PyTorch for running experiments in <https://arxiv.org/abs/1703.08961>,
but it is possible to use scattering with other frameworks (e.g. Chainer, Theano or Tensorflow) if one copies Scattering outputs to CPU (or run on CPU and convert to `numpy.ndarray` via `.numpy()`).

## Benchmarks
We do some simple timings and comparisons to the previous (multi-core CPU) implementation of scattering (ScatnetLight). We benchmark the software using a 1080 GPU. Below we show input sizes (WxHx3xBatchSize) and speed:

32 × 32 × 3 × 128 (J=2)- 0.03s (speed of 8x vs ScatNetLight)

256 × 256 × 3 × 128 (J=2) - 0.71 s (speed up of 225x vs ScatNetLight)

## Installation

The software was tested on Linux with anaconda Python 2.7 and
various GPUs, including Titan X, 1080s, 980s, K20s, and Titan X Pascal.

The first step is to install pytorch following instructions from
<http://pytorch.org>, then you can run `pip`:

```
pip install -r requirements.txt
python setup.py install
```

## Usage

Example:
  - 2D Gabor scattering:
    ```python
    import torch
    from scatwave.scattering import Scattering, SolidHarmonicScattering

    scat = Scattering(M=32, N=32, J=2).cuda()
    x = torch.randn(1, 3, 32, 32).cuda()

    print scat(x).size()

    s_harm_scat = SolidHarmonicScattering(M=64, N=64, O=64, J=2, L=1)
    x_2 = torch.zeros(1, 64, 64, 64)

    print s_harm_scat(x_2).size()
    ```
  - 3D Solid harmonic scattering:
    ```python
    import numpy as np
    import torch
    from scatwave.scattering import SolidHarmonicScattering
    from scatwave.utils import generate_sum_of_gaussians

    centers = np.zeros((1, 3))
    sigma = 4.
    M, N, O = 64, 64, 64
    signal = generate_sum_of_gaussians(centers, sigma, M, N, O).reshape(1, M, N, O)

    x = torch.from_numpy(signal)

    scat = SolidHarmonicScattering(M=M, N=N, O=O, J=2, L=2)

    print scat(x, [1, 2]).size()
    ```


## Contribution

All contributions are welcome.


## Authors

Edouard Oyallon, Eugene Belilovsky, Sergey Zagoruyko
