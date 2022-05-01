import numpy
from skimage import io

import os
import sys
from tqdm import tqdm


def write_to_file(img, output_dir, iter):
    os.umask(0)
    os.makedirs(output_dir, 0o777, exist_ok=True)
    filename = 'h5_f_{iter:010d}.{ext}'
    imgoutput = os.path.join(output_dir, filename.format(iter=iter, ext='png'))
    io.imsave(imgoutput, img)
    

# http://loli.github.io/medpy/_modules/medpy/filter/smoothing.html#anisotropic_diffusion
def anisotropic_diffusion(output_dir, img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1):
    # define conduction gradients functions
    if option == 1:
        def condgradient(delta, spacing):
            return numpy.exp(-(delta/kappa)**2.)/float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1./(1.+(delta/kappa)**2.)/float(spacing)
    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5*((1.-(delta/kappa_s)**2.)**2.)/float(spacing)
            return numpy.where(numpy.abs(delta) <= kappa_s, top, 0)

    # initialize output array
    out = numpy.array(img, dtype=numpy.float32, copy=True)

    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [numpy.zeros_like(out) for _ in range(out.ndim)]

    
    time = 0
    write_to_file(out.astype(img.dtype), output_dir, 0)

    for iter in tqdm(range(niter)):
        # calculate the diffs
        for i in range(out.ndim):
            slicer = [slice(None, -1) if j == i else slice(None) for j in range(out.ndim)]
            diff_local = numpy.diff(out, axis=i)
            deltas[i][tuple(slicer)] = diff_local

        # multiply c
        # matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxelspacing)]
        matrices = [delta for delta, spacing in zip(deltas, voxelspacing)]

        # second derivative
        for i in range(out.ndim):
            slicer = [slice(1, None) if j == i else slice(None) for j in range(out.ndim)]
            matrices[i][tuple(slicer)] = numpy.diff(matrices[i], axis=i)

        # update the image
        out += gamma * (numpy.sum(matrices, axis=0))
        time += gamma


        write_to_file(out.astype(img.dtype), output_dir, iter + 1)

    return out