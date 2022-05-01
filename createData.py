import os
import numpy as np
from anisotropicDiffusion import *
from skimage import io
import skimage
import skimage.transform
import argparse


def main():

	currentdir = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(currentdir,'data')

	parser = argparse.ArgumentParser(description='Diffusion')
	parser.add_argument('-img_path',
	                type=str, nargs=1, required=True)
	parser.add_argument('-niter',
	                type=int, nargs=1, required=True)
	args = parser.parse_args()

	img_path = args.img_path[0]
	output_dir = data_dir
	niter = args.niter[0]

	img = io.imread(img_path)

	print("-----Generating Data-----")

	img = anisotropic_diffusion(output_dir, img, niter=niter, kappa=50, gamma=0.25).astype(img.dtype)

	print("-----Finished-----")

if __name__=="__main__":
    main()