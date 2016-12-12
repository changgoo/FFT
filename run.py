#! /usr/bin/env python
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--fft',
    default='fftw',
    choices=['fftw','fftp','pfft','accfft'],
    help='select fft solver')

# Parse command-line inputs
args = vars(parser.parse_args())

import subprocess
import os
os.chdir('src/')
return_code = subprocess.call("make clean", shell=True)  
return_code = subprocess.call("./configure.py --fft=%s" % args['fft'], shell=True)  
return_code = subprocess.call("make all", shell=True)  
os.chdir('../exe')

#block decomposition

Nb=np.array([32,32,32])
Np=np.array([2,2,2])
for i in range(6):
  Nx=Np*Nb*2**i
  nproc=(Np*2**i).prod()
  nnode=(nproc-1)/28 + 1
  if nnode > 100: break
  print args['fft'],2**i,Nx,nproc,nnode

  slurm={}
  slurm['NNODE']='%d' % nnode
  slurm['NPROC']='%d' % nproc
  slurm['Nx1']='%d' % Nx[0]
  slurm['Nx2']='%d' % Nx[1]
  slurm['Nx3']='%d' % Nx[2]
  slurm['Nb1']='%d' % Nb[0]
  slurm['Nb2']='%d' % Nb[1]
  slurm['Nb3']='%d' % Nb[2]
  slurm['FFT_SOLVER']=args['fft']

  slurm_input='fft_test'
  slurm_output='%s_test' % slurm['FFT_SOLVER']
  with open(slurm_input, 'r') as current_file:
    slurm_template = current_file.read()

  for key,val in slurm.items():
    slurm_template = re.sub(r'@{0}@'.format(key), val, slurm_template)

  with open(slurm_output, 'w') as current_file:
    current_file.write(slurm_template)

  return_code = subprocess.call("sbatch %s" % slurm_output, shell=True)  
