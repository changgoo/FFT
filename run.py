#! /usr/bin/env python
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--run',
    action='store_true',
    help='submit job')

parser.add_argument('--server',
    default='perseus',
    choices=['perseus','tiger','pleiades'],
    help='server name')

parser.add_argument('--decomp',
    default='block',
    choices=['block','pencil','slab'],
    help='server name')

parser.add_argument('--fft',
    default='fftw',
    choices=['fftw','fftp','pfft','accfft'],
    help='select fft solver')

# Parse command-line inputs
args = vars(parser.parse_args())

if args['decomp']=='block' and args['fft']=='fftw':
 raise SystemExit('CONFIGURE ERROR: Block decomposition is not supported by FFTW')
if args['decomp']=='block' and args['fft']=='accfft':
 raise SystemExit('CONFIGURE ERROR: Block decomposition is not supported by ACCFFT')
if args['decomp']=='pencil' and args['fft']=='fftw':
 raise SystemExit('CONFIGURE ERROR: Pencil decomposition is not supported by ACCFFT')

import subprocess
import os
os.chdir('src/')
return_code = subprocess.call("make clean", shell=True)  
return_code = subprocess.call("./configure.py --fft=%s" % args['fft'], shell=True)  
return_code = subprocess.call("make all MACHINE=%s" % args['server'], shell=True)  
os.chdir('../exe')

if args['server']=='perseus' or args['server']=='pleiades': 
  nproc_node = 28
  Nx_set=([224,64,64],[224,128,64],[224,128,128])
elif args['server']=='tiger':
  nproc_node = 16
  Nx_set=([64,64,64],[128,64,64],[256,64,64])

Nb=np.array([64,64,64])
Nbunit=Nb.prod()
for Nx0 in Nx_set:
  for i in range(6):
    Nx=np.array(Nx0)*2**i
    if args['decomp'] == 'block':
      #block decomposition
      Np=Nx/Nb
    elif args['decomp'] == 'pencil':
      #pencil decomposition
      Nb3=Nx[2]
      Nb2=2**int(np.log2(Nbunit/Nb3)/2+1)
      Nb1=Nbunit/Nb3/Nb2
      Nb=np.array([Nb1,Nb2,Nb3])
      Np=Nx/Nb
    elif args['decomp'] == 'slab':
      #slab decomposition
      Nb3=Nx[2]
      Nb2=Nx[1]
      Nb1=Nbunit/Nx[2]/Nx[1]
      if Nb1 == 0: break
      Nb=np.array([Nb1,Nb2,Nb3])
      Np=Nx/Nb
 
    nproc=Np.prod()
    nnode=(nproc-1)/nproc_node + 1
    if nproc > 5000: break
    print args['fft'],2**i,Nx,Nb,Np,nproc,nnode
 
    slurm={}
    slurm['NNODE']='%d' % nnode
    slurm['NCPUS']='%d' % nproc_node
    slurm['NPROC']='%d' % nproc
    slurm['Nx1']='%d' % Nx[0]
    slurm['Nx2']='%d' % Nx[1]
    slurm['Nx3']='%d' % Nx[2]
    slurm['Nb1']='%d' % Nb[0]
    slurm['Nb2']='%d' % Nb[1]
    slurm['Nb3']='%d' % Nb[2]
    slurm['FFT_SOLVER']=args['fft']
    slurm['DECOMP']=args['decomp']
 
    slurm_input='fft_test_%s' % args['server']
    slurm_output='%s-%s-%d' % (slurm['FFT_SOLVER'],args['decomp'],nproc)

    outname='%s-%s-%d-%dx%dx%d.timing' % (slurm['FFT_SOLVER'],args['decomp'],nproc,Nb[0],Nb[1],Nb[2])
    slurm['OUTNAME']=outname
    with open(slurm_input, 'r') as current_file:
      slurm_template = current_file.read()
 
    for key,val in slurm.items():
      slurm_template = re.sub(r'@{0}@'.format(key), val, slurm_template)
 
    with open(slurm_output, 'w') as current_file:
      current_file.write(slurm_template)
 
    if args['run']: return_code = subprocess.call("sbatch %s" % slurm_output, shell=True)  
