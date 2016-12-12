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

defs = {}
makeoption = {}
makeoption['FFT_SOLVER']=args['fft']
if args['fft'] == 'fftw':
  defs['FFT_SOLVER'] = 'FFTW'
elif args['fft'] == 'fftp':
  defs['FFT_SOLVER'] = 'FFT_PLIMPTON'
elif args['fft'] == 'pfft':
  defs['FFT_SOLVER'] = 'PFFT'
elif args['fft'] == 'accfft':
  defs['FFT_SOLVER'] = 'ACCFFT'

defsfile_input = 'defs.h.in'
defsfile_output = 'defs.h'
makefile_input = 'Makefile.in'
makefile_output = 'Makefile'

with open(defsfile_input, 'r') as current_file:
  defs_template = current_file.read()
with open(makefile_input, 'r') as current_file:
  make_template = current_file.read()

for key,val in defs.items():
  defs_template = re.sub(r'@{0}@'.format(key), val, defs_template)
for key,val in makeoption.items():
  make_template = re.sub(r'@{0}@'.format(key), val, make_template)

with open(defsfile_output, 'w') as current_file:
  current_file.write(defs_template)
with open(makefile_output, 'w') as current_file:
  current_file.write(make_template)


