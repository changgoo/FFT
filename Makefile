CC= mpicxx
FFT = fftw
EXE = exe/c2c_$(FFT)

OBJ = c2c_$(FFT).o
INC = -I/u/cgkim/.local/include
LIB_PATH = -L/u/cgkim/.local/lib
ifeq ($(FFT),accfft)
  LIB_FFT = -laccfft
else ifeq ($(FFT),pfft)
  LIB_FFT = -lpfft
  CC = mpicc -std=c99
else ifeq ($(FFT),fftp)
  FFT_OBJ = fft_plimpton/factor.o \
            fft_plimpton/fft_2d.o \
            fft_plimpton/fft_3d.o \
            fft_plimpton/pack_2d.o \
            fft_plimpton/pack_3d.o \
            fft_plimpton/remap_2d.o \
            fft_plimpton/remap_3d.o 
  OBJ += $(FFT_OBJ)
  INC += -I/scr1/cgkim/Research/FFT/fft_plimpton
  LIB_FFT = 
  CC=mpicc
else
  LIB_FFT = 
endif
LIB_FFTW = -lfftw3_mpi -lfftw3_omp -lfftw3 -lmpi
LIB = $(LIB_PATH) $(LIB_FFT) $(LIB_FFTW)
OPT = -O3 -fopenmp

#-------------------  macro definitions  ---------------------------------------

ifeq ($(CC),mpicxx)
  SRC = $(OBJ:.o=.cpp)
else ifeq ($(CC),mpicc)
  SRC = $(OBJ:.o=.c)
endif

#--------------------  implicit rules  -----------------------------------------

.cpp.o:
	${CC} ${INC} -c $<

.c.o:
	${CC} ${INC} -c $<


#---------------------  targets  -----------------------------------------------

all:    compile

.PHONY: compile
compile: ${EXE}

${EXE}: ${OBJ}
ifeq ($(FFT),fftp)
	(cd fft_plimpton; $(MAKE) compile; cd ..)
endif
	${CC} $(OPT) -o ${EXE} ${OBJ} ${LIB}

help:
	@echo This is the FFT Makefile
	@echo Type 'make compile' to generate FFT object files
	@echo Type 'make clean'   to remove '*.o' files
	@echo OBJ=$(OBJ)

.PHONY: clean
clean:
ifeq ($(FFT),fftp)
	(cd fft_plimpton; $(MAKE) clean; cd ..)
endif
	rm -f *.o
