CC= mpicxx
FFT = fftw
MACHINE = tigress
EXE = exe/c2c_$(FFT)
#OBJ = c2c_$(FFT).o
OBJ = c2c_test.o
ifeq ($(MACHINE),local)
  FFT_PATH =/u/cgkim/.local/
else ifeq ($(MACHINE),tigress)
  FFT_PATH =/tigress/changgoo/local
endif
INC = -I$(FFT_PATH)/include
LIB_PATH = -L$(FFT_PATH)/lib

ifeq ($(FFT),accfft)
  LIB_FFT = -laccfft
else ifeq ($(FFT),openfft)
  LIB_FFT = -lopenfft
  CC = mpicc
else ifeq ($(FFT),pfft)
  LIB_FFT = -lpfft
  CC = mpicc -std=c99
else ifeq ($(FFT),fftp)
  INC += -I./fft_plimpton
  OBJ += $(FFT_OBJ)
  LIB_FFT = -L./fft_plimpton -lfftp
  CC =mpicc
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
	${CC} $(OPT) -o ${EXE} ${OBJ} ${LIB}

help:
	@echo This is the FFT Makefile
	@echo Type 'make compile' to generate FFT object files
	@echo Type 'make clean'   to remove '*.o' files
	@echo OBJ=$(OBJ)

.PHONY: clean
clean:
	rm -f *.o
