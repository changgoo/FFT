CC = mpicxx
FFT = fftw
EXE = exe/c2c_$(FFT)
OBJ = c2c_$(FFT).o
LIB = -L/u/cgkim/.local/lib -laccfft -lfftw3_mpi -lfftw3_omp -lfftw3 -lmpi
OPT = -O3 -fopenmp
INC = -I/u/cgkim/.local/include
#INC = 
#LIB = -lfftw3 -lfftw3_mpi -lmpi

#-------------------  macro definitions  ---------------------------------------

SRC = $(OBJ:.o=.cpp)

#--------------------  implicit rules  -----------------------------------------

.cpp.o:
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
