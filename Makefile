CC = mpicc
EXE = exe/fftw_example
OPT = 
INC = -I/u/cgkim/.local/include
LIB = -L/u/cgkim/.local/lib -lfftw3_mpi -lfftw3 -lmpi
#INC = 
#LIB = -lfftw3 -lfftw3_mpi -lmpi
OBJ = fftw_example.o

#-------------------  macro definitions  ---------------------------------------

SRC = $(OBJ:.o=.c)

#--------------------  implicit rules  -----------------------------------------

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
