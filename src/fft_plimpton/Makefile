#    @WARNING1@
#    @WARNING2@
#===============================================================================
# FILE: Makefile.in
#
# PURPOSE: Template for src/fftsrc/Makefile.  When 'configure' is run, a new
# Makefile will be created (overwriting the last) from this template.
#
#-------------------  object files  --------------------------------------------
CORE_OBJ = factor.o \
           fft_2d.o \
           fft_3d.o \
           pack_2d.o \
           pack_3d.o \
           remap_2d.o \
           remap_3d.o

CC = mpicc
OBJ = $(CORE_OBJ)
INC = -I/nasa/intel/Compiler/2016.2.181/mkl/include/fftw

#-------------------  macro definitions  ---------------------------------------

SRC = $(OBJ:.o=.c)


#---------------------  targets  -----------------------------------------------

FFTPLIB = ./libfftp.a


all: lib 

lib: libfftp.a

libfftp.a:  $(OFFTLIB)

%.o: %.c
	${CC} ${INC} -c $<

$(FFTPLIB): $(OBJ)
	ar rcs $(FFTPLIB) $(OBJ)

help:
	@echo This is the FFT Makefile
	@echo Type 'make compile' to generate FFT object files
	@echo Type 'make clean'   to remove '*.o' files
	@echo OBJ=$(OBJ)

.PHONY: clean
clean:
	rm -f *.o *.a
