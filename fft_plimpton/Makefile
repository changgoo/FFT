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
INC = -I/u/cgkim/.local/include

#-------------------  macro definitions  ---------------------------------------

SRC = $(OBJ:.o=.c)

#--------------------  implicit rules  -----------------------------------------

.c.o:
	${CC} ${INC} -c $<

#---------------------  targets  -----------------------------------------------

all:	compile

.PHONY: compile
compile: ${OBJ}
#	ar rcs libplimpton.a ${OBJ}

help:
	@echo This is the FFT Makefile
	@echo Type 'make compile' to generate FFT object files
	@echo Type 'make clean'   to remove '*.o' files
	@echo OBJ=$(OBJ)

.PHONY: clean
clean:
	rm -f *.o 
