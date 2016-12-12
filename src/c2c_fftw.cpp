/*
 * File: step3.cpp
 * License: Please see LICENSE file.
 * AccFFT: Massively Parallel FFT Library
 * Created by Amir Gholami on 12/23/2014
 * Email: contact@accfft.org
 */

#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <math.h> // M_PI
#include <mpi.h>
#include <fftw3-mpi.h>

#define PCOUT if(procid==0) std::cout
#define F3DI(i, j, k, nx1, nx2, nx3) ((k) + (nx3)*((j) + (nx2)*(i)))
#define F2DI(i, j, nx1, nx2) ((j) + (nx2)*(i))

#define Complex fftw_complex

void initialize(Complex *a, int*n, MPI_Comm c_comm);
void check_err(Complex* a, int*n, MPI_Comm c_comm);
void c2c_fftw(int *n, int nthreads);

inline double testcase(double X, double Y, double Z) {

	double sigma = 4;
	double pi = M_PI;
	double analytic;
	analytic = std::exp(-sigma * ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
							+ (Z - pi) * (Z - pi)));
	if (analytic != analytic)
		analytic = 0; /* Do you think the condition will be false always? */
	return analytic;
} // end testcase

void c2c_fftw(int *n, int nthreads) {
	int nprocs, procid;
	fftw_plan fplan,bplan;
	ptrdiff_t alloc_local, n1, is, i, j, k;

	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	fftw_mpi_init();

	fftw_complex *data; 

	double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;

	/* Get the local pencil size and the allocation size */
	alloc_local = fftw_mpi_local_size_3d(n[0], n[1], n[2], MPI_COMM_WORLD,
		&n1, &is);
	printf("[mpi rank %d] nlocal  %3d %3d %3d is  %3d\n", procid,
		n1,n[1],n[2],is);

	data = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local);

	/* Create FFT plan */
	setup_time = -MPI_Wtime();
	fplan = fftw_mpi_plan_dft_3d(n[0], n[1], n[2], data, data, MPI_COMM_WORLD,
		FFTW_FORWARD, FFTW_MEASURE);
	bplan = fftw_mpi_plan_dft_3d(n[0], n[1], n[2], data, data, MPI_COMM_WORLD,
		FFTW_BACKWARD, FFTW_MEASURE);
	setup_time += MPI_Wtime();

	/*  Initialize data */
	initialize(data, n, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	/* Perform forward FFT */
	f_time -= MPI_Wtime();
        fftw_execute(fplan);
	f_time += MPI_Wtime();

	MPI_Barrier(MPI_COMM_WORLD);

	/* Perform backward FFT */
	i_time-=MPI_Wtime();
        fftw_execute(bplan);
	i_time+=MPI_Wtime();

	check_err(data,n,MPI_COMM_WORLD);

	/* Compute some timings statistics */
	double g_f_time, g_i_time, g_setup_time;
	MPI_Reduce(&f_time, &g_f_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&i_time, &g_i_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&setup_time, &g_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0,
			MPI_COMM_WORLD);

	PCOUT<<"Timing for Inplace FFT of size "<<n[0]<<"*"<<n[1]<<"*"<<n[2]<<std::endl;
	PCOUT << "Setup \t" << g_setup_time << std::endl;
	PCOUT << "FFT \t" << g_f_time << std::endl;
	PCOUT << "IFFT \t" << g_i_time << std::endl;

	fftw_free(data);
	fftw_destroy_plan(fplan);
	fftw_destroy_plan(bplan);
	return;

} // end c2c_fftw

int main(int argc, char **argv) {

	int NX, NY, NZ;
	MPI_Init(&argc, &argv);
	int nprocs, procid;
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	/* Parsing Inputs  */
	if (argc == 1) {
		NX = 128;
		NY = 128;
		NZ = 128;
	} else {
		NX = atoi(argv[1]);
		NY = atoi(argv[2]);
		NZ = atoi(argv[3]);
	}
	int N[3] = { NX, NY, NZ };

	int nthreads = 1;
	c2c_fftw(N, nthreads);

	MPI_Finalize();
	return 0;
} // end main

void initialize(Complex *a, int *n, MPI_Comm c_comm) {
	double pi = M_PI;
	ptrdiff_t alloc_local, n1, is;
	alloc_local = fftw_mpi_local_size_3d(n[0], n[1], n[2], c_comm,
		&n1, &is);
	{
		double X, Y, Z;
		long int ptr;
		for (int i = 0; i < n1; i++) {
			for (int j = 0; j < n[1]; j++) {
				for (int k = 0; k < n[2]; k++) {
					X = 2 * pi / n[0] * (i + is);
					Y = 2 * pi / n[1] * j;
					Z = 2 * pi / n[2] * k;
					ptr = F3DI(i,j,k,n[0],n[1],n[2]);
					a[ptr][0] = testcase(X, Y, Z); // Real Component
					a[ptr][1] = testcase(X, Y, Z); // Imag Component
				}
			}
		}
	}
	return;
} // end initialize

void check_err(Complex* a, int*n, MPI_Comm c_comm) {
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);
	long long int size = n[0];
	size *= n[1];
	size *= n[2];
	double pi = 4 * atan(1.0);

	ptrdiff_t alloc_local, n1, is;
	alloc_local = fftw_mpi_local_size_3d(n[0], n[1], n[2], c_comm,
		&n1, &is);

	double err = 0, norm = 0;

	double X, Y, Z, numerical_r, numerical_c;
	long int ptr;
	for (int i = 0; i < n1; i++) {
		for (int j = 0; j <n[1]; j++) {
			for (int k = 0; k < n[2]; k++) {
				X = 2 * pi / n[0] * (i + is);
				Y = 2 * pi / n[1] * j;
				Z = 2 * pi / n[2] * k;
				ptr = F3DI(i,j,k,n[0],n[1],n[2]);
				numerical_r = a[ptr][0] / size;
				if (numerical_r != numerical_r)
					numerical_r = 0;
				numerical_c = a[ptr][1] / size;
				if (numerical_c != numerical_c)
					numerical_c = 0;
				err += std::abs(numerical_r - testcase(X, Y, Z))
						+ std::abs(numerical_c - testcase(X, Y, Z));
				norm += std::abs(testcase(X, Y, Z));

			}
		}
	}

	double g_err = 0, g_norm = 0;
	MPI_Reduce(&err, &g_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&norm, &g_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	PCOUT << "\nL1 Error of iFF(a)-a: " << g_err << std::endl;
	PCOUT << "Relative L1 Error of iFF(a)-a: " << g_err / g_norm << std::endl;
	if (g_err / g_norm < 1e-10)
		PCOUT << "\nResults are CORRECT!\n\n";
	else
		PCOUT << "\nResults are NOT CORRECT!\n\n";

	return;
} // end check_err
