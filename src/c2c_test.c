#include <stdio.h>
#include <stdlib.h>
#include <math.h> // M_PI
#include <mpi.h>
#include "defs.h"

static int nprocs, procid;
static int np[3];

void decompose(fft_int* Nx, fft_int* Nb, fft_int* is, int nthreads);
Complex *data_alloc(fft_int alloc_local);
void fft_plan(Complex *data, fft_int* Nx, fft_int* Nb, fft_int* is, double *fft_time, int nthreads);
void do_fft(Complex *data, double *fft_time, int nthreads);
void fft_destroy(Complex *data);

void check_err(Complex*a, fft_int* Nx, fft_int* Nb, fft_int* is);
void initialize(Complex*a, fft_int* Nx, fft_int* Nb, fft_int* is);
void timing(fft_int *Nx,double *fft_time, int Ntry);
double testcase(double X, double Y, double Z);

int main(int argc, char **argv) {

  fft_int NX, NY, NZ;
  fft_int nx, ny, nz;
  double fft_time[3];
  fft_time[0]=0.0; fft_time[1]=0.0; fft_time[2]=0.0;
  Complex *data;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* Parsing Inputs  */
  if (argc == 1) {
    NX = 128;
    NY = 128;
    NZ = 128;
    nx = 64;
    ny = 64;
    nz = 64;
  } else if (argc == 3) {
    NX = atoi(argv[1]);
    nx = atoi(argv[2]);
    NY = NX; NZ = NX;
    ny = nx; nz = nx;
  } else if (argc == 7) {
    NX = atoi(argv[1]);
    NY = atoi(argv[2]);
    NZ = atoi(argv[3]);
    nx = atoi(argv[4]);
    ny = atoi(argv[5]);
    nz = atoi(argv[6]);
  } else {
    printf("\n NUMBER of ARGUMENTS should be 2 or 6!! \n");
  }
  fft_int Nx[3] = { NX, NY, NZ };
  fft_int Nb[3] = { nx, ny, nz };
  fft_int is[3];
  int nthreads = 1;
  int Ntry=10,itry;

  decompose(Nx, Nb, is, nthreads);

  /* Allocate memory */
  fft_int alloc_local = Nb[0]*Nb[1]*Nb[2];
  data=data_alloc(alloc_local);
  fft_plan(data, Nx, Nb, is, fft_time, nthreads);

  for(itry=0;itry<Ntry;itry++){
    initialize(data, Nx, Nb, is);
    do_fft(data, fft_time, nthreads);
    check_err(data, Nx, Nb, is);
  }

  timing(Nx,fft_time, Ntry);

  fft_destroy(data);

  MPI_Finalize();
  return 0;
} // end main

void decompose(fft_int* Nx, fft_int* Nb, fft_int* is, int nthreads){
  int ip,jp,kp;
#ifndef OPENFFT
  np[0] = Nx[0]/Nb[0];
  np[1] = Nx[1]/Nb[1];
  np[2] = Nx[2]/Nb[2];
  if(nprocs != np[0]*np[1]*np[2]){
    if(procid == 0)
      printf("%d: cannot be decomposed to %d %d %d\n", nprocs,
             np[0], np[1], np[2]);
    exit(1);
  }
  /* Get parameters of data distribution */
  ip = procid/(np[2]*np[1]);
  jp = (procid-np[2]*np[1]*ip)/np[2];
  kp = procid-np[2]*np[1]*ip-np[2]*jp;
  is[2] = kp*Nb[2];
  is[1] = jp*Nb[1];
  is[0] = ip*Nb[0];
#endif

#if defined (PFFT)
  pfft_init();
  if(np[2] > 1) {
    pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d);
  } else {
    pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_3d);
  }

  /* Get parameters of data distribution */
  fft_int alloc_local,Ni[3],No[3],iis[3],ios[3];
  alloc_local = pfft_local_size_dft_3d(Nx, comm_cart_3d, PFFT_TRANSPOSED_NONE,
      Ni, iis, No, ios);
  if( is[0] != iis[0] || is[1] != iis[1] || is[2] != iis[2] ){
    printf("Decomposition is inconsistent\n");
    printf("My decomposition %d %d %d\n",is[0],is[1],is[2]);
    printf("PFFT decomposition %d %d %d\n",iis[0],iis[1],iis[2]);
    exit(1);
  }
#elif defined (FFTW)
  fftw_mpi_init();
  if (np[1] > 1 || np[2] > 1){
    printf("FFTW only support SLAB decomposition\n");
    exit(1);
  }
#elif defined (ACCFFT)
  fft_int c_dims[2] = { np[0], np[1] };
  fft_int alloc_local,Ni[3],No[3],iis[3],ios[3];
  accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);
  alloc_local = accfft_local_size_dft_c2c(Nx, Ni, iis, No, ios, c_comm);
  if( is[0] != iis[0] || is[1] != iis[1] || is[2] != iis[2] ){
    printf("Decomposition is inconsistent\n");
    printf("My decomposition %d %d %d\n",is[0],is[1],is[2]);
    printf("ACCFFT decomposition %d %d %d\n",iis[0],iis[1],iis[2]);
    exit(1);
  }
  accfft_init(nthreads);
#elif defined (OPENFFT)
  int offt_measure,measure_time,print_memory;
  int My_Max_NumGrid,My_NumGrid_In,My_NumGrid_Out;
  int My_Index_In[6],My_Index_Out[6];

  offt_measure = 0;
  measure_time = 0;
  print_memory = 0;

  /* Initialize OpenFFT */ 

  openfft_init_c2c_3d(Nx[0],Nx[1],Nx[2],
		     &My_Max_NumGrid,&My_NumGrid_In,My_Index_In,
		     &My_NumGrid_Out,My_Index_Out,
		     offt_measure,measure_time,print_memory);
/*
  printf("[OPENFFT %d] My_NumGrid %3d %3d %3d\n", procid, My_Max_NumGrid, My_NumGrid_In,My_NumGrid_Out);
  printf("[OPENFFT %d] My_Index_In %3d %3d %3d %3d %3d %3d\n", procid, My_Index_In[0], My_Index_In[1], My_Index_In[2], My_Index_In[3], My_Index_In[4], My_Index_In[5]);
  printf("[OPENFFT %d] My_Index_Out %3d %3d %3d %3d %3d %3d\n", procid, My_Index_Out[0], My_Index_Out[1], My_Index_Out[2], My_Index_Out[3], My_Index_Out[4], My_Index_Out[5]);
*/

  is[0]=My_Index_In[0];
  is[1]=My_Index_In[1];
  is[2]=My_Index_In[2];
  Nb[0]=My_Index_In[3]-My_Index_In[0]+1;
  Nb[1]=My_Index_In[4]-My_Index_In[1]+1;
  Nb[2]=My_Index_In[5]-My_Index_In[2]+1;

  np[0] = Nx[0]/Nb[0];
  np[1] = Nx[1]/Nb[1];
  np[2] = Nx[2]/Nb[2];
#endif

/*
  printf("[mpi rank %d] block size  %3d %3d %3d\n", procid,
		Nb[0],Nb[1],Nb[2]);

  printf("[mpi rank %d] istart      %3d %3d %3d\n", procid,
		is[0],is[1],is[2]);
*/

  return;
}

Complex *data_alloc(fft_int alloc_local){
#ifdef PFFT
  return pfft_alloc_complex(alloc_local);
#else
  return (Complex *) fftw_malloc(sizeof(Complex) * alloc_local);
#endif
}

void fft_plan(Complex *data, fft_int* Nx, fft_int* Nb, fft_int* is, double *fft_time, int nthreads){
  int scaled=0,permute=0,nbuf;
  fft_int ie[3];
  ie[0] = is[0] + Nb[0] - 1;
  ie[1] = is[1] + Nb[1] - 1;
  ie[2] = is[2] + Nb[2] - 1;

  fft_time[0] = -MPI_Wtime();
#if defined (FFT_PLIMPTON)
  plan = fft_3d_create_plan(MPI_COMM_WORLD, Nx[2], Nx[1], Nx[0],
                                   is[2], ie[2], is[1], ie[1], is[0], ie[0], 
	                           is[2], ie[2], is[1], ie[1], is[0], ie[0], 
                                   scaled, permute, &nbuf);
#elif defined (FFTW)
  fplan = fftw_mpi_plan_dft_3d(Nx[0], Nx[1], Nx[2], data, data, MPI_COMM_WORLD,
		FFTW_FORWARD, FFTW_MEASURE);
  bplan = fftw_mpi_plan_dft_3d(Nx[0], Nx[1], Nx[2], data, data, MPI_COMM_WORLD,
		FFTW_BACKWARD, FFTW_MEASURE);
#elif defined (PFFT)
  fplan = pfft_plan_dft_3d(Nx, data, data, comm_cart_3d,
    PFFT_FORWARD, PFFT_TRANSPOSED_NONE| PFFT_MEASURE| PFFT_DESTROY_INPUT);
  bplan = pfft_plan_dft_3d(Nx, data, data, comm_cart_3d,
    PFFT_BACKWARD, PFFT_TRANSPOSED_NONE| PFFT_MEASURE| PFFT_DESTROY_INPUT);
#elif defined (ACCFFT)
  plan = accfft_plan_dft_3d_c2c(Nx,data,data,c_comm,ACCFFT_MEASURE);
#endif
  fft_time[0] += MPI_Wtime();

  return;
}

void do_fft(Complex *data, double *fft_time, int nthreads){
  double f_time = 0, i_time = 0;
  
  /* execute parallel forward FFT */
  f_time -= MPI_Wtime();
#if defined (PFFT)
  pfft_execute(fplan);
#elif defined (FFTW)
  fftw_execute(fplan);
#elif defined (FFT_PLIMPTON)
  fft_3d(data, data, FFTW_FORWARD, plan);
#elif defined (ACCFFT)
  accfft_execute_c2c(plan,ACCFFT_FORWARD,data,data);
#elif defined (OPENFFT)
  openfft_exec_c2c_3d(data, data);
#endif
  f_time += MPI_Wtime();
//  MPI_Barrier(MPI_COMM_WORLD);

  /* Perform backward FFT */
  i_time-=MPI_Wtime();
#if defined (PFFT)
  pfft_execute(bplan);
#elif defined (FFTW)
  fftw_execute(bplan);
#elif defined (FFT_PLIMPTON)
  fft_3d(data, data, FFTW_BACKWARD, plan);
#elif defined (ACCFFT)
  accfft_execute_c2c(plan,ACCFFT_BACKWARD,data,data);
#elif defined (OPENFFT)
  openfft_exec_c2c_3d(data, data);
#endif
  i_time+=MPI_Wtime();
//  MPI_Barrier(MPI_COMM_WORLD);

  fft_time[1]+=f_time;
  fft_time[2]+=i_time;
  return;
}

void fft_destroy(Complex *data){
  /* free mem and finalize */
#if defined (PFFT)
  pfft_destroy_plan(fplan);
  pfft_destroy_plan(bplan);
  MPI_Comm_free(&comm_cart_3d);
  pfft_free(data);
#elif defined (FFTW)
  fftw_free(data);
  fftw_destroy_plan(fplan);
  fftw_destroy_plan(bplan);
#elif defined (FFT_PLIMPTON)
  fftw_free(data);
  fft_3d_destroy_plan(plan);
#elif defined (ACCFFT)
  accfft_free(data);
  accfft_destroy_plan(plan);
  accfft_cleanup();
  MPI_Comm_free(&c_comm);
#endif
  return;
}


void timing(fft_int*Nx, double *fft_time, int Ntry){
  double g_fft_time[3];
  long long int zones=Nx[0]*Nx[1]*Nx[2];
  double zcs;
  MPI_Reduce(&fft_time[0], &g_fft_time[0], 3, MPI_DOUBLE, MPI_MAX, 0,
			MPI_COMM_WORLD);
  zcs=(double)(zones*Ntry)/(g_fft_time[1]+g_fft_time[2]);
  /* Compute some timings statistics */
  if(procid == 0){
    printf("Timing for Inplace FFT of size %i %i %i\n",Nx[0],Nx[1],Nx[2]);
    printf("with MPI configuration %i %i %i\n",np[0],np[1],np[2]);
    printf("Setup\t %g\n",g_fft_time[0]);
    printf("FFT \t %g\n",g_fft_time[1]/Ntry);
    printf("IFFT \t %g\n",g_fft_time[2]/Ntry);
    printf("zcs\t %g\n",zcs);
  }
  return;
}


void initialize(Complex*a, fft_int* Nx, fft_int* Nb, fft_int* is){
  double pi = 4 * atan(1.0);
  double X, Y, Z;
  long int ptr;
  int i,j,k;
  for ( i = 0; i < Nb[0]; i++) {
    for ( j = 0; j < Nb[1]; j++) {
      for ( k = 0; k < Nb[2]; k++) {
        X = 2 * pi / Nx[0] * (i + is[0]);
        Y = 2 * pi / Nx[1] * (j + is[1]);
        Z = 2 * pi / Nx[2] * (k + is[2]);
        ptr = F3DI(i,j,k,Nb[0],Nb[1],Nb[2]);
#ifdef OPENFFT
        a[ptr].r = testcase(X, Y, Z); // Real Component
        a[ptr].i = testcase(X, Y, Z); // Imag Component
#else
        a[ptr][0] = testcase(X, Y, Z); // Real Component
        a[ptr][1] = testcase(X, Y, Z); // Imag Component
#endif
      }
    }
  }

  return;
} // end initialize

void check_err(Complex*a, fft_int* Nx, fft_int* Nb, fft_int* is){
  long long int size = Nx[0]*Nx[1]*Nx[2];
  double pi = 4 * atan(1.0);
  double err = 0, norm = 0;
  double X, Y, Z, numerical_r, numerical_c;
  long int ptr;
  int i,j,k;
  for ( i = 0; i < Nb[0]; i++) {
    for ( j = 0; j < Nb[1]; j++) {
      for ( k = 0; k < Nb[2]; k++) {
        X = 2 * pi / Nx[0] * (i + is[0]);
        Y = 2 * pi / Nx[1] * (j + is[1]);
        Z = 2 * pi / Nx[2] * (k + is[2]);
        ptr = F3DI(i,j,k,Nb[0],Nb[1],Nb[2]);
#ifdef OPENFFT
        numerical_r = a[ptr].r / size;
#else
        numerical_r = a[ptr][0] / size;
#endif
	if (numerical_r != numerical_r) numerical_r = 0;
#ifdef OPENFFT
        numerical_r = a[ptr].r / size;
#else
        numerical_c = a[ptr][1] / size;
#endif
	if (numerical_c != numerical_c) numerical_r = 0;
        err += fabs(numerical_r - testcase(X, Y, Z))
             + fabs(numerical_c - testcase(X, Y, Z));
        norm += fabs(testcase(X, Y, Z));

      }
    }
  }

  double g_err = 0, g_norm = 0;
  MPI_Reduce(&err, &g_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&norm, &g_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (g_err / g_norm > 1e-10){
    if(procid == 0){
      printf("\nL1 Error of iFF(a)-a: %g\n", g_err);
      printf("Relative L1 Error of iFF(a)-a: %g\n", g_err / g_norm);
      printf("\nResults are NOT CORRECT!\n\n");
    }
  }

  return;
} // end check_err

double testcase(double X, double Y, double Z){

  double sigma = 4;
  double pi = atan(1.0)*4;
  double analytic;
  analytic = exp(-sigma * ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
                         + (Z - pi) * (Z - pi)));
  return analytic;
} // end testcase


