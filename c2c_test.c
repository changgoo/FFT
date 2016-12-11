#include <stdio.h>
#include <stdlib.h>
#include <math.h> // M_PI
#include <mpi.h>

//#define FFT_PLIMPTON
#define PFFT

#if defined (FFT_PLIMPTON)
#include <fft_3d.h>
#define Complex fftw_complex
#define fft_int int
#elif defined (PFFT)
#include <pfft.h>
#define Complex pfft_complex
#define fft_int ptrdiff_t
#endif

#define F3DI(i, j, k, nx1, nx2, nx3) ((k) + (nx3)*((j) + (nx2)*(i)))
#define F2DI(i, j, nx1, nx2) ((j) + (nx2)*(i))


static int nprocs, procid;
static int np[3];
static struct fft_plan_3d *plan;
void check_err(Complex*a, int* Nx, int* Nb, int* is);
void initialize(Complex*a, int* Nx, int* Nb, int* is);
void decompose(int* Nx, int* Nb, int* is);
void fft_init(Complex *data, int* Nx, int* Nb, int* is, double *setup_time, int nthreads);
void do_fft(Complex *data, int *Nx, int *Nb, int *is, double *fft_time, int nthreads);

inline double testcase(double X, double Y, double Z) {

  double sigma = 4;
  double pi = atan(1.0)*4;
  double analytic;
  analytic = exp(-sigma * ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
                         + (Z - pi) * (Z - pi)));
  return analytic;
} // end testcase

int main(int argc, char **argv) {

  int NX, NY, NZ;
  int nx, ny, nz;
  double setup_time, g_setup_time, fft_time[2];
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
  int Nx[3] = { NX, NY, NZ };
  int Nb[3] = { nx, ny, nz };
  int is[3];
  int nthreads = 1;
  int Ntry=10,itry;

  decompose(Nx, Nb, is);

  /* Allocate memory */
  int alloc_local = Nb[0]*Nb[1]*Nb[2];
  data = (Complex *) fftw_malloc(sizeof(Complex) * alloc_local);

  fft_init(data, Nx, Nb, is, &setup_time, nthreads);


  fft_time[0]=0; fft_time[1]=0;
  for(itry=0;itry<Ntry;itry++) do_fft(data, Nx, Nb, is, fft_time, nthreads);

  MPI_Reduce(&setup_time, &g_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0,
			MPI_COMM_WORLD);
  double g_f_time, g_i_time;
  MPI_Reduce(&fft_time[0], &g_f_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&fft_time[1], &g_i_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  /* Compute some timings statistics */
  if(procid == 0){
    printf("Timing for Inplace FFT of size %i %i %i with configuration %i %i %i\n",Nx[0],Nx[1],Nx[2],np[0],np[1],np[2]);
    printf("Setup\t %g\n",g_setup_time);
    printf("FFT \t %g\n",g_f_time/Ntry);
    printf("IFFT \t %g\n",g_i_time/Ntry);
  }

  /* free mem and finalize */
  fftw_free(data);
  fft_3d_destroy_plan(plan);


  MPI_Finalize();
  return 0;
} // end main

void decompose(int* Nx, int* Nb, int* is){
  int ip,jp,kp;
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

  printf("[mpi rank %d] block size  %3d %3d %3d\n", procid,
		Nb[0],Nb[1],Nb[2]);

  printf("[mpi rank %d] istart      %3d %3d %3d\n", procid,
		is[0],is[1],is[2]);

  return;
}

void fft_init(Complex *data, int* Nx, int* Nb, int* is, double *setup_time, int nthreads){
  int scaled=0,permute=0,nbuf;
  int ie[3];
  ie[0] = is[0] + Nb[0] - 1;
  ie[1] = is[1] + Nb[1] - 1;
  ie[2] = is[2] + Nb[2] - 1;

  *setup_time = 0;
  *setup_time = -MPI_Wtime();
  plan = fft_3d_create_plan(MPI_COMM_WORLD, Nx[2], Nx[1], Nx[0],
                                   is[2], ie[2], is[1], ie[1], is[0], ie[0], 
	                           is[2], ie[2], is[1], ie[1], is[0], ie[0], 
                                   scaled, permute, &nbuf);
  *setup_time += MPI_Wtime();

  return;
}

void do_fft(Complex *data, int *Nx, int *Nb, int *is, double *fft_time, int nthreads){
  double f_time = 0, i_time = 0;
  
  /* Initialize input with random numbers */
  initialize(data, Nx, Nb, is);

  /* execute parallel forward FFT */
  f_time -= MPI_Wtime();
  fft_3d(data, data, FFTW_FORWARD, plan);
  f_time += MPI_Wtime();
//  MPI_Barrier(MPI_COMM_WORLD);

  /* Perform backward FFT */
  i_time-=MPI_Wtime();
  fft_3d(data, data, FFTW_BACKWARD, plan);
  i_time+=MPI_Wtime();
//  MPI_Barrier(MPI_COMM_WORLD);

  check_err(data, Nx, Nb, is);

  fft_time[0]+=f_time;
  fft_time[1]+=i_time;
  return;
}

void initialize(Complex*a, int* Nx, int* Nb, int* is){
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
        a[ptr][0] = testcase(X, Y, Z); // Real Component
        a[ptr][1] = testcase(X, Y, Z); // Imag Component
      }
    }
  }

  return;
} // end initialize

void check_err(Complex*a, int* Nx, int* Nb, int* is){
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
        numerical_r = a[ptr][0] / size;
	if (numerical_r != numerical_r) numerical_r = 0;
        numerical_c = a[ptr][1] / size;
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
