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
static struct fft_plan_3d *plan=NULL;
#elif defined (PFFT)
#include <pfft.h>
#define Complex pfft_complex
#define fft_int ptrdiff_t
pfft_plan fplan=NULL, bplan=NULL;
MPI_Comm comm_cart_3d;
#endif

#define F3DI(i, j, k, nx1, nx2, nx3) ((k) + (nx3)*((j) + (nx2)*(i)))
#define F2DI(i, j, nx1, nx2) ((j) + (nx2)*(i))


static int nprocs, procid;
static int np[3];
void check_err(Complex*a, fft_int* Nx, fft_int* Nb, fft_int* is);
void initialize(Complex*a, fft_int* Nx, fft_int* Nb, fft_int* is);
void decompose(fft_int* Nx, fft_int* Nb, fft_int* is);
Complex *data_alloc(fft_int alloc_local);
void fft_init(Complex *data, fft_int* Nx, fft_int* Nb, fft_int* is, double *fft_time, int nthreads);
void do_fft(Complex *data, fft_int *Nx, fft_int *Nb, fft_int *is, double *fft_time, int nthreads);
void fft_destroy(Complex *data);
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

  decompose(Nx, Nb, is);

  /* Allocate memory */
  fft_int alloc_local = Nb[0]*Nb[1]*Nb[2];
  data=data_alloc(alloc_local);
  fft_init(data, Nx, Nb, is, fft_time, nthreads);

  for(itry=0;itry<Ntry;itry++) do_fft(data, Nx, Nb, is, fft_time, nthreads);

  timing(Nx,fft_time, Ntry);

  fft_destroy(data);

  MPI_Finalize();
  return 0;
} // end main

void fft_destroy(Complex *data){
  /* free mem and finalize */
#if defined (PFFT)
  pfft_destroy_plan(fplan);
  pfft_destroy_plan(bplan);
  MPI_Comm_free(&comm_cart_3d);
  pfft_free(data);
#elif defined (FFT_PLIMPTON)
  fftw_free(data);
  fft_3d_destroy_plan(plan);
#endif
  return;
}


void timing(fft_int*Nx, double *fft_time, int Ntry){
  double g_fft_time[3];
  MPI_Reduce(&fft_time[0], &g_fft_time[0], 3, MPI_DOUBLE, MPI_MAX, 0,
			MPI_COMM_WORLD);
  /* Compute some timings statistics */
  if(procid == 0){
    printf("Timing for Inplace FFT of size %i %i %i\n",Nx[0],Nx[1],Nx[2]);
    printf("with MPI configuration %i %i %i\n",np[0],np[1],np[2]);
    printf("Setup\t %g\n",g_fft_time[0]);
    printf("FFT \t %g\n",g_fft_time[1]/Ntry);
    printf("IFFT \t %g\n",g_fft_time[2]/Ntry);
  }
  return;
}

void decompose(fft_int* Nx, fft_int* Nb, fft_int* is){
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

#ifdef PFFT
  pfft_init();
  if(np[2] > 1) {
    pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d);
  } else {
    pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_3d);
  }

  /* Get parameters of data distribution */
  fft_int alloc_local,Nbo[3],io[3];
  alloc_local = pfft_local_size_dft_3d(Nx, comm_cart_3d, PFFT_TRANSPOSED_NONE,
      Nb, is, Nbo, io);
#else
  /* Get parameters of data distribution */
  ip = procid/(np[2]*np[1]);
  jp = (procid-np[2]*np[1]*ip)/np[2];
  kp = procid-np[2]*np[1]*ip-np[2]*jp;
  is[2] = kp*Nb[2];
  is[1] = jp*Nb[1];
  is[0] = ip*Nb[0];
#endif

  printf("[mpi rank %d] block size  %3d %3d %3d\n", procid,
		Nb[0],Nb[1],Nb[2]);

  printf("[mpi rank %d] istart      %3d %3d %3d\n", procid,
		is[0],is[1],is[2]);


  return;
}

Complex *data_alloc(fft_int alloc_local){

#ifdef PFFT
  return pfft_alloc_complex(alloc_local);
#else
  return (Complex *) fftw_malloc(sizeof(Complex) * alloc_local);
#endif
}

void fft_init(Complex *data, fft_int* Nx, fft_int* Nb, fft_int* is, double *fft_time, int nthreads){
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
#elif defined (PFFT)
  fplan = pfft_plan_dft_3d(Nx, data, data, comm_cart_3d,
    PFFT_FORWARD, PFFT_TRANSPOSED_NONE| PFFT_MEASURE| PFFT_DESTROY_INPUT);
  bplan = pfft_plan_dft_3d(Nx, data, data, comm_cart_3d,
    PFFT_BACKWARD, PFFT_TRANSPOSED_NONE| PFFT_MEASURE| PFFT_DESTROY_INPUT);
#endif
  fft_time[0] += MPI_Wtime();

  return;
}

void do_fft(Complex *data, fft_int *Nx, fft_int *Nb, fft_int *is, double *fft_time, int nthreads){
  double f_time = 0, i_time = 0;
  
  /* Initialize input with random numbers */
  initialize(data, Nx, Nb, is);

  /* execute parallel forward FFT */
  f_time -= MPI_Wtime();
#if defined (PFFT)
  pfft_execute(fplan);
#elif defined (FFT_PLIMPTON)
  fft_3d(data, data, FFTW_FORWARD, plan);
#endif
  f_time += MPI_Wtime();
//  MPI_Barrier(MPI_COMM_WORLD);

  /* Perform backward FFT */
  i_time-=MPI_Wtime();
#if defined (PFFT)
  pfft_execute(bplan);
#elif defined (FFT_PLIMPTON)
  fft_3d(data, data, FFTW_BACKWARD, plan);
#endif
  i_time+=MPI_Wtime();
//  MPI_Barrier(MPI_COMM_WORLD);

  check_err(data, Nx, Nb, is);

  fft_time[1]+=f_time;
  fft_time[2]+=i_time;
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
        a[ptr][0] = testcase(X, Y, Z); // Real Component
        a[ptr][1] = testcase(X, Y, Z); // Imag Component
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

double testcase(double X, double Y, double Z){

  double sigma = 4;
  double pi = atan(1.0)*4;
  double analytic;
  analytic = exp(-sigma * ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
                         + (Z - pi) * (Z - pi)));
  return analytic;
} // end testcase


