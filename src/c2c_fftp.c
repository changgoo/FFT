#include <math.h>
#include <mpi.h>
#include <fft_3d.h>

#define M_PI 3.14159265358979323846
#define F3DI(i, j, k, nx1, nx2, nx3) ((k) + (nx3)*((j) + (nx2)*(i)))

static int is,js,ks;
void check_err(fftw_complex* a, int*n, int* local_n, int* local_is, MPI_Comm c_comm);
void initialize(fftw_complex *a, int* n, int* local_n, int* local_is,  MPI_Comm c_comm);

inline double testcase(double X, double Y, double Z) {

	double sigma = 4;
	double pi = M_PI;
	double analytic;
	analytic = exp(-sigma * ((X - pi) * (X - pi) + (Y - pi) * (Y - pi)
							+ (Z - pi) * (Z - pi)));
	if (analytic != analytic)
		analytic = 0; /* Do you think the condition will be false always? */
	return analytic;
} // end testcase


int main(int argc, char **argv)
{
  int np[3],procid,nprocs;
  int scaled=0,permute=0,nbuf;
  struct fft_plan_3d *plan;
  int n[3];
  int alloc_local;
  int local_ni[3], local_i_start[3];
  int local_no[3], local_o_start[3];
  int kp,jp,ip;
  double err;
  fftw_complex *in, *out; 
  
  double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;

  /* Set size of FFT and process mesh */
  n[0] = 128; n[1] = 128; n[2] = 128;
  np[0] = 2; np[1] = 2; np[2] = 2;
 
  /* Initialize MPI and PFFT */
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* Get parameters of data distribution */
  local_ni[0] = n[0]/np[0];
  local_ni[1] = n[1]/np[1];
  local_ni[2] = n[2]/np[2];
  ip = procid/(np[2]*np[1]);
  jp = (procid-np[2]*np[1]*ip)/np[2];
  kp = procid-np[2]*np[1]*ip-np[2]*jp;
  local_i_start[2] = kp*local_ni[2];
  local_i_start[1] = jp*local_ni[1];
  local_i_start[0] = ip*local_ni[0];
  alloc_local = local_ni[0]*local_ni[1]*local_ni[2];

  printf("[mpi rank %d] isize  %3d %3d %3d\n", procid,
		local_ni[0],local_ni[1],local_ni[2]);

  printf("[mpi rank %d] istart %3d %3d %3d\n", procid,
		local_i_start[0],local_i_start[1],local_i_start[2]);

  is=local_i_start[0];
  js=local_i_start[1];
  ks=local_i_start[2];
  int	ie=local_i_start[0]+local_ni[0]-1;
  int	je=local_i_start[1]+local_ni[1]-1;
  int	ke=local_i_start[2]+local_ni[2]-1;
  /* Allocate memory */
  in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local);
  out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local);

  setup_time = -MPI_Wtime();
  plan = fft_3d_create_plan(MPI_COMM_WORLD, n[2], n[1], n[0],
                                   ks, ke, js, je, is, ie, 
	                           ks, ke, js, je, is, ie, 
                                   scaled, permute, &nbuf);
  setup_time += MPI_Wtime();
  printf("%d\n",alloc_local);

  /* Initialize input with random numbers */
  initialize(in, n, local_ni, local_i_start, MPI_COMM_WORLD);
//  check_err(in, n, local_ni, local_i_start, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  
  /* execute parallel forward FFT */
  f_time -= MPI_Wtime();
  fft_3d(in, in, FFTW_FORWARD, plan);
  f_time += MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);

  /* Perform backward FFT */
  i_time-=MPI_Wtime();
  fft_3d(in, in, FFTW_BACKWARD, plan);
  i_time+=MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);

  check_err(in,n,local_ni,local_i_start,MPI_COMM_WORLD);

  /* Compute some timings statistics */
  double g_f_time, g_i_time, g_setup_time;
  MPI_Reduce(&f_time, &g_f_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&i_time, &g_i_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&setup_time, &g_setup_time, 1, MPI_DOUBLE, MPI_MAX, 0,
			MPI_COMM_WORLD);

  if(procid == 0){
    printf("Timing for Inplace FFT of size %i %i %i with configuration %i %i %i\n",n[0],n[1],n[2],np[0],np[1],np[2]);
    printf("Setup\t %g\n",g_setup_time);
    printf("FFT \t %g\n",g_f_time);
    printf("IFFT \t %g\n",g_i_time);
  }

  /* free mem and finalize */
  fftw_free(in);
  fftw_free(out);
  fft_3d_destroy_plan(plan);
  MPI_Finalize();
  return 0;
}

void initialize(fftw_complex* a, int* n, int* local_n, int* local_is, MPI_Comm c_comm) {
	double pi = M_PI;
	double X, Y, Z;
	long int ptr;
        int i,j,k;
	for (i = 0; i < local_n[0]; i++) {
		for (j = 0; j < local_n[1]; j++) {
			for (k = 0; k < local_n[2]; k++) {
				X = 2 * pi / n[0] * (i + local_is[0]);
				Y = 2 * pi / n[1] * (j + local_is[1]);
				Z = 2 * pi / n[2] * (k + local_is[2]);
				ptr = F3DI(i,j,k,local_n[0],local_n[1],local_n[2]);
				a[ptr][0] = testcase(X, Y, Z); // Real Component
				a[ptr][1] = testcase(X, Y, Z); // Imag Component
			}
		}
	}
	return;
} // end initialize


void check_err(fftw_complex* a, int* n, int* local_n, int* local_is, MPI_Comm c_comm) {
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);
	long long int size = n[0];
	size *= n[1];
	size *= n[2];
	double pi = M_PI;

	double err = 0, norm = 0;

	double X, Y, Z, numerical_r, numerical_c;
	long int ptr;
        int i,j,k;
	for (i = 0; i < local_n[0]; i++) {
		for (j = 0; j <local_n[1]; j++) {
			for (k = 0; k < local_n[2]; k++) {
				X = 2 * pi / n[0] * (i + local_is[0]);
				Y = 2 * pi / n[1] * (j + local_is[1]);
				Z = 2 * pi / n[2] * (k + local_is[2]);
				ptr = F3DI(i,j,k,local_n[0],local_n[1],local_n[2]);
				numerical_r = a[ptr][0] / size;
				if (numerical_r != numerical_r)
					numerical_r = 0;
				numerical_c = a[ptr][1] / size;
				if (numerical_c != numerical_c)
					numerical_c = 0;
				err += fabs(numerical_r - testcase(X, Y, Z))
						+ fabs(numerical_c - testcase(X, Y, Z));
				norm += fabs(testcase(X, Y, Z));

			}
		}
	}

	double g_err = 0, g_norm = 0;
	MPI_Reduce(&err, &g_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&norm, &g_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(procid == 0){
        printf("\nL1 Error of iFF(a)-a: %g\n", g_err);
	printf("Relative L1 Error of iFF(a)-a: %g\n", g_err / g_norm);
	if (g_err / g_norm < 1e-10)
		printf("\nResults are CORRECT!\n\n");
	else
		printf("\nResults are NOT CORRECT!\n\n");
        }

	return;
} // end check_err
