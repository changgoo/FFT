#include <stdio.h>
#include <stdlib.h>
#include <fftw3-mpi.h>

#define F3DI(i, j, k, nx1, nx2, nx3) ((k) + (nx3)*((j) + (nx2)*(i)))
#define F2DI(i, j, nx1, nx2) ((j) + (nx2)*(i))

int main(int argc, char **argv){
  const ptrdiff_t N1 = 128, N2 = 128, N3 = 128;
  fftw_plan plan;
  fftw_complex *data; //local data of course
  ptrdiff_t alloc_local, n1, is, i, j, k;
  int irank;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&irank);
  fftw_mpi_init();

/* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_3d(N1, N2, N3, MPI_COMM_WORLD,
    &n1, &is);
  data = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * alloc_local);
  if(irank == 0) printf("%i %i %i\n",N1, N2, N3);
  printf("%i %i %i\n",irank, n1, is);
  if(irank == 0) printf("%i %i\n",N1*N2*N3,alloc_local);
/* create plan for forward DFT */
  plan = fftw_mpi_plan_dft_3d(N1, N2, N3, data, data, MPI_COMM_WORLD,
    FFTW_FORWARD, FFTW_ESTIMATE);

/* initialize data to some function my_function(x,y) */
  for (i = 0; i < n1; i++) {
    for (j = 0; j < N1; j++){
      for (k = 0; k < N2; k++){
        data[F3DI(i,j,k,N1,N2,N3)][0]=F3DI(i,j,k,N1,N2,N3);
        data[F3DI(i,j,k,N1,N2,N3)][1]=i;
  }}}

/* compute transforms, in-place, as many times as desired */
  fftw_execute(plan);

  fftw_destroy_plan(plan);
  fftw_free(data);
  MPI_Finalize();
  if(irank == 0) printf("finalize\n");
  return 0;
}
