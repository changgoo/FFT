#include <complex.h>
#include <pfft.h>

int main(int argc, char **argv)
{
  int np[3],procid;
  ptrdiff_t n[3];
  ptrdiff_t alloc_local;
  ptrdiff_t local_ni[3], local_i_start[3];
  ptrdiff_t local_no[3], local_o_start[3];
  double err;
  pfft_complex *in, *out;
  pfft_plan plan_forw=NULL, plan_back=NULL;
  MPI_Comm comm_cart_3d;
  
  double f_time = 0 * MPI_Wtime(), i_time = 0, setup_time = 0;

  /* Set size of FFT and process mesh */
  n[0] = 128; n[1] = 128; n[2] = 128;
  np[0] = 4; np[1] = 2; np[2] = 1;
 
  /* Initialize MPI and PFFT */
  MPI_Init(&argc, &argv);
  pfft_init();

  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  /* Create three-dimensional process grid of size np[0] x np[1] x np[2], if possible */
  if(np[2] > 1) {
    pfft_create_procmesh(3, MPI_COMM_WORLD, np, &comm_cart_3d);
  } else {
    pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_3d);
  }

  
  /* Get parameters of data distribution */
  alloc_local = pfft_local_size_dft_3d(n, comm_cart_3d, PFFT_TRANSPOSED_NONE,
      local_ni, local_i_start, local_no, local_o_start);

  printf("[mpi rank %d] isize  %3d %3d %3d osize  %3d %3d %3d\n", procid,
		local_ni[0],local_ni[1],local_ni[2],
		local_no[0],local_no[1],local_no[2]);

  printf("[mpi rank %d] istart %3d %3d %3d ostart %3d %3d %3d\n", procid,
		local_i_start[0],local_i_start[1],local_i_start[2],
		local_o_start[0],local_o_start[1],local_o_start[2]);



  /* Allocate memory */
  in  = pfft_alloc_complex(alloc_local);
  out = in;

  setup_time = -MPI_Wtime();
  /* Plan parallel forward FFT */
  plan_forw = pfft_plan_dft_3d(
      n, in, out, comm_cart_3d, PFFT_FORWARD, PFFT_TRANSPOSED_NONE| PFFT_MEASURE| PFFT_DESTROY_INPUT);
  
  /* Plan parallel backward FFT */
  plan_back = pfft_plan_dft_3d(
      n, out, in, comm_cart_3d, PFFT_BACKWARD, PFFT_TRANSPOSED_NONE| PFFT_MEASURE| PFFT_DESTROY_INPUT);
  setup_time += MPI_Wtime();

  /* Initialize input with random numbers */
  pfft_init_input_complex_3d(n, local_ni, local_i_start,
      in);
  
  /* execute parallel forward FFT */
  f_time -= MPI_Wtime();
  pfft_execute(plan_forw);
  f_time += MPI_Wtime();

  /* clear the old input */
//  pfft_clear_input_complex_3d(n, local_ni, local_i_start, in);
  
  /* execute parallel backward FFT */
  i_time-=MPI_Wtime();
  pfft_execute(plan_back);
  i_time+=MPI_Wtime();
  
  /* Scale data */
  for(ptrdiff_t l=0; l < local_ni[0] * local_ni[1] * local_ni[2]; l++)
    in[l] /= (n[0]*n[1]*n[2]);

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

  /* Print error of back transformed data */
  err = pfft_check_output_complex_3d(n, local_ni, local_i_start, in, comm_cart_3d);
  pfft_printf(comm_cart_3d, "Error after one forward and backward trafo of size n=(%td, %td, %td):\n", n[0], n[1], n[2]); 
  pfft_printf(comm_cart_3d, "maxerror = %6.2e;\n", err);
  
  /* free mem and finalize */
  pfft_destroy_plan(plan_forw);
  pfft_destroy_plan(plan_back);
  MPI_Comm_free(&comm_cart_3d);
  pfft_free(in);
//  pfft_free(out);
  MPI_Finalize();
  return 0;
}
