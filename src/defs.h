
#define FFT_PLIMPTON

#if defined (FFT_PLIMPTON)
#include <fft_3d.h>
#define Complex fftw_complex
#define fft_int int
static struct fft_plan_3d *plan=NULL;
#elif defined (PFFT)
#include <pfft.h>
#define Complex pfft_complex
#define fft_int ptrdiff_t
static pfft_plan fplan=NULL, bplan=NULL;
MPI_Comm comm_cart_3d;
#elif defined (FFTW)
#include <fftw3-mpi.h>
#define Complex fftw_complex
#define fft_int ptrdiff_t
static fftw_plan fplan=NULL, bplan=NULL;
#elif defined (ACCFFT)
#include <accfft.h>
#define fft_int int
MPI_Comm c_comm;
static accfft_plan *plan=NULL;
#elif defined (OPENFFT)
#include <openfft.h>
#define Complex dcomplex
#define fft_int int
#endif

#define F3DI(i, j, k, nx1, nx2, nx3) ((k) + (nx3)*((j) + (nx2)*(i)))
#define F2DI(i, j, nx1, nx2) ((j) + (nx2)*(i))

