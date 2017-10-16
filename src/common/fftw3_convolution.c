
#ifdef HAVE_FFTW3

#include <stdlib.h>
#include "develop/imageop.h"

#include "fftw3_convolution.h"


// from:
// https://github.com/jeremyfix/FFTConvolution
//

  // Code adapted from gsl/fft/factorize.c
  void factorize (const int n,
                  int *n_factors,
                  int factors[],
                  int * implemented_factors)
  {
      int nf = 0;
      int ntest = n;
      int factor;
      int i = 0;

      if (n == 0)
      {
          printf("Length n must be positive integer\n");
          return ;
      }

      if (n == 1)
      {
          factors[0] = 1;
          *n_factors = 1;
          return ;
      }

      /* deal with the implemented factors */

      while (implemented_factors[i] && ntest != 1)
      {
          factor = implemented_factors[i];
          while ((ntest % factor) == 0)
          {
              ntest = ntest / factor;
              factors[nf] = factor;
              nf++;
          }
          i++;
      }

      // Ok that's it
      if(ntest != 1)
      {
          factors[nf] = ntest;
          nf++;
      }

      /* check that the factorization is correct */
      {
          int product = 1;

          for (i = 0; i < nf; i++)
          {
              product *= factors[i];
          }

          if (product != n)
          {
              printf("factorization failed");
          }
      }

      *n_factors = nf;
  }



  int is_optimal(int n, int * implemented_factors)
  {
      // We check that n is not a multiple of 4*4*4*2
      if(n % 4*4*4*2 == 0)
          return 0;

      int nf=0;
      int factors[64];
      int i = 0;
      factorize(n, &nf, factors,implemented_factors);

      // We just have to check if the last factor belongs to GSL_FACTORS
      while(implemented_factors[i])
      {
          if(factors[nf-1] == implemented_factors[i])
              return 1;
          ++i;
      }
      return 0;
  }

  int find_closest_factor(int n, int * implemented_factor)
  {
      int j;
      if(is_optimal(n,implemented_factor))
          return n;
      else
      {
          j = n+1;
          while(!is_optimal(j,implemented_factor))
              ++j;
          return j;
      }
  }
  

void fftw3_convolve(const float *const src, const int width_src, const int height_src, const int ch, const int channel, 
		const float *const kernel, const int width_kernel, const int height_kernel, 
		float *dest, const int width_dest, const int height_dest, 
		const int mode, double *time)
{
	double start = 0;

#ifdef HAVE_FFTW3_OMPx
#ifdef _OPENMP
	start = dt_get_wtime();
	dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
  *time += dt_get_wtime() - start;
  
	if (fftwf_init_threads())
		fftwf_plan_with_nthreads(omp_get_max_threads());
	else
	printf("fftw3_convolve: fftwf_init_threads()\n");
	
	start = dt_get_wtime();
	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
  *time += dt_get_wtime() - start;
#endif
#endif
	
  int FFTW_FACTORS[7] = {13,11,7,5,3,2,0}; // end with zero to detect the end of the array

	float *in_src = NULL;
	fftwf_complex *out_src = NULL;
	fftwf_plan plan_src = NULL;
	
	float *in_kernel = NULL;
	fftwf_complex *out_kernel = NULL;
	fftwf_plan plan_kernel = NULL;
	
	fftwf_complex *in_inv = NULL;
	float *out_inv = NULL;
	fftwf_plan plan_inv = NULL;
	
	int width_fft =  0;
	int height_fft =  0;
	
	if (mode == fftw3_convolve_full)
	{
		height_fft = height_src + height_kernel - 1;
    width_fft = width_src + width_kernel - 1;
    if (height_dest != (height_src + height_kernel-1) || width_dest != (width_src + width_kernel-1))
    {
    	printf("fftw3_convolve: invalid dest size\n");
  		goto cleanup;
    }
	}
	else if (mode == fftw3_convolve_valid)
	{
		height_fft = height_src;
		width_fft = width_src;
		if (height_dest != (height_src - height_kernel+1) || width_dest != (width_src - width_kernel+1))
    {
    	printf("fftw3_convolve: invalid dest size\n");
  		goto cleanup;
    }
	}
	else
	{
		printf("fftw3_convolve: unknown mode=%i\n", mode);
		goto cleanup;
	}
	
	height_fft = find_closest_factor(height_fft, FFTW_FACTORS);
	width_fft = find_closest_factor(width_fft, FFTW_FACTORS);

	float scale = 1.0 / (width_fft * height_fft);
	
	start = dt_get_wtime();
	dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
  *time += dt_get_wtime() - start;

	in_src = (float*)fftwf_malloc(sizeof(float) * width_fft * height_fft);
	out_src = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * width_fft * height_fft);
	
	in_kernel = (float*)fftwf_malloc(sizeof(float) * width_fft * height_fft);
	out_kernel = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * width_fft * height_fft);
	
	out_inv = (float*)fftwf_malloc(sizeof(float) * width_fft * height_fft);
	in_inv = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * width_fft * height_fft);
	
	start = dt_get_wtime();
	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
  *time += dt_get_wtime() - start;

	memset(in_src, 0, sizeof(float) * width_fft * height_fft);
	memset(out_src, 0, sizeof(fftwf_complex) * width_fft * height_fft);
	
	memset(in_kernel, 0, sizeof(float) * width_fft * height_fft);
	memset(out_kernel, 0, sizeof(fftwf_complex) * width_fft * height_fft);
	
	memset(out_inv, 0, sizeof(float) * width_fft * height_fft);
	memset(in_inv, 0, sizeof(fftwf_complex) * width_fft * height_fft);
	
	
	for (int y = 0; y < height_src; y++)
	{
		for (int x = 0; x < width_src; x++)
		{
			in_src[y * width_fft + x] = src[y * width_src * ch + x * ch + channel];
		}
	}

	
	for (int y = 0; y < height_kernel; y++)
	{
		for (int x = 0; x < width_kernel; x++)
		{
			in_kernel[y * width_fft + x] = kernel[y * width_kernel * ch + x * ch + channel];
		}
	}
	
	start = dt_get_wtime();
	dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
  *time += dt_get_wtime() - start;
  
	plan_src = fftwf_plan_dft_r2c_2d(height_fft, width_fft, in_src, out_src, FFTW_ESTIMATE);
	plan_kernel = fftwf_plan_dft_r2c_2d(height_fft, width_fft, in_kernel, out_kernel, FFTW_ESTIMATE);

	plan_inv = fftwf_plan_dft_c2r_2d(height_fft, width_fft, in_inv, out_inv, FFTW_ESTIMATE);
	
  start = dt_get_wtime();
	fftwf_execute(plan_src);
	fftwf_execute(plan_kernel);
	time[1] += dt_get_wtime() - start;

	start = dt_get_wtime();
	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
  *time += dt_get_wtime() - start;
	
  for (int i = 0; i < height_fft; ++i)
  {
       for (int j = 0; j < width_fft/2+1; ++j)
       {
         int ij = i*(width_fft/2+1) + j;
         int ij_in = i*(width_fft/2+1) + j;
            in_inv[ij_in][0] = (out_src[ij][0] * out_kernel[ij][0]
                          - out_src[ij][1] * out_kernel[ij][1]) * scale;
            in_inv[ij_in][1] = (out_src[ij][0] * out_kernel[ij][1]
                          + out_src[ij][1] * out_kernel[ij][0]) * scale;
       }
  }
	
  start = dt_get_wtime();
  dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
	*time += dt_get_wtime() - start;
  
  
  start = dt_get_wtime();
	fftwf_execute(plan_inv);
	time[1] += dt_get_wtime() - start;

	
  start = dt_get_wtime();
	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
	*time += dt_get_wtime() - start;
	
	
	if (mode == fftw3_convolve_full)
	{
		// return [0:height_dest-1; 0:width_dest-1]
		for (int y = 0; y < height_dest; y++)
		{
			for (int x = 0; x < width_dest; x++)
			{
				dest[y * width_dest * ch + x * ch + channel] = out_inv[y * width_fft + x];
			}
		}
	}
	else if (mode == fftw3_convolve_valid)
	{
		// return [height_dest; width_dest] starting at [height_kernel-1; width_kernel-1]
		for (int y = 0; y < height_dest; y++)
		{
			for (int x = 0; x < width_dest; x++)
			{
				dest[y * width_dest * ch + x * ch + channel] = out_inv[(y+height_kernel-1) * width_fft + (x+width_kernel-1)];
			}
		}
	}
	
cleanup:
	
	start = dt_get_wtime();
	dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
	*time += dt_get_wtime() - start;

	if (plan_src) fftwf_destroy_plan(plan_src);
	if (plan_kernel) fftwf_destroy_plan(plan_kernel);
	if (plan_inv) fftwf_destroy_plan(plan_inv);
	
	if (in_src) fftwf_free(in_src);
	if (out_src) fftwf_free(out_src);
	if (in_kernel) fftwf_free(in_kernel);
	if (out_kernel) fftwf_free(out_kernel);
	if (in_inv) fftwf_free(in_inv);
	if (out_inv) fftwf_free(out_inv);
	
	start = dt_get_wtime();
	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
	*time += dt_get_wtime() - start;

#ifdef HAVE_FFTW3_OMPx
#ifdef _OPENMP
	start = dt_get_wtime();
	dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
  *time += dt_get_wtime() - start;
  
  fftwf_cleanup_threads();
	
	start = dt_get_wtime();
	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
  *time += dt_get_wtime() - start;
#endif
#endif

}



#endif
