
#include <stdlib.h>
#include "develop/imageop.h"
#include "convolution.h"

#ifdef HAVE_FFTW3

//-----------------------------------------------------------------------

#define TIME_FFTW3_LOCK 0
#define TIME_FFTW3_EXEC 1
#define TIME_FFTW3_COPY 2
#define TIME_FFTW3_MULT 3
#define TIME_FFTW3_PLAN 4

//#define TIME_FFTW3

#ifdef TIME_FFTW3

#define FFTW3_TIME_DECL double start = 0;

#define FFTW3_TIME_BEGIN start = dt_get_wtime();

#define FFTW3_TIME_END(time_index) time[time_index] += dt_get_wtime() - start;

#else

#define FFTW3_TIME_DECL ;

#define FFTW3_TIME_BEGIN ;

#define FFTW3_TIME_END(time_index) ;

#endif

//-----------------------------------------------------------------------

// from:
// https://github.com/jeremyfix/FFTConvolution
//

// Code adapted from gsl/fft/factorize.c
static void fftw3_factorize (const int n, int *n_factors, int factors[], int * implemented_factors)
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

static int fftw3_is_optimal(int n, int * implemented_factors)
{
	// We check that n is not a multiple of 4*4*4*2
	if(n % 4*4*4*2 == 0)
		return 0;

	int nf = 0;
	int factors[64];
	int i = 0;
	
	fftw3_factorize(n, &nf, factors,implemented_factors);

	// We just have to check if the last factor belongs to GSL_FACTORS
	while(implemented_factors[i])
	{
		if(factors[nf-1] == implemented_factors[i])
			return 1;
		++i;
	}
	
	return 0;
}

static int fftw3_find_closest_factor(int n, int * implemented_factor)
{
	int j;
	if (fftw3_is_optimal(n,implemented_factor))
		return n;
	else
	{
		j = n+1;
		while (!fftw3_is_optimal(j,implemented_factor))
			++j;
		return j;
	}
}
  
void fftw3_convolve_free(fftw3_convolve_t *fftw_conv, double *time)
{
  FFTW3_TIME_DECL

	if (fftw_conv)
	{
		FFTW3_TIME_BEGIN
		dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
		FFTW3_TIME_END(TIME_FFTW3_LOCK)

		FFTW3_TIME_BEGIN
		if (fftw_conv->plan_src) fftwf_destroy_plan(fftw_conv->plan_src);
		if (fftw_conv->plan_kernel) fftwf_destroy_plan(fftw_conv->plan_kernel);
		if (fftw_conv->plan_inv) fftwf_destroy_plan(fftw_conv->plan_inv);
		FFTW3_TIME_END(TIME_FFTW3_PLAN)
		
		if (fftw_conv->in_src) fftwf_free(fftw_conv->in_src);
		if (fftw_conv->out_src) fftwf_free(fftw_conv->out_src);
		if (fftw_conv->in_kernel) fftwf_free(fftw_conv->in_kernel);
		if (fftw_conv->out_kernel) fftwf_free(fftw_conv->out_kernel);

		FFTW3_TIME_BEGIN
		dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
		FFTW3_TIME_END(TIME_FFTW3_LOCK)

		free(fftw_conv);
	}
}

fftw3_convolve_t * fftw3_convolve_init(const int width_src, const int height_src, const int ch, 
		const int width_kernel, const int height_kernel, 
		const int width_dest, const int height_dest, const int mode, double *time)
{
	fftw3_convolve_t *fftw_conv = calloc(1, sizeof(fftw3_convolve_t));
	if (fftw_conv)
	{
#ifdef HAVE_FFTW3_OMP
#ifdef _OPENMP
  
		fftwf_plan_with_nthreads(dt_get_num_threads());
	
#endif
#endif
		
		FFTW3_TIME_DECL

    int FFTW_FACTORS[7] = {13,11,7,5,3,2,0}; // end with zero to detect the end of the array

    fftw_conv->width_src = width_src;
    fftw_conv->height_src = height_src;
    fftw_conv->ch = ch;
    fftw_conv->width_kernel = width_kernel;
    fftw_conv->height_kernel = height_kernel;
    fftw_conv->width_dest = width_dest;
    fftw_conv->height_dest = height_dest;
    fftw_conv->mode = mode;

  	fftw_conv->width_fft =  0;
    fftw_conv->height_fft =  0;
  	
  	if (fftw_conv->mode == convolve_mode_full)
  	{
  		fftw_conv->height_fft = fftw_conv->height_src + fftw_conv->height_kernel - 1;
  		fftw_conv->width_fft = fftw_conv->width_src + fftw_conv->width_kernel - 1;
  	}
  	else if (fftw_conv->mode == convolve_mode_valid)
  	{
  		fftw_conv->height_fft = fftw_conv->height_src;
  		fftw_conv->width_fft = fftw_conv->width_src;
  	}
  	else
  	{
  		printf("fftw3_convolve: unknown mode=%i\n", fftw_conv->mode);
  		goto cleanup;
  	}
  	
  	{
  	  int w = 0, h = 0;
  	  convolve_get_dest_size(fftw_conv->width_src, fftw_conv->height_src, 
  	      fftw_conv->width_kernel, fftw_conv->height_kernel, 
  	      &w, &h, fftw_conv->mode);
      if (fftw_conv->height_dest != h || fftw_conv->width_dest != w)
      {
        printf("fftw3_convolve: invalid dest size\n");
        goto cleanup;
      }
  	}
  	
  	fftw_conv->height_fft = fftw3_find_closest_factor(fftw_conv->height_fft, FFTW_FACTORS);
  	fftw_conv->width_fft = fftw3_find_closest_factor(fftw_conv->width_fft, FFTW_FACTORS);

  	fftw_conv->width_fft_complex = fftw_conv->width_fft/2+1;
  	fftw_conv->height_fft_complex = fftw_conv->height_fft;
  	
  	FFTW3_TIME_BEGIN
  	dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
    FFTW3_TIME_END(TIME_FFTW3_LOCK)

    fftw_conv->in_src = (float*)fftwf_malloc(sizeof(float) * fftw_conv->width_fft * fftw_conv->height_fft);
    fftw_conv->out_src = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fftw_conv->width_fft_complex * fftw_conv->height_fft_complex);
  	
    fftw_conv->in_kernel = (float*)fftwf_malloc(sizeof(float) * fftw_conv->width_fft * fftw_conv->height_fft);
    fftw_conv->out_kernel = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fftw_conv->width_fft_complex * fftw_conv->height_fft_complex);
  	
  	FFTW3_TIME_BEGIN
  	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
    FFTW3_TIME_END(TIME_FFTW3_LOCK)

  	FFTW3_TIME_BEGIN
  	dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
    FFTW3_TIME_END(TIME_FFTW3_LOCK)
    
    FFTW3_TIME_BEGIN
    fftw_conv->plan_src = fftwf_plan_dft_r2c_2d(fftw_conv->height_fft, fftw_conv->width_fft, fftw_conv->in_src, fftw_conv->out_src, FFTW_ESTIMATE);
    fftw_conv->plan_kernel = fftwf_plan_dft_r2c_2d(fftw_conv->height_fft, fftw_conv->width_fft, fftw_conv->in_kernel, fftw_conv->out_kernel, FFTW_ESTIMATE);

    fftw_conv->plan_inv = fftwf_plan_dft_c2r_2d(fftw_conv->height_fft, fftw_conv->width_fft, fftw_conv->out_src, fftw_conv->in_src, FFTW_ESTIMATE);
    FFTW3_TIME_END(TIME_FFTW3_PLAN)
    
  	FFTW3_TIME_BEGIN
  	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
    FFTW3_TIME_END(TIME_FFTW3_LOCK)

	}
	
	return fftw_conv;
	
cleanup:
	
	fftw3_convolve_free(fftw_conv, time);
	return NULL;
	
}

void fftw3_convolve(fftw3_convolve_t *fftw_conv, const float *const image_src, const int width_src, const int height_src, const int ch, 
    const float *const image_kernel, const int width_kernel, const int height_kernel, 
    float *image_dest, const int width_dest, const int height_dest, const int channel, double *time)
{
  FFTW3_TIME_DECL

  if (width_src != fftw_conv->width_src || height_src != fftw_conv->height_src || ch != fftw_conv->ch)
    printf("fftw3_convolve: image_src different from defined\n");
  if (width_kernel != fftw_conv->width_kernel || height_kernel != fftw_conv->height_kernel)
    printf("fftw3_convolve: image_kernel different from defined\n");
  if (width_dest != fftw_conv->width_dest || height_dest != fftw_conv->height_dest)
    printf("fftw3_convolve: image_dest different from defined\n");
  
  const float scale = 1.0 / (fftw_conv->width_fft * fftw_conv->height_fft);

  memset(fftw_conv->in_src, 0, sizeof(float) * fftw_conv->width_fft * fftw_conv->height_fft);
  memset(fftw_conv->out_src, 0, sizeof(fftwf_complex) * fftw_conv->width_fft_complex * fftw_conv->height_fft_complex);

  memset(fftw_conv->in_kernel, 0, sizeof(float) * fftw_conv->width_fft * fftw_conv->height_fft);
  memset(fftw_conv->out_kernel, 0, sizeof(fftwf_complex) * fftw_conv->width_fft_complex * fftw_conv->height_fft_complex);

  FFTW3_TIME_BEGIN
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(fftw_conv) schedule(static)
#endif
  for (int y = 0; y < fftw_conv->height_src; y++)
  {
    float *in_src = fftw_conv->in_src + y * fftw_conv->width_fft;
    const float *src = image_src + y * fftw_conv->width_src * fftw_conv->ch;

    for (int x = 0; x < fftw_conv->width_src; x++)
    {
      in_src[x] = src[x * fftw_conv->ch + channel];
    }
  }

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(fftw_conv) schedule(static)
#endif
  for (int y = 0; y < fftw_conv->height_kernel; y++)
  {
    float *in_kernel = fftw_conv->in_kernel + y * fftw_conv->width_fft;
    const float *kernel = image_kernel + y * fftw_conv->width_kernel * fftw_conv->ch;

    for (int x = 0; x < fftw_conv->width_kernel; x++)
    {
      in_kernel[x] = kernel[x * fftw_conv->ch + channel];
    }
  }
  FFTW3_TIME_END(TIME_FFTW3_COPY)
  
  FFTW3_TIME_BEGIN
//  dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
  FFTW3_TIME_END(TIME_FFTW3_LOCK)

  FFTW3_TIME_BEGIN
  fftwf_execute(fftw_conv->plan_src);
  fftwf_execute(fftw_conv->plan_kernel);
  FFTW3_TIME_END(TIME_FFTW3_EXEC)

  FFTW3_TIME_BEGIN
//  dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
  FFTW3_TIME_END(TIME_FFTW3_LOCK)

  FFTW3_TIME_BEGIN
  
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(fftw_conv) schedule(static)
#endif
  for (int i = 0; i < fftw_conv->height_fft_complex; ++i)
  {
    fftwf_complex *in_inv = fftw_conv->out_src + i * fftw_conv->width_fft_complex;
    fftwf_complex *out_src = fftw_conv->out_src + i * fftw_conv->width_fft_complex;
    fftwf_complex *out_kernel = fftw_conv->out_kernel + i * fftw_conv->width_fft_complex;

    for (int j = 0; j < fftw_conv->width_fft_complex; ++j)
    {
      float in_r = (out_src[j][0] * out_kernel[j][0] - out_src[j][1] * out_kernel[j][1]);// * scale;
      float in_c = (out_src[j][0] * out_kernel[j][1] + out_src[j][1] * out_kernel[j][0]);// * scale;

      in_inv[j][0] = in_r;
      in_inv[j][1] = in_c;
    }
  }
  
  FFTW3_TIME_END(TIME_FFTW3_MULT)
  
  memset(fftw_conv->in_src, 0, sizeof(float) * fftw_conv->width_fft * fftw_conv->height_fft);

  FFTW3_TIME_BEGIN
//  dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
  FFTW3_TIME_END(TIME_FFTW3_LOCK)


  FFTW3_TIME_BEGIN
  fftwf_execute(fftw_conv->plan_inv);
  FFTW3_TIME_END(TIME_FFTW3_EXEC)


  FFTW3_TIME_BEGIN
//  dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
  FFTW3_TIME_END(TIME_FFTW3_LOCK)

  FFTW3_TIME_BEGIN
  if (fftw_conv->mode == convolve_mode_full)
  {
    // return [0:height_dest-1; 0:width_dest-1]
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(fftw_conv, image_dest) schedule(static)
#endif
    for (int y = 0; y < fftw_conv->height_dest; y++)
    {
      float *dest = image_dest + y * fftw_conv->width_dest * fftw_conv->ch;
      float *out_inv = fftw_conv->in_src + y * fftw_conv->width_fft;

      for (int x = 0; x < fftw_conv->width_dest; x++)
      {
        dest[x * fftw_conv->ch + channel] = out_inv[x] * scale;
      }
    }
  }
  else if (fftw_conv->mode == convolve_mode_valid)
  {
    // return [height_dest; width_dest] starting at [height_kernel-1; width_kernel-1]
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(fftw_conv, image_dest) schedule(static)
#endif
    for (int y = 0; y < fftw_conv->height_dest; y++)
    {
      float *dest = image_dest + y * fftw_conv->width_dest * fftw_conv->ch;
      float *out_inv = fftw_conv->in_src + (y + fftw_conv->height_kernel - 1) * fftw_conv->width_fft;

      for (int x = 0; x < fftw_conv->width_dest; x++)
      {
        dest[x * fftw_conv->ch + channel] = out_inv[x + fftw_conv->width_kernel - 1] * scale;
      }
    }
  }
  FFTW3_TIME_END(TIME_FFTW3_COPY)

}

#endif	// #ifdef HAVE_FFTW3

void convolve_get_dest_size(const int width_src, const int height_src, 
    const int width_kernel, const int height_kernel, 
    int *width_dest, int *height_dest, const int mode)
{
  if (mode == convolve_mode_full)
  {
    *height_dest = height_src + height_kernel-1;
    *width_dest = width_src + width_kernel-1;
  }
  else if (mode == convolve_mode_valid)
  {
    *height_dest = height_src - height_kernel+1;
    *width_dest = width_src - width_kernel+1;
  }
  else
  {
    printf("convolve_get_dest_size: unknown mode=%i\n", mode);
  }

}

#ifdef HAVE_FFTW3

static fftw3_convolve_t * fftw3_find_conv_kernel(convolve_data_t *fft_conv_data, 
    const int width_src, const int height_src, const int ch, 
    const int width_kernel, const int height_kernel, 
    const int width_dest, const int height_dest, 
    const int mode, double *time)
{
  fftw3_convolve_t * kernel = NULL;
  int null_index = -1;
  
  for (int i = 0; i < RLUCY_CONV_DATA_MAX; i++)
  {
    fftw3_convolve_t *fftw3_conv = fft_conv_data->conv_kern[i];
    
    if (fftw3_conv)
    {
      if (width_src == fftw3_conv->width_src && height_src == fftw3_conv->height_src && ch == fftw3_conv->ch && 
          width_kernel == fftw3_conv->width_kernel && height_kernel == fftw3_conv->height_kernel && 
          width_dest == fftw3_conv->width_dest && height_dest == fftw3_conv->height_dest)
       {
        kernel = fftw3_conv;
        break;
       }
    }
    else
    {
      null_index = i;
      break;
    }
  }
  
  if (kernel == NULL && null_index >= 0)
  {
    kernel = fftw3_convolve_init(width_src, height_src, ch, 
        width_kernel, height_kernel, 
        width_dest, height_dest, 
        mode, time);
    
    fft_conv_data->conv_kern[null_index] = kernel;
  }
  
  return kernel;
}

void convolve(convolve_data_t *fft_conv_data, const float *const src, const int width_src, const int height_src, const int ch, 
    const float *const kernel, const int width_kernel, const int height_kernel, 
    float *dest, const int width_dest, const int height_dest, 
    const int mode, double *time)
{  
  const int ch1 = (ch == 4) ? 3: ch;
  int free_conv = 0;
  fftw3_convolve_t *fftw3_conv = NULL;

  fftw3_conv = fftw3_find_conv_kernel(fft_conv_data, 
      width_src, height_src, ch,
      width_kernel, height_kernel,
      width_dest, height_dest, mode, time);
  
  if (fftw3_conv == NULL)
  {
    fftw3_conv = fftw3_convolve_init(width_src, height_src, ch,
        width_kernel, height_kernel,
        width_dest, height_dest, 
            mode, time);
    
    free_conv = 1;
  }
  
  if (fftw3_conv)
  {
    for (int c = 0; c < ch1; c++)
    {
      fftw3_convolve(fftw3_conv, src, width_src, height_src, ch,
          kernel, width_kernel, height_kernel,
          dest, width_dest, height_dest, c, time);
    }
  }
  else
  {
    printf("rlucy_convolve: error initializing fftw3\n");
  }
  
  if (free_conv)
  {
    fftw3_convolve_free(fftw3_conv, time);
  }
  
}

 
#else

static void rlucy_convolve_internal(const float *const src, const int width_src, const int height_src, const int ch, 
    const float *const kernel, const int width_kernel, const int height_kernel, 
    float *dest, const int width_dest, const int height_dest, double *time)
{
  const int pad_w = (width_kernel - 1) / 2;
  const int pad_h = (height_kernel - 1) / 2;
  
  const int offset_x = (width_src - width_dest) / 2;
  const int offset_y = (height_src - height_dest) / 2;
  
  const int start_x = MAX(pad_w-offset_x, 0);
  const int start_y = MAX(pad_h-offset_y, 0);
  const int end_x = width_dest - start_x;
  const int end_y = height_dest - start_y;
  
  for (int y = start_y; y < end_y; y++)
  {
    for (int x = start_x; x < end_x; x++)
    {
      for (int c = 0; c < ch; c++)
      {
        float pixel = 0.f;
  
        for (int j = 0; j < height_kernel; j++)
        {
          for (int i = 0; i < width_kernel; i++)
          {
//            pixel += image[(x + i - (roi_kernel->width - 1) / 2) + (y + j - (roi_kernel->height - 1) / 2) * roi_image->width * ch] * kernel[j * roi_kernel->width + i];
            const int x1 = x + offset_x + (i-pad_w);
            const int y1 = y + offset_y + (j-pad_h);
            if (x1 >= width_src) printf("x1=%i >= roi_image->width=%i\n", x1, width_src);
            if (y1 >= height_src) printf("y1=%i >= roi_image->height=%i\n", y1, height_src);
            
            pixel += src[c + x1 * ch + y1 * width_src * ch] * kernel[j * width_kernel + i];
          }
        }
        
        if (x >= width_dest) printf("x=%i >= roi_dest->width=%i\n", x, width_dest);
        if (y >= height_dest) printf("y=%i >= roi_dest->height=%i\n", y, height_dest);
        
        dest[c + x * ch + y * width_dest * ch] = pixel;
      }
    }
  }
  
}

void convolve(convolve_data_t *fft_conv_data, const float *const src, const int width_src, const int height_src, const int ch, 
    const float *const kernel, const int width_kernel, const int height_kernel, 
    float *dest, const int width_dest, const int height_dest, 
    const int mode, double *time)
{
  float *im_padded = NULL;
  
  int width_fft =  0;
  int height_fft =  0;
  int pad_v = 0;
  int pad_h = 0;

  if (mode == convolve_mode_full)
  {
    height_fft = height_src + height_kernel - 1;
    width_fft = width_src + width_kernel - 1;
  }
  else if (mode == convolve_mode_valid)
  {
    height_fft = height_src;
    width_fft = width_src;
  }
  else
  {
    printf("convolve: unknown mode=%i\n", mode);
    goto cleanup;
  }

  {
    int w = 0, h = 0;
    convolve_get_dest_size(width_src, height_src, 
        width_kernel, height_kernel, 
        &w, &h, mode);
    if (height_dest != h || width_dest != w)
    {
      printf("fftw3_convolve: invalid dest size\n");
      goto cleanup;
    }
  }

  pad_v = floor((height_fft-height_src) / 2);
  pad_h = floor((width_fft-width_src) / 2);

  if (pad_v > 0 || pad_h > 0)
  {
    const int width_padded = width_src + pad_h * 2;
    const int height_padded = height_src + pad_h * 2;
    
    im_padded = dt_alloc_align(64, width_padded * height_padded * ch * sizeof(float));
    if (im_padded == NULL) goto cleanup;
    
    convolve_pad_image(src, width_src, height_src, ch, pad_h, pad_v, im_padded, width_padded, height_padded, time);
    
    rlucy_convolve_internal(im_padded, width_padded, height_padded, ch, 
        kernel, width_kernel, height_kernel, 
        dest, width_dest, height_dest, time);
  }
  else
  {
    rlucy_convolve_internal(src, width_src, height_src, ch, 
        kernel, width_kernel, height_kernel, 
        dest, width_dest, height_dest, time);
  }

  
cleanup:
  
  if (im_padded) dt_free_align(im_padded);
  
}
#endif

void convolve_free(convolve_data_t *fft_conv_data, double *time)
{
#ifdef HAVE_FFTW3
  for (int i = 0; i < RLUCY_CONV_DATA_MAX; i++)
  {
    if (fft_conv_data->conv_kern[i])
    {
      fftw3_convolve_free(fft_conv_data->conv_kern[i], time);
      fft_conv_data->conv_kern[i] = NULL;
    }
  }
#endif
}

void convolve_pad_image(const float *const img_src, const int width_src, const int height_src, const int ch, 
    const int pad_h, const int pad_v, 
    float *img_dest, const int width_dest, const int height_dest, double *time)
{
  
  // copy image
  for (int y = 0; y < height_src; y++)
  {
    const float *const src = img_src + y * width_src * ch;
    float *dest = img_dest + (y + pad_v) * width_dest * ch + pad_h * ch;
    
    memcpy(dest, src, width_src * ch * sizeof(float));
  }
  
  // pad left and right
  for (int y = 0; y < height_src; y++)
  {
    const float *const src_l = img_src + y * width_src * ch;
    float *dest_l = img_dest + (y + pad_v) * width_dest * ch;
    
    const float *const src_r = img_src + y * width_src * ch + (width_src-1) * ch;
    float *dest_r = img_dest + (y + pad_v) * width_dest * ch + (width_src+pad_h) * ch;
    
    for (int x = 0; x < pad_h; x++)
    {
      for (int c = 0; c < ch; c++)
      {
        dest_l[x*ch + c] = src_l[c];
        dest_r[x*ch + c] = src_r[c];
      }
    }
  }

  // pad top and bottom
  const float *const src_t = img_dest + pad_v * width_dest * ch;
  const float *const src_b = img_dest + (height_src + pad_v - 1) * width_dest * ch;
  
  for (int y = 0; y < pad_v; y++)
  {
    float *dest_t = img_dest + y * width_dest * ch;
    float *dest_b = img_dest + (y + height_src + pad_v) * width_dest * ch;
    
    memcpy(dest_t, src_t, width_dest * ch * sizeof(float));
    memcpy(dest_b, src_b, width_dest * ch * sizeof(float));
  }

}

