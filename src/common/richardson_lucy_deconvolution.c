
#include <stdlib.h>
#include "develop/imageop.h"
#include "common/interpolation.h"

#include "common/convolution.h"

#include "richardson_lucy_deconvolution.h"

//-----------------------------------------------------------------------------------------

#define TIME_CONV_FFT_LOCK 1
#define TIME_CONV_FFT_EXEC 2
#define TIME_CONV_FFT_COPY 3
#define TIME_CONV_FFT_MULT 4
#define TIME_CONV_FFT_PLAN 5

#define TIME_CONVOLVE 6
#define TIME_SUM 0
#define TIME_SUBTRACT 0
#define TIME_MULSUB 0
#define TIME_ADD 0
#define TIME_PADIMAGE 0
#define TIME_UNPADIMAGE 0
#define TIME_TRIMMASK 0
#define TIME_DIVTV 0
#define TIME_ROTATE 0
#define TIME_GRADIENT 0
#define TIME_DIVIDE 0
#define TIME_PROCESS 18
#define TIME_BUILD_KERNEL 19

#define MAX_TIME 20

const char *rlucy_time_desc[MAX_TIME] = { 
                                  "dummy         "
                                 ,"fft lock    "
                                 ,"fft execute "
                                 ,"fft copy    "
                                 ,"fft mult    "
                                 ,"fft plan    "
                                 ,"convolve      "
                                 ,"sum           "
                                 ,"subtract      "
                                 ,"mulsub        "
                                 ,"add           "
                                 ,"pad image     "
                                 ,"unpad image   "
                                 ,"trim mask     "
                                 ,"divTV         "
                                 ,"rotate        "
                                 ,"gradient      "
                                 ,"divide        "
                                 ,"process       "
                                 ,"build kernel  "
};

#define RLUCY_TIME

#ifdef RLUCY_TIME

#define RLUCY_TIME_DECL double start = 0;

#define RLUCY_TIME_BEGIN start = dt_get_wtime();

#define RLUCY_TIME_END(time_index) time[time_index] += dt_get_wtime() - start;

#else

#define RLUCY_TIME_DECL ;

#define RLUCY_TIME_BEGIN ;

#define RLUCY_TIME_END(time_index) ;

#endif

//-----------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------

//#define RLUCY_NAN

#ifdef RLUCY_NAN

static void rlucy_check_nan(rlucy_image_t* im, const char *str)
{
  int i_nan = 0;
  
  for (int i = 0; i < im->w*im->h*im->h; i++) if ( isnan(im->im[i]) ) i_nan++;
  
  if (i_nan > 0) printf("%s nan: %i\n", str, i_nan);
}

#else
 
#define rlucy_check_nan(im, str) {}

#endif


//-----------------------------------------------------------------------------------------


/*
'''
Created on 30 avr. 2017

@author: aurelien

This script shows an implementation of the Richardson-Lucy deconvolution.

In theory, blurred and noisy pictures can be perfectly sharpened if we perfectly 
know the [*Point spread function*](https://en.wikipedia.org/wiki/Point_spread_function) 
of their maker. In practice, we can only estimate it.
One of the means to do so is the [Richardson-Lucy deconvolution](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution).

The Richardson-Lucy algorithm used here has a damping coefficient wich allows to remove from 
the iterations the pixels which deviate to much (x times the standard deviation of the difference
source image - deconvoluted image) from the original image. This pixels are considered 
noise and would be amplificated from iteration to iteration otherwise.
*/


typedef struct
{
  float *im;
  int w;
  int h;
  int c;
  size_t alloc_size;
} rlucy_image_t;


static void rlucy_free_image(rlucy_image_t *image)
{
  if (image->im)
  {
    dt_free_align(image->im);
    image->im = NULL;
  }
  image->w = image->h = image->c = 0;
  image->alloc_size = 0;
}

static void rlucy_alloc_image(rlucy_image_t *image, const int w, const int h, const int c)
{
  // re-alloc
  if (image->im != NULL)
  {
    if (image->alloc_size >= w * h * c)
    {
      image->w = w;
      image->h = h;
      image->c = c;
    }
    else
    {
      rlucy_free_image(image);
    }
  }
  
  if (image->im == NULL)
  {
    image->w = w;
    image->h = h;
    image->c = c;
    image->alloc_size = w * h * c;
    image->im = dt_alloc_align(64, image->alloc_size * sizeof(float));
  }
}

static void rlucy_img_sum(const rlucy_image_t *const image, float *sum, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
	
	const int size = image->w * image->h * image->c;

	for (int c = 0; c < image->c; c++)
		sum[c] = 0.f;

	for (int i = 0; i < size; i += image->c)
	{
		for (int c = 0; c < image->c; c++)
			sum[c] += image->im[i+c];
	}
	
	RLUCY_TIME_END(TIME_SUM)
	
}

static void rlucy_img_divide(rlucy_image_t *image, float *div, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
	
	const int size = image->w * image->h * image->c;
	float mult[4] = {0};
	
	for (int c = 0; c < image->c; c++)
	{
		if (div[c] == 0.f)
			mult[c] = 0.f;
		else
			mult[c] = 1.f / div[c];
	}
	
	float *im = image->im;
	const int ch = image->c;
	
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(im, mult) schedule(static)
#endif
	for (int i = 0; i < size; i+=ch)
	{
		for (int c = 0; c < ch; c++)
		{
			im[i+c] *= mult[c];
		}
	}
	
	RLUCY_TIME_END(TIME_DIVIDE)
	
}

static void rlucy_img_outer(float *vector, const int size, float *outer, const int ch)
{
  const int size_c = size * ch;
  
	for (int i = 0; i < size; i++)
	{
	  const int idx_outer = i * size_c;
	  
		for (int j = 0; j < size; j++)
		{
			const float mult = vector[i] * vector[j];
			
			for (int c = 0; c < ch; c++)
			{
				outer[idx_outer + j * ch + c] = mult;
			}
		}
	}

}

// from:
// https://github.com/johnglover/simpl/blob/master/src/loris/KaiserWindow.C
//

//  Compute the zeroeth order modified Bessel function of the first kind 
//  at x using the series expansion, used to compute the Kasier window
//  function.
//
static float rlucy_zeroethOrderBessel(const float x)
{
  const float eps = 0.000001;

  //  initialize the series term for m=0 and the result
  float besselValue = 0;
  float term = 1;
  float m = 0;

  //  accumulate terms as long as they are significant
  while(term  > eps * besselValue)
  {
    besselValue += term;

    //  update the term
    ++m;
    term *= (x*x) / (4*m*m);
  }

  return besselValue;
}

//! Build a new Kaiser analysis window having the specified shaping
//! parameter. See Oppenheim and Schafer:  "Digital Signal Processing" 
//! (1975), p. 452 for further explanation of the Kaiser window. Also, 
//! see Kaiser and Schafer, 1980.
//!
//! \param      win is the vector that will store the window
//!             samples. The number of samples computed will be
//!             equal to the length of this vector. Any previous
//!             contents will be overwritten.
//! \param      shape is the Kaiser shaping parameter, controlling
//!             the sidelobe rejection level.
//
static void rlucy_buildKaiserWindow(float *win, const int win_size, const float shape)
{   
  //  Pre-compute the shared denominator in the Kaiser equation. 
  const float oneOverDenom = 1.0f / rlucy_zeroethOrderBessel( shape );

  const int N = win_size - 1;
  const float oneOverN = 1.0 / N;

  for ( int n = 0; n <= N; ++n )
  {
    const float K = (2.0 * n * oneOverN) - 1.0;
    const float arg = sqrtf( 1.0 - (K * K) );

    win[n] = rlucy_zeroethOrderBessel( shape * arg ) * oneOverDenom;
  }
}

static void rlucy_kaiser_kernel(rlucy_image_t *kern, const float beta, double *time)
{
	float *win = NULL;
	const int radius = kern->w;
	
	if (kern->w != kern->h)
		printf("rlucy_kaiser_kernel: invalid kernel size\n");
			
//	window = np.kaiser(radius, beta)
	win = calloc(radius, sizeof(float));
	if (win)
	{
		rlucy_buildKaiserWindow(win, radius, beta);
	
//	kern = np.outer(window, window)
		rlucy_img_outer(win, radius, kern->im, kern->c);
		
		float kern_sum[4] = {0};
		
//  kern = kern / kern.sum()
		rlucy_img_sum(kern, kern_sum, time);
		rlucy_img_divide(kern, kern_sum, time);
	}
	
	if (win) free(win);
}

// from:
// https://github.com/scipy/scipy/blob/v0.19.1/scipy/signal/windows.py#L1159-L1219
//

static void rlucy_buildGaussianWindow(float *win, const int win_size, const float shape)
{   
	if (win_size == 1)
	{
		win[0] = 1.f;
	}
	else
	{
		for (int i = 0; i < win_size; i++)
		{
			win[i] = i - (win_size - 1.f) / 2.f;
		}
		
		float sig2 = 2.f * shape * shape;
		
		for (int i = 0; i < win_size; i++)
		{
			win[i] = expf( -(win[i] * win[i]) / sig2 );
		}
	}
	
}

static void rlucy_gaussian_kernel(rlucy_image_t *kern, const float beta, double *time)
{
	float *win = NULL;
	const int radius = kern->w;
	
	if (kern->w != kern->h)
		printf("rlucy_gaussian_kernel: invalid kernel size\n");
			
//	window = np.kaiser(radius, beta)
	win = calloc(radius, sizeof(float));
	if (win)
	{
		rlucy_buildGaussianWindow(win, radius, beta);
	
//	kern = np.outer(window, window)
		rlucy_img_outer(win, radius, kern->im, kern->c);
		
		float kern_sum[4] = {0};
		
//  kern = kern / kern.sum()
		rlucy_img_sum(kern, kern_sum, time);
		rlucy_img_divide(kern, kern_sum, time);
	}
	
	if (win) free(win);
}

/*def uniform_kernel(size):
    kern = np.ones((size, size))
    kern /= np.sum(kern)
    return kern*/
static void rlucy_uniform_kernel(rlucy_image_t *kern)
{
  const int radius = kern->w;
	const float radius2 = 1.f / (radius * radius);
  const int size = kern->w * kern->h * kern->c;

	if (kern->w != kern->h)
		printf("rlucy_uniform_kernel: invalid kernel size\n");
			
	for (int i = 0; i < size; i++)
	{
		kern->im[i] = radius2;
	}
}

static void rlucy_img_subtract(rlucy_image_t *img_in1, 
		const rlucy_image_t *const img_in2, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
	
	const int size = img_in1->w * img_in1->h * img_in1->c;
	
	if (img_in1->w != img_in2->w || img_in1->h != img_in2->h)
		printf("rlucy_img_subtract: invalid image size\n");

  float *img_in1_im = img_in1->im;
  const float *const img_in2_im = img_in2->im;
	
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(img_in1_im) schedule(static)
#endif
	for (int i = 0; i < size; i++)
	{
	  img_in1_im[i] -= img_in2_im[i];
	}
	
	RLUCY_TIME_END(TIME_SUBTRACT)
	
}

static void rlucy_img_mulsub(rlucy_image_t *img_in1, 
		const rlucy_image_t *const img_in2, 
		const float *const mult, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
	
	const int size = img_in1->w * img_in1->h * img_in1->c;
	
	if (img_in1->w != img_in2->w || img_in1->h != img_in2->h)
		printf("rlucy_img_mulsub: invalid image size\n");

  float *img_in1_im = img_in1->im;
  const float *const img_in2_im = img_in2->im;
  const int ch = img_in1->c;
  
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(img_in1_im) schedule(static)
#endif
	for (int i = 0; i < size; i+=ch)
	{
		for (int c = 0; c < ch; c++)
		{
		  img_in1_im[i+c] -= img_in2_im[i+c] * mult[c];
		}
	}
	
	RLUCY_TIME_END(TIME_MULSUB)
	
}

static void rlucy_img_max(const rlucy_image_t *const image, float *max, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
	
	const int size = image->w * image->h * image->c;
	
	for (int c = 0; c < image->c; c++)
	{
		max[c] = -INFINITY;
	}
		
	for (int i = 0; i < size; i+=image->c)
	{
		for (int c = 0; c < image->c; c++)
		{
			max[c] = MAX(max[c], image->im[i+c]);
		}
	}
	
	RLUCY_TIME_END(0)
	
}

static void rlucy_img_absmax(const rlucy_image_t *const image, float *max, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
	
	const int size = image->w * image->h * image->c;
	
	for (int c = 0; c < image->c; c++)
	{
		max[c] = 0.f;
	}
	
	for (int i = 0; i < size; i+=image->c)
	{
		for (int c = 0; c < image->c; c++)
		{
			max[c] = MAX(max[c], fabs(image->im[i+c]));
		}
	}
	
	RLUCY_TIME_END(0)
	
}

static void rlucy_img_rotate90(const rlucy_image_t *const img_in, 
    rlucy_image_t *img_dest, 
    const int times, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
    
  if (times == 1)
  {
    rlucy_alloc_image(img_dest, img_in->h, img_in->w, img_in->c);
    
    if (img_in->w != img_dest->h || img_in->h != img_dest->w)
      printf("rlucy_img_rotate90: invalid image size\n");

    for (int y = 0; y < img_in->h; y++)
    {
      const int x1 = (img_dest->w - y - 1) * img_dest->c;
      const int y2 = y * img_in->w * img_in->c;
      
      for (int x = 0; x < img_in->w; x++)
      {
        const int y1 = (x * img_dest->w * img_dest->c) + x1;
        const int x2 = y2 + (x * img_in->c);
        
        for (int c = 0; c < img_in->c; c++)
        {
          img_dest->im[y1 + c] = img_in->im[x2 + c];
        }
      }
    }
  }
  else if (times == 2)
  {
    rlucy_alloc_image(img_dest, img_in->w, img_in->h, img_in->c);
    
    if (img_in->w != img_dest->w || img_in->h != img_dest->h)
      printf("rlucy_img_rotate90: invalid image size\n");

    for (int y = 0; y < img_in->h; y++)
    {
      const int y1 = (img_dest->h - y - 1) * img_dest->w * img_dest->c;
      const int y2 = (y * img_in->w * img_in->c);
      
      for (int x = 0; x < img_in->w; x++)
      {
        const int x1 = y1 + (img_dest->w - x - 1) * img_dest->c;
        const int x2 = y2 + (x * img_in->c);
        
        for (int c = 0; c < img_in->c; c++)
        {
          img_dest->im[x1 + c] = img_in->im[x2 + c];
        }
      }
    }
  }
  else
    printf("rlucy_img_rotate90: invalid argument times\n");
  
  RLUCY_TIME_END(TIME_ROTATE)
  
}

static void rlucy_pad_image(const rlucy_image_t *const img_src, 
    const int pad_h, const int pad_v, 
    rlucy_image_t *img_dest, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
    
  rlucy_alloc_image(img_dest, img_src->w + pad_h * 2, img_src->h + pad_v * 2, img_src->c);
  
  convolve_pad_image(img_src->im, img_src->w, img_src->h, img_src->c, 
      pad_h, pad_v, 
      img_dest->im, img_dest->w, img_dest->h, time);
  
  RLUCY_TIME_END(TIME_PADIMAGE)

}

static void rlucy_copy_image(rlucy_image_t * img_dest, rlucy_image_t *img_src)
{
  
  if (img_src->w == img_dest->w && img_src->h == img_dest->h && img_src->c == img_dest->c)
  {
    memcpy(img_dest->im, img_src->im, img_src->w * img_src->h * img_src->c * sizeof(float));
    return;
  }
  
  if (img_src->c == img_dest->c)
  {
    const int h = MIN(img_src->h, img_dest->h);
    const int size = MIN(img_src->w * img_src->c, img_dest->w * img_dest->c) * sizeof(float);
    
    for (int y = 0; y < h; y++)
    {
      float *dest = img_dest->im + y * img_dest->w * img_dest->c;
      float *src = img_src->im + y * img_src->w * img_src->c;
      
      memcpy(dest, src, size);
    }
  }
  else
    printf("rlucy_copy_image: different channels not implementd\n");
  
}

static void rlucy_unpad_image(rlucy_image_t * img_dest, 
		const int pad_h, const int pad_v, 
		rlucy_image_t *img_src, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
  
  rlucy_alloc_image(img_dest, img_src->w - pad_h * 2, img_src->h - pad_v * 2, img_src->c);
  
	// copy image
	for (int y = 0; y < img_dest->h; y++)
	{
		float *dest = img_dest->im + y * img_dest->w * img_dest->c;
		float *src = img_src->im + (y + pad_v) * img_src->w * img_src->c + pad_h * img_src->c;
		
		memcpy(dest, src, img_dest->w * img_dest->c * sizeof(float));
	}
	
  RLUCY_TIME_END(TIME_UNPADIMAGE)
}

// mask[left, top, width, height]
static void rlucy_trim_mask(const rlucy_image_t *const img_src, rlucy_image_t *img_dest, 
    const int im_x, const int im_y, const int im_w, const int im_h, rlucy_image_t *im_tmp1,
		double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
  
  if (img_src->im == img_dest->im)
  {
    rlucy_image_t tmp = {0};
    
    rlucy_trim_mask(img_src, im_tmp1, im_x, im_y, im_w, im_h, &tmp, time);
    rlucy_alloc_image(img_dest, im_tmp1->w, im_tmp1->h, im_tmp1->c);
    rlucy_copy_image(img_dest, im_tmp1);
    
    rlucy_free_image(&tmp);
  }
  else
  {
    rlucy_alloc_image(img_dest, im_w, im_h, img_src->c);
  
    if (img_dest->w > img_src->w || img_dest->h > img_src->h)
      printf("rlucy_trim_mask: roi_dest > roi_src\n");
    if (img_dest->w+im_x > img_src->w || img_dest->h+im_y > img_src->h)
      printf("rlucy_trim_mask: roi_dest+x > roi_src\n");
      
    // copy image
    for (int y = 0; y < img_dest->h; y++)
    {
      const float *const src = img_src->im + (y + im_y) * img_src->w * img_src->c + im_x * img_src->c;
      float *dest = img_dest->im + y * img_dest->w * img_dest->c;
      
      memcpy(dest, src, img_dest->w * img_dest->c * sizeof(float));
    }
  }
	
  RLUCY_TIME_END(TIME_TRIMMASK)

}

// from:
// https://github.com/numpy/numpy/blob/v1.13.0/numpy/lib/function_base.py#L1502-L1840
//

void rlucy_gradient(const rlucy_image_t *const f, rlucy_image_t *outvals, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
  
	int edge_order = 2;
  float *yy = NULL;

//	f = np.asanyarray(f)
//	    N = f.ndim  # number of dimensions
//	int N = 2;

/*	    axes = kwargs.pop('axis', None)
	    if axes is None:
	        axes = tuple(range(N))
	    else:
	        axes = _nx.normalize_axis_tuple(axes, N)*/
//	int axes[2] = {0, 1};

//	    len_axes = len(axes)
	int len_axes = 2;
	
/*	    n = len(varargs)
	    if n == 0:
	        dx = [1.0] * len_axes
	    elif n == len_axes or (n == 1 and np.isscalar(varargs[0])):
	        dx = list(varargs)
	        for i, distances in enumerate(dx):
	            if np.isscalar(distances):
	                continue
	            if len(distances) != f.shape[axes[i]]:
	                raise ValueError("distances must be either scalars or match "
	                                 "the length of the corresponding dimension")
	            diffx = np.diff(dx[i])
	            # if distances are constant reduce to the scalar case
	            # since it brings a consistent speedup
	            if (diffx == diffx[0]).all():
	                diffx = diffx[0]
	            dx[i] = diffx
	        if len(dx) == 1:
	            dx *= len_axes
	    else:
	        raise TypeError("invalid number of arguments")*/
	const float dx[2] = {1.0f, 1.0f};
	
/*	    edge_order = kwargs.pop('edge_order', 1)
	    if kwargs:
	        raise TypeError('"{}" are not valid keyword arguments.'.format(
	                                                  '", "'.join(kwargs.keys())))
	    if edge_order > 2:
	        raise ValueError("'edge_order' greater than 2 not supported")*/

/*	
	    # use central differences on interior and one-sided differences on the
	    # endpoints. This preserves second order-accuracy over the full domain.
*/
//	    outvals = []

/*	    # create slice objects --- initially all are [:, :, ..., :]
	    slice1 = [slice(None)]*N
	    slice2 = [slice(None)]*N
	    slice3 = [slice(None)]*N
	    slice4 = [slice(None)]*N

	    otype = f.dtype.char
	    if otype not in ['f', 'd', 'F', 'D', 'm', 'M']:
	        otype = 'd'

	    # Difference of datetime64 elements results in timedelta64
	    if otype == 'M':
	        # Need to use the full dtype name because it contains unit information
	        otype = f.dtype.name.replace('datetime', 'timedelta')
	    elif otype == 'm':
	        # Needs to keep the specific units, can't be a general unit
	        otype = f.dtype

	    # Convert datetime64 data into ints. Make dummy variable `y`
	    # that is a view of ints if the data is datetime64, otherwise
	    # just set y equal to the array `f`.
	    if f.dtype.char in ["M", "m"]:
	        y = f.view('int64')
	    else:
	        y = f
*/
	    yy = dt_alloc_align(64, f->w * f->h * f->c * sizeof(float));
	    if (yy == NULL) goto cleanup;
	    
	    memcpy(yy, f->im, f->w * f->h * f->c * sizeof(float));
	    
//	    for i, axis in enumerate(axes):
  for (int i = 0, axis = 0; i < len_axes; i++, axis++)
  {
/*	        if y.shape[axis] < edge_order + 1:
	            raise ValueError(
	                "Shape of array too small to calculate a numerical gradient, "
	                "at least (edge_order + 1) elements are required.")*/
	        // result allocation
//	        out = np.empty_like(y, dtype=otype)
//    rlucy_image_t out = *f;
    rlucy_image_t out = {0};
//    out.im = dt_alloc_align(64, f->w * f->h * f->c * sizeof(float));
    rlucy_alloc_image(&out, f->w, f->h, f->c);

//	        uniform_spacing = np.isscalar(dx[i])
  	int uniform_spacing = 1;

  	// Numerical differentiation: 2nd order interior
/*	        slice1[axis] = slice(1, -1)
	        slice2[axis] = slice(None, -2)
	        slice3[axis] = slice(1, -1)
	        slice4[axis] = slice(2, None)*/

/*	        if uniform_spacing:
	            out[slice1] = (f[slice4] - f[slice2]) / (2. * dx[i])
	        else:
	            dx1 = dx[i][0:-1]
	            dx2 = dx[i][1:]
	            a = -(dx2)/(dx1 * (dx1 + dx2))
	            b = (dx2 - dx1) / (dx1 * dx2)
	            c = dx1 / (dx2 * (dx1 + dx2))
	            # fix the shape for broadcasting
	            shape = np.ones(N, dtype=int)
	            shape[axis] = -1
	            a.shape = b.shape = c.shape = shape
	            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
	            out[slice1] = a * f[slice2] + b * f[slice3] + c * f[slice4]*/

  	if (uniform_spacing)
  	{
  	  const float dx_2 = 1.f / (2.f * dx[i]);
  	  
			if (axis == 0)
			{
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(out, i) schedule(static)
#endif
				for (int y = 0; y < f->h; y++)
				{
					for (int x = 1; x < f->w - 1; x++)
					{
						const int idx_out = y * f->w * f->c + x * f->c;
						const int idx_f2 = y * f->w * f->c + (x - 1) * f->c;
						const int idx_f4 = y * f->w * f->c + (x + 1) * f->c;
						
						for (int c = 0; c < f->c; c++)
						{
							out.im[idx_out + c] = (f->im[idx_f4 + c] - f->im[idx_f2 + c]) * dx_2;
						}
					}
  			}
  		}
			else
			{
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(out, i) schedule(static)
#endif
				for (int y = 1; y < f->h - 1; y++)
				{
					for (int x = 0; x < f->w; x++)
					{
						const int idx_out = y * f->w * f->c + x * f->c;
						const int idx_f2 = (y - 1) * f->w * f->c + x * f->c;
						const int idx_f4 = (y + 1) * f->w * f->c + x * f->c;
						
						for (int c = 0; c < f->c; c++)
						{
							out.im[idx_out + c] = (f->im[idx_f4 + c] - f->im[idx_f2 + c]) * dx_2;
						}
					}
  			}
			}
  	}
  	
/*	        # Numerical differentiation: 1st order edges
	        if edge_order == 1:
	            slice1[axis] = 0
	            slice2[axis] = 1
	            slice3[axis] = 0
	            dx_0 = dx[i] if uniform_spacing else dx[i][0]
	            # 1D equivalent -- out[0] = (y[1] - y[0]) / (x[1] - x[0])
	            out[slice1] = (y[slice2] - y[slice3]) / dx_0

	            slice1[axis] = -1
	            slice2[axis] = -1
	            slice3[axis] = -2
	            dx_n = dx[i] if uniform_spacing else dx[i][-1]
	            # 1D equivalent -- out[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
	            out[slice1] = (y[slice2] - y[slice3]) / dx_n*/

  	// Numerical differentiation: 2nd order edges
//	        else:
/*	            slice1[axis] = 0
	            slice2[axis] = 0
	            slice3[axis] = 1
	            slice4[axis] = 2*/
  	
/*	            if uniform_spacing:
	                a = -1.5 / dx[i]
	                b = 2. / dx[i]
	                c = -0.5 / dx[i]
	            else:
	                dx1 = dx[i][0]
	                dx2 = dx[i][1]
	                a = -(2. * dx1 + dx2)/(dx1 * (dx1 + dx2))
	                b = (dx1 + dx2) / (dx1 * dx2)
	                c = - dx1 / (dx2 * (dx1 + dx2))
	            # 1D equivalent -- out[0] = a * y[0] + b * y[1] + c * y[2]
	            out[slice1] = a * y[slice2] + b * y[slice3] + c * y[slice4]*/
  	if (edge_order == 1)
  	{
  	}
  	else
  	{
			if (uniform_spacing)
			{
        const float ia = -1.5f / dx[i];
        const float ib = 2.f / dx[i];
        const float ic = -0.5f / dx[i];

        if (axis == 0)
        {
					for (int y = 0; y < f->h; y++)
					{
						const int idx_out = y * f->w * f->c;
						const int idx_y2 = y * f->w * f->c;
						const int idx_y3 = y * f->w * f->c + 1 * f->c;
						const int idx_y4 = y * f->w * f->c + 2 * f->c;
						
						for (int c = 0; c < f->c; c++)
						{
							out.im[idx_out + c] = ia * yy[idx_y2 + c] + ib * yy[idx_y3 + c] + ic * yy[idx_y4 + c];
						}
					}
        }
        else
        {
					for (int x = 0; x < f->w; x++)
					{
						const int idx_out = x * f->c;
						const int idx_y2 = x * f->c;
						const int idx_y3 = x * f->c + 1 * f->w * f->c;
						const int idx_y4 = x * f->c + 2 * f->w * f->c;
						
						for (int c = 0; c < f->c; c++)
						{
							out.im[idx_out + c] = ia * yy[idx_y2 + c] + ib * yy[idx_y3 + c] + ic * yy[idx_y4 + c];
						}
					}
        }
			}
  	}
  	
/*	            slice1[axis] = -1
	            slice2[axis] = -3
	            slice3[axis] = -2
	            slice4[axis] = -1
	            if uniform_spacing:
	                a = 0.5 / dx[i]
	                b = -2. / dx[i]
	                c = 1.5 / dx[i]
	            else:
	                dx1 = dx[i][-2]
	                dx2 = dx[i][-1]
	                a = (dx2) / (dx1 * (dx1 + dx2))
	                b = - (dx2 + dx1) / (dx1 * dx2)
	                c = (2. * dx2 + dx1) / (dx2 * (dx1 + dx2))
	            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
	            out[slice1] = a * y[slice2] + b * y[slice3] + c * y[slice4]*/

		if (uniform_spacing)
		{
      const float ia = 0.5f / dx[i];
      const float ib = -2.f / dx[i];
      const float ic = 1.5f / dx[i];

      if (axis == 0)
      {
				for (int y = 0; y < f->h; y++)
				{
					const int idx_out = y * f->w * f->c + (f->w - 1) * f->c;
					const int idx_y2 = y * f->w * f->c + (f->w - 3) * f->c;
					const int idx_y3 = y * f->w * f->c + (f->w - 2) * f->c;
					const int idx_y4 = y * f->w * f->c + (f->w - 1) * f->c;
					
					for (int c = 0; c < f->c; c++)
					{
						out.im[idx_out + c] = ia * yy[idx_y2 + c] + ib * yy[idx_y3 + c] + ic * yy[idx_y4 + c];
					}
				}
      }
      else
      {
				for (int x = 0; x < f->w; x++)
				{
					const int idx_out = (f->h - 1) * f->w * f->c + x * f->c;
					const int idx_y2 = (f->h - 3) * f->w * f->c + x * f->c;
					const int idx_y3 = (f->h - 2) * f->w * f->c + x * f->c;
					const int idx_y4 = (f->h - 1) * f->w * f->c + x * f->c;
					
					for (int c = 0; c < f->c; c++)
					{
						out.im[idx_out + c] = ia * yy[idx_y2 + c] + ib * yy[idx_y3 + c] + ic * yy[idx_y4 + c];
					}
				}
      }
		}

		outvals[axis] = out;
/*	        outvals.append(out)

	        # reset the slice object in this dimension to ":"
	        slice1[axis] = slice(None)
	        slice2[axis] = slice(None)
	        slice3[axis] = slice(None)
	        slice4[axis] = slice(None)*/
  }
  
/*	    if len_axes == 1:
	        return outvals[0]
	    else:
	return outvals*/
  
cleanup:

  if (yy) dt_free_align(yy);
  
  RLUCY_TIME_END(TIME_GRADIENT)

}

// Compute the second-order divergence of the pixel matrix, known as the Total Variation.
static void rlucy_divTV(const rlucy_image_t *const image, 
    rlucy_image_t *grad, double *time)
{
  RLUCY_TIME_DECL
	
	rlucy_image_t outvals[2] = {0};
	
  // grad = np.array(np.gradient(image, edge_order=2))
	rlucy_gradient(image, outvals, time);
	if (outvals[0].im == NULL || outvals[1].im == NULL) goto cleanup;
	
  RLUCY_TIME_BEGIN

  // grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
	const int size = image->w * image->h * image->c;
	
  rlucy_alloc_image(grad, image->w, image->h, image->c);
  
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(grad, outvals) schedule(static)
#endif
	for (int i = 0; i < size; i+=image->c)
	{
		for (int c = 0; c < image->c; c++)
		{
			grad->im[i + c] = sqrtf(outvals[0].im[i + c] * outvals[0].im[i + c] + outvals[1].im[i + c] * outvals[1].im[i + c]);
		}
	}

  RLUCY_TIME_END(TIME_DIVTV)
  
  // grad = grad / np.amax(grad)
	float max_grad[4] = {0};
	rlucy_img_absmax(grad, max_grad, time);
	rlucy_img_divide(grad, max_grad, time);
	
cleanup:
	
  rlucy_free_image(&(outvals[0]));
  rlucy_free_image(&(outvals[1]));
  
}

static void rlucy_convolve(const rlucy_image_t *const image, 
    const rlucy_image_t *const kernel, 
    const int mode, rlucy_image_t *img_dest, 
    convolve_data_t *fft_conv_data, 
    double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
  
  int w = 0, h = 0;

  convolve_get_dest_size(image->w, image->h, 
      kernel->w, kernel->h,
      &w, &h, mode);

  rlucy_alloc_image(img_dest, w, h, image->c);
  
  memset(img_dest->im, 0, img_dest->w * img_dest->h * img_dest->c * sizeof(float));
  
  convolve(fft_conv_data, image->im, image->w, image->h, image->c, 
      kernel->im, kernel->w, kernel->h, 
      img_dest->im, img_dest->w, img_dest->h, 
      mode, &(time[TIME_CONV_FFT_LOCK]));
  
  RLUCY_TIME_END(TIME_CONVOLVE)
  
  rlucy_check_nan(img_dest, "rlucy_convolve fft");
  
}

/*
def convolve_kernel(u, psf, image):
    return fftconvolve(np.rot90(u, 2), fftconvolve(u, psf, "valid") - image, "valid").astype(np.float32)
*/
static void rlucy_convolve_kernel(const rlucy_image_t *const u, const rlucy_image_t *const psf, const rlucy_image_t *const image, rlucy_image_t *dest,
    rlucy_image_t *tmp1, rlucy_image_t *tmp2,
    convolve_data_t *fft_conv_data, double *time)
{
  rlucy_convolve(u, psf, convolve_mode_valid, tmp1, fft_conv_data, time);
  rlucy_img_subtract(tmp1, image, time);

  rlucy_img_rotate90(u, tmp2, 2, time);
  rlucy_convolve(tmp2, tmp1, convolve_mode_valid, dest, fft_conv_data, time);

}

/*
@jit(float32[:, :](float32[:, :], float32[:, :], float32[:, :]), cache=True)
def convolve_image(u, psf, image):
    return fftconvolve(fftconvolve(u, psf, "valid") - image, np.rot90(psf), "full").astype(np.float32)
*/
static void rlucy_convolve_image(const rlucy_image_t *const u, const rlucy_image_t *const psf, const rlucy_image_t *const image, rlucy_image_t *dest,
    rlucy_image_t *tmp1, rlucy_image_t *tmp2,
        convolve_data_t *fft_conv_data, double *time)
{
  rlucy_convolve(u, psf, convolve_mode_valid, tmp1, fft_conv_data, time);
  rlucy_img_subtract(tmp1, image, time);
  rlucy_img_rotate90(psf, tmp2, 1, time);
  rlucy_convolve(tmp1, tmp2, convolve_mode_full, dest, fft_conv_data, time);
  
}

/*
def weight_update(factor: float, array: np.ndarray, grad_array: np.ndarray) -> np.ndarray:
    return (factor * np.amax(array) / np.maximum(1e-31, np.amax(np.abs(grad_array)))).astype(np.float32)
*/
static void rlucy_weight_update(const float factor, rlucy_image_t *array, rlucy_image_t *grad_array, float *dest, double *time)
{
  float max_array[4] = {0};
  rlucy_img_max(array, max_array, time);
  float max_grad_array[4] = {0};
  rlucy_img_absmax(grad_array, max_grad_array, time);

  for (int c = 0; c < array->c; c++)
  {
    dest[c] = factor * max_array[c] / MAX(1e-31, max_grad_array[c]);
  }

}

/*
def update_values(target: np.ndarray, factor: float, source: np.ndarray) -> np.ndarray:
    return (target - factor * source).astype(np.float32)
*/
static void rlucy_update_values(rlucy_image_t *target, const float *factor, rlucy_image_t *source, double *time)
{
  rlucy_img_mulsub(target, source, factor, time);
}

static void rlucy_update_image(const rlucy_image_t *const image, 
    rlucy_image_t *u, 
		const float *lambd, 
		const rlucy_image_t *const psf, 
		convolve_data_t *fft_conv_data, double *time)
{
  rlucy_image_t gradUdata = {0};
  rlucy_image_t u_divTV = {0};
  rlucy_image_t tmp1 = {0};
  rlucy_image_t tmp2 = {0};
	
  // # Total Variation Regularization
  // gradUdata = convolve_image(u, psf, image)
	rlucy_convolve_image(u, psf, image, &gradUdata, &tmp1, &tmp2, fft_conv_data, time);
	
	// TODO: [:gradUdata.shape[0], :gradUdata.shape[1]]
  // gradu = gradUdata - lambd * divTV(u)[:gradUdata.shape[0], :gradUdata.shape[1]]
  rlucy_divTV(u, &u_divTV, time);
	rlucy_img_mulsub(&gradUdata, &u_divTV, lambd, time);
	
  // u = update_values(u[:gradUdata.shape[0], :gradUdata.shape[1]],
  //                  weight_update(5e-3, u[:gradUdata.shape[0], :gradUdata.shape[1]], gradu), gradu)
  
  float weight_upd[4] = {0.f};
  rlucy_weight_update(5e-3, u, &gradUdata, weight_upd, time);
	rlucy_update_values(u, weight_upd, &gradUdata, time);
	
  rlucy_free_image(&gradUdata);
  rlucy_free_image(&u_divTV);
  rlucy_free_image(&tmp1);
  rlucy_free_image(&tmp2);
  
}

static void rlucy_normalize_kernel(rlucy_image_t *kern, double *time)
{
  //  kern[kern < 0] = 0
  for (int i = 0; i < kern->w * kern->h * kern->c; i++)
  {
    if (kern->im[i] < 0.f) kern->im[i] = 0.f;
  }
  
  //  return (kern / np.sum(kern)).astype(np.float32)
  float kern_sum[4] = {0};
  rlucy_img_sum(kern, kern_sum, time);
  rlucy_img_divide(kern, kern_sum, time);

}

/*
# @jit(float32[:,:](float32[:,:], float32[:,:], float32, int16), cache=True)
def richardson_lucy(image: np.ndarray, psf: np.ndarray, lambd: float, iterations: int,
                    blind: bool = False) -> np.ndarray:
    """Richardson-Lucy Blind and non Blind Deconvolution with Total Variation Regularization.

    Based on Matlab sourcecode of D. Perrone and P. Favaro: "Total Variation Blind Deconvolution:
    The Devil is in the Details", IEEE Conference on Computer Vision
    and Pattern Recognition (CVPR), 2014: http://www.cvg.unibe.ch/dperrone/tvdb/

    :param ndarray image : Input 3 channels image.
    :param ndarray psf : The point spread function.
    :param int iterations : Number of iterations.
    :param float lambd : Lambda parameter of the total Variation regularization
    :param bool blind : Determine if it is a blind deconvolution is launched, thus if the PSF is updated
        between two iterations
    :returns ndarray: deconvoluted image

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] http://www.cvg.unibe.ch/dperrone/tvdb/perrone2014tv.pdf
    .. [3] http://hal.archives-ouvertes.fr/docs/00/43/75/81/PDF/preprint.pdf
    .. [4] http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html
    .. [5] http://www.cs.sfu.ca/~pingtan/Papers/pami10_deblur.pdf
    """
*/
void rlucy_richardson_lucy(rlucy_image_t *image, 
     rlucy_image_t *u, 
     rlucy_image_t *psf, 
		float *lambd, const int iterations, const int blind, const int *mask, 
		convolve_data_t *fft_conv_data, double *time)
{
	rlucy_image_t grad_psf = {0};
	rlucy_image_t u_padded = {0};
	rlucy_image_t masked_image = {0};
	rlucy_image_t trim_u = {0};
  rlucy_image_t masked_u = {0};
  rlucy_image_t tmp1 = {0};
  rlucy_image_t tmp2 = {0};
	
	const int pad_v = floor(psf->h / 2);
	const int pad_h = floor(psf->w / 2);
	
	// Pad the image on each channel with data to avoid border effects
	//	u = pad_image(image, pad_v, pad_h)
	rlucy_pad_image(u, pad_h, pad_v, &u_padded, time);
	
	if (blind)
	{
	  // # Blind or myopic deconvolution
	  
		if (mask != NULL)
		{
			rlucy_trim_mask(image, &masked_image, mask[0], mask[1], mask[2], mask[3], &tmp1, time);
		}

		for (int i = 0; i < iterations; i++)
		{
			rlucy_update_image(image, &u_padded, lambd, psf, fft_conv_data, time);
      
      for (int c = 0; c < image->c; c++) lambd[c] *= .99f;

      // # Extract the portion of the source image and the deconvolved image under the mask
			if (mask != NULL)
			{
			  // masked_u = pad_image(trim_mask(unpad_image(u, pad), mask), pad).astype(np.float32)
			  // int mask_u[4] = { mask[0]+pad_h, mask[1]+pad_v, mask[2], mask[3] };
			  
				rlucy_trim_mask(&u_padded, &trim_u, mask[0]+pad_h, mask[1]+pad_v, mask[2], mask[3], &tmp1, time);
				rlucy_pad_image(&trim_u, pad_h, pad_v, &masked_u, time);
			}

			// Update blur kernel
			if (mask != NULL)
			{
			  rlucy_convolve_kernel(&masked_u, psf, &masked_image, &grad_psf, &tmp1, &tmp2, fft_conv_data, time);
			}
			else
			{
        rlucy_convolve_kernel(&u_padded, psf, image, &grad_psf, &tmp1, &tmp2, fft_conv_data, time);
			}
			
			// psf = normalize_kernel(update_values(psf, weight_update(1e-3, psf, grad_psf), grad_psf))
			float weight_upd[4] = {0.f};
			rlucy_weight_update(1e-3, psf, &grad_psf, weight_upd, time);
			rlucy_update_values(psf, weight_upd, &grad_psf, time);
			rlucy_normalize_kernel(psf, time);
		}
	}
	else
	{
	  // # Regular non-blind RL deconvolution
    for (int i = 0; i < iterations; i++)
    {
      rlucy_update_image(image, &u_padded, lambd, psf, fft_conv_data, time);
      
      for (int c = 0; c < image->c; c++) lambd[c] *= .99f;
    }
	}
	
	// u = unpad_image(u, pad)
	rlucy_unpad_image(u, pad_h, pad_v, &u_padded, time);

  rlucy_free_image(&grad_psf);
  rlucy_free_image(&u_padded);
  rlucy_free_image(&masked_image);
  rlucy_free_image(&trim_u);
  rlucy_free_image(&masked_u);
  rlucy_free_image(&tmp1);
  rlucy_free_image(&tmp2);
  
}

static void rlucy_resize_1c(rlucy_image_t *image_src, rlucy_image_t *image_dest, const float scale, const int new_w, const int new_h, 
    rlucy_image_t *im_tmp_in, rlucy_image_t *im_tmp_out)
{
  float *out = NULL;
  float *in = NULL;
  dt_iop_roi_t roi_out = {0};
  dt_iop_roi_t roi_in = {0};
  int32_t out_stride = 0;
  int32_t in_stride = 0;
  
  // in-place
  if (image_src->im == image_dest->im)
  {
    if (scale == 1.f) return;
  }
  else
  {
    if (scale == 1.f)
    {
      rlucy_alloc_image(image_dest, image_src->w, image_src->h, image_src->c);
      memcpy(image_dest->im, image_src->im, image_dest->w * image_dest->h * image_dest->c * sizeof(float));
      return;
    }
  }
  
  roi_in.width = image_src->w;
  roi_in.height = image_src->h;
  roi_in.scale = 1.0f;

  roi_out.width = new_w;
  roi_out.height = new_h;
  roi_out.scale = scale;

  out_stride = roi_out.width * 4 * sizeof(float);
  in_stride = roi_in.width * 4 * sizeof(float);

  rlucy_alloc_image(im_tmp_out, roi_out.width, roi_out.height, 4);
  rlucy_alloc_image(im_tmp_in, roi_in.width, roi_in.height, 4);
  memset(im_tmp_in->im, 0, im_tmp_in->alloc_size * sizeof(float));
  
  out = im_tmp_out->im;
  in = im_tmp_in->im;
  
  int size = image_src->w * image_src->h;
  for (int i = 0; i < size; i++)
  {
    in[i*4] = image_src->im[i];
  }
  
  const struct dt_interpolation *itor = dt_interpolation_new(DT_INTERPOLATION_USERPREF);
  
  dt_interpolation_resample(itor, out, &roi_out, out_stride, in, &roi_in, in_stride);

  rlucy_alloc_image(image_dest, roi_out.width, roi_out.height, image_src->c);
  
  size = image_dest->w * image_dest->h;
  for (int i = 0; i < size; i++)
  {
    image_dest->im[i] = out[i*4];
  }

}

static void rlucy_resize(rlucy_image_t *image_src, rlucy_image_t *image_dest, const float scale, const int new_w, const int new_h, 
    rlucy_image_t *im_tmp_in, rlucy_image_t *im_tmp_out)
{
  if (image_src->c == 1)
  {
    rlucy_resize_1c(image_src, image_dest, scale, new_w, new_h, im_tmp_in, im_tmp_out);
    return;
  }
  
  float *out = NULL;
  float *in = NULL;
  dt_iop_roi_t roi_out = {0};
  dt_iop_roi_t roi_in = {0};
  int32_t out_stride = 0;
  int32_t in_stride = 0;
  
  rlucy_image_t im_dest = {0};
  
  // in-place
  if (image_src->im == image_dest->im)
  {
    if (scale == 1.f) return;
    
    rlucy_alloc_image(&im_dest, new_w, new_h, image_src->c);
  }
  else
  {
    rlucy_alloc_image(image_dest, new_w, new_h, image_src->c);
    
    if (scale == 1.f)
    {
      memcpy(image_dest->im, image_src->im, image_dest->w * image_dest->h * image_dest->c * sizeof(float));
      return;
    }
    
    im_dest = *image_dest;
  }
  
  roi_in.width = image_src->w;
  roi_in.height = image_src->h;
  roi_in.scale = 1.0f;

  roi_out.width = im_dest.w;
  roi_out.height = im_dest.h;
  roi_out.scale = scale;

  out_stride = roi_out.width * image_src->c * sizeof(float);
  in_stride = roi_in.width * image_src->c * sizeof(float);

  out = im_dest.im;
  in = image_src->im;

  const struct dt_interpolation *itor = dt_interpolation_new(DT_INTERPOLATION_USERPREF);
  
  dt_interpolation_resample(itor, out, &roi_out, out_stride, in, &roi_in, in_stride);

  // in-place
  if (image_src->im == image_dest->im)
  {
    rlucy_alloc_image(image_dest, roi_out.width, roi_out.height, image_src->c);
    memcpy(image_dest->im, im_dest.im, image_dest->w * image_dest->h * image_dest->c * sizeof(float));
    
    rlucy_free_image(&im_dest);
  }
  else
  {
    image_dest->w = roi_out.width;
    image_dest->h = roi_out.height;
  }
  
}

static void rlucy_build_pyramid(const int kernel_size, const float lambd, const float scaling /*= 1.9*/, const float max_lambd /*= 1*/,
    int **kernels_size, float **images_scaling, float **lambdas, int *pyramid_len)
{
  int len = 1;
  float lambda_tmp = lambd;
  int kernel_tmp = kernel_size;
  
  int *arr_kernels_size = NULL;
  float *arr_images_scaling = NULL;
  float *arr_lambdas = NULL;
  
  while (lambda_tmp * scaling < max_lambd && kernel_tmp - 2 >= 3)
  {
    lambda_tmp *= scaling;
    kernel_tmp -= 2;
    
    len++;
  }
  
  arr_kernels_size = calloc(len, sizeof(int));
  arr_images_scaling = calloc(len, sizeof(float));
  arr_lambdas = calloc(len, sizeof(float));
  if (arr_kernels_size == NULL || arr_images_scaling == NULL || arr_lambdas == NULL)
    goto cleanup;
  
  arr_kernels_size[0] = kernel_size;
  arr_images_scaling[0] = 1.f;
  arr_lambdas[0] = lambd;
  
  for (int i = 1; i < len; i++)
  {
    arr_lambdas[i] = arr_lambdas[i-1] * scaling;
    arr_kernels_size[i] = arr_kernels_size[i-1] - 2;
    arr_images_scaling[i] = arr_images_scaling[i-1] / scaling;
  }
  
  if (!(arr_lambdas[len-1] < max_lambd && arr_kernels_size[len-1] >= 3) && len > 1)
    printf("rlucy_build_pyramid: error in construction, len=%i\n", len);
  
  *kernels_size = arr_kernels_size;
  *images_scaling = arr_images_scaling;
  *lambdas = arr_lambdas;
  *pyramid_len = len;
  
  return;
  
cleanup:
  if (arr_kernels_size) free(arr_kernels_size);
  if (arr_images_scaling) free(arr_images_scaling);
  if (arr_lambdas) free(arr_lambdas);
  
  *pyramid_len = 0;
  
  return;
}

static void rlucy_auto_kernel(rlucy_image_t *kern, rlucy_image_t *image, const float lambd, const int quality, 
    convolve_data_t *fft_conv_data, double *time)
{
  rlucy_image_t im = {0};
  rlucy_image_t u = {0};
  rlucy_image_t im_tmp1 = {0};
  rlucy_image_t im_tmp2 = {0};
  
  int kernel_size = kern->w;
  const float scaling = 1.9f;
  const float max_lambd = 1.f;
  
  int *kernels_size = NULL;
  float *images_scaling = NULL;
  float *lambdas = NULL;
  int pyramid_len = 0;
  
  rlucy_build_pyramid(kernel_size, lambd, scaling, max_lambd, &kernels_size, &images_scaling, &lambdas, &pyramid_len);
  if (pyramid_len <= 0) goto cleanup;
  
  rlucy_alloc_image(&u, image->w, image->h, image->c);
  if (u.im == NULL) goto cleanup;
  rlucy_copy_image(&u, image);

  rlucy_uniform_kernel(kern);

  int iterations = MAX(ceil(quality / pyramid_len), kernel_size);
  
  for (int i = pyramid_len-1; i >= 0; i--)
  {
    // # Scale the previous deblured image to the dimensions of images_scaling[i]
    
    rlucy_resize(image, &im, images_scaling[i], image->w * images_scaling[i], image->h * images_scaling[i], &im_tmp1, &im_tmp2);
    const float scale_u = (float)(im.w) / (float)(u.w);
    rlucy_resize(&u, &u, scale_u, im.w, im.h, &im_tmp1, &im_tmp2);
    
    if (im.w != u.w || im.h != u.h)
    {
      printf("rlucy_auto_kernel: invalid image size im.w=%i, im.h=%i, u.w=%i, u.h=%i\n", im.w, im.h, u.w, u.h);
      printf("rlucy_auto_kernel: images_scaling[i]=%f, scale_u=%f\n", images_scaling[i], scale_u);
    }

    // # Scale the PSF
    if (kernels_size[i] != kern->w)
    {
      rlucy_resize(kern, kern, (float)kernels_size[i] / (float)kern->w, kernels_size[i], kernels_size[i], &im_tmp1, &im_tmp2);
      
      if (kernels_size[i] != kern->w || kernels_size[i] != kern->h)
      {
        printf("rlucy_auto_kernel: invalid kernel size kernels_size[i]=%i, kern->w=%i, kern->h=%i\n", kernels_size[i], kern->w, kern->h);
      }
      
      rlucy_normalize_kernel(kern, time);
    }
    
    // # Make a blind Richardson- Lucy deconvolution on the RGB signal
    float lambd4[4] = { lambdas[i], lambdas[i], lambdas[i], lambdas[i] };
      
    rlucy_richardson_lucy(&im, &u, kern, lambd4, iterations, 1, NULL, fft_conv_data, time);
  }

cleanup:
  if (kernels_size) free(kernels_size);
  if (images_scaling) free(images_scaling);
  if (lambdas) free(lambdas);
  
  rlucy_free_image(&im);
  rlucy_free_image(&u);
  rlucy_free_image(&im_tmp1);
  rlucy_free_image(&im_tmp2);
  
}


/*
 def deblur_module(image: np.ndarray, filename: str, blur_type: str, quality: int, artifacts_damping: float,
                   deblur_strength: float, blur_width: int = 3,
                   blur_strength: int = 1, refine: bool = False, mask: np.ndarray = None, backvsmask_ratio: float = 0,
                   debug: bool = False):
     """This mimics a Darktable module inputs where the technical specs are hidden and translated into human-readable parameters

     :param image: Blured image in RGB 8 bits
     :param filename: File name to save the sharp picture
     :param blur_type: kind of blur or "auto" to perform a blind deconvolution. Use "auto" for motion blur or composite blur
     :param blur_width: the width of the blur in px
     :param blur_strength: the strength of the blur, thus the standard deviation of the blur kernel
     :param quality: the quality of the refining (ie the total number of iterations to compute). (10 to 100)
     :param artifacts_damping: the noise and artifacts reduction factor lambda. Typically between 0.00003 and 1. Increase it if smudges, noise, or ringing appear
     :param deblur_strength: the number of debluring iterations to perform,
     :param refine: True or False, decide if the blur kernel should be refined through myopic deconvolution
     :param mask: the coordinates of the rectangular mask to apply on the image to refine the blur kernel from the top-left corner of the image
         in list [y_top, y_bottom, x_left, x_right]
     :param backvsmask_ratio: when a mask is used, the ratio  of weights of the whole image / the masked zone.
         0 means only the masked zone is used, 1 means the masked zone is ignored and only the whole image is taken. 0 is
         runs faster, 1 runs much slower.
     :return:
     """
*/
void richardson_lucy_build_kernel(float *image_src, const int width, const int height, const int ch, 
    float *kernel, const int kernel_width, const int blur_strength, 
    const int blur_type, const int quality, const float artifacts_damping)
{
  double time[MAX_TIME] = {0};
  
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
  
  convolve_data_t fft_conv_data = {0};
  
  rlucy_image_t psf = {0};
  rlucy_image_t image = {0};
  
  image.im = image_src;
  image.w = width;
  image.h = height;
  image.c = ch;
  
  rlucy_alloc_image(&psf, kernel_width, kernel_width, ch);
  if (psf.im == NULL) goto cleanup;

  // build the PSF
  if (blur_type == blur_type_auto)
  {
    rlucy_auto_kernel(&psf, &image, artifacts_damping, quality, &fft_conv_data, time);
  }
  else if (blur_type == blur_type_gaussian)
  {
    rlucy_gaussian_kernel(&psf, blur_strength, time);
  }
  else if (blur_type == blur_type_kaiser)
  {
    rlucy_kaiser_kernel(&psf, blur_strength, time);
  }
  else
  {
    printf("richardson_lucy: invalid blur type\n");
  }
  
  if (psf.w != kernel_width || psf.h != kernel_width)
  {
    printf("richardson_lucy_build_kernel: invalid kernel size psf.w=%i, psf.h=%i, kernel_width=%i\n", psf.w, psf.h, kernel_width);
  }
  
  memcpy(kernel, psf.im, psf.w * psf.h * psf.c * sizeof(float));
  
  RLUCY_TIME_END(TIME_BUILD_KERNEL)
  
#ifdef RLUCY_TIME
  printf("\n");
  for (int i = 1; i < MAX_TIME; i++)
    if (time[i] > 0.0) printf("rlucy %i:%s took %0.06f sec\n", i, rlucy_time_desc[i], time[i]);
  printf("\n");
#endif

cleanup:
  
  convolve_free(&fft_conv_data, time);
              
  rlucy_free_image(&psf);
  
}
 
void richardson_lucy(float *image_src, const int width, const int height, const int ch, 
    float *kernel, const int kernel_width, 
    const int quality, const float artifacts_damping, const int deblur_strength, 
    const int refine, const int *mask, const float backvsmask_ratio)
{
  double time[MAX_TIME] = {0};
  
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
  
  convolve_data_t fft_conv_data = {0};
  
  rlucy_image_t psf = {0};
  rlucy_image_t u = {0};
  rlucy_image_t image = {0};
  
  image.im = image_src;
  image.w = width;
  image.h = height;
  image.c = ch;
  
  rlucy_alloc_image(&u, image.w, image.h, image.c);
  if (u.im == NULL) goto cleanup;
  rlucy_copy_image(&u, &image);

  rlucy_alloc_image(&psf, kernel_width, kernel_width, image.c);
  if (psf.im == NULL) goto cleanup;
  memcpy(psf.im, kernel, psf.w * psf.h * psf.c * sizeof(float));
  
  if (refine)
  {
    if (mask)
    {
      // # Masked blind or myopic deconvolution
      int iter_background = (quality * backvsmask_ratio);
      int iter_mask = (quality - iter_background);
      
      // u, psf = richardson_lucy(pic, u, psf, artifacts_damping, iter_mask, blind=True, mask=mask)
      float lambd[4] = { artifacts_damping, artifacts_damping, artifacts_damping, artifacts_damping };
      
      rlucy_richardson_lucy(&image, &u, &psf, lambd, iter_mask, 1, mask, &fft_conv_data, time);
      
      if (iter_background != 0)
      {
        // u, psf = richardson_lucy(pic, u, psf, artifacts_damping, iter_background, blind=True)
        rlucy_richardson_lucy(&image, &u, &psf, lambd, iter_background, 1, NULL, &fft_conv_data, time);
      }
    }
    else
    {
      // # Unmasked blind or myopic deconvolution
      // u, psf = richardson_lucy(pic, u, psf, artifacts_damping, quality, blind=True)
      float lambd[4] = { artifacts_damping, artifacts_damping, artifacts_damping, artifacts_damping };
      
      rlucy_richardson_lucy(&image, &u, &psf, lambd, quality, 1, NULL, &fft_conv_data, time);
    }
    
    if (deblur_strength > 0)
    {
      // # Apply a last round of non-blind optimization to enhance the sharpness
      // u, psf = richardson_lucy(pic, u, psf, artifacts_damping, deblur_strength)
      float lambd[4] = { artifacts_damping, artifacts_damping, artifacts_damping, artifacts_damping };
      
      rlucy_richardson_lucy(&image, &u, &psf, lambd, deblur_strength, 0, NULL, &fft_conv_data, time);
    }
  }
  else
  {
    // # Regular Richardson-Lucy deconvolution
    // u, psf = richardson_lucy(pic, u, psf, artifacts_damping, deblur_strength)
    float lambd[4] = { artifacts_damping, artifacts_damping, artifacts_damping, artifacts_damping };
    
    rlucy_richardson_lucy(&image, &u, &psf, lambd, deblur_strength, 0, NULL, &fft_conv_data, time);
  }
  
//  memcpy(image.im, u.im, u.w * u.h * u.c * sizeof(float));
  rlucy_copy_image(&image, &u);

  RLUCY_TIME_END(TIME_PROCESS)
  
#ifdef RLUCY_TIME
  printf("\n");
  for (int i = 1; i < MAX_TIME; i++)
    if (time[i] > 0.0) printf("rlucy %i:%s took %0.06f sec\n", i, rlucy_time_desc[i], time[i]);
  printf("\n");
#endif

cleanup:
  
  convolve_free(&fft_conv_data, time);
              
  rlucy_free_image(&psf);
  rlucy_free_image(&u);
  
}
 
void test_rlucy_auto_kernel(rlucy_image_t *image, convolve_data_t *fft_conv_data, double *time)
{
  rlucy_image_t im = {0};
  rlucy_image_t u = {0};
  rlucy_image_t im_tmp1 = {0};
  rlucy_image_t im_tmp2 = {0};
  
  int kernel_size = 5;
  const float scaling = 1.9f;
  const float max_lambd = 1.f;
  
  int *kernels_size = NULL;
  float *images_scaling = NULL;
  float *lambdas = NULL;
  int pyramid_len = 0;
  
  rlucy_build_pyramid(kernel_size, .00003f, scaling, max_lambd, &kernels_size, &images_scaling, &lambdas, &pyramid_len);
  if (pyramid_len <= 0) goto cleanup;
  
  rlucy_alloc_image(&u, image->w, image->h, image->c);
  if (u.im == NULL) goto cleanup;
  rlucy_copy_image(&u, image);

//  int iterations = MAX(ceil(quality / pyramid_len), kernel_size);
  
  for (int i = pyramid_len-1; i >= 0; i--)
  {
    // # Scale the previous deblured image to the dimensions of images_scaling[i]
    
    rlucy_resize(image, &im, images_scaling[i], image->w * images_scaling[i], image->h * images_scaling[i], &im_tmp1, &im_tmp2);
    const float scale_u = (float)(im.w) / (float)(u.w);
    rlucy_resize(&u, &u, scale_u, im.w, im.h, &im_tmp1, &im_tmp2);
    
    if (im.w != u.w || im.h != u.h)
    {
      printf("rlucy_auto_kernel: invalid image size im.w=%i, im.h=%i, u.w=%i, u.h=%i\n", im.w, im.h, u.w, u.h);
      printf("rlucy_auto_kernel: images_scaling[i]=%f, scale_u=%f\n", images_scaling[i], scale_u);
    }

  }

  if (image->w != im.w || image->h != im.h)
  {
    printf("test_rlucy_auto_kernel: invalid im size image->w=%i, image->h=%i, im.w=%i, im.h=%i\n", image->w, image->h, im.w, im.h);
  }
  if (image->w != u.w || image->h != u.h)
  {
    printf("test_rlucy_auto_kernel: invalid u size image->w=%i, image->h=%i, u.w=%i, u.h=%i\n", image->w, image->h, u.w, u.h);
  }

  rlucy_copy_image(image, &u);
  
cleanup:
  if (kernels_size) free(kernels_size);
  if (images_scaling) free(images_scaling);
  if (lambdas) free(lambdas);
  
  rlucy_free_image(&im);
  rlucy_free_image(&u);
  rlucy_free_image(&im_tmp1);
  rlucy_free_image(&im_tmp2);
  
}

void _richardson_lucy(float *image_src, const int width, const int height, const int ch, 
    const int blur_type, const int quality, const float artifacts_damping, const int deblur_strength,
    const int blur_width, const int blur_strength, const int refine, const int *mask, const float backvsmask_ratio)
{
  double time[MAX_TIME] = {0};
  
  convolve_data_t fft_conv_data = {0};

  rlucy_image_t image = {0};
  rlucy_image_t tmp = {0};
  rlucy_image_t conv_1 = {0};
  rlucy_image_t im_padded = {0};
  rlucy_image_t im_tmp_in = {0};
  rlucy_image_t im_tmp_out = {0};
  
  image.w = width;
  image.h = height;
  image.c = ch;
  image.im = image_src;
  
/*  rlucy_alloc_image(&tmp, 3, 3, ch);
  if (tmp.im == NULL) goto cleanup;
  
  int i = 0;
  if (1)
  {
    i = 0;
    for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f; for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f; for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f;
    for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f; for (int c = 0; c < ch; c++) tmp.im[i++] = 5.f; for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f;
    for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f; for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f; for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f;
  }
  if (0)
  {
    i = 0;
    for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f; for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f; for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f;
    for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f; for (int c = 0; c < ch; c++) tmp.im[i++] = 8.f; for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f;
    for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f; for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f; for (int c = 0; c < ch; c++) tmp.im[i++] = -1.f;
  }
  if (0)
  {
    i = 0;
    for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f; for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f; for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f;
    for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f; for (int c = 0; c < ch; c++) tmp.im[i++] = 1.f; for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f;
    for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f; for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f; for (int c = 0; c < ch; c++) tmp.im[i++] = 0.f;
  }
  
  const int pad_v = floor(tmp.h / 2);
  const int pad_h = floor(tmp.w / 2);
  
  rlucy_pad_image(&image, pad_h, pad_v, &im_padded, time);
  
  rlucy_convolve(&im_padded, &tmp, convolve_valid, &conv_1, fft_conv_data, 0, time);
  
  rlucy_resize(&conv_1, &tmp, .5f, conv_1.w * .5f, conv_1.h * .5f, &im_tmp_in, &im_tmp_out);
  
  rlucy_clear_image(&image);
  rlucy_copy_image(&image, &tmp);
  */
  
  test_rlucy_auto_kernel(&image, &fft_conv_data, time);
  
//cleanup:
    
  convolve_free(&fft_conv_data, time);
            
  rlucy_free_image(&tmp);
  rlucy_free_image(&conv_1);
  rlucy_free_image(&im_padded);
  rlucy_free_image(&im_tmp_in);
  rlucy_free_image(&im_tmp_out);
  
}

