
#include <stdlib.h>
#include "develop/imageop.h"

#ifdef HAVE_FFTW3
#include "common/fftw3_convolution.h"
#endif

#include "richardson_lucy_deconvolution.h"

/*
#define TIME_CONV_FFTW3 3
#define TIME_CONV_FFT_BIG 5
#define TIME_CONV_FFT_SMALL 6
#define TIME_CONV_CPU_BIG 7
#define TIME_CONV_CPU_SMALL 8

#define MAX_TIME 9
*/

#define TIME_CONVOLVE 1
#define TIME_CONV_FFTW3_LOCK 2
#define TIME_CONV_FFTW3_EXEC 3

#define MAX_TIME 4

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
'''
import multiprocessing
from os.path import join

import numpy as np
from PIL import Image
from numba import float32, int16, jit
from scipy.signal import fftconvolve

from lib import utils
*/

typedef enum
{
	rlucy_convolve_valid = 1,
	rlucy_convolve_full = 2
} rlucy_convolve_modes;

static void rlucy_check_nan(const float* im, const int size, const char *str)
{
	int i_nan = 0;
	
	for (int i = 0; i < size; i++) if ( isnan(im[i]) ) i_nan++;
	
	if (i_nan > 0) printf("%s nan: %i\n", str, i_nan);
}


static void rlucy_img_sum(const float *const image, const dt_iop_roi_t *const roi_image, const int ch, float *sum, double *time)
{
	double start = dt_get_wtime();
	
	const int size = roi_image->width * roi_image->height * ch;

	for (int c = 0; c < ch; c++)
		sum[c] = 0.f;

	for (int i = 0; i < size; i+=ch)
	{
		for (int c = 0; c < ch; c++)
			sum[c] += image[i+c];
	}
	
	*time += dt_get_wtime() - start;
	
}

static void rlucy_img_divide(float *image, const dt_iop_roi_t *const roi_image, const int ch, float *div, double *time)
{
	double start = dt_get_wtime();
	
	const int size = roi_image->width * roi_image->height * ch;
	float mult[4] = {0};
	
	for (int c = 0; c < ch; c++)
	{
		if (div[c] == 0.f)
			mult[c] = 0.f;
		else
			mult[c] = 1.f / div[c];
	}
	
	for (int i = 0; i < size; i+=ch)
	{
		for (int c = 0; c < ch; c++)
		{
			image[i+c] *= mult[c];
		}
	}
	
	*time += dt_get_wtime() - start;
	
}

static void rlucy_img_outer(float *vector, const int size, float *outer, const int ch)
{

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			const float mult = vector[i] * vector[j];
			
			for (int c = 0; c < ch; c++)
			{
				outer[i * size * ch + j * ch + c] = mult;
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
static float rlucy_zeroethOrderBessel( float x )
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

 void rlucy_kaiser_kernel(float *kern, const dt_iop_roi_t *const roi_kern, const int ch, const float beta)
{
	double time = 0;
	
	float *win = NULL;
	const int radius = roi_kern->width;
	
	if (roi_kern->width != roi_kern->height)
		printf("rlucy_kaiser_kernel: invalid kernel size\n");
			
//	window = np.kaiser(radius, beta)
	win = calloc(radius, sizeof(float));
	if (win)
	{
		rlucy_buildKaiserWindow(win, radius, beta);
	
//	kern = np.outer(window, window)
		rlucy_img_outer(win, radius, kern, ch);
		
		float kern_sum[4] = {0};
		
		rlucy_img_sum(kern, roi_kern, ch, kern_sum, &time);
		rlucy_img_divide(kern, roi_kern, ch, kern_sum, &time);
//	kern = kern / kern.sum()
	}
	
	if (win) free(win);
}

static void rlucy_img_subtract(const float *const img_in1, const dt_iop_roi_t *const roi_in1, 
		const float *const img_in2, const dt_iop_roi_t *const roi_in2, 
		float *img_dest, const dt_iop_roi_t *const roi_dest, const int ch, double *time)
{
	double start = dt_get_wtime();
	
	const int size = roi_in1->width * roi_in1->height * ch;
	
	if (roi_in1->width != roi_in2->width || roi_in1->height != roi_in2->height)
		printf("rlucy_img_subtract: invalid image size\n");
	if (roi_in1->width != roi_dest->width || roi_in1->height != roi_dest->height)
		printf("rlucy_img_subtract: invalid dest size\n");

	for (int i = 0; i < size; i++)
	{
		img_dest[i] = img_in1[i] - img_in2[i];
	}
	
	*time += dt_get_wtime() - start;
	
}

static void rlucy_img_mulsub(const float *const img_in1, const dt_iop_roi_t *const roi_in1,
		const float *const img_in2, const dt_iop_roi_t *const roi_in2,
		float *img_dest, const dt_iop_roi_t *const roi_dest, const int ch, const float *mult, double *time)
{
	double start = dt_get_wtime();
	
	const int size = roi_in1->width * roi_in1->height * ch;
	
	if (roi_in1->width != roi_in2->width || roi_in1->height != roi_in2->height)
		printf("rlucy_img_mulsub: invalid image size\n");
	if (roi_in1->width != roi_dest->width || roi_in1->height != roi_dest->height)
		printf("rlucy_img_mulsub: invalid dest size\n");

	for (int i = 0; i < size; i+=ch)
	{
		for (int c = 0; c < ch; c++)
		{
			img_dest[i+c] = img_in1[i+c] - img_in2[i+c] * mult[c];
		}
	}
	
	*time += dt_get_wtime() - start;
	
}

static void rlucy_img_add(const float *const img_in1, const dt_iop_roi_t *const roi_in1,
		const float *const img_in2, const dt_iop_roi_t *const roi_in2,
		float *img_dest, const dt_iop_roi_t *const roi_dest, const int ch, double *time)
{
	double start = dt_get_wtime();
	
	const int size = roi_in1->width * roi_in1->height * ch;
	
	if (roi_in1->width != roi_in2->width || roi_in1->height != roi_in2->height)
		printf("rlucy_img_add: invalid image size\n");
	if (roi_in1->width != roi_dest->width || roi_in1->height != roi_dest->height)
		printf("rlucy_img_add: invalid dest size\n");

	for (int i = 0; i < size; i++)
	{
		img_dest[i] = img_in1[i] + img_in2[i];
	}
	
	*time += dt_get_wtime() - start;
	
}

static void rlucy_img_max(const float *const image, const dt_iop_roi_t *const roi_image, const int ch, float *max, double *time)
{
	double start = dt_get_wtime();
	
	const int size = roi_image->width * roi_image->height * ch;
	
	for (int c = 0; c < ch; c++)
	{
		max[c] = -INFINITY;
	}
		
	for (int i = 0; i < size; i+=ch)
	{
		for (int c = 0; c < ch; c++)
		{
			max[c] = MAX(max[c], image[i+c]);
		}
	}
	
	*time += dt_get_wtime() - start;
	
}

static void rlucy_img_absmax(const float *const image, const dt_iop_roi_t *const roi_image, const int ch, float *max, double *time)
{
	double start = dt_get_wtime();
	
	const int size = roi_image->width * roi_image->height * ch;
	
	for (int c = 0; c < ch; c++)
	{
		max[c] = 0.f;
	}
	
	for (int i = 0; i < size; i+=ch)
	{
		for (int c = 0; c < ch; c++)
			max[c] = MAX(max[c], fabs(image[i+c]));
	}
	
	*time += dt_get_wtime() - start;
	
}

static void rlucy_img_rotate90(const float *const img_in, const dt_iop_roi_t *const roi_in, 
		float *img_dest, const dt_iop_roi_t *const roi_dest, const int ch, 
		const int times, double *time)
{
	double start = dt_get_wtime();
		
	if (times == 1)
	{
		if (roi_in->width != roi_dest->height || roi_in->height != roi_dest->width)
			printf("rlucy_img_rotate90: invalid image size\n");

		for (int y = 0; y < roi_in->height; y++)
		{
			for (int x = 0; x < roi_in->width; x++)
			{
				for (int c = 0; c < ch; c++)
				{
					const int x1 = roi_dest->width - y - 1;
					const int y1 = x;
					
					img_dest[y1 * roi_dest->width * ch + x1 * ch + c] = img_in[y * roi_in->width * ch + x * ch + c];
				}
			}
		}
	}
	else if (times == 2)
	{
		if (roi_in->width != roi_dest->width || roi_in->height != roi_dest->height)
			printf("rlucy_img_rotate90: invalid image size\n");

		for (int y = 0; y < roi_in->height; y++)
		{
			for (int x = 0; x < roi_in->width; x++)
			{
				for (int c = 0; c < ch; c++)
				{
					const int x1 = roi_dest->width - x - 1;
					const int y1 = roi_dest->height - y - 1;
					
					img_dest[y1 * roi_dest->width * ch + x1 * ch + c] = img_in[y * roi_in->width * ch + x * ch + c];
				}
			}
		}
	}
	else
		printf("rlucy_img_rotate90: invalid argument times\n");
	
	*time += dt_get_wtime() - start;
	
}

/*
@jit(float32[:, :](float32[:, :], int16, int16), cache=True)
def pad_image(image: np.ndarray, pad_v: int, pad_h: int):
    R = np.pad(image[..., 0], (pad_v, pad_h), mode="edge")
    G = np.pad(image[..., 1], (pad_v, pad_h), mode="edge")
    B = np.pad(image[..., 2], (pad_v, pad_h), mode="edge")
    u = np.dstack((R, G, B))
    return u
*/

static void rlucy_pad_image(const float *const img_src, const dt_iop_roi_t *const roi_src, const int ch,
		const int pad_h, const int pad_v, 
		float **img_dest, dt_iop_roi_t * roi_dest)
{
	float *img_padded = NULL;
	
	*roi_dest = *roi_src;
	
	roi_dest->width = roi_src->width + pad_h * 2;
	roi_dest->height = roi_src->height + pad_v * 2;
	
	img_padded = dt_alloc_align(64, roi_dest->width * roi_dest->height * ch * sizeof(float));
	if (img_padded == NULL) goto cleanup;
	
	// copy image
	for (int y = 0; y < roi_src->height; y++)
	{
		const float *const src = img_src + y * roi_src->width * ch;
		float *dest = img_padded + (y + pad_v) * roi_dest->width * ch + pad_h * ch;
		
		memcpy(dest, src, roi_src->width * ch * sizeof(float));
	}
	
	// pad left and right
	for (int y = 0; y < roi_src->height; y++)
	{
		const float *const src_l = img_src + y * roi_src->width * ch;
		float *dest_l = img_padded + (y + pad_v) * roi_dest->width * ch;
		
		const float *const src_r = img_src + y * roi_src->width * ch + (roi_src->width-1) * ch;
		float *dest_r = img_padded + (y + pad_v) * roi_dest->width * ch + (roi_src->width+pad_h-1) * ch;
		
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
	float *src_t = img_padded + pad_v * roi_dest->width * ch;
	float *src_b = img_padded + (roi_src->height + pad_v - 1) * roi_dest->width * ch;
	
	for (int y = 0; y < pad_v; y++)
	{
		float *dest_t = img_padded + y * roi_dest->width * ch;
		float *dest_b = img_padded + (y + roi_src->height + pad_v) * roi_dest->width * ch;
		
		memcpy(dest_t, src_t, roi_dest->width * ch * sizeof(float));
		memcpy(dest_b, src_b, roi_dest->width * ch * sizeof(float));
	}

cleanup:
	
	*img_dest = img_padded;
}

static void rlucy_unpad_image(float * img_dest, const dt_iop_roi_t *const roi_dest, const int ch,
		const int pad_h, const int pad_v, 
		float *img_src, dt_iop_roi_t * roi_src)
{
	// copy image
	for (int y = 0; y < roi_dest->height; y++)
	{
		float *dest = img_dest + y * roi_dest->width * ch;
		float *src = img_src + (y + pad_v) * roi_src->width * ch + pad_h * ch;
		
		memcpy(dest, src, roi_dest->width * ch * sizeof(float));
	}
	
}

// from:
// https://github.com/numpy/numpy/blob/v1.13.0/numpy/lib/function_base.py#L1502-L1840
//

void rlucy_gradient(const float *const f, const dt_iop_roi_t *const roi_f, const int ch, float **outvals)
{
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
	float dx[2] = {1.0f, 1.0f};
	
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
	    yy = dt_alloc_align(64, roi_f->width * roi_f->height * ch * sizeof(float));
	    if (yy == NULL) goto cleanup;
	    
	    memcpy(yy, f, roi_f->width * roi_f->height * ch * sizeof(float));
	    
//	    for i, axis in enumerate(axes):
  for (int i = 0, axis = 0; i < len_axes; i++, axis++)
  {
/*	        if y.shape[axis] < edge_order + 1:
	            raise ValueError(
	                "Shape of array too small to calculate a numerical gradient, "
	                "at least (edge_order + 1) elements are required.")*/
	        // result allocation
//	        out = np.empty_like(y, dtype=otype)
  	float *out = dt_alloc_align(64, roi_f->width * roi_f->height * ch * sizeof(float));

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
			if (axis == 0)
			{
				for (int y = 0; y < roi_f->height; y++)
				{
					for (int x = 1; x < roi_f->width - 1; x++)
					{
						const int idx_out = y * roi_f->width * ch + x * ch;
						const int idx_f2 = y * roi_f->width * ch + (x - 1) * ch;
						const int idx_f4 = y * roi_f->width * ch + (x + 1) * ch;
						
						for (int c = 0; c < ch; c++)
						{
							out[idx_out + c] = (f[idx_f4 + c] - f[idx_f2 + c]) / (2.f * dx[i]);
						}
					}
  			}
  		}
			else
			{
				for (int y = 1; y < roi_f->height - 1; y++)
				{
					for (int x = 0; x < roi_f->width; x++)
					{
						const int idx_out = y * roi_f->width * ch + x * ch;
						const int idx_f2 = (y - 1) * roi_f->width * ch + x * ch;
						const int idx_f4 = (y + 1) * roi_f->width * ch + x * ch;
						
						for (int c = 0; c < ch; c++)
						{
							out[idx_out + c] = (f[idx_f4 + c] - f[idx_f2 + c]) / (2.f * dx[i]);
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
        float ia = -1.5f / dx[i];
        float ib = 2.f / dx[i];
        float ic = -0.5f / dx[i];

        if (axis == 0)
        {
					for (int y = 0; y < roi_f->height; y++)
					{
						const int idx_out = y * roi_f->width * ch;
						const int idx_y2 = y * roi_f->width * ch;
						const int idx_y3 = y * roi_f->width * ch + 1 * ch;
						const int idx_y4 = y * roi_f->width * ch + 2 * ch;
						
						for (int c = 0; c < ch; c++)
						{
							out[idx_out] = ia * yy[idx_y2] + ib * yy[idx_y3] + ic * yy[idx_y4];
						}
					}
        }
        else
        {
					for (int x = 0; x < roi_f->width; x++)
					{
						const int idx_out = x * ch;
						const int idx_y2 = x * ch;
						const int idx_y3 = x * ch + 1 * roi_f->width * ch;
						const int idx_y4 = x * ch + 2 * roi_f->width * ch;
						
						for (int c = 0; c < ch; c++)
						{
							out[idx_out] = ia * yy[idx_y2] + ib * yy[idx_y3] + ic * yy[idx_y4];
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
      float ia = 0.5f / dx[i];
      float ib = -2.f / dx[i];
      float ic = 1.5f / dx[i];

      if (axis == 0)
      {
				for (int y = 0; y < roi_f->height; y++)
				{
					const int idx_out = y * roi_f->width * ch + (roi_f->width - 1) * ch;
					const int idx_y2 = y * roi_f->width * ch + (roi_f->width - 3) * ch;
					const int idx_y3 = y * roi_f->width * ch + (roi_f->width - 2) * ch;
					const int idx_y4 = y * roi_f->width * ch + (roi_f->width - 1) * ch;
					
					for (int c = 0; c < ch; c++)
					{
						out[idx_out] = ia * yy[idx_y2] + ib * yy[idx_y3] + ic * yy[idx_y4];
					}
				}
      }
      else
      {
				for (int x = 0; x < roi_f->width; x++)
				{
					const int idx_out = (roi_f->height - 1) * roi_f->width * ch + x * ch;
					const int idx_y2 = (roi_f->height - 3) * roi_f->width * ch + x * ch;
					const int idx_y3 = (roi_f->height - 2) * roi_f->width * ch + x * ch;
					const int idx_y4 = (roi_f->height - 1) * roi_f->width * ch + x * ch;
					
					for (int c = 0; c < ch; c++)
					{
						out[idx_out] = ia * yy[idx_y2] + ib * yy[idx_y3] + ic * yy[idx_y4];
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
  
}

/*
@jit(float32[:, :](float32[:, :]), cache=True)
def divTV(image: np.ndarray) -> np.ndarray:
    """Compute the second-order divergence of the pixel matrix, known as the Total Variation.

    :param ndarray image: Input array of pixels
    :return: div(Total Variation regularization term) as described in [3]
    :rtype: ndarray

    References
    ----------
*/

static void rlucy_divTV(const float *const image, const dt_iop_roi_t *const roi_image, 
		float *grad, const dt_iop_roi_t *const roi_grad, const int ch, double *time)
{
	double start = dt_get_wtime();
	
	float *outvals[2] = {0};
	
/*	grad = np.array(np.gradient(image, edge_order=2))
	grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
	grad = grad / np.amax(grad)
	*/
	rlucy_gradient(image, roi_image, ch, outvals);
	if (outvals[0] == NULL || outvals[1] == NULL) goto cleanup;
	
	const int size = roi_image->width * roi_image->height * ch;
	
	for (int i = 0; i < size; i+=ch)
	{
		for (int c = 0; c < ch; c++)
		{
			grad[i + c] = sqrtf(outvals[0][i + c] * outvals[0][i + c] + outvals[1][i + c] * outvals[1][i + c]);
		}
	}

	float max_grad[4] = {0};
	rlucy_img_absmax(grad, roi_grad, ch, max_grad, time);
	rlucy_img_divide(grad, roi_grad, ch, max_grad, time);
	
	*time += dt_get_wtime() - start;
	
cleanup:
	
	if (outvals[0]) dt_free_align(outvals[0]);
	if (outvals[1]) dt_free_align(outvals[1]);
	
}

/*
static float rlucy_norm(const float *const image, const dt_iop_roi_t *const roi_image, const int ch)
{
	float norm = 0.f;
	const int ch1 = (ch == 4) ? 3: ch;
	const int size = roi_image->width * roi_image->height * ch;
	
	for (int i = 0; i < size; i+=ch)
	{
		for (int c = 0; c < ch1; c++)
		{
			norm += image[i + c] * image[i + c];
		}
	}
	
	return sqrtf(norm);
}
*/

/*
@jit(cache=True)
def convergence(image_after: np.ndarray, image_before: np.ndarray) -> float:
    """Compute the convergence rate between 2 iterations

    :param ndarray image_after: Image @ iteration n
    :param ndarray image_before: Image @ iteration n-1
    :param int padding: Number of pixels to ignore along each side
    :return float: convergence rate
    """
*/
/*
static float rlucy_convergence(const float *const image_after, const float *const image_before,
		const dt_iop_roi_t *const roi_image, const int ch)
{
	float convergence = 0.f;
	
//	convergence = np.log(np.linalg.norm(image_after) / np.linalg.norm((image_before)))
	convergence = log(rlucy_norm(image_after, roi_image, ch) / rlucy_norm(image_before, roi_image, ch));
//	print("Convergence :", convergence)
	return convergence;
}
*/

#ifdef HAVE_FFTW3

static void rlucy_convolve(const float *const image, const dt_iop_roi_t *const roi_image, const int ch, 
		const float *const kernel, const dt_iop_roi_t *const roi_kernel,
		const int mode, float *img_dest, const dt_iop_roi_t *const roi_dest, double *time)
{
	double start = dt_get_wtime();
	
	const int ch1 = (ch == 4) ? 3: ch;
	int fftw3_mode = fftw3_convolve_valid;
	
	if (mode == rlucy_convolve_full) fftw3_mode = fftw3_convolve_full;
	
  for (int c = 0; c < ch1; c++)
	{
		fftw3_convolve(image, roi_image->width, roi_image->height, ch, c,
				kernel, roi_kernel->width, roi_kernel->height, 
				img_dest, roi_dest->width, roi_dest->height, 
				fftw3_mode, &(time[TIME_CONV_FFTW3_LOCK]));
	}
	
	time[TIME_CONVOLVE] += dt_get_wtime() - start;
	
	rlucy_check_nan(img_dest, roi_dest->width * roi_dest->height * ch, "rlucy_convolve");
	
}

#else

static void rlucy_convolve_internal(const float *const image, const dt_iop_roi_t *const roi_image, const int ch, 
		const float *const kernel, const dt_iop_roi_t *const roi_kernel,
		const int mode, float *img_dest, const dt_iop_roi_t *const roi_dest, double *time)
{
	double start = dt_get_wtime();
	
	const int pad_w = (roi_kernel->width - 1) / 2;
	const int pad_h = (roi_kernel->height - 1) / 2;
	
	const int offset_x = (roi_image->width - roi_dest->width) / 2;
	const int offset_y = (roi_image->height - roi_dest->height) / 2;
	
	const int start_x = MAX(pad_w-offset_x, 0);
	const int start_y = MAX(pad_h-offset_y, 0);
	const int end_x = roi_dest->width - start_x;
	const int end_y = roi_dest->height - start_y;
	
	for (int y = start_y; y < end_y; y++)
	{
		for (int x = start_x; x < end_x; x++)
		{
			for (int c = 0; c < ch; c++)
			{
				float pixel = 0.f;
	
				for (int j = 0; j < roi_kernel->height; j++)
				{
					for (int i = 0; i < roi_kernel->width; i++)
					{
//						pixel += image[(x + i - (roi_kernel->width - 1) / 2) + (y + j - (roi_kernel->height - 1) / 2) * roi_image->width * ch] * kernel[j * roi_kernel->width + i];
						const int x1 = x + offset_x + (i-pad_w);
						const int y1 = y + offset_y + (j-pad_h);
						if (x1 >= roi_image->width) printf("x1=%i >= roi_image->width=%i\n", x1, roi_image->width);
						if (y1 >= roi_image->height) printf("y1=%i >= roi_image->height=%i\n", y1, roi_image->height);
						
						pixel += image[c + x1 * ch + y1 * roi_image->width * ch] * kernel[j * roi_kernel->width + i];
					}
				}
				
				if (x >= roi_dest->width) printf("x=%i >= roi_dest->width=%i\n", x, roi_dest->width);
				if (y >= roi_dest->height) printf("y=%i >= roi_dest->height=%i\n", y, roi_dest->height);
				
				img_dest[c + x * ch + y * roi_dest->width * ch] = pixel;
			}
		}
	}
	
//	printf("rlucy_convolve took %0.04f sec\n", dt_get_wtime() - start);
	*time += dt_get_wtime() - start;
	
	rlucy_check_nan(img_dest, roi_dest->width * roi_dest->height * ch, "rlucy_convolve");
	
}

static void rlucy_convolve_cpu(const float *const image, const dt_iop_roi_t *const roi_image, const int ch, 
		const float *const kernel, const dt_iop_roi_t *const roi_kernel,
		const int mode, float *img_dest, const dt_iop_roi_t *const roi_dest, double *time)
{
	float *im_padded = NULL;
	dt_iop_roi_t roi_padded = {0};
	
	int width_fft =  0;
	int height_fft =  0;
	int pad_v = 0;
	int pad_h = 0;

	if (mode == rlucy_convolve_full)
	{
		height_fft = roi_image->height + roi_kernel->height - 1;
    width_fft = roi_image->width + roi_kernel->width - 1;
    if (roi_dest->height != (roi_image->height + roi_kernel->height-1) || roi_dest->width != (roi_image->width + roi_kernel->width-1))
    {
    	printf("fftw3_convolve: invalid dest size\n");
  		goto cleanup;
    }
	}
	else if (mode == rlucy_convolve_valid)
	{
		height_fft = roi_image->height;
		width_fft = roi_image->width;
		if (roi_dest->height != (roi_image->height - roi_kernel->height+1) || roi_dest->width != (roi_image->width - roi_kernel->width+1))
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

	pad_v = floor((height_fft-roi_image->height) / 2);
	pad_h = floor((width_fft-roi_image->width) / 2);

	if (pad_v > 0 || pad_h > 0)
	{
		rlucy_pad_image(image, roi_image, ch, pad_h, pad_v, &im_padded, &roi_padded);
		
		rlucy_convolve_internal(im_padded, &roi_padded, ch, 
				kernel, roi_kernel,
				mode, img_dest, roi_dest, time);
	}
	else
	{
		rlucy_convolve_internal(image, roi_image, ch, 
				kernel, roi_kernel,
				mode, img_dest, roi_dest, time);
	}

	
cleanup:
	
	if (im_padded && (pad_v > 0 || pad_h > 0)) dt_free_align(im_padded);
	
}
#endif

/*
static void rlucy_convolve(const float *const image, const dt_iop_roi_t *const roi_image, const int ch, 
		const float *const kernel, const dt_iop_roi_t *const roi_kernel,
		const int mode, float *img_dest, const dt_iop_roi_t *const roi_dest, double *time)
{
	double *time_fft = NULL;
	double *time_cpu = NULL;
	
	if (roi_kernel->width > 50)
//	if (mode == rlucy_convolve_full)
	{
		time_fft = &(time[TIME_CONV_FFT_BIG]);
		time_cpu = &(time[TIME_CONV_CPU_BIG]);
	}
	else
	{
		time_fft = &(time[TIME_CONV_FFT_SMALL]);
		time_cpu = &(time[TIME_CONV_CPU_SMALL]);
	}
	
	rlucy_convolve_cpu(image, roi_image, ch, 
			kernel, roi_kernel,
			mode, img_dest, roi_dest, time_cpu);

	rlucy_convolve_fftw3(image, roi_image, ch, 
			kernel, roi_kernel,
			mode, img_dest, roi_dest, time_fft);

}
*/


/*
@jit(float32[:, :](float32[:, :], float32[:, :], int16, float32[:, :]), cache=True)
def update_image(image, u, lambd, psf):
    """Update one channel only (R, G or B)

    :param image:
    :param u:
    :param lambd:
    :param psf:
    :return:
    """
*/
static void rlucy_update_image(const float *const image, const dt_iop_roi_t *const roi_image, const int ch,
		float *u, const dt_iop_roi_t *const roi_u,
		const float *lambd, 
		const float *const psf, const dt_iop_roi_t *const roi_psf, double *time)
{
	float *conv_1 = NULL;
	float *gradUdata = NULL;
	float *gradu = NULL;
	float *psf_rotate = NULL;
	float *u_divTV = NULL;
	
	dt_iop_roi_t roi_psf_rotate = *roi_psf;
	
	conv_1 = dt_alloc_align(64, roi_image->width * roi_image->height * ch * sizeof(float));
	if (conv_1 == NULL) goto cleanup;
	
	gradUdata = dt_alloc_align(64, roi_u->width * roi_u->height * ch * sizeof(float));
	if (gradUdata == NULL) goto cleanup;
	
	gradu = dt_alloc_align(64, roi_u->width * roi_u->height * ch * sizeof(float));
	if (gradu == NULL) goto cleanup;
	
	u_divTV = dt_alloc_align(64, roi_u->width * roi_u->height * ch * sizeof(float));
	if (u_divTV == NULL) goto cleanup;
	
	roi_psf_rotate.width = roi_psf->height;
	roi_psf_rotate.height = roi_psf->width;
	
	psf_rotate = dt_alloc_align(64, roi_psf_rotate.width * roi_psf_rotate.height * ch * sizeof(float));
	if (psf_rotate == NULL) goto cleanup;
	
	// Richardson-Lucy deconvolution
//	gradUdata = fftconvolve(fftconvolve(u, psf, "valid") - image, np.rot90(psf), "full")
	rlucy_convolve(u, roi_u, ch, psf, roi_psf, rlucy_convolve_valid, conv_1, roi_image, time);
	rlucy_img_subtract(conv_1, roi_image, image, roi_image, conv_1, roi_image, ch, time);
	rlucy_img_rotate90(psf, roi_psf, psf_rotate, &roi_psf_rotate, ch, 1, time);
	rlucy_convolve(conv_1, roi_image, ch, psf_rotate, &roi_psf_rotate, rlucy_convolve_full, gradUdata, roi_u, time);
	
	// Total Variation Regularization
//	gradu = gradUdata - lambd * divTV(u)
	rlucy_divTV(u, roi_u, u_divTV, roi_u, ch, time);
	rlucy_img_mulsub(gradUdata, roi_u, u_divTV, roi_u, gradu, roi_u, ch, lambd, time);
	
//	sf = 5E-3 * np.max(u) / np.maximum(1E-31, np.amax(np.abs(gradu)))
	float max_u[4] = {0};
	rlucy_img_max(u, roi_u, ch, max_u, time);
	float max_gradu[4] = {0};
	rlucy_img_absmax(gradu, roi_u, ch, max_gradu, time);
	float sf[4] = {0};
	
	for (int c = 0; c < ch; c++)
	{
		sf[c] = 5e-3 * max_u[c] / MAX(1e-31, max_gradu[c]);
	}
	
//	u = u - sf * gradu
	rlucy_img_mulsub(u, roi_u, gradu, roi_u, u, roi_u, ch, sf, time);
	
	// Normalize for 8 bits RGB values
//    u = np.clip(u, 0.0000001, 255)

cleanup:
	
	if (conv_1) dt_free_align(conv_1);
	if (gradUdata) dt_free_align(gradUdata);
	if (gradu) dt_free_align(gradu);
	if (u_divTV) dt_free_align(u_divTV);
	if (psf_rotate) dt_free_align(psf_rotate);
	
//    return u;
}
/*
@jit(float32[:, :](float32[:, :], float32[:, :], int16, float32[:, :], int16), cache=True)
def loop_update_image(image, u, lambd, psf, iterations):
    for i in range(iterations):
        # Richardson-Lucy deconvolution
        gradUdata = fftconvolve(fftconvolve(u, psf, "valid") - image, np.rot90(psf), "full")

        # Total Variation Regularization
        gradu = gradUdata - lambd * divTV(u)
        sf = 5E-3 * np.max(u) / np.maximum(1E-31, np.amax(np.abs(gradu)))
        u = u - sf * gradu

        # Normalize for 8 bits RGB values
        u = np.clip(u, 0.0000001, 255)

        lambd = lambd / 2

    return u
*/
static void rlucy_loop_update_image(const float *const image, const dt_iop_roi_t *const roi_image, const int ch,
		float *u, const dt_iop_roi_t *const roi_u,
		float *lambd, 
		const float *const psf, const dt_iop_roi_t *const roi_psf, 
		const int iterations, double *time)
{
	for (int i = 0; i < iterations; i++)
	{
		rlucy_update_image(image, roi_image, ch, u, roi_u, lambd, psf, roi_psf, time);
		
		for (int c = 0; c < ch; c++)
			lambd[c] /= 2.f;
	}
}

/*
@jit(float32[:, :](float32[:, :], float32[:, :], float32[:, :], float32[:, :]), cache=True)
def update_kernel(gradk: np.ndarray, u: np.ndarray, psf: np.ndarray, image: np.ndarray) -> np.ndarray:
*/
static void rlucy_update_kernel(float *gradk, const dt_iop_roi_t *const roi_gradk,
		const float *const u, const dt_iop_roi_t *const roi_u,
		float *psf, const dt_iop_roi_t *const roi_psf,
		const float *const image, const dt_iop_roi_t *const roi_image, const int ch, double *time)
{
	float *conv_1 = NULL;
	float *conv_2 = NULL;
	float *u_rotate = NULL;
	
	conv_1 = dt_alloc_align(64, roi_image->width * roi_image->height * ch * sizeof(float));
	if (conv_1 == NULL) goto cleanup;
	
	conv_2 = dt_alloc_align(64, roi_gradk->width * roi_gradk->height * ch * sizeof(float));
	if (conv_2 == NULL) goto cleanup;
	
	u_rotate = dt_alloc_align(64, roi_u->width * roi_u->height * ch * sizeof(float));
	if (u_rotate == NULL) goto cleanup;
	

//	gradk = gradk + fftconvolve(np.rot90(u, 2), fftconvolve(u, psf, "valid") - image, "valid")
	rlucy_convolve(u, roi_u, ch, psf, roi_psf, rlucy_convolve_valid, conv_1, roi_image, time);
	rlucy_img_subtract(conv_1, roi_image, image, roi_image, conv_1, roi_image, ch, time);
	
	rlucy_img_rotate90(u, roi_u, u_rotate, roi_u, ch, 2, time);
	rlucy_convolve(u_rotate, roi_u, ch, conv_1, roi_image, rlucy_convolve_valid, conv_2, roi_gradk, time);
	
	rlucy_img_add(gradk, roi_gradk, conv_2, roi_gradk, gradk, roi_gradk, ch, time);
	
//	sh = 1e-3 * np.amax(psf) / np.maximum(1e-31, np.amax(np.abs(gradk)))
	float max_psf[4] = {0};
	rlucy_img_max(psf, roi_psf, ch, max_psf, time);
	float max_gradk[4] = {0};
	rlucy_img_absmax(gradk, roi_gradk, ch, max_gradk, time);
	float sh[4] = {0};
	
	for (int c = 0; c < ch; c++)
	{
		sh[c] = 1e-3 * max_psf[c] / MAX(1e-31, max_gradk[c]);
	}

//	psf = psf - sh * gradk
	rlucy_img_mulsub(psf, roi_psf, gradk, roi_gradk, psf, roi_psf, ch, sh, time);
	
//	psf = psf / np.sum(psf)
	float sum_psf[4] = {0};
	rlucy_img_sum(psf, roi_psf, ch, sum_psf, time);
	rlucy_img_divide(psf, roi_psf, ch, sum_psf, time);
	
cleanup:
	
	if (conv_1) dt_free_align(conv_1);
	if (conv_2) dt_free_align(conv_2);
	if (u_rotate) dt_free_align(u_rotate);
	
//	return psf, gradk
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
 void rlucy_richardson_lucy(float *image, const dt_iop_roi_t *const roi_image, const int ch, 
		float *psf, const dt_iop_roi_t *const roi_psf,
		float *lambd, const int iterations, const int blind)
{
	double time[MAX_TIME] = {0};
	
	dt_iop_roi_t roi_u = {0};
	dt_iop_roi_t roi_gradk = {0};
	
	float *u = NULL;
	float *gradk = NULL;
	
	// image dimensions
//	M, N, C = image.shape
//	MK, NK = psf.shape
//	pad_v = np.floor(MK / 2).astype(int)
//	pad_h = np.floor(NK / 2).astype(int)
	const int pad_v = floor(roi_psf->height / 2);
	const int pad_h = floor(roi_psf->width / 2);
	
	// Pad the image on each channel with data to avoid border effects
//	u = pad_image(image, pad_v, pad_h)
	rlucy_pad_image(image, roi_image, ch, pad_h, pad_v, &u, &roi_u);
	
	if (blind)
	{
//			gradk = np.zeros((MK, NK))
		roi_gradk = *roi_psf;
		gradk = dt_alloc_align(64, roi_gradk.width * roi_gradk.width * ch * sizeof(float));
		if (gradk == NULL) goto cleanup;
	
		memset(gradk, 0, roi_gradk.width * roi_gradk.width * ch * sizeof(float));
		
//			for i in range(iterations):
		for (int i = 0; i < iterations; i++)
		{
			// Update sharp image
/*			with multiprocessing.Pool(processes=3) as pool:
					u = np.dstack(pool.starmap(
							update_image,
							[(image[..., chan], u[..., chan], lambd, psf) for chan in range(C)]
					)
					)*/
			rlucy_update_image(image, roi_image, ch, u, &roi_u, lambd, psf, roi_psf, time);
			
			// Update blur kernel
/*			for chan in range(3):
					psf, gradk = update_kernel(gradk, u[..., chan], psf, image[..., chan])
*/
			rlucy_update_kernel(gradk, &roi_gradk, u, &roi_u, psf, roi_psf, image, roi_image, ch, time);
			
			for (int c = 0; c < ch; c++)
				lambd[c] /= 2.f;
		}
	}
	else
	{
			// Update sharp image
/*			with multiprocessing.Pool(processes=3) as pool:
					u = np.dstack(pool.starmap(
							loop_update_image,
							[(image[..., chan], u[..., chan], lambd, psf, iterations) for chan in range(C)]
					)
					)*/
		rlucy_loop_update_image(image, roi_image, ch, u, &roi_u, lambd, psf, roi_psf, iterations, time);
	}
	
	rlucy_unpad_image(image, roi_image, ch, pad_h, pad_v, u, &roi_u);

	printf("\n");
	for (int i = 0; i < MAX_TIME; i++)
		printf("rlucy %i took %0.04f sec\n", i, time[i]);
	printf("\n");
	
cleanup:
	
	if (u) dt_free_align(u);
	if (gradk) dt_free_align(gradk);
	
//	return u[pad_v:-pad_v, pad_h:-pad_h, ...], psf
}

#if 1
void richardson_lucy(float *image, const int width, const int height, const int ch, const int process_type)
{
	double time = 0;
	
	float *psf = NULL;
	
	dt_iop_roi_t roi_image = {0};
	dt_iop_roi_t roi_kernel = {0};
	
	roi_image.width = width;
	roi_image.height = height;
	
	if (process_type == rlucy_type_fast)
	{
		// Generate a blur kernel as point spread function
		roi_kernel.width = roi_kernel.height = 11;

		psf = (float*)dt_alloc_align(64, roi_kernel.width * roi_kernel.height * ch * sizeof(float));
		if (psf == NULL) goto cleanup;

		rlucy_kaiser_kernel(psf, &roi_kernel, ch, 8);
		
		// Make a non-blind Richardson- Lucy deconvolution on the RGB signal
		float lambd[4] = { 12.f, 12.f, 12.f, 12.f };
		rlucy_richardson_lucy(image, &roi_image, ch, psf, &roi_kernel, lambd, 50, 0);
	}
	else if (process_type == rlucy_type_blind)
	{
		// Generate a dumb blur kernel as point spread function
		roi_kernel.width = 7;
		roi_kernel.height = 7;
//		psf = np.ones((7, 7))
		psf = (float*)dt_alloc_align(64, roi_kernel.width * roi_kernel.height * ch * sizeof(float));
		if (psf == NULL) goto cleanup;
		
		for (int i = 0; i < roi_kernel.width * roi_kernel.height * ch; i++)
			psf[i] = 1.f;
		
//		psf /= np.sum(psf)
			float sum_psf[4] = {0};
			rlucy_img_sum(psf, &roi_kernel, ch, sum_psf, &time);
			rlucy_img_divide(psf, &roi_kernel, ch, sum_psf, &time);

		// Make a blind Richardson- Lucy deconvolution on the RGB signal
			float lambd[4] = { 0.006f, 0.006f, 0.006f, 0.006f };
			rlucy_richardson_lucy(image, &roi_image, ch, psf, &roi_kernel, lambd, 50, 1);
	}
	else if (process_type == rlucy_type_myope)
	{
		// Generate a guessed blur kernel as point spread function
		roi_kernel.width = roi_kernel.height = 11;
		
		psf = (float*)dt_alloc_align(64, roi_kernel.width * roi_kernel.height * ch * sizeof(float));
		if (psf == NULL) goto cleanup;
		
		rlucy_kaiser_kernel(psf, &roi_kernel, ch, 8);
		
		// Make a blind Richardson- Lucy deconvolution on the RGB signal
		float lambd[4] = { 1.f, 1.f, 1.f, 1.f };
		rlucy_richardson_lucy(image, &roi_image, ch, psf, &roi_kernel, lambd, 50, 1);
	}
	
cleanup:
	
	if (psf) dt_free_align(psf);
}
#else
void richardson_lucy(float *image, const int width, const int height, const int ch, const int process_type)
{
/*	double time[MAX_TIME] = {0};
	
	float *tmp = NULL;
	
	dt_iop_roi_t roi_image = {0};
	dt_iop_roi_t roi_tmp = {0};
	
	roi_image.width = width;
	roi_image.height = height;

	const int pad_v = 35;
	const int pad_h = 55;

	rlucy_pad_image(image, &roi_image, ch, pad_h, pad_v, &tmp, &roi_tmp);
	if (tmp == NULL) goto cleanup;
	rlucy_unpad_image(image, &roi_image, ch, pad_h, pad_v, tmp, &roi_tmp);

	if (tmp) dt_free_align(tmp);
	
	roi_tmp.width = height;
	roi_tmp.height = width;
	
	tmp = (float*)dt_alloc_align(64, roi_tmp.width * roi_tmp.height * ch * sizeof(float));
	if (tmp == NULL) goto cleanup;
	
	rlucy_img_rotate90(image, &roi_image, tmp, &roi_tmp, ch, 1, time);
	rlucy_img_rotate90(tmp, &roi_tmp, image, &roi_image, ch, 1, time);

cleanup:
		
	if (tmp) dt_free_align(tmp);
*/
	
	double time[MAX_TIME] = {0};
	
	float *tmp = NULL;
	float *conv_1 = NULL;
	float *im_padded = NULL;
	
	dt_iop_roi_t roi_image = {0};
	dt_iop_roi_t roi_tmp = {0};
	
	roi_image.width = width;
	roi_image.height = height;
	
	roi_tmp.width = 3;
	roi_tmp.height = 3;
	
	tmp = (float*)dt_alloc_align(64, roi_tmp.width * roi_tmp.height * ch * sizeof(float));
	if (tmp == NULL) goto cleanup;
	
	int i = 0;
	if (0)
	{
		i = 0;
		for (int c = 0; c < ch; c++) tmp[i++] = -1.f; for (int c = 0; c < ch; c++) tmp[i++] = -1.f; for (int c = 0; c < ch; c++) tmp[i++] = -1.f;
		for (int c = 0; c < ch; c++) tmp[i++] = -1.f; for (int c = 0; c < ch; c++) tmp[i++] = 8.f; for (int c = 0; c < ch; c++) tmp[i++] = -1.f;
		for (int c = 0; c < ch; c++) tmp[i++] = -1.f; for (int c = 0; c < ch; c++) tmp[i++] = -1.f; for (int c = 0; c < ch; c++) tmp[i++] = -1.f;
	}
	if (1)
	{
		i = 0;
		for (int c = 0; c < ch; c++) tmp[i++] = 0.f; for (int c = 0; c < ch; c++) tmp[i++] = 0.f; for (int c = 0; c < ch; c++) tmp[i++] = 0.f;
		for (int c = 0; c < ch; c++) tmp[i++] = 0.f; for (int c = 0; c < ch; c++) tmp[i++] = 1.f; for (int c = 0; c < ch; c++) tmp[i++] = 0.f;
		for (int c = 0; c < ch; c++) tmp[i++] = 0.f; for (int c = 0; c < ch; c++) tmp[i++] = 0.f; for (int c = 0; c < ch; c++) tmp[i++] = 0.f;
	}
	
	const int pad_v = floor(roi_tmp.height / 2);
	const int pad_h = floor(roi_tmp.width / 2);
	
	dt_iop_roi_t roi_padded = {0};
	rlucy_pad_image(image, &roi_image, ch, pad_h, pad_v, &im_padded, &roi_padded);
	
	dt_iop_roi_t roi_conv_1 = {0};

	roi_conv_1 = roi_image;
	
	conv_1 = (float*)dt_alloc_align(64, roi_conv_1.width * roi_conv_1.height * ch * sizeof(float));
	if (conv_1 == NULL) goto cleanup;
	
	rlucy_convolve(im_padded, &roi_padded, ch, tmp, &roi_tmp, fftw3_convolve_valid, conv_1, &roi_conv_1, time);
	
	memcpy(image, conv_1, roi_conv_1.width * roi_conv_1.height * ch * sizeof(float));
	
cleanup:
		
	if (tmp) dt_free_align(tmp);
	if (conv_1) dt_free_align(conv_1);
	if (im_padded) dt_free_align(im_padded);
	
}
#endif

/*
@utils.timeit
def processing_FAST(pic):
    # Open the picture
    pic = np.array(pic).astype(np.float32)
    
    # Generate a blur kernel as point spread function
    psf = utils.kaiser_kernel(11, 8)

    # Make a non-blind Richardson- Lucy deconvolution on the RGB signal
    pic, psf = richardson_lucy(pic, psf, 12, 50, blind=False)

    return pic.astype(np.uint8)

@utils.timeit
def processing_BLIND(pic):
    # Open the picture
    pic = np.array(pic).astype(np.float32)

    # Generate a dumb blur kernel as point spread function
    psf = np.ones((7, 7))
    psf /= np.sum(psf)

    # Make a blind Richardson- Lucy deconvolution on the RGB signal
    pic, psf = richardson_lucy(pic, psf, 0.006, 50, blind=True)

    return pic.astype(np.uint8)


@utils.timeit
def processing_MYOPE(pic):
    # Open the picture
    pic = np.array(pic).astype(np.float32)

    # Generate a guessed blur kernel as point spread function
    psf = utils.kaiser_kernel(11, 8)

    # Make a blind Richardson- Lucy deconvolution on the RGB signal
    pic, psf = richardson_lucy(pic, psf, 1, 50, blind=True)

    return pic.astype(np.uint8)


def save(pic, name):
    with Image.fromarray(pic) as output:
        output.save(join(dest_path, picture + "-" + name + ".jpg"),
                    format="jpeg",
                    optimize=True,
                    progressive=True,
                    quality=90)


if __name__ == '__main__':

    source_path = "img"
    dest_path = "img/richardson-lucy-deconvolution"

    images = ["blured.jpg"]

    for picture in images:

        with Image.open(join(source_path, picture)) as pic:

            """
            The "BEST" algorithm resamples the image × 2, applies the deconvolution and
            then sample it back. It's good to dilute the noise on low res images.

            The "FAST" algorithm is a direct method, more suitable for high res images
            that will be resized anyway. It's twice as fast and almost as good.
            """

            pic_fast = processing_FAST(pic)
            save(pic_fast, "fast-v3")

            pic_myope = processing_MYOPE(pic)
            save(pic_myope, "myope-v3")

            pic_blind = processing_BLIND(pic)
            save(pic_blind, "blind-v3")
*/
