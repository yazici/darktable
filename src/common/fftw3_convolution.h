
#ifndef CONVOLUTION_FFTW_H
#define CONVOLUTION_FFTW_H

#ifdef HAVE_FFTW3

#include <fftw3.h>

typedef enum
{
	fftw3_convolve_valid = 1,
	fftw3_convolve_full = 2
} fftw3_convolve_modes;

void fftw3_convolve(const float *const src, const int width_src, const int height_src, const int ch, const int channel, 
		const float *const kernel, const int width_kernel, const int height_kernel, 
		float *dest, const int width_dest, const int height_dest, 
		const int mode, double *time);

#endif

#endif
