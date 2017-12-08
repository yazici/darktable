
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#ifdef HAVE_FFTW3
#include <fftw3.h>
#endif

typedef enum
{
  convolve_mode_valid = 1,
  convolve_mode_full = 2
} convolve_modes_t;


#ifdef HAVE_FFTW3

typedef struct
{
  float *in_src;
  fftwf_complex *out_src;
  fftwf_plan plan_src;

  float *in_kernel;
  fftwf_complex *out_kernel;
  fftwf_plan plan_kernel;

  fftwf_plan plan_inv;

  int width_src;
  int height_src;
  int ch;

  int width_kernel;
  int height_kernel;

  int width_dest;
  int height_dest;

  int mode;

  int width_fft;
  int height_fft;
  int width_fft_complex;
  int height_fft_complex;

} fftw3_convolve_t;

#endif // HAVE_FFTW3

#ifdef HAVE_FFTW3

#define RLUCY_CONV_DATA_MAX 6

typedef struct
{
  fftw3_convolve_t *conv_kern[RLUCY_CONV_DATA_MAX];
} convolve_data_t;

#else // HAVE_FFTW3

typedef struct
{
  int dummy;
} convolve_data_t;

#endif

void convolve_pad_image(const float *const img_src, const int width_src, const int height_src, const int ch,
                        const int pad_h, const int pad_v, float *img_dest, const int width_dest,
                        const int height_dest, double *time);

void convolve_get_dest_size(const int width_src, const int height_src, const int width_kernel,
                            const int height_kernel, int *width_dest, int *height_dest, const int mode);

void convolve(convolve_data_t *fft_conv_data, const float *const src, const int width_src, const int height_src,
              const int ch, const float *const kernel, const int width_kernel, const int height_kernel,
              float *dest, const int width_dest, const int height_dest, const int mode, double *time);

void convolve_free(convolve_data_t *fft_conv_data, double *time);

#endif // CONVOLUTION_H
