/*
    This file is part of darktable,
    copyright (c) 2018 Aurélien Pierre.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/


/** Local laplacian filter adapted from https://www.di.ens.fr/willow/pdfscurrent/Aubry14tog.pdf
 **/


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "common/opencl.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include "common/iop_group.h"

#define KERNEL_SIZE 7
#define PADDING 3
#define CHANNELS 8
#define PIXEL_CHAN 12

// For every pixel luminance, the sum of the gaussian masks
// evenly spaced by 2 EV with 1 EV std should be this constant
#define w_sum 1.7726372048266452f

/** Note :
 * we use finite-math-only and fast-math because divisions by zero are manually avoided in the code
 * fp-contract=fast enables hardware-accelerated Fused Multiply-Add
 * the rest is loop reorganization and vectorization optimization
 **/

#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("tree-vectorize", "unroll-loops", "tree-loop-if-convert", \
                      "tree-loop-distribution", "no-strict-aliasing",\
                      "loop-interchange", "loop-nest-optimize", "tree-loop-im", \
                      "unswitch-loops", "tree-loop-ivcanon", "ira-loop-pressure", \
                      "split-ivs-in-unroller", "variable-expansion-in-unroller", \
                      "split-loops", "ivopts", "predictive-commoning",\
                      "tree-loop-linear", "loop-block", "loop-strip-mine", \
                      "finite-math-only", "fp-contract=fast", "fast-math")

#pragma GCC target("sse2", "sse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2")
#endif


DT_MODULE_INTROSPECTION(1, dt_iop_toneequalizer_params_t)

typedef enum dt_iop_toneequalizer_method_t
{
  DT_TONEEQ_MEAN = 0,
  DT_TONEEQ_LIGHTNESS,
  DT_TONEEQ_VALUE,
  DT_TONEEQ_NORM_1,
  DT_TONEEQ_NORM_2,
  DT_TONEEQ_NORM_POWER
} dt_iop_toneequalizer_method_t;

typedef struct dt_iop_toneequalizer_params_t
{
  float noise, ultra_deep_blacks, deep_blacks, blacks, shadows, midtones, highlights, whites;
  float blending;
  int details;
  dt_iop_toneequalizer_method_t method;
} dt_iop_toneequalizer_params_t;

typedef struct dt_iop_toneequalizer_data_t
{
  dt_iop_toneequalizer_method_t method;
  int details;
  float blending;
  float factors[PIXEL_CHAN] __attribute__((aligned(64)));
} dt_iop_toneequalizer_data_t;

typedef struct dt_iop_toneequalizer_global_data_t
{
  int kernel_zonesystem;
} dt_iop_toneequalizer_global_data_t;


typedef struct dt_iop_toneequalizer_gui_data_t
{
  GtkWidget *noise, *ultra_deep_blacks, *deep_blacks, *blacks, *shadows, *midtones, *highlights, *whites;
  GtkWidget *blending, *scales;
  GtkWidget *method;
  GtkWidget *details;
} dt_iop_toneequalizer_gui_data_t;


const char *name()
{
  return _("tone equalizer");
}

int default_group()
{
  return IOP_GROUP_BASIC;
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}


// Build the luma channels : band-pass filters with gaussian windows of std 2 EV, spaced by 2 EV
const float centers[PIXEL_CHAN] __attribute__((aligned(64))) = {
                                                         -18.0f,
                                                         -16.0f,
                                                         -14.0f,
                                                         -12.0f,
                                                         -10.0f,
                                                         -8.0f,
                                                         -6.0f,
                                                         -4.0f,
                                                         - 2.0f,
                                                         0.0f,
                                                         2.0f,
                                                         4.0f};

/***
 * Maths
 ***/

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float square(const float x)
{
  return x * x;
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float gauss(const float x, const float mean)
{
  // gaussian coefficient un-normalized of std = 1.0
  return expf(-square(x - mean)/ 4.0f);
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float gaussian_coef(const float x, const float mean, const float std)
{
  // gaussiun coefficient normalized of arbitrary std
  return expf(-square((x - mean)/ std)/ 2.0f) / (sqrtf(2.0f * M_PI) * std);
}


/***
 *
 * Lightness map computation
 *
 ***/

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float _RGB_mean(const float pixel[4])
{
  return (pixel[0] + pixel[1] + pixel[2]) / 3.0f;
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float _RGB_value(const float pixel[4])
{
  return fmaxf(fmaxf(pixel[0], pixel[1]), pixel[2]);
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float _RGB_lightness(const float pixel[4])
{
  const float max_rgb = _RGB_value(pixel);
  const float min_rgb = fminf(pixel[0], fminf(pixel[1], pixel[2]));
  return (max_rgb + min_rgb) / 2.0f;
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float _RGB_norm_1(const float pixel[4])
{
  return pixel[0] + pixel[1] + pixel[2];
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float _RGB_norm_2(const float pixel[4])
{
  float RGB_square[4] __attribute__((aligned(16)));

#ifdef _OPENMP
#pragma omp simd
#endif
  for(int i = 0; i < 4; ++i) RGB_square[i] = square(pixel[i]);

  return sqrtf(RGB_square[0] + RGB_square[1] + RGB_square[2]);
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float _RGB_norm_power(const float pixel[4])
{
  float RGB_square[4] __attribute__((aligned(16)));
  float RGB_cubic[4] __attribute__((aligned(16)));

#ifdef _OPENMP
#pragma omp simd
#endif
  for(int i = 0; i < 4; ++i) RGB_square[i] = square(pixel[i]);

#ifdef _OPENMP
#pragma omp simd
#endif
  for(int i = 0; i < 4; ++i) RGB_cubic[i] = RGB_square[i] * pixel[i];

  return (RGB_cubic[0] + RGB_cubic[1] + RGB_cubic[2]) / (RGB_square[0] + RGB_square[1] + RGB_square[2]);
}


#ifdef _OPENMP
#pragma omp declare simd
#endif
static inline float RGB_light(const float pixel[4], const dt_iop_toneequalizer_method_t method)
{
  /** compute an estimation of the pixel lightness using several color models **/

  if(method == DT_TONEEQ_MEAN)
  {
    /** Pixel intensity **/
    return _RGB_mean(pixel);
  }

  else if(method == DT_TONEEQ_LIGHTNESS)
  {
    /** Pixel HSL lightness **/
    return _RGB_lightness(pixel);
  }

  else if(method == DT_TONEEQ_VALUE)
  {
    /** Pixel HSV value **/
    return _RGB_value(pixel);
  }

  else if(method == DT_TONEEQ_NORM_1)
  {
    /** Energy of the pixel **/
    return _RGB_norm_1(pixel);
  }

  else if(method == DT_TONEEQ_NORM_2)
  {
    /** Radius of the pixel **/
    return _RGB_norm_2(pixel);
  }

  else if(method == DT_TONEEQ_NORM_POWER)
  {
    /** Black magic : means nothing but looks good **/
    return _RGB_norm_power(pixel);
  }
  return -1.0;
}


static inline void image_luminance(const float *const restrict in, float *const restrict out,
                                   size_t width, size_t height, const dt_iop_toneequalizer_method_t method)
{
  /** Build the luminance map of an RGBA picture.
   * The output is monochrome,  **/

  const size_t ch = 4;

#ifdef _OPENMP
#pragma omp parallel for simd collapse(2) schedule(static)
#endif
  for(size_t i = 0; i < height; ++i)
  {
    for(size_t j = 0; j < width; ++j)
    {
      const size_t index = (i * width + j);
      const float *pixel_in = __builtin_assume_aligned(in + index * ch, 16); // RGBa
      float *pixel_out = __builtin_assume_aligned(out + index, 4); // single channel
      *pixel_out = RGB_light(pixel_in, method);
    }
  }
}


#ifdef _OPENMP
#pragma omp declare simd
#endif
static float compute_correction(const float factors[PIXEL_CHAN], const float exposure)
{
  // build the correction for the current pixel
  // as the sum of the contribution of each luminance channel
  float correction = 0.0f;
  float weights[PIXEL_CHAN] __attribute__((aligned(64)));

#ifdef _OPENMP
#pragma omp simd
#endif
  for (int c = 0; c < PIXEL_CHAN; ++c) weights[c] = gauss(centers[c], exposure);

#ifdef _OPENMP
#pragma omp simd reduction(+:correction)
#endif
  for (int c = 0; c < PIXEL_CHAN; ++c) correction += weights[c] * factors[c];

  return correction /= w_sum;
}


static inline void process_image(const float *const restrict in, float *const restrict out,
                                  size_t width, size_t height,
                                  const float factors[PIXEL_CHAN], const dt_iop_toneequalizer_method_t method,
                                  float *const restrict luminance)
{
  const size_t ch = 4;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t k = 0; k < (size_t)ch * width * height; k += 4 * ch)
  {
    /** Process 4 pixels at every loop step to saturate the cache and SIMD **/

    const float *const pixel_in = __builtin_assume_aligned(in + k, 64);
    float *const pixel_out = __builtin_assume_aligned(out + k, 64);
    float *const pixel_lum = (luminance) ? __builtin_assume_aligned(luminance + k / ch, 16) : NULL;

    // Get the lightness/luminance estimator
    float luma[4] __attribute__((aligned(16)));

#ifdef _OPENMP
#pragma omp simd
#endif
    for(int i = 0; i < 4; ++i) luma[i] = RGB_light(pixel_in + i * ch, method);

    // Save the luminance map - used in laplacian processing only
    if(luminance)
    {
#ifdef _OPENMP
#pragma omp simd
#endif
      for(int i = 0; i < 4; ++i) pixel_lum[i] = luma[i];
    }

    // Get the exposure
    float exposure[4] __attribute__((aligned(16)));

#ifdef _OPENMP
#pragma omp simd
#endif
    for(int i = 0; i < 4; ++i) exposure[i] = fmaxf(log2f(luma[i]), -20.0f);

    // Get the correction
    float correction[4] __attribute__((aligned(16)));

#ifdef _OPENMP
#pragma omp simd
#endif
    for(int i = 0; i < 4; ++i) correction[i] = compute_correction(factors, exposure[i]);

    // Apply the correction
#ifdef _OPENMP
#pragma omp simd
#endif
    for(int i = 0; i < 16; ++i) pixel_out[i] = correction[i / ch] * pixel_in[i];
  }
}



static inline void laplacian_filter(  const float *const restrict in_toned,
                                      const float *const restrict luminance_in,
                                      const float *const restrict luminance_toned,
                                      float *const restrict out,
                                      size_t width, size_t height,
                                      const float kernel[KERNEL_SIZE][KERNEL_SIZE])
{
  const size_t ch = 4;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for(size_t i = PADDING; i < height - PADDING - 1; ++i)
  {
    for(size_t j = PADDING; j < width - PADDING - 1; ++j)
    {
      // Monochrome pictures - luminance maps
      size_t index = (i * width + j);
      const float *const pixel_lum_in = __builtin_assume_aligned(luminance_in + index, 4);
      const float *const pixel_lum_toned = __builtin_assume_aligned(luminance_toned + index, 4);

      // RGBa pictures
      index *= ch;
      const float *const pixel_in = __builtin_assume_aligned(in_toned + index, 16);
      float *const pixel_out = __builtin_assume_aligned(out + index, 16);

      // Values
      const float in_value = *pixel_lum_in ;
      const float toned_value = *pixel_lum_toned;

      // Convolution filter
      float weight = 0.0f;

#ifdef _OPENMP
#pragma omp simd reduction(+:weight)
#endif
      for(size_t m = 0; m < KERNEL_SIZE; ++m)
      {
        for(size_t n = 0; n < KERNEL_SIZE; ++n)
        {
          const size_t index_2 = (- PADDING + n) + (- PADDING + m) * width;

          // Neighbours
          const float *const neighbour_in = __builtin_assume_aligned(pixel_lum_in + index_2, 4);
          const float *const neighbour_toned  = __builtin_assume_aligned(pixel_lum_toned + index_2, 4);

          // Values - We need to copy them to use hardware FMA
          const float in_nei_value = *neighbour_in;
          const float toned_nei_value = *neighbour_toned;

          weight += (toned_nei_value - toned_value) * (in_nei_value - in_value) * kernel[m][n];
        }
      }

      // Corrective ratio to apply on the toned image
      const float ratio =  toned_value / (toned_value + weight);

#ifdef _OPENMP
#pragma omp simd
#endif
      for(int c = 0; c < 4; ++c) pixel_out[c] = pixel_in[c] * ratio;
    }
  }
}



static inline void process_scale( const float *const restrict in,
                                  float *const restrict out,
                                  size_t width, size_t height,
                                  const float kernel[KERNEL_SIZE][KERNEL_SIZE],
                                  const float factors[PIXEL_CHAN],
                                  const dt_iop_toneequalizer_method_t method)
{
  const size_t ch = 4;

  float *in_toned = NULL;
  float *luma_in = NULL;
  float *luma_toned = NULL;

  in_toned = dt_alloc_align(64, width * height * ch * sizeof(float));
  luma_in = dt_alloc_align(64, width * height * sizeof(float));
  luma_toned = dt_alloc_align(64, width * height * sizeof(float));

  if(in_toned == NULL || luma_in == NULL || luma_toned == NULL)
  {
    dt_control_log(_("tone equalizer : unable to allocate memory"));
    goto clean;
  }

  process_image(in, in_toned, width, height, factors, method, luma_in);
  image_luminance(in_toned, luma_toned, width, height, method);
  laplacian_filter(in_toned, luma_in, luma_toned, out, width, height, kernel);

clean:
  dt_free_align(in_toned);
  dt_free_align(luma_in);
  dt_free_align(luma_toned);
  return;
}


static inline void pad_image( const float *const restrict in,
                               float *const restrict out,
                               size_t width, size_t height)
{
  /**
   * expand an RGBa input image of size (width × height × ch)
   * to an output image of size ((width + 2 × padding) × (height + 2 × padding) × ch)
   * using a periodic boundary condition
   **/

  const size_t ch = 4;
  const size_t corrected_width = (width + 2 * PADDING);

  // Duplicate the valid region shifted of (padding ; padding)
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
#endif
  for(size_t i = 0; i < height; ++i)
  {
    for(size_t j = 0; j < width; ++j)
    {
      // Note : i and j are relative to the smaller (input) space
      const size_t i_padded = i + PADDING;
      const size_t j_padded = j + PADDING;
      size_t index = (i * width + j) * ch;
      size_t index_padded = (i_padded * corrected_width + j_padded) * ch;
      const float *const pixel_in  = __builtin_assume_aligned(in + index, 16);
      float *const pixel_out = __builtin_assume_aligned(out + index_padded, 16);

#ifdef _OPENMP
#pragma omp simd
#endif
      for(int c = 0; c < 4; ++c) pixel_out[c] = pixel_in[c];
    }
  }

  // Symmetrize top rows
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
#endif
  for(size_t i = 1; i < PADDING; ++i)
  {
    for(size_t j = 0; j < width; ++j)
    {
      // Note : i and j are relative to the smaller (input) space
      const size_t i_padded = PADDING - i;
      const size_t j_padded = j + PADDING;
      size_t index = (i * width + j) * ch;
      size_t index_padded = (i_padded * corrected_width + j_padded) * ch;
      const float *const pixel_in  = __builtin_assume_aligned(in + index, 16);
      float *const pixel_out = __builtin_assume_aligned(out + index_padded, 16);

#ifdef _OPENMP
#pragma omp simd
#endif
      for(int c = 0; c < 4; ++c) pixel_out[c] = pixel_in[c];
    }
  }

  // Symmetrize bottom rows
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
#endif
  for(size_t i = height - PADDING - 1; i < height; ++i)
  {
    for(size_t j = 0; j < width; ++j)
    {
      // Note : i and j are relative to the smaller (input) space
      const size_t i_padded = 2 * height - i;
      const size_t j_padded = j + PADDING;
      size_t index = (i * width + j) * ch;
      size_t index_padded = (i_padded * corrected_width + j_padded) * ch;
      const float *const pixel_in  = __builtin_assume_aligned(in + index, 16);
      float *const pixel_out = __builtin_assume_aligned(out + index_padded, 16);

#ifdef _OPENMP
#pragma omp simd
#endif
      for(int c = 0; c < 4; ++c) pixel_out[c] = pixel_in[c];
    }
  }

  // Symmetrize left columns
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
#endif
  for(size_t i = 0; i < height; ++i)
  {
    for(size_t j = 1; j < PADDING; ++j)
    {
      // Note : i and j are relative to the smaller (input) space
      const size_t i_padded = i + PADDING;
      const size_t j_padded = PADDING - j;
      size_t index = (i * width + 0) * ch;
      size_t index_padded = (i_padded * corrected_width + j_padded) * ch;
      const float *const pixel_in  = __builtin_assume_aligned(in + index, 16);
      float *const pixel_out = __builtin_assume_aligned(out + index_padded, 16);

#ifdef _OPENMP
#pragma omp simd
#endif
      for(int c = 0; c < 4; ++c) pixel_out[c] = pixel_in[c];
    }
  }

  // Symmetrize right columns
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
#endif
  for(size_t i = 0; i < height; ++i)
  {
    for(size_t j = width - PADDING - 1; j < width; ++j)
    {
      // Note : i and j are relative to the smaller (input) space
      const size_t i_padded = i + PADDING;
      const size_t j_padded = 2 * width - j;
      size_t index = (i * width + j) * ch;
      size_t index_padded = (i_padded * corrected_width + j_padded) * ch;
      const float *const pixel_in  = __builtin_assume_aligned(in + index, 16);
      float *const pixel_out = __builtin_assume_aligned(out + index_padded, 16);

#ifdef _OPENMP
#pragma omp simd
#endif
      for(int c = 0; c < 4; ++c) pixel_out[c] = pixel_in[c];
    }
  }
}


static inline void unpad_image( const float *const restrict in,
                                float *const restrict out,
                                size_t width, size_t height)
{
  /**
   * reduce an RGBa input image of size ((width + 2 × padding) × (height + 2 × padding) × ch)
   * to an output image of size (width × height × ch)
   **/

  const size_t ch = 4;
  const size_t corrected_width = (width + 2 * PADDING);

  // Duplicate the inner valid region shifted by (padding;padding)
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
#endif
  for(size_t i = 0; i < height; ++i)
  {
    for(size_t j = 0; j < width; ++j)
    {
      // Note : i and j are relative to the smaller (output) space
      const size_t i_padded = i + PADDING;
      const size_t j_padded = j + PADDING;

      size_t index = (i * width + j) * ch;
      size_t index_padded = (i_padded * corrected_width + j_padded) * ch;

      const float *const pixel_in  = __builtin_assume_aligned(in + index_padded, 16);
      float *const pixel_out = __builtin_assume_aligned(out + index, 16);

#ifdef _OPENMP
#pragma omp simd
#endif
      for(int c = 0; c < 4; ++c) pixel_out[c] = pixel_in[c];
    }
  }
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static inline void normalize_kernel(float kernel[KERNEL_SIZE])
{
  float sum = 0.0f;

#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
  for(int i = 0; i < KERNEL_SIZE; ++i) sum += kernel[i];

#ifdef _OPENMP
#pragma omp simd
#endif
  for(int i = 0; i < KERNEL_SIZE; ++i) kernel[i] /= sum;
}


void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  const dt_iop_toneequalizer_data_t *const d = (const dt_iop_toneequalizer_data_t *const)piece->data;

  const float *const in  = (const float *const)ivoid;
  float *const out  = (float *const)ovoid;

  assert(piece->colors == 4);

  if(!d->details)
  {
    process_image(in, out, roi_out->width, roi_out->height, d->factors, d->method, NULL);
  }
  else
  {
    const float scale = piece->iscale / roi_in->scale;
    const float sigma = d->blending * scale;

    float gauss_1D[KERNEL_SIZE] __attribute__((aligned(64)));
    for(int m = 0; m < KERNEL_SIZE; ++m)
      gauss_1D[m] = gaussian_coef((float)(m - PADDING), 0.0f, sigma);

    normalize_kernel(gauss_1D);

    float gauss_kernel[KERNEL_SIZE][KERNEL_SIZE] __attribute__((aligned(64)));
    for(int m = 0; m < KERNEL_SIZE; ++m)
      for(int n = 0; n < KERNEL_SIZE; ++n)
        gauss_kernel[m][n] = gauss_1D[m] * gauss_1D[n];

    const size_t width_padded = roi_out->width + 2 * PADDING;
    const size_t height_padded = roi_out->height + 2 * PADDING;
    const size_t ch = 4;

    float *in_padded = dt_alloc_align(64, width_padded * height_padded * ch * sizeof(float));
    float *out_padded = dt_alloc_align(64, width_padded * height_padded * ch * sizeof(float));

    pad_image(in, in_padded, roi_out->width, roi_out->height);
    process_scale(in_padded, out_padded, width_padded, height_padded, gauss_kernel, d->factors, d->method);
    unpad_image(out_padded, out, roi_out->width, roi_out->height);

    dt_free_align(in_padded);
    dt_free_align(out_padded);
  }

  if(piece->pipe->mask_display & DT_DEV_PIXELPIPE_DISPLAY_MASK) dt_iop_alpha_copy(in, out, roi_in->width, roi_in->height);

}

#if defined(__GNUC__)
#pragma GCC pop_options
#endif

void init_global(dt_iop_module_so_t *module)
{
  dt_iop_toneequalizer_global_data_t *gd
      = (dt_iop_toneequalizer_global_data_t *)malloc(sizeof(dt_iop_toneequalizer_global_data_t));
  module->data = gd;
}

void cleanup_global(dt_iop_module_so_t *module)
{
  free(module->data);
  module->data = NULL;
}

void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)p1;
  dt_iop_toneequalizer_data_t *d = (dt_iop_toneequalizer_data_t *)piece->data;

  float factors[PIXEL_CHAN] __attribute__((aligned(64))) = {0.0f,             // -18 EV
                                                    p->noise,             // -16 EV
                                                    p->noise,             // -14 EV
                                                    p->ultra_deep_blacks, // -12 EV
                                                    p->deep_blacks,       // -10 EV
                                                    p->blacks,            //  -8 EV
                                                    p->shadows,           //  -6 EV
                                                    p->midtones,          //  -4 EV
                                                    p->highlights,        //  -2 EV
                                                    p->whites,            //  0 EV
                                                    p->whites,            //  2 EV
                                                    0.0f};           //  4 EV

#ifdef _OPENMP
#pragma omp simd
#endif
  for(int c = 0; c < PIXEL_CHAN; ++c) d->factors[c] = exp2f(factors[c]);

  d->blending = p->blending;
  d->method = p->method;
  d->details = p->details;
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_toneequalizer_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(struct dt_iop_module_t *self)
{
  dt_iop_module_t *module = (dt_iop_module_t *)self;
  dt_iop_toneequalizer_gui_data_t *g = (dt_iop_toneequalizer_gui_data_t *)self->gui_data;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)module->params;

  dt_bauhaus_slider_set(g->noise, p->noise);
  dt_bauhaus_slider_set(g->ultra_deep_blacks, p->ultra_deep_blacks);
  dt_bauhaus_slider_set(g->deep_blacks, p->deep_blacks);
  dt_bauhaus_slider_set(g->blacks, p->blacks);
  dt_bauhaus_slider_set(g->shadows, p->shadows);
  dt_bauhaus_slider_set(g->midtones, p->midtones);
  dt_bauhaus_slider_set(g->highlights, p->highlights);
  dt_bauhaus_slider_set(g->whites, p->whites);

  dt_bauhaus_combobox_set(g->details, p->details);
  dt_bauhaus_combobox_set(g->method, p->method);
  dt_bauhaus_slider_set(g->blending, p->blending);
}

void init(dt_iop_module_t *module)
{
  module->params = calloc(1, sizeof(dt_iop_toneequalizer_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_toneequalizer_params_t));
  module->default_enabled = 0;
  module->priority = 158; // module order created by iop_dependencies.py, do not edit!
  module->params_size = sizeof(dt_iop_toneequalizer_params_t);
  module->gui_data = NULL;
  dt_iop_toneequalizer_params_t tmp = (dt_iop_toneequalizer_params_t){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, FALSE, DT_TONEEQ_NORM_2};
  memcpy(module->params, &tmp, sizeof(dt_iop_toneequalizer_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_toneequalizer_params_t));
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

static void noise_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->noise = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void ultra_deep_blacks_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->ultra_deep_blacks = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void deep_blacks_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->deep_blacks = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void blacks_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->blacks = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void shadows_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->shadows = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void midtones_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->midtones = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void highlights_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->highlights = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void whites_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->whites = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void blending_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->blending = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void method_changed(GtkWidget *widget, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->method = dt_bauhaus_combobox_get(widget);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void details_changed(GtkWidget *widget, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->details = dt_bauhaus_combobox_get(widget);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_init(struct dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_toneequalizer_gui_data_t));
  dt_iop_toneequalizer_gui_data_t *g = (dt_iop_toneequalizer_gui_data_t *)self->gui_data;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_GUI_IOP_MODULE_CONTROL_SPACING);

  const float top = 4.0;
  const float bottom = -4.0;

  g->noise = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->noise, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->noise, NULL, _("-14 EV : HDR noise"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->noise, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->noise), "value-changed", G_CALLBACK(noise_callback), self);

  g->ultra_deep_blacks = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->ultra_deep_blacks, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->ultra_deep_blacks, NULL, _("-12 EV : HDR ultra-deep blacks"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->ultra_deep_blacks, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->ultra_deep_blacks), "value-changed", G_CALLBACK(ultra_deep_blacks_callback), self);

  g->deep_blacks = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->deep_blacks, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->deep_blacks, NULL, _("-10 EV : HDR deep blacks"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->deep_blacks, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->deep_blacks), "value-changed", G_CALLBACK(deep_blacks_callback), self);

  g->blacks = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->blacks, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->blacks, NULL, _("-08 EV : blacks"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->blacks, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->blacks), "value-changed", G_CALLBACK(blacks_callback), self);

  g->shadows = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->shadows, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->shadows, NULL, _("-06 EV : shadows"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->shadows, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->shadows), "value-changed", G_CALLBACK(shadows_callback), self);

  g->midtones = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->midtones, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->midtones, NULL, _("-04 EV : midtones"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->midtones, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->midtones), "value-changed", G_CALLBACK(midtones_callback), self);

  g->highlights = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->highlights, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->highlights, NULL, _("-02 EV : highlights"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->highlights, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->highlights), "value-changed", G_CALLBACK(highlights_callback), self);

  g->whites = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->whites, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->whites, NULL, _("-00 EV : whites"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->whites, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->whites), "value-changed", G_CALLBACK(whites_callback), self);

  g->method = dt_bauhaus_combobox_new(NULL);
  dt_bauhaus_widget_set_label(g->method, NULL, _("color model"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->method, TRUE, TRUE, 0);
  dt_bauhaus_combobox_add(g->method, "RGB mean (intensity)");
  dt_bauhaus_combobox_add(g->method, "RGB mean of extrema (lightness)");
  dt_bauhaus_combobox_add(g->method, "RGB max (value)");
  dt_bauhaus_combobox_add(g->method, "RGB norm L1 (energy)");
  dt_bauhaus_combobox_add(g->method, "RGB norm L2 (radius)");
  dt_bauhaus_combobox_add(g->method, "RGB power norm (magic)");
  g_signal_connect(G_OBJECT(g->method), "value-changed", G_CALLBACK(method_changed), self);

  g->details = dt_bauhaus_combobox_new(NULL);
  dt_bauhaus_widget_set_label(g->details, NULL, _("details preservation"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->details, TRUE, TRUE, 0);
  dt_bauhaus_combobox_add(g->details, "none (fast)");
  dt_bauhaus_combobox_add(g->details, "laplacian (slow)");
  g_signal_connect(G_OBJECT(g->details), "value-changed", G_CALLBACK(details_changed), self);

  g->blending = dt_bauhaus_slider_new_with_range(self, 0.5, 35.0, 0.1, 5.0, 2);
  dt_bauhaus_slider_set_format(g->blending, "%.2f px");
  dt_bauhaus_widget_set_label(g->blending, NULL, _("details size"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->blending, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->blending), "value-changed", G_CALLBACK(blending_callback), self);
}

void gui_cleanup(struct dt_iop_module_t *self)
{
  self->request_color_pick = DT_REQUEST_COLORPICK_OFF;
  free(self->gui_data);
  self->gui_data = NULL;
}


// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
