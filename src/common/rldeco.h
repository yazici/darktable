
#ifndef DT_RLDECO_H
#define DT_RLDECO_H

typedef enum
{
  rldeco_blur_type_auto = 0,
  rldeco_blur_type_gaussian = 1,
  rldeco_blur_type_kaiser = 2
} rldeco_blur_types;

typedef enum
{
  rldeco_method_fast = 0,
  rldeco_method_best = 1
} rldeco_methods;

void rldeco_deblur_module(float *pic_in,
    int blur_type,
    int blur_width,
    float noise_reduction_factor,
    int deblur_strength,
    int blur_strength,
    float auto_quality,
    float ringing_factor,
    int refine,
    int refine_quality,
    int *mask,
    float effect_strength,
    float *psf_in,
    int denoise,
    int method,
    float *pic_out,
    int width,
    int height,
    int ch);

#endif

