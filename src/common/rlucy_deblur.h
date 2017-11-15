
#ifndef DT_RLDECO_H
#define DT_RLDECO_H

typedef enum
{
  rlucy_blur_type_auto = 0,
  rlucy_blur_type_gaussian = 1,
  rlucy_blur_type_kaiser = 2
} rlucy_blur_types;

typedef enum
{
  rlucy_method_fast = 0,
  rlucy_method_best = 1
} rlucy_methods;

void rlucy_deblur_module(float *pic_in,
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

