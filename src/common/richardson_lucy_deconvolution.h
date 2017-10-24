
#ifndef DT_RLUCY_DEC_H
#define DT_RLUCY_DEC_H

typedef enum
{
	blur_type_auto = 0,
	blur_type_gaussian = 1,
	blur_type_kaiser = 2
} rlucy_blur_types;


void richardson_lucy_build_kernel(float *image_src, const int width, const int height, const int ch, 
    float *kernel, const int kernel_width, const int blur_strength, 
    const int blur_type, const int quality, const float artifacts_damping);

void richardson_lucy(float *image_src, const int width, const int height, const int ch, 
    float *kernel, const int kernel_width, 
    const int quality, const float artifacts_damping, const int deblur_strength, 
    const int refine, const int *mask, const float backvsmask_ratio);

#endif

