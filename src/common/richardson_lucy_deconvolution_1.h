
#ifndef DT_RLUCY_DEC_H
#define DT_RLUCY_DEC_H

/*
typedef enum
{
	rlucy_type_fast = 0,
	rlucy_type_blind = 1,
	rlucy_type_myope = 2
} rlucy_process_types;
*/

typedef enum
{
	blur_type_auto = 0,
	blur_type_gaussian = 1,
	blur_type_kaiser = 2
} rlucy_blur_types;


//void richardson_lucy(float *image, const int width, const int height, const int ch, const int process_type);
void richardson_lucy(float *image, const int width, const int height, const int ch, 
		const int blur_type, const int quality, const float artifacts_damping, const float deblur_strength,
		const int blur_width, const int blur_strength, const int refine, const int *mask, const float backvsmask_ratio);

#endif

