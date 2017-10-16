
#ifndef DT_RLUCY_DEC_H
#define DT_RLUCY_DEC_H

typedef enum
{
	rlucy_type_fast = 0,
	rlucy_type_blind = 1,
	rlucy_type_myope = 2
} rlucy_process_types;


void richardson_lucy(float *image, const int width, const int height, const int ch, const int process_type);

#endif

