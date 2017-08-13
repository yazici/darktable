
#ifndef DT_DEVELOP_LOCALLAPLACIAN_DEN_1_H
#define DT_DEVELOP_LOCALLAPLACIAN_DEN_1_H

//void loclapden_get_L_from_rgb(float *image, float *pL, const int width, const int height, const int forward);
//void loclapden_get_L_from_lab(float *image, float *pL, const int width, const int height, const int forward);

void loclapden_LocalLaplacian(float *im_in, float *im_out, const int width, const int height, 
															const float alpha, const float beta, const float *const scales, const float preview_scale, const int use_sse);
void loclapden_LocalLaplacianDen(float *im_in, float *im_out, const int width, const int height, 
															const float _alpha, const float beta, const float *const scales, const float _preview_scale, const int use_sse);

#endif

