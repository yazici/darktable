
#ifndef DT_DEVELOP_LOCALLAPLACIAN_1_H
#define DT_DEVELOP_LOCALLAPLACIAN_1_H

//void loclap_get_L_from_rgb(float *image, float *pL, const int width, const int height, const int forward);
//void loclap_get_L_from_lab(float *image, float *pL, const int width, const int height, const int forward);

void loclap_LocalLaplacian(float *im_in, float *im_out, const int width, const int height, float alpha, float beta);

#endif

