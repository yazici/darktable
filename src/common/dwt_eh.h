
#ifndef DT_DEVELOP_DWT_H
#define DT_DEVELOP_DWT_H


typedef struct dwt_params_t
{
  float *layers; // buffer for internal use
  float *image;
  int ch;
  int width;
  int height;
  int scales;
  int return_layer;
  float blend_factor;
  void *user_data;
  float preview_scale;
  int use_sse;
} dwt_params_t;

typedef void(_dwt_layer_func)(float *layer, dwt_params_t *p, const int scale);

int dwt_get_max_scale(const int width, const int height, const float preview_scale);

void dwt_decompose(dwt_params_t *p, _dwt_layer_func layer_func);

#endif
