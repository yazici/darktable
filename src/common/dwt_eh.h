
#ifndef DT_DEVELOP_DWT_H
#define DT_DEVELOP_DWT_H


typedef struct dwt_params_t
{
  float *image;
  int ch;
  int width;
  int height;
  int width_unscale;
  int height_unscale;
  int scales;
  int return_layer;
  float blend_factor;
  void *user_data;
  float preview_scale;
  int use_sse;
} dwt_params_t;

typedef void(_dwt_layer_func)(float *layer, dwt_params_t *const p, const int scale);

int dwt_get_max_scale(dwt_params_t *p);

void dwt_decompose(dwt_params_t *p, _dwt_layer_func layer_func);

#endif
