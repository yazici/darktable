
#ifndef DT_DEVELOP_DWT_H
#define DT_DEVELOP_DWT_H


typedef struct dwt_params_t
{
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

typedef void(_dwt_layer_func)(float *layer, dwt_params_t *const p, const int scale);

dwt_params_t *dt_dwt_init(float *image, 
														const int width, 
														const int height, 
														const int ch, 
														const int scales, 
														const int return_layer, 
														const float blend_factor, 
														void *user_data, 
														const float preview_scale, 
														const int use_sse);
void dt_dwt_free(dwt_params_t *p);

int dwt_get_max_scale(dwt_params_t *p);

void dwt_decompose(dwt_params_t *p, _dwt_layer_func layer_func);

#ifdef HAVE_OPENCL
typedef struct dt_dwt_cl_global_t
{
  int kernel_dwt_add_img_to_layer;
  int kernel_dwt_subtract_layer;
  int kernel_dwt_hat_transform_col;
  int kernel_dwt_hat_transform_row;
  int kernel_dwt_init_buffer;
} dt_dwt_cl_global_t;

typedef struct dwt_params_cl_t
{
  dt_dwt_cl_global_t *global;
  int devid;
  cl_mem image;
  int width;
  int height;
  int ch;
  int scales;
  int return_layer;
  float blend_factor;
  void *user_data;
  float preview_scale;
} dwt_params_cl_t;

typedef cl_int(_dwt_layer_func_cl)(cl_mem layer, dwt_params_cl_t *const p, const int scale);

dt_dwt_cl_global_t *dt_dwt_init_cl_global(void);
void dt_dwt_free_cl_global(dt_dwt_cl_global_t *g);

dwt_params_cl_t *dt_dwt_init_cl(const int devid, 
																	cl_mem image, 
																	const int width, 
																	const int height, 
																	const int scales, 
																	const int return_layer, 
																	const float blend_factor, 
																	void *user_data, 
																	const float preview_scale);
void dt_dwt_free_cl(dwt_params_cl_t *p);

int dwt_get_max_scale_cl(dwt_params_cl_t *p);

cl_int dwt_decompose_cl(dwt_params_cl_t *p, _dwt_layer_func_cl layer_func);

#endif

#endif
