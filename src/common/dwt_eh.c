
#include "control/control.h"
#include "develop/imageop.h"
#include "dwt_eh.h"

/* Based on the original source code of GIMP's Wavelet Decompose plugin, by Marco Rossini 
 * 
 * http://registry.gimp.org/node/11742
 * 
 * */

//#define _FFT_MULTFR_

int dwt_get_max_scale(const int width, const int height, const float preview_scale)
{
  int maxscale = 0;
  
  /* smallest edge must be higher than or equal to 2^scales */
  unsigned int size = MIN(width, height);
  while ((size >>= 1) * preview_scale) maxscale++;

  return maxscale;
} 

#define INDEX_WT_IMAGE(index) ((index)*p->ch)+c
#define INDEX_WT_IMAGE_SSE(index) ((index)*p->ch)

/* code copied from UFRaw (which originates from dcraw) */
#if defined(__SSE__)
static void dwt_hat_transform_sse(float *temp, float *const base, const int st, const int size, int sc, dwt_params_t *const p)
{
  int i;
  const __m128 hat_mult = _mm_set1_ps(2.f);
  __m128 valb_1, valb_2, valb_3;
  sc = (int)(sc * p->preview_scale);
  if (sc > size) sc = size;

  for (i = 0; i < sc; i++, temp +=4)
  {
    valb_1 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * i)]);
    valb_2 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (sc - i))]);
    valb_3 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (i + sc))]);
    
    _mm_store_ps(temp, _mm_add_ps(_mm_add_ps(_mm_mul_ps(hat_mult, valb_1), valb_2), valb_3));
  }
  for (; i + sc < size; i++, temp +=4)
  {
      valb_1 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * i)]);
      valb_2 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (i - sc))]);
      valb_3 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (i + sc))]);
      
      _mm_store_ps(temp, _mm_add_ps(_mm_add_ps(_mm_mul_ps(hat_mult, valb_1), valb_2), valb_3));
  }
  for (; i < size; i++, temp +=4)
  {
      valb_1 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * i)]);
      valb_2 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (i - sc))]);
      valb_3 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (2 * size - 2 - (i + sc)))]);
      
      _mm_store_ps(temp, _mm_add_ps(_mm_add_ps(_mm_mul_ps(hat_mult, valb_1), valb_2), valb_3));
  }
  
}
#endif

static void dwt_hat_transform(float *temp, float *const base, const int st, const int size, int sc, dwt_params_t *const p)
{
  int i, c;
  const float hat_mult = 2.f;
  sc = (int)(sc * p->preview_scale);
  if (sc > size) sc = size;

  for (i = 0; i < sc; i++)
  {
    for (c = 0; c < p->ch; c++, temp++)
    {
      *temp = hat_mult * base[INDEX_WT_IMAGE(st * i)] + base[INDEX_WT_IMAGE(st * (sc - i))] + base[INDEX_WT_IMAGE(st * (i + sc))];
    }
  }
  for (; i + sc < size; i++)
  {
    for (c = 0; c < p->ch; c++, temp++)
    {
      *temp = hat_mult * base[INDEX_WT_IMAGE(st * i)] + base[INDEX_WT_IMAGE(st * (i - sc))] + base[INDEX_WT_IMAGE(st * (i + sc))];
    }
  }
  for (; i < size; i++)
  {
    for (c = 0; c < p->ch; c++, temp++)
    {
      *temp = hat_mult * base[INDEX_WT_IMAGE(st * i)] + base[INDEX_WT_IMAGE(st * (i - sc))]
                                                             + base[INDEX_WT_IMAGE(st * (2 * size - 2 - (i + sc)))];
    }
  }
  
}

#if defined(__SSE__)
void dwt_add_layer_sse(float *img, dwt_params_t *const p, const int n_scale)
{
  float *l = p->layers;
  
  if (n_scale == p->scales+1)
  {
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
#endif
    for (int i=0; i<p->width*p->height; i++, l+=4, img+=4)
    {
      _mm_store_ps(l, _mm_add_ps(_mm_load_ps(l), _mm_load_ps(img)));
    }
  }
  else
  {
    const __m128 lpass_sub = _mm_set1_ps(p->blend_factor /*.128f*/);
    
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
#endif
    for (int i=0; i<p->width*p->height; i++, l+=4, img+=4)
    {
      _mm_store_ps(l, _mm_add_ps(_mm_load_ps(l), _mm_sub_ps(_mm_load_ps(img), lpass_sub)));
    }
  }

}
#endif

void dwt_add_layer(float *const img, dwt_params_t *const p, const int n_scale)
{
  const float lpass_sub = p->blend_factor; //.128f;
  
  if (n_scale == p->scales+1)
  {
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
#endif
    for (int i=0; i<p->width*p->height*p->ch; i++)
      p->layers[i] += img[i];
  }
  else
  {
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
#endif
    for (int i=0; i<p->width*p->height*p->ch; i++)
      p->layers[i] += img[i] - lpass_sub;
  }

}

void dwt_get_image_layer(float *const layer, dwt_params_t *const p)
{
  if (p->image == layer) return;
  
  memcpy(p->image, layer, p->width * p->height* p->ch * sizeof(float));
}

/* actual decomposing algorithm */
#if defined(__SSE__)
void dwt_wavelet_decompose_sse(float *img, dwt_params_t *const p, _dwt_layer_func layer_func)
{
  float *temp = NULL;
  unsigned int lpass, hpass;
  float *buffer[2] = {0, 0};
  int bcontinue = 1;
  
  const __m128 v4_lpass_add = _mm_set1_ps(0.f);
  const __m128 v4_lpass_mult = _mm_set1_ps((1.f / 16.f));
  const __m128 v4_lpass_sub = _mm_set1_ps(p->blend_factor);
  

  const int size = p->width * p->height * p->ch;

  /* image buffers */
  buffer[0] = img;
  /* temporary storage */
  buffer[1] = dt_alloc_align(64, size * sizeof(float));
  if (buffer[1] == NULL)
  {
    dt_control_log(_("not enough memory for wavelet decomposition"));
    goto cleanup;
  }

  temp = dt_alloc_align(64, MAX(p->width, p->height) * p->ch * sizeof(float));
  if (temp == NULL)
  {
    dt_control_log(_("not enough memory for wavelet decomposition"));
    goto cleanup;
  }

  /* iterate over wavelet scales */
  lpass = 1;
  hpass = 0;
  for (unsigned int lev = 0; lev < p->scales && bcontinue; lev++) 
  {
    lpass = (1 - (lev & 1));

#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(buffer, temp, lev, lpass, hpass) schedule(static)
#endif
#endif
    for (int row = 0; row < p->height; row++) 
    {
      dwt_hat_transform_sse(temp, buffer[hpass] + (row * p->width * p->ch), 1, p->width, 1 << lev, p);
      memcpy(&(buffer[lpass][row * p->width * p->ch]), temp, p->width * p->ch * sizeof(float));
    }
      
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(buffer, temp, lev, lpass, hpass) schedule(static)
#endif
#endif
    for (int col = 0; col < p->width; col++) 
    {
      dwt_hat_transform_sse(temp, buffer[lpass] + col*p->ch, p->width, p->height, 1 << lev, p);
      for (int row = 0; row < p->height; row++) 
      {
        for (int c = 0; c < p->ch; c++) 
          buffer[lpass][INDEX_WT_IMAGE(row * p->width + col)] = temp[INDEX_WT_IMAGE(row)];
      }
    }
    
    float *bl = buffer[lpass];
    float *bh = buffer[hpass];
    
    for (int i = 0; i < p->width * p->height; i++, bl+=4, bh+=4)
    {
      /* rounding errors introduced here (division by 16) */
      _mm_store_ps(bl, _mm_mul_ps(_mm_add_ps(_mm_load_ps(bl), v4_lpass_add), v4_lpass_mult));
      _mm_store_ps(bh, _mm_sub_ps(_mm_load_ps(bh), _mm_sub_ps(_mm_load_ps(bl), v4_lpass_sub)));
    }

    if (layer_func) layer_func(buffer[hpass], p, lev + 1);
    
    if (p->return_layer == 0)
    {
      dwt_add_layer_sse(buffer[hpass], p, lev + 1);
    }
    else if (p->return_layer == lev + 1)
    {
      dwt_get_image_layer(buffer[hpass], p);
  
      bcontinue = 0;
    }
    
    hpass = lpass;
  }

  if (p->return_layer == p->scales+1)
  {
    if (layer_func) layer_func(buffer[lpass], p, p->scales+1);
    
    dwt_get_image_layer(buffer[lpass], p);
  }
  else if (p->return_layer == 0 && p->scales > 0)
  {
    if (layer_func) layer_func(buffer[hpass], p, p->scales+1);
    
    dwt_add_layer_sse(buffer[hpass], p, p->scales+1);
    
    if (layer_func) layer_func(p->layers, p, p->scales+2);
    
    dwt_get_image_layer(p->layers, p);
  }

cleanup:
  if (temp) dt_free_align(temp);
  if (buffer[1]) dt_free_align(buffer[1]);

}
#endif

void dwt_wavelet_decompose(float *img, dwt_params_t *const p, _dwt_layer_func layer_func)
{
  float *temp = NULL;
  unsigned int lpass, hpass;
  float *buffer[2] = {0, 0};
  int bcontinue = 1;
  
  const float lpass_add = 0.f;
  const float lpass_mult = (1.f / 16.f);
  const float lpass_sub = p->blend_factor;
  
  const int size = p->width * p->height * p->ch;

  /* image buffers */
  buffer[0] = img;
  /* temporary storage */
  buffer[1] = dt_alloc_align(64, size * sizeof(float));
  if (buffer[1] == NULL)
  {
    dt_control_log(_("not enough memory for wavelet decomposition"));
    goto cleanup;
  }

  temp = dt_alloc_align(64, MAX(p->width, p->height) * p->ch * sizeof(float));
  if (temp == NULL)
  {
    dt_control_log(_("not enough memory for wavelet decomposition"));
    goto cleanup;
  }

  /* iterate over wavelet scales */
  lpass = 1;
  hpass = 0;
  for (unsigned int lev = 0; lev < p->scales && bcontinue; lev++) 
  {
    lpass = (1 - (lev & 1));

#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(buffer, temp, lev, lpass, hpass) schedule(static)
#endif
#endif
    for (int row = 0; row < p->height; row++) 
    {
      dwt_hat_transform(temp, buffer[hpass] + (row * p->width * p->ch), 1, p->width, 1 << lev, p);
      memcpy(&(buffer[lpass][row * p->width * p->ch]), temp, p->width * p->ch * sizeof(float));
    }
    
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(buffer, temp, lev, lpass, hpass) schedule(static)
#endif
#endif
    for (int col = 0; col < p->width; col++) 
    {
      dwt_hat_transform(temp, buffer[lpass] + col*p->ch, p->width, p->height, 1 << lev, p);
      for (int row = 0; row < p->height; row++) 
      {
        for (int c = 0; c < p->ch; c++) 
        buffer[lpass][INDEX_WT_IMAGE(row * p->width + col)] = temp[INDEX_WT_IMAGE(row)];
      }
    }
      
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(buffer, lev, lpass, hpass) schedule(static)
#endif
#endif
    for (int i = 0; i < size; i++) 
    {
      /* rounding errors introduced here (division by 16) */
      buffer[lpass][i] = ( buffer[lpass][i] + lpass_add ) * lpass_mult;
      buffer[hpass][i] -= buffer[lpass][i] - lpass_sub;
    }

    if (layer_func) layer_func(buffer[hpass], p, lev + 1);
    
    if (p->return_layer == 0)
    {
      dwt_add_layer(buffer[hpass], p, lev + 1);
    }
    else if (p->return_layer == lev + 1)
    {
      dwt_get_image_layer(buffer[hpass], p);
  
      bcontinue = 0;
    }
    
    hpass = lpass;
  }

  //  Wavelet residual
  if (p->return_layer == p->scales+1)
  {
    if (layer_func) layer_func(buffer[lpass], p, p->scales+1);
    
    dwt_get_image_layer(buffer[lpass], p);
  }
  else if (p->return_layer == 0 && p->scales > 0)
  {
    if (layer_func) layer_func(buffer[hpass], p, p->scales+1);
    
    dwt_add_layer(buffer[hpass], p, p->scales+1);
    
    if (layer_func) layer_func(p->layers, p, p->scales+2);
    
    dwt_get_image_layer(p->layers, p);
  }

cleanup:
  if (temp) dt_free_align(temp);
  if (buffer[1]) dt_free_align(buffer[1]);

}

void dwt_decompose(dwt_params_t *p, _dwt_layer_func layer_func)
{
  double start = dt_get_wtime();
  
  if (layer_func) layer_func(p->image, p, 0);
  
  if (p->scales == 0) return;
  
  /* this function prepares for decomposing, which is done in the function dwt_wavelet_decompose() */
  int max_scale = dwt_get_max_scale(p->width, p->height, p->preview_scale);
  
  if (p->preview_scale == 0.f) p->preview_scale = 1.f;
  if (p->scales > max_scale) p->scales = max_scale;
  if (p->return_layer > p->scales+1) p->return_layer = p->scales+1;
  
  p->layers = dt_alloc_align(64, p->width * p->height * p->ch * sizeof(float));
  if (p->layers == NULL)
  {
    dt_control_log(_("not enough memory for wavelet decomposition"));
    goto cleanup;
  }
  
#if defined(__SSE__)
  if (p->ch == 4 && p->use_sse)
    dwt_wavelet_decompose_sse(p->image, p, layer_func);
  else
#endif
    dwt_wavelet_decompose(p->image, p, layer_func);

  
cleanup:
    if (p->layers) 
    {
      dt_free_align(p->layers);
      p->layers = NULL;
    }

    if(darktable.unmuted & DT_DEBUG_PERF) printf("dwt_decompose took %0.04f sec\n", dt_get_wtime() - start);
}

