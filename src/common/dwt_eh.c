
#include "control/control.h"
#include "develop/imageop.h"
#include "dwt_eh.h"

/* Based on the original source code of GIMP's Wavelet Decompose plugin, by Marco Rossini 
 * 
 * http://registry.gimp.org/node/11742
 * 
 * */


static int _get_max_scale(const int width, const int height, const float preview_scale)
{
  int maxscale = 0;
  
  // smallest edge must be higher than or equal to 2^scales
  unsigned int size = MIN(width, height);
  while ((size >>= 1) * preview_scale) maxscale++;

  // avoid rounding issues...
  size = MIN(width, height);
  while ( (maxscale > 0) && ((1 << maxscale) * preview_scale >= size) ) maxscale--;
  
  return maxscale;
} 

int dwt_get_max_scale(dwt_params_t *p)
{
  return _get_max_scale(p->width/p->preview_scale, p->height/p->preview_scale, p->preview_scale);
}

#define INDEX_WT_IMAGE(index, num_channels, channel) (((index)*(num_channels))+(channel))
#define INDEX_WT_IMAGE_SSE(index, num_channels) ((index)*(num_channels))

/* code copied from UFRaw (which originates from dcraw) */
#if defined(__SSE__)
static void dwt_hat_transform_sse(float *temp, const float *const base, const int st, const int size, int sc, 
		const dwt_params_t *const p)
{
  int i;
  const __m128 hat_mult = _mm_set1_ps(2.f);
  __m128 valb_1, valb_2, valb_3;
  sc = (int)(sc * p->preview_scale);
  if (sc > size) sc = size;

  for (i = 0; i < sc; i++, temp +=4)
  {
    valb_1 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * i, p->ch)]);
    valb_2 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (sc - i), p->ch)]);
    valb_3 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (i + sc), p->ch)]);
    
    _mm_store_ps(temp, _mm_add_ps(_mm_add_ps(_mm_mul_ps(hat_mult, valb_1), valb_2), valb_3));
  }
  for (; i + sc < size; i++, temp +=4)
  {
		valb_1 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * i, p->ch)]);
		valb_2 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (i - sc), p->ch)]);
		valb_3 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (i + sc), p->ch)]);
		
		_mm_store_ps(temp, _mm_add_ps(_mm_add_ps(_mm_mul_ps(hat_mult, valb_1), valb_2), valb_3));
  }
  for (; i < size; i++, temp +=4)
  {
		valb_1 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * i, p->ch)]);
		valb_2 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (i - sc), p->ch)]);
		valb_3 = _mm_load_ps(&base[INDEX_WT_IMAGE_SSE(st * (2 * size - 2 - (i + sc)), p->ch)]);
		
		_mm_store_ps(temp, _mm_add_ps(_mm_add_ps(_mm_mul_ps(hat_mult, valb_1), valb_2), valb_3));
  }
  
}
#endif

static void dwt_hat_transform(float *temp, const float *const base, const int st, const int size, int sc, 
		dwt_params_t *const p)
{
#if defined(__SSE__)
  if (p->ch == 4 && p->use_sse)
  {
    dwt_hat_transform_sse(temp, base, st, size, sc, p);
    return;
  }
#endif
    
  int i, c;
  const float hat_mult = 2.f;
  sc = (int)(sc * p->preview_scale);
  if (sc > size) sc = size;

  for (i = 0; i < sc; i++)
  {
    for (c = 0; c < p->ch; c++, temp++)
    {
      *temp = hat_mult * base[INDEX_WT_IMAGE(st * i, p->ch, c)] + 
      									base[INDEX_WT_IMAGE(st * (sc - i), p->ch, c)] + 
												base[INDEX_WT_IMAGE(st * (i + sc), p->ch, c)];
    }
  }
  for (; i + sc < size; i++)
  {
    for (c = 0; c < p->ch; c++, temp++)
    {
      *temp = hat_mult * base[INDEX_WT_IMAGE(st * i, p->ch, c)] + 
      									base[INDEX_WT_IMAGE(st * (i - sc), p->ch, c)] + 
												base[INDEX_WT_IMAGE(st * (i + sc), p->ch, c)];
    }
  }
  for (; i < size; i++)
  {
    for (c = 0; c < p->ch; c++, temp++)
    {
      *temp = hat_mult * base[INDEX_WT_IMAGE(st * i, p->ch, c)] + 
      									base[INDEX_WT_IMAGE(st * (i - sc), p->ch, c)] + 
												base[INDEX_WT_IMAGE(st * (2 * size - 2 - (i + sc)), p->ch, c)];
    }
  }
  
}

#if defined(__SSE__)
static void dwt_add_layer_sse(float *const img, float *layers, dwt_params_t *const p, const int n_scale)
{
  const int i_size = p->width*p->height*4;
  
  if (n_scale == p->scales+1)
  {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(layers) schedule(static)
#endif
    for (int i=0; i<i_size; i+=4)
    {
      _mm_store_ps(&(layers[i]), _mm_add_ps(_mm_load_ps(&(layers[i])), _mm_load_ps(&(img[i]))));
    }
  }
  else
  {
    const __m128 lpass_sub = _mm_set1_ps(p->blend_factor);
    
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(layers) schedule(static)
#endif
    for (int i=0; i<i_size; i+=4)
    {
      _mm_store_ps(&(layers[i]), _mm_add_ps(_mm_load_ps(&(layers[i])), _mm_sub_ps(_mm_load_ps(&(img[i])), lpass_sub)));
    }
  }

}
#endif

static void dwt_add_layer(float *const img, float *layers, dwt_params_t *const p, const int n_scale)
{
#if defined(__SSE__)
  if (p->ch == 4 && p->use_sse)
  {
    dwt_add_layer_sse(img, layers, p, n_scale);
    return;
  }
#endif
    
  const int i_size = p->width*p->height*p->ch;
  const float lpass_sub = p->blend_factor;
  
  if (n_scale == p->scales+1)
  {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(layers) schedule(static)
#endif
    for (int i=0; i<i_size; i++)
      layers[i] += img[i];
  }
  else
  {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(layers) schedule(static)
#endif
    for (int i=0; i<i_size; i++)
      layers[i] += img[i] - lpass_sub;
  }

}

static void dwt_get_image_layer(float *const layer, dwt_params_t *const p)
{
  if (p->image == layer) return;
  
  memcpy(p->image, layer, p->width * p->height * p->ch * sizeof(float));
}

#if defined(__SSE__)
static void dwt_subtract_layer_sse(float *bl, float *bh, dwt_params_t *const p)
{
//  const __m128 v4_lpass_add = _mm_set1_ps(0.f);
  const __m128 v4_lpass_mult = _mm_set1_ps((1.f / 16.f));
  const __m128 v4_lpass_sub = _mm_set1_ps(p->blend_factor);
  const int size = p->width * p->height * 4;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(bl, bh) schedule(static)
#endif
  for (int i = 0; i < size; i+=4)
  {
    // rounding errors introduced here (division by 16)
//    _mm_store_ps(&(bl[i]), _mm_mul_ps(_mm_add_ps(_mm_load_ps(&(bl[i])), v4_lpass_add), v4_lpass_mult));
    _mm_store_ps(&(bl[i]), _mm_mul_ps(_mm_load_ps(&(bl[i])), v4_lpass_mult));
    _mm_store_ps(&(bh[i]), _mm_sub_ps(_mm_load_ps(&(bh[i])), _mm_sub_ps(_mm_load_ps(&(bl[i])), v4_lpass_sub)));
  }
}
#endif

static void dwt_subtract_layer(float *bl, float *bh, dwt_params_t *const p)
{
#if defined(__SSE__)
  if (p->ch == 4 && p->use_sse)
  {
    dwt_subtract_layer_sse(bl, bh, p);
    return;
  }
#endif
    
//  const float lpass_add = 0.f;
  const float lpass_mult = (1.f / 16.f);
  const float lpass_sub = p->blend_factor;
  const int size = p->width * p->height * p->ch;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(bl, bh) schedule(static)
#endif
    for (int i = 0; i < size; i++) 
    {
      // rounding errors introduced here (division by 16)
      bl[i] = ( bl[i] /*+ lpass_add*/ ) * lpass_mult;
      bh[i] -= bl[i] - lpass_sub;
    }
}

/* actual decomposing algorithm */
static void dwt_wavelet_decompose(float *img, dwt_params_t *const p, _dwt_layer_func layer_func)
{
  float *temp = NULL;
  float *layers = NULL;
  unsigned int lpass, hpass;
  float *buffer[2] = {0, 0};
  int bcontinue = 1;
  const int size = p->width * p->height * p->ch;

  if (layer_func) layer_func(img, p, 0);
  
  if (p->scales <= 0)
    goto cleanup;
  
  /* image buffers */
  buffer[0] = img;
  /* temporary storage */
  buffer[1] = dt_alloc_align(64, size * sizeof(float));
  if (buffer[1] == NULL)
  {
    dt_control_log(_("not enough memory for wavelet decomposition"));
    goto cleanup;
  }

  // setup a temp buffer
  temp = dt_alloc_align(64, MAX(p->width, p->height) * p->ch * sizeof(float));
  if (temp == NULL)
  {
    dt_control_log(_("not enough memory for wavelet decomposition"));
    goto cleanup;
  }

  // buffer to reconstruct the image
  layers = dt_alloc_align(64, p->width * p->height * p->ch * sizeof(float));
  if (layers == NULL)
  {
    dt_control_log(_("not enough memory for wavelet decomposition"));
    goto cleanup;
  }
  
  // iterate over wavelet scales
  lpass = 1;
  hpass = 0;
  for (unsigned int lev = 0; lev < p->scales && bcontinue; lev++) 
  {
    lpass = (1 - (lev & 1));

    for (int row = 0; row < p->height; row++) 
    {
      dwt_hat_transform(temp, buffer[hpass] + (row * p->width * p->ch), 1, p->width, 1 << lev, p);
      memcpy(&(buffer[lpass][row * p->width * p->ch]), temp, p->width * p->ch * sizeof(float));
    }
    
    for (int col = 0; col < p->width; col++) 
    {
      dwt_hat_transform(temp, buffer[lpass] + col*p->ch, p->width, p->height, 1 << lev, p);
      for (int row = 0; row < p->height; row++) 
      {
        for (int c = 0; c < p->ch; c++) 
          buffer[lpass][INDEX_WT_IMAGE(row * p->width + col, p->ch, c)] = temp[INDEX_WT_IMAGE(row, p->ch, c)];
      }
    }

    dwt_subtract_layer(buffer[lpass], buffer[hpass], p);
    
    if (layer_func) layer_func(buffer[hpass], p, lev + 1);
    
    if (p->return_layer == 0)
    {
      dwt_add_layer(buffer[hpass], layers, p, lev + 1);
    }
    else if (p->return_layer == lev + 1)
    {
      dwt_get_image_layer(buffer[hpass], p);
  
      bcontinue = 0;
    }
    
    hpass = lpass;
  }

  // Wavelet residual
  if (p->return_layer == p->scales+1)
  {
    if (layer_func) layer_func(buffer[lpass], p, p->scales+1);
    
    dwt_get_image_layer(buffer[lpass], p);
  }
  else if (p->return_layer == 0 && p->scales > 0)
  {
    if (layer_func) layer_func(buffer[hpass], p, p->scales+1);
    
    dwt_add_layer(buffer[hpass], layers, p, p->scales+1);
    
    if (layer_func) layer_func(layers, p, p->scales+2);
    
    dwt_get_image_layer(layers, p);
  }

cleanup:
  if (layers) dt_free_align(layers);
  if (temp) dt_free_align(temp);
  if (buffer[1]) dt_free_align(buffer[1]);

}

/* this function prepares for decomposing, which is done in the function dwt_wavelet_decompose() */
void dwt_decompose(dwt_params_t *p, _dwt_layer_func layer_func)
{
  double start = dt_get_wtime();
  
  // this is a zoom scale, not a wavelet scale
  if (p->preview_scale <= 0.f) p->preview_scale = 1.f;
  
  // if a single scale is requested it cannot be grather than the residual
  if (p->return_layer > p->scales+1)
  {
    p->return_layer = p->scales+1;
  }
  
  const int max_scale = dwt_get_max_scale(p);
  
  // if requested scales is grather than max scales adjust it
  if (p->scales > max_scale)
  {
    // residual shoud be returned
    if (p->return_layer > p->scales)
      p->return_layer = max_scale+1;
    // a scale should be returned, it cannot be grather than max scales
    else if (p->return_layer > max_scale)
      p->return_layer = max_scale;
    
    p->scales = max_scale;
  }
  
  // call the actual decompose
  dwt_wavelet_decompose(p->image, p, layer_func);

  if(darktable.unmuted & DT_DEBUG_PERF) printf("dwt_decompose took %0.04f sec\n", dt_get_wtime() - start);
}

