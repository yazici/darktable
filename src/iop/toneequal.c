/*
    This file is part of darktable,
    copyright (c) 2018 Aurélien Pierre.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "common/opencl.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "dtgtk/drawingarea.h"
#include "dtgtk/gradientslider.h"
#include "dtgtk/togglebutton.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include "common/iop_group.h"

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

DT_MODULE_INTROSPECTION(1, dt_iop_toneequalizer_params_t)

typedef enum dt_iop_toneequalizer_method_t
{
  DT_TONEEQ_MEAN = 0,
  DT_TONEEQ_LIGHTNESS,
  DT_TONEEQ_VALUE,
  DT_TONEEQ_NORM
} dt_iop_toneequalizer_method_t;

typedef struct dt_iop_toneequalizer_params_t
{
  float noise, ultra_deep_blacks, deep_blacks, blacks, shadows, midtones, highlights, whites;
  float blending;
  int scales;
  dt_iop_toneequalizer_method_t method;
} dt_iop_toneequalizer_params_t;

typedef struct dt_iop_toneequalizer_data_t
{
  dt_iop_toneequalizer_params_t params;
  float factors[16] __attribute__((aligned(16)));
} dt_iop_toneequalizer_data_t;

typedef struct dt_iop_toneequalizer_global_data_t
{
  int kernel_zonesystem;
} dt_iop_toneequalizer_global_data_t;


typedef struct dt_iop_toneequalizer_gui_data_t
{
  GtkWidget *noise, *ultra_deep_blacks, *deep_blacks, *blacks, *shadows, *midtones, *highlights, *whites;
  GtkWidget *blending, *scales;
  GtkWidget *method;
  GtkWidget *details;
} dt_iop_toneequalizer_gui_data_t;


const char *name()
{
  return _("tone equalizer");
}

int default_group()
{
  return IOP_GROUP_BASIC;
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

#define GAUSS(b, x) (expf((-(x - b) * (x - b) / 4.0f)))

// Build the luma channels : band-pass filters with gaussian windows of std 2 EV, spaced by 2 EV
const float centers[16] __attribute__((aligned(16))) = { -22.0f,
                                                         -20.0f,
                                                         -18.0f,
                                                         -16.0f,
                                                         -14.0f,
                                                         -12.0f,
                                                         -10.0f,
                                                         -8.0f,
                                                         -6.0f,
                                                         -4.0f,
                                                         - 2.0f,
                                                         0.0f,
                                                         2.0f,
                                                         4.0f,
                                                         6.0f,
                                                         8.0f};

// For every pixel luminance, the sum of the gaussian masks
// evenly spaced by 2 EV with 2 EV std should be this constant
#define w_sum 1.772637204826214f

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float RGB_light(const float pixel[4], const dt_iop_toneequalizer_method_t method)
{
  // compute an estimation of the pixel lightness using several color models
  switch(method)
  {
    case DT_TONEEQ_MEAN:
      return (pixel[0] + pixel[1] + pixel[2]) / 3.0f;

    case DT_TONEEQ_LIGHTNESS:
    {
      const float pixel_min = fminf(pixel[0], fminf(pixel[1], pixel[2]));
      const float pixel_max = fmaxf(pixel[0], fmaxf(pixel[1], pixel[2]));
      return (pixel_min + pixel_max) / 2.0f;
    }

    case DT_TONEEQ_VALUE:
      return fmaxf(pixel[0], fmaxf(pixel[1], pixel[2]));

    case DT_TONEEQ_NORM:
      return sqrtf(pixel[0] * pixel[0] + pixel[1] * pixel[1] + pixel[2] * pixel[2]);

    default:
      return -1.0;
  }
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float compute_exposure(const float pixel[4], const dt_iop_toneequalizer_method_t method)
{
  // compute the exposure (in EV) of the current pixel
  return fmaxf(log2f(RGB_light(pixel, method)), -18.0f);
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static float compute_correction(const float factors[16], const float luma)
{
  // build the correction for the current pixel
  // as the sum of the contribution of each luminance channel
  float correction = 0.0f;
  for (int c = 0; c < 16; ++c) correction += GAUSS(centers[c], luma) * factors[c];
  return correction /= w_sum;
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static void process_pixel(const float pixel_in[4], float pixel_out[4],
                          const float correction)
{
  // Apply the weighted contribution of each channel to the current pixel
  const float factor[4] __attribute__((aligned(16))) = { correction, correction, correction, 1.0f };
  for(int c = 0; c < 4; ++c) pixel_out[c] = factor[c] * pixel_in[c];
}

static inline void process_image_loop(const float *const restrict in, float *const restrict out,
                                      size_t width, size_t height, size_t ch,
                                      const float factors[16],
                                      const dt_iop_toneequalizer_method_t method)
{
#ifdef _OPENMP
#pragma omp parallel for simd
#endif
  for(size_t k = 0; k < (size_t)ch * width * height; k += 8 * ch)
  {
    // Manually unroll loops to saturate the cache, and be savage about it
    // we process 8 contiguous pixels at a time and give every hint to the compiler
    // to vectorize using the maximum SSE size available

    // Input vectors
    const float *const pixel_in_0 = in + k;
    const float *const pixel_in_1 = pixel_in_0 + ch;
    const float *const pixel_in_2 = pixel_in_1 + ch;
    const float *const pixel_in_3 = pixel_in_2 + ch;

    const float *const pixel_in_4 = pixel_in_3 + ch;
    const float *const pixel_in_5 = pixel_in_4 + ch;
    const float *const pixel_in_6 = pixel_in_5 + ch;
    const float *const pixel_in_7 = pixel_in_6 + ch;

    // Output vectors
    float *const pixel_out_0 = out + k;
    float *const pixel_out_1 = pixel_out_0 + ch;
    float *const pixel_out_2 = pixel_out_1 + ch;
    float *const pixel_out_3 = pixel_out_2 + ch;

    float *const pixel_out_4 = pixel_out_3 + ch;
    float *const pixel_out_5 = pixel_out_4 + ch;
    float *const pixel_out_6 = pixel_out_5 + ch;
    float *const pixel_out_7 = pixel_out_6 + ch;

    // Pixel lightness
    const float luma_0 = compute_exposure(pixel_in_0, method);
    const float luma_1 = compute_exposure(pixel_in_1, method);
    const float luma_2 = compute_exposure(pixel_in_2, method);
    const float luma_3 = compute_exposure(pixel_in_3, method);

    const float luma_4 = compute_exposure(pixel_in_4, method);
    const float luma_5 = compute_exposure(pixel_in_5, method);
    const float luma_6 = compute_exposure(pixel_in_6, method);
    const float luma_7 = compute_exposure(pixel_in_7, method);

    // Pixel corrections
    const float correction_0 = compute_correction(factors, luma_0);
    const float correction_1 = compute_correction(factors, luma_1);
    const float correction_2 = compute_correction(factors, luma_2);
    const float correction_3 = compute_correction(factors, luma_3);

    const float correction_4 = compute_correction(factors, luma_4);
    const float correction_5 = compute_correction(factors, luma_5);
    const float correction_6 = compute_correction(factors, luma_6);
    const float correction_7 = compute_correction(factors, luma_7);

    // Actual processing
    process_pixel(pixel_in_0, pixel_out_0, correction_0);
    process_pixel(pixel_in_1, pixel_out_1, correction_1);
    process_pixel(pixel_in_2, pixel_out_2, correction_2);
    process_pixel(pixel_in_3, pixel_out_3, correction_3);

    process_pixel(pixel_in_4, pixel_out_4, correction_4);
    process_pixel(pixel_in_5, pixel_out_5, correction_5);
    process_pixel(pixel_in_6, pixel_out_6, correction_6);
    process_pixel(pixel_in_7, pixel_out_7, correction_7);
  }
}

static inline void process_image(const float *const in, float *const out,
                                  size_t width, size_t height, size_t ch,
                                  const float factors[16], const dt_iop_toneequalizer_method_t method)
{
  // Force the compiler to compile static variants of the loop and avoid inner branching at pixel-level
  switch(method)
  {
    case DT_TONEEQ_MEAN:
    {
      process_image_loop(in, out, width, height, ch, factors, DT_TONEEQ_MEAN);
      break;
    }
    case DT_TONEEQ_LIGHTNESS:
    {
      process_image_loop(in, out, width, height, ch, factors, DT_TONEEQ_LIGHTNESS);
      break;
    }
    case DT_TONEEQ_VALUE:
    {
      process_image_loop(in, out, width, height, ch, factors, DT_TONEEQ_VALUE);
      break;
    }
    case DT_TONEEQ_NORM:
    {
      process_image_loop(in, out, width, height, ch, factors, DT_TONEEQ_NORM);
      break;
    }
  }
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  const dt_iop_toneequalizer_data_t *const d = (const dt_iop_toneequalizer_data_t *const)piece->data;

  const int ch = piece->colors;
  const float *const in = (const float *const)ivoid;
  float *const out = (float *const)ovoid;

  process_image(in, out, roi_out->width, roi_out->height, ch, d->factors, d->params.method);
}

void init_global(dt_iop_module_so_t *module)
{
  dt_iop_toneequalizer_global_data_t *gd
      = (dt_iop_toneequalizer_global_data_t *)malloc(sizeof(dt_iop_toneequalizer_global_data_t));
  module->data = gd;
}

void cleanup_global(dt_iop_module_so_t *module)
{
  free(module->data);
  module->data = NULL;
}

void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)p1;
  dt_iop_toneequalizer_data_t *d = (dt_iop_toneequalizer_data_t *)piece->data;

  d->params = *p;

  float factors[16] __attribute__((aligned(16))) = { d->params.noise,             // -22 EV
                                                                d->params.noise,             // -20 EV
                                                                d->params.noise,             // -18 EV
                                                                d->params.noise,             // -16 EV
                                                                d->params.noise,             // -14 EV
                                                                d->params.ultra_deep_blacks, // -12 EV
                                                                d->params.deep_blacks,       // -10 EV
                                                                d->params.blacks,            //  -8 EV
                                                                d->params.shadows,           //  -6 EV
                                                                d->params.midtones,          //  -4 EV
                                                                d->params.highlights,        //  -2 EV
                                                                d->params.whites,            //  0 EV
                                                                d->params.whites,            //  2 EV
                                                                d->params.whites,            //  4 EV
                                                                d->params.whites,            //  6 EV
                                                                d->params.whites};           //  8 EV
#ifdef _OPENMP
#pragma omp for simd
#endif
  for(int c = 0; c < 16; ++c) d->factors[c] = exp2f(factors[c]);
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_toneequalizer_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(struct dt_iop_module_t *self)
{
  dt_iop_module_t *module = (dt_iop_module_t *)self;
  dt_iop_toneequalizer_gui_data_t *g = (dt_iop_toneequalizer_gui_data_t *)self->gui_data;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)module->params;

  dt_bauhaus_slider_set(g->noise, p->noise);
  dt_bauhaus_slider_set(g->ultra_deep_blacks, p->ultra_deep_blacks);
  dt_bauhaus_slider_set(g->deep_blacks, p->deep_blacks);
  dt_bauhaus_slider_set(g->blacks, p->blacks);
  dt_bauhaus_slider_set(g->shadows, p->shadows);
  dt_bauhaus_slider_set(g->midtones, p->midtones);
  dt_bauhaus_slider_set(g->highlights, p->highlights);
  dt_bauhaus_slider_set(g->whites, p->whites);
}

void init(dt_iop_module_t *module)
{
  module->params = calloc(1, sizeof(dt_iop_toneequalizer_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_toneequalizer_params_t));
  module->default_enabled = 0;
  module->priority = 158; // module order created by iop_dependencies.py, do not edit!
  module->params_size = sizeof(dt_iop_toneequalizer_params_t);
  module->gui_data = NULL;
  dt_iop_toneequalizer_params_t tmp = (dt_iop_toneequalizer_params_t){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  memcpy(module->params, &tmp, sizeof(dt_iop_toneequalizer_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_toneequalizer_params_t));
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

static void noise_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->noise = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void ultra_deep_blacks_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->ultra_deep_blacks = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void deep_blacks_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->deep_blacks = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void blacks_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->blacks = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void shadows_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->shadows = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void midtones_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->midtones = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void highlights_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->highlights = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void whites_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->whites = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void method_changed(GtkWidget *widget, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_toneequalizer_params_t *p = (dt_iop_toneequalizer_params_t *)self->params;
  p->method = dt_bauhaus_combobox_get(widget);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_init(struct dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_toneequalizer_gui_data_t));
  dt_iop_toneequalizer_gui_data_t *g = (dt_iop_toneequalizer_gui_data_t *)self->gui_data;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_GUI_IOP_MODULE_CONTROL_SPACING);

  const float top = 2.0;
  const float bottom = -2.0;

  g->noise = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->noise, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->noise, NULL, _("-14 EV : HDR noise"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->noise, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->noise), "value-changed", G_CALLBACK(noise_callback), self);

  g->ultra_deep_blacks = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->ultra_deep_blacks, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->ultra_deep_blacks, NULL, _("-12 EV : HDR ultra-deep blacks"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->ultra_deep_blacks, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->ultra_deep_blacks), "value-changed", G_CALLBACK(ultra_deep_blacks_callback), self);

  g->deep_blacks = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->deep_blacks, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->deep_blacks, NULL, _("-10 EV : HDR deep blacks"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->deep_blacks, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->deep_blacks), "value-changed", G_CALLBACK(deep_blacks_callback), self);

  g->blacks = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->blacks, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->blacks, NULL, _("-08 EV : blacks"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->blacks, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->blacks), "value-changed", G_CALLBACK(blacks_callback), self);

  g->shadows = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->shadows, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->shadows, NULL, _("-06 EV : shadows"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->shadows, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->shadows), "value-changed", G_CALLBACK(shadows_callback), self);

  g->midtones = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->midtones, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->midtones, NULL, _("-04 EV : midtones"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->midtones, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->midtones), "value-changed", G_CALLBACK(midtones_callback), self);

  g->highlights = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->highlights, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->highlights, NULL, _("-02 EV : highlights"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->highlights, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->highlights), "value-changed", G_CALLBACK(highlights_callback), self);

  g->whites = dt_bauhaus_slider_new_with_range(self, bottom, top, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->whites, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->whites, NULL, _("-00 EV : whites"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->whites, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->whites), "value-changed", G_CALLBACK(whites_callback), self);

  g->method = dt_bauhaus_combobox_new(NULL);
  dt_bauhaus_widget_set_label(g->method, NULL, _("compute pixel exposure using"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->method, TRUE, TRUE, 0);
  dt_bauhaus_combobox_add(g->method, "RGB average");
  dt_bauhaus_combobox_add(g->method, "RGB lightness");
  dt_bauhaus_combobox_add(g->method, "RGB value");
  dt_bauhaus_combobox_add(g->method, "RGB norm");
  g_signal_connect(G_OBJECT(g->method), "value-changed", G_CALLBACK(method_changed), self);
}

void gui_cleanup(struct dt_iop_module_t *self)
{
  self->request_color_pick = DT_REQUEST_COLORPICK_OFF;
  free(self->gui_data);
  self->gui_data = NULL;
}


// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
