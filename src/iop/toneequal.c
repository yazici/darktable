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
#include "common/colorspaces_inline_conversions.h"
#include "common/darktable.h"
#include "common/gaussian.h"
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

typedef struct dt_iop_toneequalizer_params_t
{
  float ultra_deep_blacks, deep_blacks, blacks, shadows, midtones, highlights, whites;
  float blending;
  int scales;
} dt_iop_toneequalizer_params_t;

typedef struct dt_iop_toneequalizer_data_t
{
  dt_iop_toneequalizer_params_t params;
} dt_iop_toneequalizer_data_t;

typedef struct dt_iop_toneequalizer_global_data_t
{
  int kernel_zonesystem;
} dt_iop_toneequalizer_global_data_t;


typedef struct dt_iop_toneequalizer_gui_data_t
{
  GtkWidget *ultra_deep_blacks, *deep_blacks, *blacks, *shadows, *midtones, *highlights, *whites;
  GtkWidget *blending, *scales;
} dt_iop_toneequalizer_gui_data_t;


const char *name()
{
  return _("tone equalizer");
}

int groups()
{
  return dt_iop_get_group("tone equalizer", IOP_GROUP_BASIC);
}

#ifdef _OPENMP
#pragma omp declare simd inbranch
#endif
#define GAUSS(b, x) (expf((-(x - b) * (x - b) / 4.0f)))

// From data/kernels/extended.cl
static inline float fastlog2(float x)
{
  union { float f; unsigned int i; } vx = { x };
  union { unsigned int i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };

  float y = vx.i;

  y *= 1.1920928955078125e-7f;

  return y - 124.22551499f
    - 1.498030302f * mx.f
    - 1.72587999f / (0.3520887068f + mx.f);
}

// Build the luma channels : band-pass filters with gaussian windows of std 2 EV, spaced by 2 EV
const float centers[12] = { -16.0f, -14.0f, -12.0f, -10.0f, -8.0f, -6.0f, \
                            -4.0f,  - 2.0f,   0.0f,  2.0f,   4.0f,  8.0f};

// For every pixel luminance, the sum of the gaussian masks
// evenly spaced by 2 EV with 2 EV std should be this constant
#define w_sum 1.772637204826214f

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  const dt_iop_toneequalizer_data_t *const d = (const dt_iop_toneequalizer_data_t *const)piece->data;

  const float factors[12] = { 1.0f,
                              1.0f,
                              exp2f(d->params.ultra_deep_blacks),
                              exp2f(d->params.deep_blacks),
                              exp2f(d->params.blacks),
                              exp2f(d->params.shadows),
                              exp2f(d->params.midtones),
                              exp2f(d->params.highlights),
                              exp2f(d->params.whites),
                              1.0f,
                              1.0f,
                              1.0f };

  const int ch = piece->colors;
  const float *const in = (const float *const)ivoid;
  float *const out = (float *const)ovoid;

  // correct the luminance
#ifdef _OPENMP
#pragma omp parallel for SIMD() default(none) schedule(simd:static)
#endif
  for(size_t k = 0; k < (size_t)ch * roi_out->width * roi_out->height; k += ch)
  {
    // get the luminance of the pixel
    const float luma = fastlog2( (fmaxf(fmaxf(in[k], in[k+1]), in[k+2]) +
                                  fminf(fminf(in[k], in[k+1]), in[k+2])) / 2.0f );

    // build the pixel correction
    float correction = 0.0f;

#ifdef _OPENMP
#pragma omp ordered simd
#endif
    for (int c = 0; c < 12; ++c) correction += GAUSS(centers[c], luma) * factors[c];

    correction /= w_sum;

    // Apply the weighted contribution of each channel to the current pixel
#ifdef _OPENMP
#pragma omp ordered simd
#endif
    for (int c = 0; c < 3; ++c) out[k + c] = correction * in[k + c];
    out[k + 4] = in[k + 4];
  }
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
  dt_iop_toneequalizer_params_t tmp = (dt_iop_toneequalizer_params_t){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  memcpy(module->params, &tmp, sizeof(dt_iop_toneequalizer_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_toneequalizer_params_t));
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
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

void gui_init(struct dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_toneequalizer_gui_data_t));
  dt_iop_toneequalizer_gui_data_t *g = (dt_iop_toneequalizer_gui_data_t *)self->gui_data;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_GUI_IOP_MODULE_CONTROL_SPACING);

  g->ultra_deep_blacks = dt_bauhaus_slider_new_with_range(self, -2.0, 2.0, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->ultra_deep_blacks, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->ultra_deep_blacks, NULL, _("-12 EV : HDR ultra-deep blacks"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->ultra_deep_blacks, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->ultra_deep_blacks), "value-changed", G_CALLBACK(ultra_deep_blacks_callback), self);

  g->deep_blacks = dt_bauhaus_slider_new_with_range(self, -2.0, 2.0, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->deep_blacks, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->deep_blacks, NULL, _("-10 EV : HDR deep blacks"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->deep_blacks, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->deep_blacks), "value-changed", G_CALLBACK(deep_blacks_callback), self);

  g->blacks = dt_bauhaus_slider_new_with_range(self, -2.0, 2.0, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->blacks, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->blacks, NULL, _("-08 EV : blacks"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->blacks, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->blacks), "value-changed", G_CALLBACK(blacks_callback), self);

  g->shadows = dt_bauhaus_slider_new_with_range(self, -2.0, 2.0, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->shadows, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->shadows, NULL, _("-06 EV : shadows"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->shadows, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->shadows), "value-changed", G_CALLBACK(shadows_callback), self);

  g->midtones = dt_bauhaus_slider_new_with_range(self, -2.0, 2.0, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->midtones, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->midtones, NULL, _("-04 EV : midtones"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->midtones, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->midtones), "value-changed", G_CALLBACK(midtones_callback), self);

  g->highlights = dt_bauhaus_slider_new_with_range(self, -2.0, 2.0, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->highlights, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->highlights, NULL, _("-02 EV : highlights"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->highlights, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->highlights), "value-changed", G_CALLBACK(highlights_callback), self);

  g->whites = dt_bauhaus_slider_new_with_range(self, -2.0, 2.0, 0.1, 0.0, 2);
  dt_bauhaus_slider_set_format(g->whites, "%+.2f EV");
  dt_bauhaus_widget_set_label(g->whites, NULL, _("-00 EV : whites"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->whites, TRUE, TRUE, 0);
  g_signal_connect(G_OBJECT(g->whites), "value-changed", G_CALLBACK(whites_callback), self);
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
