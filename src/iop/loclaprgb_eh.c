/*
    This file is part of darktable,
    copyright (c) 2009--2010 johannes hanika.
    copyright (c) 2014-2016 Roman Lebedev.

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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#include "bauhaus/bauhaus.h"
#include "common/histogram.h"
#include "common/image_cache.h"
#include "common/mipmap_cache.h"
#include "common/opencl.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/pixelpipe.h"
#include "dtgtk/paint.h"
#include "dtgtk/resetlabel.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"

//#include "common/loclap_eh.h"
#include "common/LocalLaplacian_den_eh.h"

DT_MODULE_INTROSPECTION(1, dt_iop_loclaprgb_params_t)

typedef struct dt_iop_loclaprgb_params_t
{
  float alpha[4];
  float beta[4];
  int channel;
} dt_iop_loclaprgb_params_t;

typedef struct dt_iop_loclaprgb_gui_data_t
{
  GtkWidget *sl_alpha;
  GtkWidget *sl_beta;
  GtkWidget *cmb_channel;
} dt_iop_loclaprgb_gui_data_t;

typedef struct dt_iop_loclaprgb_params_t dt_iop_loclaprgb_data_t;

const char *name()
{
  return _("loclaprgb_eh");
}

int groups()
{
  return IOP_GROUP_BASIC;
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING;
}


void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *params, dt_dev_pixelpipe_t *pipe,
    								dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, params, sizeof(dt_iop_loclaprgb_params_t));
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_loclaprgb_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(struct dt_iop_module_t *self)
{
  dt_iop_loclaprgb_gui_data_t *g = (dt_iop_loclaprgb_gui_data_t *)self->gui_data;
  dt_iop_loclaprgb_params_t *p = (dt_iop_loclaprgb_params_t *)self->params;

  dt_bauhaus_slider_set(g->sl_alpha, p->alpha[p->channel]);
  dt_bauhaus_slider_set(g->sl_beta, p->beta[p->channel]);
  dt_bauhaus_combobox_set(g->cmb_channel, p->channel);
  
}

void init(dt_iop_module_t *module)
{
	module->data = NULL;
  module->params = calloc(1, sizeof(dt_iop_loclaprgb_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_loclaprgb_params_t));
  module->default_enabled = 0;
  module->priority = 164; // module order created by iop_dependencies.py, do not edit! // from exposure
//  module->priority = 671; // from tonecurve
//  module->priority = 582; // module order created by iop_dependencies.py, do not edit! // from bilat (local contrast)
  module->params_size = sizeof(dt_iop_loclaprgb_params_t);
  module->gui_data = NULL;
  
  dt_iop_loclaprgb_params_t tmp = {0};
  
  memcpy(module->params, &tmp, sizeof(dt_iop_loclaprgb_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_loclaprgb_params_t));

}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

static void alpha_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;
  dt_iop_loclaprgb_params_t *p = (dt_iop_loclaprgb_params_t *)self->params;

  p->alpha[p->channel] = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void beta_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;
  dt_iop_loclaprgb_params_t *p = (dt_iop_loclaprgb_params_t *)self->params;

  p->beta[p->channel] = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void channel_callback(GtkComboBox *combo, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;
  
  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_loclaprgb_params_t *p = (dt_iop_loclaprgb_params_t *)self->params;
  dt_iop_loclaprgb_gui_data_t *g = (dt_iop_loclaprgb_gui_data_t *)self->gui_data;

  p->channel = dt_bauhaus_combobox_get((GtkWidget *)combo);
  
  dt_bauhaus_slider_set(g->sl_alpha, p->alpha[p->channel]);
  dt_bauhaus_slider_set(g->sl_beta, p->beta[p->channel]);
  
  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_init(struct dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_loclaprgb_gui_data_t));
  dt_iop_loclaprgb_gui_data_t *g = (dt_iop_loclaprgb_gui_data_t *)self->gui_data;

  self->widget = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE));

  g->cmb_channel = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->cmb_channel, NULL, _("channel"));
  dt_bauhaus_combobox_add(g->cmb_channel, _("all"));
  dt_bauhaus_combobox_add(g->cmb_channel, _("red"));
  dt_bauhaus_combobox_add(g->cmb_channel, _("green"));
  dt_bauhaus_combobox_add(g->cmb_channel, _("blue"));
  g_object_set(g->cmb_channel, "tooltip-text", _("channel."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->cmb_channel), "value-changed", G_CALLBACK(channel_callback), self);

  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->cmb_channel), TRUE, TRUE, 0);
  
  g->sl_alpha = dt_bauhaus_slider_new_with_range(self, -5.0, 5.0, 0.05, 0.0, 3);
  dt_bauhaus_widget_set_label(g->sl_alpha, _("alpha"), _("alpha"));
  g_object_set(g->sl_alpha, "tooltip-text", _("specifies how much of an effect to apply.\n"
            " 0 gives no effect,\n1 produces a moderate amount of contrast"
            " enhancement,\n-1 produces a piecewise flattening."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_alpha), "value-changed", G_CALLBACK(alpha_callback), self);
  
  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_alpha, TRUE, TRUE, 0);

  g->sl_beta = dt_bauhaus_slider_new_with_range(self, -5.0, 5.0, 0.05, 0.0, 3);
  dt_bauhaus_widget_set_label(g->sl_beta, _("beta"), _("beta"));
  g_object_set(g->sl_beta, "tooltip-text", _("specifies how the effect should change with respect to"
            " scale.\n0 applies the same effect at all scales,\n1 applies the effect only at fine scales,\n"
            "-1 applies the effectonly at coarse scales."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_beta), "value-changed", G_CALLBACK(beta_callback), self);
  
  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_beta, TRUE, TRUE, 0);

}

void gui_cleanup(struct dt_iop_module_t *self)
{
  free(self->gui_data);
  self->gui_data = NULL;
}

void process_internal(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out, const int use_sse)
{
  const dt_iop_loclaprgb_data_t *const p = (const dt_iop_loclaprgb_data_t *const)piece->data;

/*  loclap_eh((const float *const)ivoid,   // input buffer in some Labx or yuvx format
      (float *const) ovoid,           // output buffer with colour
			roi_out->width,               // width and
			roi_out->height,               // height of the input buffer
			0.2f,          // user param: separate shadows/midtones/highlights
      1,        // user param: lift shadows
      1,     // user param: compress highlights
      0,
			p->alpha);
*/
  
  const dt_iop_colorspace_type_t cst = dt_iop_module_colorspace(self);
  const int stride = roi_in->width * roi_in->height;
  const int ch = piece->colors;

  // just to be sure...
  if (cst != iop_cs_rgb) return;
  
  memcpy(ovoid, ivoid, roi_in->width * roi_in->height * ch * sizeof(float));
  
  float *image_L = NULL;
  image_L = dt_alloc_align(64, roi_in->width * roi_in->height * sizeof(float));
  
  for (int channel = 0; channel < 3; channel++)
  {
		float *in = (float*)ivoid;
		
		for (int i = 0; i < stride; i++, in += ch)
			image_L[i] = in[channel];
	
		loclapden_LocalLaplacian(image_L, image_L, roi_out->width, roi_out->height, 
															p->alpha[0]+p->alpha[channel+1], p->beta[0]+p->beta[channel+1], NULL, (roi_in->scale / piece->iscale), use_sse);
		
		float *out = (float*)ovoid;
		
		for (int i = 0; i < stride; i++, out += ch)
			out[channel] = image_L[i];
  }
  
  if (image_L) dt_free_align(image_L);
  
  if(piece->pipe->mask_display) dt_iop_alpha_copy(ivoid, ovoid, roi_out->width, roi_out->height);
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
	process_internal(self, piece, ivoid, ovoid, roi_in, roi_out, 0);
}

#if defined(__SSE__)
void process_sse2(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
    				const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
	process_internal(self, piece, ivoid, ovoid, roi_in, roi_out, 1);
}
#endif

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
