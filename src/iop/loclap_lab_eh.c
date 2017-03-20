/*
    This file is part of darktable,
    copyright (c) 2009--2014 johannes hanika.
    copyright (c) 2014 Ulrich Pegelow.

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
#include <stdlib.h>
#include "bauhaus/bauhaus.h"
#include "develop/LocalLaplacian.h"


DT_MODULE_INTROSPECTION(1, dt_iop_loclap_params_t)

typedef struct dt_iop_loclap_params_t
{
  float alpha;
  float beta;
} dt_iop_loclap_params_t;


typedef struct dt_iop_loclap_gui_data_t
{
  GtkWidget *sl_alpha;
  GtkWidget *sl_beta;
} dt_iop_loclap_gui_data_t;

typedef struct dt_iop_loclap_params_t dt_iop_loclap_data_t;


const char *name()
{
  return _("loclap_lab_eh");
}

int groups()
{
  return IOP_GROUP_TONE;
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING;
}

void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *params, dt_dev_pixelpipe_t *pipe,
    dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, params, sizeof(dt_iop_loclap_params_t));
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_loclap_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(struct dt_iop_module_t *self)
{
  dt_iop_loclap_gui_data_t *g = (dt_iop_loclap_gui_data_t *)self->gui_data;
  dt_iop_loclap_params_t *p = (dt_iop_loclap_params_t *)self->params;

  dt_bauhaus_slider_set(g->sl_alpha, p->alpha);
  dt_bauhaus_slider_set(g->sl_beta, p->beta);
  
}


void init(dt_iop_module_t *module)
{
  // we don't need global data:
  module->data = NULL;
  module->params = calloc(1, sizeof(dt_iop_loclap_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_loclap_params_t));
  module->default_enabled = 0;
  module->priority = 671; // module order created by iop_dependencies.py, do not edit!
  module->params_size = sizeof(dt_iop_loclap_params_t);
  module->gui_data = NULL;
  dt_iop_loclap_params_t tmp = (dt_iop_loclap_params_t){ 0 };
  memcpy(module->params, &tmp, sizeof(dt_iop_loclap_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_loclap_params_t));
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

static void _alpha_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  
  dt_iop_loclap_params_t *p = (dt_iop_loclap_params_t *)self->params;
  p->alpha = dt_bauhaus_slider_get(slider);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _beta_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  
  dt_iop_loclap_params_t *p = (dt_iop_loclap_params_t *)self->params;
  p->beta = dt_bauhaus_slider_get(slider);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_init(struct dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_loclap_gui_data_t));
  dt_iop_loclap_gui_data_t *g = (dt_iop_loclap_gui_data_t *)self->gui_data;

  self->widget = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE));

  g->sl_alpha = dt_bauhaus_slider_new_with_range(self, -5.0, 5.0, .02, 0, 3);
  dt_bauhaus_widget_set_label(g->sl_alpha, _("alpha"), _("alpha"));
  g_object_set(g->sl_alpha, "tooltip-text", _("specifies how much of an effect to apply.\n"
            " 0 gives no effect,\n1 produces a moderate amount of contrast"
            " enhancement,\n-1 produces a piecewise flattening."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_alpha), "value-changed", G_CALLBACK(_alpha_callback), self);

  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_alpha, TRUE, TRUE, 0);

  g->sl_beta = dt_bauhaus_slider_new_with_range(self, -5.0, 5.0, .02, 0, 3);
  dt_bauhaus_widget_set_label(g->sl_beta, _("beta"), _("beta"));
  g_object_set(g->sl_beta, "tooltip-text", _("specifies how the effect should change with respect to"
            " scale.\n0 applies the same effect at all scales,\n1 applies the effect only at fine scales,\n"
            "-1 applies the effectonly at coarse scales."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_beta), "value-changed", G_CALLBACK(_beta_callback), self);

  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_beta, TRUE, TRUE, 0);

}

void gui_cleanup(struct dt_iop_module_t *self)
{
  free(self->gui_data);
  self->gui_data = NULL;
}
/*
void loclap_copy_in_to_out(float *const in, const struct dt_iop_roi_t *const roi_ft, float *const out, const struct dt_iop_roi_t *const roi_out, const int ch)
{
  const int rowsize = roi_out->width*ch*sizeof(float);
  const int xoffs = roi_out->x - roi_ft->x;
  const int yoffs = roi_out->y - roi_ft->y;
  const int iwidth = roi_ft->width;

#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
#endif
  for (int y=0; y < roi_out->height; y++) // copy each row
  {
    size_t iindex = ((size_t)(y + yoffs) * iwidth + xoffs) * ch;
    size_t oindex = (size_t)y * roi_out->width * ch;
    float *in = (float *)in + iindex;
    float *out1 = (float *)out + oindex;

    memcpy(out1, in, rowsize);
  }
}
*/
void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
    void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_loclap_data_t *p = (dt_iop_loclap_data_t *)piece->data;
  const int ch = piece->colors;
  const dt_iop_colorspace_type_t cst = dt_iop_module_colorspace(self);
  const int stride = roi_in->width * roi_in->height;
  
  // just to be sure...
  if (cst != iop_cs_Lab) return;

//  float * image = NULL;
  float *image_L = NULL;
  
  memcpy(ovoid, ivoid, roi_in->width * roi_in->height * ch * sizeof(float));
  
//  image = dt_alloc_align(64, roi_in->width * roi_in->height * ch * sizeof(float));
//  memcpy(image, ivoid, roi_in->width * roi_in->height * ch * sizeof(float));

  image_L = dt_alloc_align(64, roi_in->width * roi_in->height * sizeof(float));
//  loclap_get_L_from_lab(image, image_L, roi_in->width, roi_in->height, 1);
  
  float *in = (float*)ivoid;
  
  for (int i = 0; i < stride; i++, in += ch)
    image_L[i] = *in;

  loclap_LocalLaplacian(image_L, image_L, roi_in->width, roi_in->height, p->alpha, p->beta);
//  loclap_get_L_from_lab(image, image_L, roi_in->width, roi_in->height, 0);

  float *out = (float*)ovoid;
  
  for (int i = 0; i < stride; i++, out += ch)
    (*out) = image_L[i];

//  loclap_copy_in_to_out(image, roi_in, ovoid, roi_out, ch);
  
//  if (image) dt_free_align(image);
  if (image_L) dt_free_align(image_L);

}

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-space on;
