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

#include "bauhaus/bauhaus.h"
#include "common/richardson_lucy_deconvolution.h"
//#include "develop/tiling.h"

DT_MODULE_INTROSPECTION(1, dt_iop_rlucydec_params_t)


typedef struct dt_iop_rlucydec_params_t
{
  int process_type;
} dt_iop_rlucydec_params_t;

typedef struct dt_iop_rlucydec_gui_data_t
{
  GtkWidget *cmb_process_type;
} dt_iop_rlucydec_gui_data_t;

typedef struct dt_iop_rlucydec_params_t dt_iop_rlucydec_data_t;

const char *name()
{
  return _("Richardson-Lucy deconvolution");
}

int groups()
{
  return IOP_GROUP_TONE;
}

int flags()
{
  return /*IOP_FLAGS_ALLOW_TILING |*/ IOP_FLAGS_SUPPORTS_BLENDING;
}

static void rt_process_type_callback(GtkComboBox *combo, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;
  
  dt_iop_rlucydec_params_t *p = (dt_iop_rlucydec_params_t *)self->params;

  p->process_type = dt_bauhaus_combobox_get((GtkWidget *)combo);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

/*
void tiling_callback(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in, const dt_iop_roi_t *roi_out,
                     struct dt_develop_tiling_t *tiling)
{
//  dt_iop_rlucydec_data_t *d = (dt_iop_rlucydec_data_t *)piece->data;

  tiling->factor = 3.0f; // in + out + tmp
  tiling->maxbuf = 1.0f;
  tiling->overhead = 0;
  tiling->overlap = 11 / 2;
  tiling->xalign = 1;
  tiling->yalign = 1;
  return;
}
*/
void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *params, dt_dev_pixelpipe_t *pipe,
    								dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, params, sizeof(dt_iop_rlucydec_params_t));
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_rlucydec_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(struct dt_iop_module_t *self)
{
  dt_iop_rlucydec_gui_data_t *g = (dt_iop_rlucydec_gui_data_t *)self->gui_data;
  dt_iop_rlucydec_params_t *p = (dt_iop_rlucydec_params_t *)self->params;

  dt_bauhaus_combobox_set(g->cmb_process_type, p->process_type);
  
}

void init(dt_iop_module_t *module)
{
	module->data = NULL;
  module->params = calloc(1, sizeof(dt_iop_rlucydec_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_rlucydec_params_t));
  module->default_enabled = 0;
//  module->priority = 164; // module order created by iop_dependencies.py, do not edit! // from exposure
  module->priority = 582; // module order created by iop_dependencies.py, do not edit! // from bilat
  module->params_size = sizeof(dt_iop_rlucydec_params_t);
  module->gui_data = NULL;
  
  dt_iop_rlucydec_params_t tmp = {0};
  
  tmp.process_type = rlucy_type_fast;
  
  memcpy(module->params, &tmp, sizeof(dt_iop_rlucydec_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_rlucydec_params_t));

}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

void gui_init(struct dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_rlucydec_gui_data_t));
  dt_iop_rlucydec_gui_data_t *g = (dt_iop_rlucydec_gui_data_t *)self->gui_data;

  self->widget = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE));

  g->cmb_process_type = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->cmb_process_type, NULL, _("process type"));
  dt_bauhaus_combobox_add(g->cmb_process_type, _("fast"));
  dt_bauhaus_combobox_add(g->cmb_process_type, _("blind"));
  dt_bauhaus_combobox_add(g->cmb_process_type, _("myope"));
  g_object_set(g->cmb_process_type, "tooltip-text", _("process type."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->cmb_process_type), "value-changed", G_CALLBACK(rt_process_type_callback), self);

  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->cmb_process_type), TRUE, TRUE, 0);

  
}

void gui_cleanup(struct dt_iop_module_t *self)
{
  free(self->gui_data);
  self->gui_data = NULL;
}

static void check_nan(const float* im, const int size)
{
	int i_nan = 0;
	
	for (int i = 0; i < size; i++) if ( isnan(im[i]) ) i_nan++;
	
	if (i_nan > 0) printf("Richardson-Lucy deconvolution nan: %i\n", i_nan);
}

void process_internal(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out, const int use_sse)
{
  const dt_iop_rlucydec_data_t *const p = (const dt_iop_rlucydec_data_t *const)piece->data;

  const dt_iop_colorspace_type_t cst = dt_iop_module_colorspace(self);
  const int ch = piece->colors;

  float *im_tmp = NULL;
  
  if (cst == iop_cs_rgb)
  {
		memcpy(ovoid, ivoid, roi_in->width * roi_in->height * ch * sizeof(float));
		
		richardson_lucy((float*)ovoid, roi_out->width, roi_out->height, ch, p->process_type);
  }
  else if (cst == iop_cs_Lab)
  {
  	memcpy(ovoid, ivoid, roi_in->width * roi_in->height * ch * sizeof(float));
  	
  	im_tmp = (float*)dt_alloc_align(64, roi_out->width * roi_out->height * sizeof(float));
		if (im_tmp)
		{
			float *im_src = (float*)ivoid;
			
			for (int i = 0; i < roi_out->width * roi_out->height; i++)
			{
				im_tmp[i] = im_src[i*ch];
			}
			
			richardson_lucy(im_tmp, roi_out->width, roi_out->height, 1, p->process_type);
			
			float *im_dest = (float*)ovoid;
			
			for (int i = 0; i < roi_out->width * roi_out->height; i++)
			{
				im_dest[i*ch] = im_tmp[i];
			}
		}
  }
  else
  {
  	memcpy(ovoid, ivoid, roi_in->width * roi_in->height * ch * sizeof(float));
  }
  
  check_nan((float*)ovoid, roi_out->width*roi_out->height*ch);
  
  if (im_tmp) dt_free_align(im_tmp);
  
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
	process_internal(self, piece, ivoid, ovoid, roi_in, roi_out, 0);
}

#if defined(__SSE__x)
void process_sse2(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
    				const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
	process_internal(self, piece, ivoid, ovoid, roi_in, roi_out, 1);
}
#endif

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
