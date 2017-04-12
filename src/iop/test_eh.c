/*
    This file is part of darktable,
    copyright (c) 2010 Henrik Andersson.

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
#include "bauhaus/bauhaus.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "dtgtk/button.h"
#include "dtgtk/resetlabel.h"
#include "dtgtk/togglebutton.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include <assert.h>
#include <gtk/gtk.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


DT_MODULE_INTROSPECTION(2, dt_iop_test_eh_params_t)

typedef struct dt_iop_test_eh_params_v1_t
{
  int num;
  int num2;
  char *text;
} dt_iop_test_eh_params_v1_t;

typedef struct dt_iop_test_eh_params_t
{
  int num;
  int num2;
  int num3;
  int num4;
  char *text;
} dt_iop_test_eh_params_t;

typedef struct dt_iop_test_eh_gui_data_t
{
  GtkWidget *text;
  GtkWidget *sl_num;
  GtkWidget *sl_num2;
} dt_iop_test_eh_gui_data_t;

typedef struct dt_iop_test_eh_params_t dt_iop_test_eh_data_t;


const char *name()
{
  return _("test_eh");
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

int groups()
{
  return IOP_GROUP_EFFECT;
}




static void text_callback(GtkWidget *entry, gpointer user_data)
{
  if(darktable.gui->reset) return;
  
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  dt_iop_test_eh_params_t *p = (dt_iop_test_eh_params_t *)self->params;
  
  size_t len = strlen(gtk_entry_get_text(GTK_ENTRY(entry))) + 1;
  p->text = realloc(p->text, len);
  snprintf(p->text, len, "%s", gtk_entry_get_text(GTK_ENTRY(entry)));
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_num_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_test_eh_params_t *p = (dt_iop_test_eh_params_t *)self->params;
  dt_iop_test_eh_gui_data_t *g = (dt_iop_test_eh_gui_data_t *)self->gui_data;
  int num = dt_bauhaus_slider_get(slider);
  
  size_t len = 1 + 3;
  if (p->text) len += strlen(p->text);
  p->text = realloc(p->text, len);
  snprintf(p->text, len, "%i %s", num, gtk_entry_get_text(GTK_ENTRY(g->text)));

  if (p->text)
    gtk_entry_set_text(GTK_ENTRY(g->text), p->text);
  else
    gtk_entry_set_text(GTK_ENTRY(g->text), "");

  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_num2_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  dt_iop_test_eh_params_t *p = (dt_iop_test_eh_params_t *)self->params;
  
  p->num2 = dt_bauhaus_slider_get(slider);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}



void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *params, dt_dev_pixelpipe_t *pipe,
    dt_dev_pixelpipe_iop_t *piece)
{
  copy_params(piece->data, params, sizeof(dt_iop_test_eh_params_t));
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_test_eh_data_t));
  memset(piece->data, 0, sizeof(dt_iop_test_eh_data_t));
  
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_test_eh_data_t *d = (dt_iop_test_eh_data_t *)piece->data;
  
  free_params(d);
  
  free(piece->data);
  piece->data = NULL;
}

void gui_update(struct dt_iop_module_t *self)
{
  dt_iop_module_t *module = (dt_iop_module_t *)self;
  dt_iop_test_eh_gui_data_t *g = (dt_iop_test_eh_gui_data_t *)self->gui_data;
  dt_iop_test_eh_params_t *p = (dt_iop_test_eh_params_t *)module->params;

  if (p->text)
    gtk_entry_set_text(GTK_ENTRY(g->text), p->text);
  else
    gtk_entry_set_text(GTK_ENTRY(g->text), "");
  
  dt_bauhaus_slider_set(g->sl_num, p->num);
  dt_bauhaus_slider_set(g->sl_num2, p->num2);
}

void init(dt_iop_module_t *module)
{
  module->params = calloc(1, sizeof(dt_iop_test_eh_params_t));
  module->params_size = sizeof(dt_iop_test_eh_params_t);
  module->default_params = calloc(1, sizeof(dt_iop_test_eh_params_t));
  module->default_enabled = 0;
  module->priority = 970; // module order created by iop_dependencies.py, do not edit!
  module->params_size = sizeof(dt_iop_test_eh_params_t);
  module->gui_data = NULL;
  dt_iop_test_eh_params_t tmp = (dt_iop_test_eh_params_t){0};
  memcpy(module->params, &tmp, sizeof(dt_iop_test_eh_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_test_eh_params_t));
}

void cleanup(dt_iop_module_t *module)
{
  dt_iop_test_eh_data_t *p = (dt_iop_test_eh_data_t *)module->params;
  
  free_params(p);
  
  free(module->params);
  module->params = NULL;
}

void gui_init(struct dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_test_eh_gui_data_t));
  dt_iop_test_eh_gui_data_t *g = (dt_iop_test_eh_gui_data_t *)self->gui_data;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  
  // Simple text
  GtkWidget *label = gtk_label_new(_("text"));
  gtk_widget_set_halign(label, GTK_ALIGN_START);
  g->text = gtk_entry_new();
  gtk_entry_set_width_chars(GTK_ENTRY(g->text), 1);
  gtk_widget_set_tooltip_text(g->text, _("text string, tag:\n$(WATERMARK_TEXT)"));
  dt_gui_key_accel_block_on_focus_connect(g->text);

  g_signal_connect(G_OBJECT(g->text), "changed", G_CALLBACK(text_callback), self);
  
  gtk_box_pack_start(GTK_BOX(self->widget), label, FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), g->text, FALSE, TRUE, 0);
  
  g->sl_num = dt_bauhaus_slider_new_with_range(self, 0.0, 15, 1, 0.0, 0);
  dt_bauhaus_widget_set_label(g->sl_num, _("num"), _("num"));
  g_object_set(g->sl_num, "tooltip-text", _("num."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_num), "value-changed", G_CALLBACK(rt_num_callback), self);
  
  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_num, TRUE, TRUE, 0);
  
  g->sl_num2 = dt_bauhaus_slider_new_with_range(self, 0.0, 15, 1, 0.0, 0);
  dt_bauhaus_widget_set_label(g->sl_num2, _("num2"), _("num2"));
  g_object_set(g->sl_num2, "tooltip-text", _("num2."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_num2), "value-changed", G_CALLBACK(rt_num2_callback), self);
  
  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_num2, TRUE, TRUE, 0);
}

void gui_cleanup(struct dt_iop_module_t *self)
{
  free(self->gui_data);
  self->gui_data = NULL;
}

static void decode_single_param(void *param, const void **source, int32_t param_size, int32_t *total_size)
{
  if (*total_size >= param_size)
  {
    memcpy(param, *source, param_size);
    *source += param_size;
    *total_size -= param_size;
  }

}

static void decode_single_param_txt(char **param_txt, const void **source, int32_t *total_size)
{
  int32_t tmp_size = 0;
  
  // decode string len
  decode_single_param(&tmp_size, source, sizeof(tmp_size), total_size);
  
  // actual text
  if (tmp_size > 0 && tmp_size >= *total_size)
  {
    *param_txt = realloc(*param_txt, tmp_size);
    
    memcpy(*param_txt, *source, tmp_size);
    
    *source += tmp_size;
    *total_size -= tmp_size;
  }
}

static void copy_string_param(char **dest_text, char *src_text)
{
  if (src_text == NULL)
  {
    if (*dest_text)
    {
      free(*dest_text);
      *dest_text = NULL;
    }
  }
  else
  {
    size_t len = strlen(src_text)+1;
    *dest_text = realloc(*dest_text, len);
    memcpy(*dest_text, src_text, len * sizeof(char));
  }

}

static void encode_single_param(void **encoded_params, void *param, int32_t param_size)
{
  memcpy(*encoded_params, param, param_size);
  *encoded_params += param_size;
}

static void encode_single_param_txt(void **encoded_params, char *param_txt)
{
  int32_t tmp_size = 0;
  
  if (param_txt) tmp_size = (strlen(param_txt) + 1) * sizeof(char);

  encode_single_param(encoded_params, &tmp_size, sizeof(tmp_size));
  
  if (tmp_size > 0)
  {
    memcpy(*encoded_params, param_txt, tmp_size);
    *encoded_params += tmp_size;
  }
}

static int32_t get_param_size_txt(char *param_txt)
{
  int32_t encoded_size = sizeof(int32_t);
  if (param_txt) encoded_size += (strlen(param_txt) + 1) * sizeof(char);
  return encoded_size;
}

static int single_para_equal_txt(char *param1, char *param2)
{
  int equal = 1;

  if (param1 && param2)
    equal &= (strcmp(param1, param2) == 0);
  else if (param1 || param2)
    equal = 0;

  return equal;
}

int32_t get_params_size_variable(struct dt_iop_module_t *self, void *const params)
{
  int32_t encoded_size = 0;
  dt_iop_test_eh_params_t *const p = (dt_iop_test_eh_params_t *const)params;
  
  encoded_size += sizeof(p->num);
  encoded_size += sizeof(p->num2);
  encoded_size += sizeof(p->num3);
  encoded_size += sizeof(p->num4);

  // text len
  encoded_size += get_param_size_txt(p->text);
  
  return encoded_size;
}

void encode_params(struct dt_iop_module_t *self, void *const decoded_params, void *encoded_params)
{
  dt_iop_test_eh_params_t *const decoded = (dt_iop_test_eh_params_t *const)decoded_params;
  void *encoded = encoded_params;
  
  // set new values to param struct
  
  // num
  encode_single_param(&encoded, &decoded->num, sizeof(decoded->num));
  // num2
  encode_single_param(&encoded, &decoded->num2, sizeof(decoded->num2));
  // num3
  encode_single_param(&encoded, &decoded->num3, sizeof(decoded->num3));
  // num4
  encode_single_param(&encoded, &decoded->num4, sizeof(decoded->num4));
  // text
  encode_single_param_txt(&encoded, decoded->text);
  
}

static int decode_params_v1_to_v2(struct dt_iop_module_t *self, const void *encoded_params, const int32_t encoded_size, const int encoded_version,
    void *decoded_params, const int decoded_version)
{
  if (encoded_version == 1 && decoded_version == 2)
  {
    const void *encoded = encoded_params;
    dt_iop_test_eh_params_t *decoded = (dt_iop_test_eh_params_t *)decoded_params;
  
    int32_t old_size = encoded_size;
    
    // num
    decode_single_param(&decoded->num, &encoded, sizeof(decoded->num), &old_size);
    // num2
    decode_single_param(&decoded->num2, &encoded, sizeof(decoded->num2), &old_size);
    // num3
    decoded->num3 = 0;
    // num4
    decoded->num4 = 0;
    // text
    decode_single_param_txt(&decoded->text, &encoded, &old_size);
    
    // if missing something return error
    if (old_size != 0)
    {
      printf("decode_params_v1_to_v2: size mismatch encoded_size=%i, old_size=%i\n", encoded_size, old_size);
      return 1;
    }
    
    return 0;
  }
  
  return 1;
}

int decode_params(struct dt_iop_module_t *self, const void *encoded_params, const int32_t encoded_size, const int encoded_version,
    void *decoded_params, const int decoded_version)
{
  if (encoded_version == 2 && decoded_version == 2)
  {
    const void *encoded = encoded_params;
    dt_iop_test_eh_params_t *decoded = (dt_iop_test_eh_params_t *)decoded_params;
  
    int32_t old_size = encoded_size;
    
    // num
    decode_single_param(&decoded->num, &encoded, sizeof(decoded->num), &old_size);
    // num2
    decode_single_param(&decoded->num2, &encoded, sizeof(decoded->num2), &old_size);
    // num3
    decode_single_param(&decoded->num3, &encoded, sizeof(decoded->num3), &old_size);
    // num4
    decode_single_param(&decoded->num4, &encoded, sizeof(decoded->num4), &old_size);
    // text
    decode_single_param_txt(&decoded->text, &encoded, &old_size);
    
    // if missing something return error
    if (old_size != 0)
    {
      printf("decode_params: size mismatch encoded_size=%i, old_size=%i\n", encoded_size, old_size);
      return 1;
    }
    
    return 0;
  }
  else if (encoded_version == 1 && decoded_version == 2)
  {
    return decode_params_v1_to_v2(self, encoded_params, encoded_size, encoded_version, decoded_params, decoded_version);
  }
  
  printf("decode_params() unknown version encoded_version=%i, decoded_version=%i\n", encoded_version, decoded_version);
  
  return 1;
}

void copy_params(void *dest_params, void *const source_params, int32_t params_size)
{
  dt_iop_test_eh_params_t *dest = (dt_iop_test_eh_params_t*)dest_params;
  dt_iop_test_eh_params_t *src = (dt_iop_test_eh_params_t*)source_params;
  
  dest->num = src->num;
  dest->num2 = src->num2;
  dest->num3 = src->num3;
  dest->num4 = src->num4;
  copy_string_param(&dest->text, src->text);
  
}

void free_params(void *params)
{
  if (params == NULL) return;
  
  dt_iop_test_eh_params_t *p = (dt_iop_test_eh_params_t*)params;
  
  if (p->text)
  {
    free(p->text);
    p->text = NULL;
  }
}

int params_equal(void *params1, void *params2)
{
  int equal = 1;
  dt_iop_test_eh_params_t *p1 = (dt_iop_test_eh_params_t*)params1;
  dt_iop_test_eh_params_t *p2 = (dt_iop_test_eh_params_t*)params2;
  
  equal &= (p1->num == p2->num);
  equal &= (p1->num2 == p2->num2);
  equal &= (p1->num3 == p2->num3);
  equal &= (p1->num4 == p2->num4);
  equal &= single_para_equal_txt(p1->text, p2->text);

  return equal;
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_test_eh_data_t *p = (dt_iop_test_eh_data_t *)piece->data;
  float *in = (float *)ivoid;
  float *out = (float *)ovoid;
  const int ch = piece->colors;
  
  memcpy(out, in, roi_out->width * roi_out->height * ch * sizeof(float));
  
  if (p->text)
    printf("%s\n", p->text);
  
}


// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
