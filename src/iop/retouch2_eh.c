/*
    This file is part of darktable,
    copyright (c) 2011 johannes hanika.

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
#include "develop/masks.h"
#include "develop/blend.h"
#include "develop/imageop_math.h"
#include "common/opencl.h"
#include "gui/accelerators.h"
#include "common/gaussian.h"
#include "common/heal_eh.h"
#include "common/dwt_eh.h"


// this is the version of the modules parameters,
// and includes version information about compile-time dt
DT_MODULE_INTROSPECTION(1, dt_iop_retouch2_params_t)

#define RETOUCH2_NO_FORMS 300
#define RETOUCH2_MAX_SCALES 15
#define RETOUCH2_NO_SCALES (RETOUCH2_MAX_SCALES+2)


typedef enum dt_iop_retouch2_preview_types_t
{
  dt_iop_rt_preview_final_image = 1,
  dt_iop_rt_preview_current_scale = 2,
  dt_iop_rt_preview_keep_current_scale = 3
} dt_iop_retouch2_preview_types_t;

typedef enum dt_iop_retouch2_fill_modes_t
{
  dt_iop_rt_fill_erase = 0,
  dt_iop_rt_fill_color = 1
} dt_iop_retouch2_fill_modes_t;

typedef enum dt_iop_retouch2_algo_type_t
{
  dt_iop_retouch2_clone = 1, 
  dt_iop_retouch2_heal = 2,
  dt_iop_retouch2_gaussian_blur = 3, 
  dt_iop_retouch2_fill = 4
} dt_iop_retouch2_algo_type_t;

typedef struct dt_iop_retouch2_form_data_t
{
  int formid; // from masks, form->formid
  int scale; // 0==original image; 1..RETOUCH2_MAX_SCALES==scale; RETOUCH2_MAX_SCALES+1==residual
  dt_iop_retouch2_algo_type_t algorithm; // clone, heal, blur, fill
  
  float blur_radius; // radius for blur algorithm
  
  int fill_mode; // mode for fill algorithm, erase or fill with color
  float fill_color[3]; // color for fill algorithm
  float fill_delta; // value to be added to the color
} dt_iop_retouch2_form_data_t;

typedef struct _rt_user_data_t
{
  dt_iop_module_t *self;
  dt_dev_pixelpipe_iop_t *piece;
  dt_iop_roi_t roi;
  int mask_display;
  int display_scale;
} _rt_user_data_t;


typedef struct dt_iop_retouch2_params_t
{
  dt_iop_retouch2_form_data_t rt_forms[RETOUCH2_NO_FORMS]; // array of masks index and additional data
  
  dt_iop_retouch2_algo_type_t algorithm; // clone, heal, blur, fill

  int num_scales; // number of wavelets scales
  int curr_scale; // current wavelet scale
  
  float blend_factor; // value to be added to each scale (for preview only)
  
  dt_iop_retouch2_preview_types_t preview_type; // final image, current scale...
  
  float blur_radius; // radius for blur algorithm
  
  int fill_mode; // mode for fill algorithm, erase or fill with color
  float fill_color[3]; // color for fill algorithm
  float fill_delta; // value to be added to the color
} dt_iop_retouch2_params_t;

typedef struct dt_iop_retouch2_gui_data_t
{
  dt_pthread_mutex_t lock;
  
  int copied_scale; // scale to be copied to another scale

  GtkLabel *label_form; // display number of forms
  GtkLabel *label_form_selected; // display number of forms selected
  GtkWidget *bt_edit_masks, *bt_path, *bt_circle, *bt_ellipse, *bt_brush; // shapes
  GtkWidget *bt_clone, *bt_heal, *bt_gaussian_blur, *bt_fill; // algorithms
  GtkWidget *bt_showmask, *bt_suppress; // supress & show masks
  
  GtkWidget *sl_num_scales; // number of scales to decompose
  GtkWidget *sl_curr_scale; // current wavelet scale
  GtkWidget *sl_blend_factor; // blend factor
  
  GtkWidget *bt_show_final_image; // show recoposed image
  GtkWidget *bt_show_current_scale; // show decomposed scale
  GtkWidget *bt_keep_current_scale; // keep showing decomposed scale even if module is not active

  GtkWidget *bt_copy_scale; // copy all shapes from one scale to another
  GtkWidget *bt_paste_scale;

  GtkWidget *hbox_blur;
  GtkWidget *sl_blur_radius;
 
  GtkWidget *hbox_color;
  GtkWidget *hbox_color_pick;
  GtkWidget *colorpick; // select a specific color
  GtkToggleButton *color_picker; // pick a color from the picture

  GtkWidget *cmb_fill_mode;
  GtkWidget *sl_fill_delta;
  
  GtkWidget *sl_mask_opacity; // draw mask opacity
} dt_iop_retouch2_gui_data_t;

typedef struct dt_iop_retouch2_params_t dt_iop_retouch2_data_t;

typedef struct dt_iop_retouch2_global_data_t
{
  int kernel_retouch_clear_alpha;
  int kernel_retouch_copy_alpha;
  int kernel_retouch_copy_buffer_to_buffer;
  int kernel_retouch_copy_buffer_to_image;
  int kernel_retouch_fill;
  int kernel_retouch_copy_image_to_buffer_masked;
  int kernel_retouch_copy_buffer_to_buffer_masked;
} dt_iop_retouch2_global_data_t;


// this returns a translatable name
const char *name()
{
  return _("retouch2_eh");
}

int groups()
{
  return IOP_GROUP_CORRECT;
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_NO_MASKS;
}

static void check_nan(const float* im, const int size)
{
	int i_nan = 0;
	
	for (int i = 0; i < size; i++) if ( isnan(im[i]) ) i_nan++;
	
	if (i_nan > 0) printf("retouch2_eh nan: %i\n", i_nan);
}


//---------------------------------------------------------------------------------
// draw buttons
//---------------------------------------------------------------------------------

#define PREAMBLE                                        \
  cairo_save (cr);                                      \
  const gint s = MIN (w, h);                            \
  cairo_translate (cr, x + (w / 2.0) - (s / 2.0),       \
                   y + (h / 2.0) - (s / 2.0));          \
  cairo_scale (cr, s, s);                               \
  cairo_push_group (cr);                                \
  cairo_set_source_rgba (cr, 1.0, 1.0, 1.0, 1.0);       \
  cairo_set_line_cap (cr, CAIRO_LINE_CAP_ROUND);        \
  cairo_set_line_width (cr, 0.1);

#define POSTAMBLE                                               \
  cairo_pop_group_to_source (cr);                               \
  cairo_paint_with_alpha (cr, flags & CPF_ACTIVE ? 1.0 : 0.5);  \
  cairo_restore (cr);

static void _retouch2_cairo_paint_tool_clone(cairo_t *cr,
                                          const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  cairo_arc(cr, 0.65, 0.35, 0.35, 0, 2 * M_PI);
  cairo_stroke(cr);
  
  cairo_arc(cr, 0.35, 0.65, 0.35, 0, 2 * M_PI);
  cairo_stroke(cr);

  POSTAMBLE;
}

static void _retouch2_cairo_paint_tool_heal(cairo_t *cr,
                                          const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  cairo_rectangle(cr, 0., 0., 1., 1.);
  cairo_fill(cr);

  cairo_set_source_rgba(cr, .74, 0.13, 0.13, 1.0);
  cairo_set_line_width(cr, 0.3);

  cairo_move_to(cr, 0.5, 0.18);
  cairo_line_to(cr, 0.5, 0.82);
  cairo_move_to(cr, 0.18, 0.5);
  cairo_line_to(cr, 0.82, 0.5);
  cairo_stroke(cr);

  POSTAMBLE;
}

static void _retouch2_cairo_paint_tool_fill(cairo_t *cr,
                                          const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  cairo_move_to(cr, 0.1, 0.1);
  cairo_line_to(cr, 0.2, 0.1);
  cairo_line_to(cr, 0.2, 0.9);
  cairo_line_to(cr, 0.8, 0.9);
  cairo_line_to(cr, 0.8, 0.1);
  cairo_line_to(cr, 0.9, 0.1);
  cairo_stroke(cr);
  cairo_rectangle(cr, 0.2, 0.4, .6, .5);
  cairo_fill(cr);
  cairo_stroke(cr);

  POSTAMBLE;
}

static void _retouch2_cairo_paint_tool_blur(cairo_t *cr, gint x, gint y, gint w, gint h, gint flags)
{
  PREAMBLE;
  
  cairo_pattern_t *pat = NULL;
  pat = cairo_pattern_create_radial(.5, .5, 0.005, .5, .5, .5);
  cairo_pattern_add_color_stop_rgba(pat, 0.0, 1, 1, 1, 1);
  cairo_pattern_add_color_stop_rgba(pat, 1.0, 1, 1, 1, 0.1);
  cairo_set_source(cr, pat);

  cairo_set_line_width(cr, 0.125);
  cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
  cairo_arc(cr, 0.5, 0.5, 0.45, 0, 2 * M_PI);
  cairo_fill(cr);

  cairo_pattern_destroy(pat);

  POSTAMBLE;
}

static void _retouch2_cairo_paint_paste_forms(cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  if((flags & CPF_ACTIVE))
  {
    cairo_set_source_rgba(cr, .75, 0.75, 0.75, 1.0);
    cairo_arc(cr, 0.5, 0.5, 0.40, 0, 2 * M_PI);
    cairo_fill(cr);
  }
  else
  {
    cairo_move_to(cr, 0.1, 0.5);
    cairo_line_to(cr, 0.9, 0.5);
    cairo_line_to(cr, 0.5, 0.9);
    cairo_line_to(cr, 0.1, 0.5);
    cairo_stroke(cr);
    cairo_move_to(cr, 0.1, 0.5);
    cairo_line_to(cr, 0.9, 0.5);
    cairo_line_to(cr, 0.5, 0.9);
    cairo_line_to(cr, 0.1, 0.5);
    cairo_fill(cr);
    
    cairo_move_to(cr, 0.4, 0.1);
    cairo_line_to(cr, 0.6, 0.1);
    cairo_line_to(cr, 0.6, 0.5);
    cairo_line_to(cr, 0.4, 0.5);
    cairo_stroke(cr);
    cairo_move_to(cr, 0.4, 0.1);
    cairo_line_to(cr, 0.6, 0.1);
    cairo_line_to(cr, 0.6, 0.5);
    cairo_line_to(cr, 0.4, 0.5);
    cairo_fill(cr);
  }

  POSTAMBLE;
}

static void _retouch2_cairo_paint_cut_forms(cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  if((flags & CPF_ACTIVE))
  {
    cairo_move_to(cr, 0.11, 0.25);
    cairo_line_to(cr, 0.89, 0.75);
    cairo_move_to(cr, 0.25, 0.11);
    cairo_line_to(cr, 0.75, 0.89);
    cairo_stroke(cr);
    
    cairo_arc(cr, 0.89, 0.53, 0.17, 0, 2 * M_PI);
    cairo_stroke(cr);
    
    cairo_arc(cr, 0.53, 0.89, 0.17, 0, 2 * M_PI);
    cairo_stroke(cr);
  }
  else
  {
    cairo_move_to(cr, 0.01, 0.35);
    cairo_line_to(cr, 0.99, 0.65);
    cairo_move_to(cr, 0.35, 0.01);
    cairo_line_to(cr, 0.65, 0.99);
    cairo_stroke(cr);
    
    cairo_arc(cr, 0.89, 0.53, 0.17, 0, 2 * M_PI);
    cairo_stroke(cr);
    
    cairo_arc(cr, 0.53, 0.89, 0.17, 0, 2 * M_PI);
    cairo_stroke(cr);
  }

  POSTAMBLE;
}

static void _retouch2_cairo_paint_show_final_image(cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  cairo_move_to(cr, 0.08, 1.);
  cairo_curve_to(cr, 0.4, 0.05, 0.6, 0.05, 1., 1.);
  cairo_line_to(cr, 0.08, 1.);
  cairo_fill(cr);
  
  cairo_set_line_width(cr, 0.1);
  cairo_rectangle(cr, 0., 0., 1., 1.);
  cairo_stroke(cr);

  POSTAMBLE;
}

static void _retouch2_cairo_paint_show_current_scale(cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  float x1 = 0.0f;
  float y1 = 1.f;
  
  cairo_move_to(cr, x1, y1);
  
  const int steps = 3;
  const float delta = 1. / (float)steps;
  for (int i=0; i<steps; i++)
  {
    y1 -= delta;
    cairo_line_to(cr, x1, y1);
    x1 += delta;
    cairo_line_to(cr, x1, y1);
  }
  cairo_stroke(cr);
  
  POSTAMBLE;
}

static void _retouch2_cairo_paint_keep_current_scale(cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  float x1 = 0.2f;
  float y1 = 1.f;
  
  cairo_move_to(cr, x1, y1);
  
  const int steps = 3;
  const float delta = 1. / (float)steps;
  for (int i=0; i<steps; i++)
  {
    y1 -= delta;
    cairo_line_to(cr, x1, y1);
    x1 += delta;
    cairo_line_to(cr, x1, y1);
  }
  cairo_stroke(cr);
  
  cairo_set_line_width(cr, 0.1);
  cairo_rectangle(cr, 0., 0., 1., 1.);
  cairo_stroke(cr);

  POSTAMBLE;
}


//---------------------------------------------------------------------------------
// shape selection
//---------------------------------------------------------------------------------

static int rt_get_index_from_formid(dt_iop_retouch2_params_t *p, const int formid)
{
  int index = -1;
  if (formid > 0)
  {
    int i = 0;
    
    while (index == -1 && i < RETOUCH2_NO_FORMS)
    {
      if (p->rt_forms[i].formid == formid) index = i;
      i++;
    }
  }
  return index;
}

static dt_iop_retouch2_algo_type_t rt_get_algorithm_from_formid(dt_iop_retouch2_params_t *p, const int formid)
{
  dt_iop_retouch2_algo_type_t algo = 0;
  if (formid > 0)
  {
    int i = 0;
    
    while (algo == 0 && i < RETOUCH2_NO_FORMS)
    {
      if (p->rt_forms[i].formid == formid) algo = p->rt_forms[i].algorithm;
      i++;
    }
  }
  return algo;
}

static int rt_get_selected_shape_id()
{
  return darktable.develop->mask_form_selected_id;
}

static dt_masks_point_group_t * rt_get_mask_point_group(dt_iop_module_t *self, int formid)
{
  dt_masks_point_group_t *form_point_group = NULL;
  
  dt_develop_blend_params_t *bp = self->blend_params;
  if (!bp) return form_point_group;
  
  dt_masks_form_t *grp = dt_masks_get_from_id(self->dev, bp->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    GList *forms = g_list_first(grp->points);
    while(forms)
    {
      dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;
      if (grpt->formid == formid)
      {
        form_point_group = grpt;
        break;
      }
      forms = g_list_next(forms);
    }
  }
  
  return form_point_group;
}

static float rt_get_shape_opacity(dt_iop_module_t *self, const int formid)
{
  float opacity = 0.f;
  
  dt_masks_point_group_t *grpt = rt_get_mask_point_group(self, formid);
  if (grpt) opacity = grpt->opacity;
  
  return opacity;
}

static void rt_display_selected_fill_color(dt_iop_retouch2_gui_data_t *g, dt_iop_retouch2_params_t *p)
{
  GdkRGBA c = (GdkRGBA){.red = p->fill_color[0], .green = p->fill_color[1], 
                          .blue = p->fill_color[2], .alpha = 1.0 };
  gtk_color_chooser_set_rgba(GTK_COLOR_CHOOSER(g->colorpick), &c);
}

static void rt_show_hide_controls(dt_iop_retouch2_gui_data_t *d, dt_iop_retouch2_params_t *p)
{
  switch (p->algorithm)
  {
  case dt_iop_retouch2_heal:
      gtk_widget_hide(GTK_WIDGET(d->hbox_blur));
      gtk_widget_hide(GTK_WIDGET(d->hbox_color));
      break;
    case dt_iop_retouch2_gaussian_blur:
      gtk_widget_show(GTK_WIDGET(d->hbox_blur));
      gtk_widget_hide(GTK_WIDGET(d->hbox_color));
      break;
    case dt_iop_retouch2_fill:
      gtk_widget_hide(GTK_WIDGET(d->hbox_blur));
      gtk_widget_show(GTK_WIDGET(d->hbox_color));
      if (p->fill_mode == dt_iop_rt_fill_color)
        gtk_widget_show(GTK_WIDGET(d->hbox_color_pick));
      else
        gtk_widget_hide(GTK_WIDGET(d->hbox_color_pick));
      break;
    case dt_iop_retouch2_clone:
    default:
      gtk_widget_hide(GTK_WIDGET(d->hbox_blur));
      gtk_widget_hide(GTK_WIDGET(d->hbox_color));
      break;
  }
}

static void rt_display_selected_shapes_lbl(dt_iop_retouch2_gui_data_t *g)
{
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, rt_get_selected_shape_id());
  if (form)
    gtk_label_set_text(g->label_form_selected, form->name);
  else
    gtk_label_set_text(g->label_form_selected, _(" "));
}

static int rt_get_selected_shape_index(dt_iop_retouch2_params_t *p)
{
  return rt_get_index_from_formid(p, rt_get_selected_shape_id());
}

static void rt_shape_selection_changed(dt_iop_module_t *self)
{
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  int selection_changed = 0;
  
  const int index = rt_get_selected_shape_index(p);
  if (index >= 0)
  {
    dt_bauhaus_slider_set(g->sl_mask_opacity, rt_get_shape_opacity(self, p->rt_forms[index].formid));
    
    if (p->rt_forms[index].algorithm == dt_iop_retouch2_gaussian_blur)
    {
      p->blur_radius = p->rt_forms[index].blur_radius;
      
      dt_bauhaus_slider_set(g->sl_blur_radius, p->blur_radius);
      
      selection_changed = 1;
    }
    else if (p->rt_forms[index].algorithm == dt_iop_retouch2_fill)
    {
      p->fill_mode = p->rt_forms[index].fill_mode;
      p->fill_delta = p->rt_forms[index].fill_delta;
      p->fill_color[0] = p->rt_forms[index].fill_color[0];
      p->fill_color[1] = p->rt_forms[index].fill_color[1];
      p->fill_color[2] = p->rt_forms[index].fill_color[2];
      
      dt_bauhaus_slider_set(g->sl_fill_delta, p->fill_delta);
      dt_bauhaus_combobox_set(g->cmb_fill_mode, p->fill_mode);
      rt_display_selected_fill_color(g, p);
      
      selection_changed = 1;
    }
  
    if (p->algorithm != p->rt_forms[index].algorithm)
    {
      p->algorithm = p->rt_forms[index].algorithm;
      
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_clone), (p->algorithm == dt_iop_retouch2_clone));
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_heal), (p->algorithm == dt_iop_retouch2_heal));
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_gaussian_blur), (p->algorithm == dt_iop_retouch2_gaussian_blur));
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_fill), (p->algorithm == dt_iop_retouch2_fill));
      
      selection_changed = 1;
    }
    
    if (selection_changed)
      rt_show_hide_controls(g, p);
  }
  
  rt_display_selected_shapes_lbl(g);
  
  darktable.gui->reset = reset;
  
  if (selection_changed)
    dt_dev_add_history_item(darktable.develop, self, TRUE);
}

//---------------------------------------------------------------------------------
// helpers
//---------------------------------------------------------------------------------

static void rt_masks_form_change_opacity(dt_iop_module_t *self, int formid, float opacity)
{
  if (opacity < 0.f || opacity > 1.f) return;
  
  dt_masks_point_group_t *grpt = rt_get_mask_point_group(self, formid);
  if (grpt)
  {
    grpt->opacity = opacity;
    
    dt_develop_blend_params_t *bp = self->blend_params;
    dt_masks_form_t *grp = dt_masks_get_from_id(self->dev, bp->mask_id);
    dt_masks_write_form(grp, darktable.develop);
    
    dt_dev_masks_list_update(darktable.develop);
  }
}

static void rt_paste_forms_from_scale(dt_iop_retouch2_params_t *p, const int source_scale, const int dest_scale)
{
  if (source_scale != dest_scale && source_scale >= 0 && dest_scale >= 0)
  {
    for (int i = 0; i < RETOUCH2_NO_FORMS; i++)
    {
      if (p->rt_forms[i].scale == source_scale)
        p->rt_forms[i].scale = dest_scale;
    }
  }
}

static int rt_allow_create_form(dt_iop_module_t *self)
{
  int allow = 1;
  
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  if (p)
  {
    allow = (p->rt_forms[RETOUCH2_NO_FORMS-1].formid == 0);
  }
  return allow;
}

static void rt_reset_form_creation(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;
  
  if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_path)) ||
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_circle)) ||
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_ellipse)) ||
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_brush)))
  {
    // we unset the creation mode
    dt_masks_change_form_gui(NULL);
  }
  
  if (widget != g->bt_path) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), FALSE);
  if (widget != g->bt_circle) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_circle), FALSE);
  if (widget != g->bt_ellipse) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_ellipse), FALSE);
  if (widget != g->bt_brush) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), FALSE);
  
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_showmask), FALSE);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_suppress), FALSE);
  gtk_toggle_button_set_active(g->color_picker, FALSE);
}

static void rt_show_forms_for_current_scale(dt_iop_module_t *self)
{
  if (!self->enabled || darktable.develop->gui_module != self) return;
  
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  if (bd == NULL) return;
  
  int scale = p->curr_scale;
  int count = 0;
  
  // check if there is a shape on this scale
  for (int i = 0; i < RETOUCH2_NO_FORMS && count == 0; i++)
  {
    if (p->rt_forms[i].formid != 0 && p->rt_forms[i].scale == scale) count++;
  }

  // if no shapes on this scale, we hide all
  if (bd->masks_shown == DT_MASKS_EDIT_OFF || count == 0)
  {
    dt_masks_change_form_gui(NULL);
    dt_control_queue_redraw_center();
    return;
  }
  
  // else, we create a new from group with the shapes and display it
  dt_masks_form_t *grp = dt_masks_create(DT_MASKS_GROUP);
  for (int i = 0; i < RETOUCH2_NO_FORMS; i++)
  {
    if (p->rt_forms[i].scale == scale)
    {
      int grid = self->blend_params->mask_id;
      int formid = p->rt_forms[i].formid;
      dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, formid);
      if(form)
      {
        dt_masks_point_group_t *fpt = (dt_masks_point_group_t *)malloc(sizeof(dt_masks_point_group_t));
        fpt->formid = formid;
        fpt->parentid = grid;
        fpt->state = DT_MASKS_STATE_USE;
        fpt->opacity = 1.0f;
        grp->points = g_list_append(grp->points, fpt);
      }
    }
  }

  dt_masks_form_t *grp2 = dt_masks_create(DT_MASKS_GROUP);
  grp2->formid = 0;
  dt_masks_group_ungroup(grp2, grp);
  dt_masks_change_form_gui(grp2);
  darktable.develop->form_gui->edit_mode = bd->masks_shown;
  dt_control_queue_redraw_center();
}

// called if a shape is added or deleted
static void rt_resynch_params(struct dt_iop_module_t *self)
{
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  dt_develop_blend_params_t *bp = self->blend_params;
  
  dt_iop_retouch2_form_data_t forms_d[RETOUCH2_NO_FORMS] = { 0 };
  
  // we go through all forms in blend params
  dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, bp->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    GList *forms = g_list_first(grp->points);
    int new_form_index = 0;
    while((new_form_index < RETOUCH2_NO_FORMS) && forms)
    {
      dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;
      
      // search for the form on the shapes array
      const int form_index = rt_get_index_from_formid(p, grpt->formid);
      
      // if it exists copy it to the new array
      if (form_index >= 0)
      {
        forms_d[new_form_index] = p->rt_forms[form_index];
 
        new_form_index++;
      }
      else
      {
        // if it does not exists add it to the new array
        dt_masks_form_t *parent_form = dt_masks_get_from_id(darktable.develop, grpt->formid);
        if (parent_form)
        {
        	forms_d[new_form_index].formid = grpt->formid;
        	forms_d[new_form_index].scale = p->curr_scale;
        	forms_d[new_form_index].algorithm = p->algorithm;
          
          switch (forms_d[new_form_index].algorithm)
          {
            case dt_iop_retouch2_gaussian_blur:
            	forms_d[new_form_index].blur_radius = p->blur_radius;
              break;
            case dt_iop_retouch2_fill:
            	forms_d[new_form_index].fill_mode = p->fill_mode;
            	forms_d[new_form_index].fill_color[0] = p->fill_color[0];
            	forms_d[new_form_index].fill_color[1] = p->fill_color[1];
            	forms_d[new_form_index].fill_color[2] = p->fill_color[2];
            	forms_d[new_form_index].fill_delta = p->fill_delta;
              break;
            default:
              break;
          }
          
          new_form_index++;
        }
      }
        
      forms = g_list_next(forms);
    }
  }

  // we reaffect params
  for(int i = 0; i < RETOUCH2_NO_FORMS; i++)
  {
    p->rt_forms[i] = forms_d[i];
  }
  
}

static gboolean rt_masks_form_is_in_roi(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
                                     dt_masks_form_t *form, const dt_iop_roi_t *roi_in,
                                     const dt_iop_roi_t *roi_out)
{
  // we get the area for the form
  int fl, ft, fw, fh;

  if(!dt_masks_get_area(self, piece, form, &fw, &fh, &fl, &ft)) return FALSE;

  // is the form outside of the roi?
  fw *= roi_in->scale, fh *= roi_in->scale, fl *= roi_in->scale, ft *= roi_in->scale;
  if(ft >= roi_out->y + roi_out->height || ft + fh <= roi_out->y || fl >= roi_out->x + roi_out->width
     || fl + fw <= roi_out->x)
    return FALSE;

  return TRUE;
}

static void rt_masks_point_denormalize(dt_dev_pixelpipe_iop_t *piece, const dt_iop_roi_t *roi,
                                    const float *points, size_t points_count, float *new)
{
  const float scalex = piece->pipe->iwidth * roi->scale, scaley = piece->pipe->iheight * roi->scale;

  for(size_t i = 0; i < points_count * 2; i += 2)
  {
    new[i] = points[i] * scalex;
    new[i + 1] = points[i + 1] * scaley;
  }
}

static int rt_masks_point_calc_delta(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
                                  const dt_iop_roi_t *roi, const float *target, const float *source, int *dx, int *dy)
{
  float points[4];
  rt_masks_point_denormalize(piece, roi, target, 1, points);
  rt_masks_point_denormalize(piece, roi, source, 1, points + 2);

  int res = dt_dev_distort_transform_plus(self->dev, piece->pipe, 0, self->priority, points, 2);
  if(!res) return res;

  *dx = points[0] - points[2];
  *dy = points[1] - points[3];

  return res;
}

static int rt_masks_get_delta(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const dt_iop_roi_t *roi,
                           dt_masks_form_t *form, int *dx, int *dy)
{
  int res = 0;

  if(form->type & DT_MASKS_PATH)
  {
    dt_masks_point_path_t *pt = (dt_masks_point_path_t *)form->points->data;

    res = rt_masks_point_calc_delta(self, piece, roi, pt->corner, form->source, dx, dy);
  }
  else if(form->type & DT_MASKS_CIRCLE)
  {
    dt_masks_point_circle_t *pt = (dt_masks_point_circle_t *)form->points->data;

    res = rt_masks_point_calc_delta(self, piece, roi, pt->center, form->source, dx, dy);
  }
  else if(form->type & DT_MASKS_ELLIPSE)
  {
    dt_masks_point_ellipse_t *pt = (dt_masks_point_ellipse_t *)form->points->data;

    res = rt_masks_point_calc_delta(self, piece, roi, pt->center, form->source, dx, dy);
  }
  else if(form->type & DT_MASKS_BRUSH)
  {
    dt_masks_point_brush_t *pt = (dt_masks_point_brush_t *)form->points->data;

    res = rt_masks_point_calc_delta(self, piece, roi, pt->corner, form->source, dx, dy);
  }

  return res;
}

//---------------------------------------------------------------------------------
// GUI callbacks
//---------------------------------------------------------------------------------

static void rt_request_pick_toggled_callback(GtkToggleButton *togglebutton, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  self->request_color_pick
      = (gtk_toggle_button_get_active(togglebutton) ? DT_REQUEST_COLORPICK_MODULE : DT_REQUEST_COLORPICK_OFF);

  // set the area sample size 
  if(self->request_color_pick != DT_REQUEST_COLORPICK_OFF)
  {
    dt_lib_colorpicker_set_point(darktable.lib, 0.5, 0.5);
    dt_dev_reprocess_all(self->dev);
  }
  else
  {
    dt_control_queue_redraw();
  }

  if(self->off) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(self->off), 1);
  dt_iop_request_focus(self);
}

static void rt_colorpick_color_set_callback(GtkColorButton *widget, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;

  // turn off the other color picker
  gtk_toggle_button_set_active(g->color_picker, FALSE);

  GdkRGBA c = (GdkRGBA){.red = p->fill_color[0], .green = p->fill_color[1], 
                          .blue = p->fill_color[2], .alpha = 1.0 };
  gtk_color_chooser_get_rgba(GTK_COLOR_CHOOSER(widget), &c);
  p->fill_color[0] = c.red;
  p->fill_color[1] = c.green;
  p->fill_color[2] = c.blue;

  const int index = rt_get_selected_shape_index(p);
  if (index >= 0)
  {
    if (p->rt_forms[index].algorithm == dt_iop_retouch2_fill)
    {
    	p->rt_forms[index].fill_color[0] = p->fill_color[0];
    	p->rt_forms[index].fill_color[1] = p->fill_color[1];
    	p->rt_forms[index].fill_color[2] = p->fill_color[2];
    }
  }
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static gboolean rt_draw_callback(GtkWidget *widget, cairo_t *cr, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return FALSE;
  if(self->picked_output_color_max[0] < 0) return FALSE;
  if(self->request_color_pick == DT_REQUEST_COLORPICK_OFF) return FALSE;
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;

  // interrupt if no valid color reading
  if(self->picked_output_color_min[0] == INFINITY) return FALSE;

  if(fabsf(p->fill_color[0] - self->picked_output_color[0]) < 0.0001f
     && fabsf(p->fill_color[1] - self->picked_output_color[1]) < 0.0001f
     && fabsf(p->fill_color[2] - self->picked_output_color[2]) < 0.0001f)
  {
    // interrupt infinite loops
    return FALSE;
  }

  p->fill_color[0] = self->picked_output_color[0];
  p->fill_color[1] = self->picked_output_color[1];
  p->fill_color[2] = self->picked_output_color[2];

  const int index = rt_get_selected_shape_index(p);
  if (index >= 0)
  {
    if (p->rt_forms[index].algorithm == dt_iop_retouch2_fill)
    {
    	p->rt_forms[index].fill_color[0] = p->fill_color[0];
    	p->rt_forms[index].fill_color[1] = p->fill_color[1];
    	p->rt_forms[index].fill_color[2] = p->fill_color[2];
    }
  }
  
  rt_display_selected_fill_color(g, p);
    
  dt_dev_add_history_item(darktable.develop, self, TRUE);
  
  return FALSE;
}

static void rt_copypaste_scale_callback(GtkToggleButton *togglebutton, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  int scale_copied = 0;
  int active = gtk_toggle_button_get_active(togglebutton);
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;

  if (togglebutton == (GtkToggleButton *)g->bt_copy_scale)
  {
    g->copied_scale = (active) ? p->curr_scale: -1;
  }
  else if (togglebutton == (GtkToggleButton *)g->bt_paste_scale)
  {
    rt_paste_forms_from_scale(p, g->copied_scale, p->curr_scale);
    scale_copied = 1;
    g->copied_scale = -1;
  }

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_copy_scale), g->copied_scale >= 0);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_paste_scale), g->copied_scale < 0);

  darktable.gui->reset = reset;
  
  if (scale_copied)
    dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_num_scales_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  p->num_scales = dt_bauhaus_slider_get(slider);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_preview_image_callback(GtkToggleButton *togglebutton, dt_iop_module_t *module)
{
  if(darktable.gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)module->params;
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)module->gui_data;

  if (togglebutton == (GtkToggleButton *)g->bt_show_final_image)
    p->preview_type = dt_iop_rt_preview_final_image;
  else if (togglebutton == (GtkToggleButton *)g->bt_show_current_scale)
    p->preview_type = dt_iop_rt_preview_current_scale;
  else if (togglebutton == (GtkToggleButton *)g->bt_keep_current_scale)
    p->preview_type = dt_iop_rt_preview_keep_current_scale;
  
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_show_final_image), (p->preview_type == dt_iop_rt_preview_final_image));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_show_current_scale), (p->preview_type == dt_iop_rt_preview_current_scale));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_keep_current_scale), (p->preview_type == dt_iop_rt_preview_keep_current_scale));

  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, module, TRUE);
}

static void rt_curr_scale_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  p->curr_scale = dt_bauhaus_slider_get(slider);
  
  rt_show_forms_for_current_scale(self);

  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_blend_factor_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  p->blend_factor = dt_bauhaus_slider_get(slider);
  
  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_mask_opacity_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  if (rt_get_selected_shape_id() > 0)
  {
    float opacity = dt_bauhaus_slider_get(slider);
    rt_masks_form_change_opacity(self, rt_get_selected_shape_id(), opacity);
  }

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static gboolean rt_edit_masks_callback(GtkWidget *widget, GdkEventButton *event, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return FALSE;
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;

  if(event->button == 1)
  {
    darktable.gui->reset = 1;

    dt_iop_request_focus(self);
    
    self->request_color_pick = DT_REQUEST_COLORPICK_OFF;
    gtk_toggle_button_set_active(g->color_picker, FALSE);

    dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, self->blend_params->mask_id);
    if(grp && (grp->type & DT_MASKS_GROUP) && g_list_length(grp->points) > 0)
    {
      const int control_button_pressed = event->state & GDK_CONTROL_MASK;

      switch(bd->masks_shown)
      {
        case DT_MASKS_EDIT_FULL:
          bd->masks_shown = control_button_pressed ? DT_MASKS_EDIT_RESTRICTED : DT_MASKS_EDIT_OFF;
          break;

        case DT_MASKS_EDIT_RESTRICTED:
          bd->masks_shown = !control_button_pressed ? DT_MASKS_EDIT_FULL : DT_MASKS_EDIT_OFF;
          break;

        default:
        case DT_MASKS_EDIT_OFF:
          bd->masks_shown = control_button_pressed ? DT_MASKS_EDIT_RESTRICTED : DT_MASKS_EDIT_FULL;
      }
    }
    else
      bd->masks_shown = DT_MASKS_EDIT_OFF;


    rt_show_forms_for_current_scale(self);
    
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), (bd->masks_shown != DT_MASKS_EDIT_OFF) && (darktable.develop->gui_module == self));
    
    darktable.gui->reset = 0;

    return TRUE;
  }

  return FALSE;
}

static gboolean rt_add_shape_callback(GtkWidget *widget, GdkEventButton *e, dt_iop_module_t *self)
{
  const int allow = rt_allow_create_form(self);
  if (allow)
  {
    rt_reset_form_creation(widget, self);
   
    if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget))) return FALSE;
    
    dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
    dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;

    // we want to be sure that the iop has focus
    dt_iop_request_focus(self);
    
    dt_masks_type_t type = DT_MASKS_CIRCLE;
    if (widget == g->bt_path)
      type = DT_MASKS_PATH;
    else if (widget == g->bt_circle)
      type = DT_MASKS_CIRCLE;
    else if (widget == g->bt_ellipse)
      type = DT_MASKS_ELLIPSE;
    else if (widget == g->bt_brush)
      type = DT_MASKS_BRUSH;
   
		// we create the new form
		dt_masks_form_t *spot = NULL;
		if (p->algorithm == dt_iop_retouch2_clone || p->algorithm == dt_iop_retouch2_heal)
		  spot = dt_masks_create(type | DT_MASKS_CLONE);
		else
		  spot = dt_masks_create(type);
		
		dt_masks_change_form_gui(spot);
		darktable.develop->form_gui->creation = TRUE;
		darktable.develop->form_gui->creation_module = self;
		dt_control_queue_redraw_center();
  }
  else
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget), FALSE);
    
  return !allow;
}

static void rt_select_algorithm_callback(GtkToggleButton *togglebutton, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;

  if (togglebutton == (GtkToggleButton *)g->bt_gaussian_blur)
    p->algorithm = dt_iop_retouch2_gaussian_blur;
  else if (togglebutton == (GtkToggleButton *)g->bt_clone)
    p->algorithm = dt_iop_retouch2_clone;
  else if (togglebutton == (GtkToggleButton *)g->bt_heal)
  {
    p->algorithm = dt_iop_retouch2_heal;
  }
  else if (togglebutton == (GtkToggleButton *)g->bt_fill)
    p->algorithm = dt_iop_retouch2_fill;

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_clone), (p->algorithm == dt_iop_retouch2_clone));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_heal), (p->algorithm == dt_iop_retouch2_heal));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_gaussian_blur), (p->algorithm == dt_iop_retouch2_gaussian_blur));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_fill), (p->algorithm == dt_iop_retouch2_fill));

  rt_show_hide_controls(g, p);
  
  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_showmask_callback(GtkToggleButton *togglebutton, dt_iop_module_t *module)
{
  module->request_mask_display = gtk_toggle_button_get_active(togglebutton);
  if(darktable.gui->reset) return;

  if(module->off) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(module->off), 1);
  dt_iop_request_focus(module);

  dt_dev_reprocess_all(module->dev);
}

static void rt_suppress_callback(GtkToggleButton *togglebutton, dt_iop_module_t *module)
{
  module->suppress_mask = gtk_toggle_button_get_active(togglebutton);
  if(darktable.gui->reset) return;

  if(module->off) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(module->off), 1);
  dt_iop_request_focus(module);

  dt_dev_reprocess_all(module->dev);
}

static void rt_blur_radius_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  
  p->blur_radius = dt_bauhaus_slider_get(slider);
  
  const int index = rt_get_selected_shape_index(p);
  if (index >= 0)
  {
    if (p->rt_forms[index].algorithm == dt_iop_retouch2_gaussian_blur)
    {
    	p->rt_forms[index].blur_radius = p->blur_radius;
    }
  }
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_fill_mode_callback(GtkComboBox *combo, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;
  
  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;

  p->fill_mode = dt_bauhaus_combobox_get((GtkWidget *)combo);
  
  const int index = rt_get_selected_shape_index(p);
  if (index >= 0)
  {
    if (p->rt_forms[index].algorithm == dt_iop_retouch2_fill)
    {
    	p->rt_forms[index].fill_mode = p->fill_mode;
    }
  }
  
  rt_show_hide_controls(g, p);
  
  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_fill_delta_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;

  p->fill_delta = dt_bauhaus_slider_get(slider);
  
  const int index = rt_get_selected_shape_index(p);
  if (index >= 0)
  {
    if (p->rt_forms[index].algorithm == dt_iop_retouch2_fill)
    {
    	p->rt_forms[index].fill_delta = p->fill_delta;
    }
  }
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

//--------------------------------------------------------------------------------------------------
// GUI
//--------------------------------------------------------------------------------------------------

void masks_selection_changed(struct dt_iop_module_t *self, const int form_selected_id)
{
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;
  if (!g) return;

  dt_pthread_mutex_lock(&g->lock);
  
  rt_shape_selection_changed(self);
  
  dt_pthread_mutex_unlock(&g->lock);
}

void init(dt_iop_module_t *module)
{
  module->params = calloc(1, sizeof(dt_iop_retouch2_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_retouch2_params_t));
  // our module is disabled by default
  module->default_enabled = 0;
  module->priority = 189; // module order created by iop_dependencies.py, do not edit!
  module->params_size = sizeof(dt_iop_retouch2_params_t);
  module->gui_data = NULL;
  
  // init defaults:
  dt_iop_retouch2_params_t tmp = {0};
  
  tmp.algorithm = dt_iop_retouch2_clone, 
  tmp.num_scales = 0, 
  tmp.curr_scale = 0, 
  tmp.blend_factor = 0.128f,
                    
  tmp.preview_type = dt_iop_rt_preview_final_image;
  
  tmp.blur_radius = 0.0f;
  
  tmp.fill_mode = dt_iop_rt_fill_erase;
  tmp.fill_color[0] = tmp.fill_color[1] = tmp.fill_color[2] = 0.f;
  tmp.fill_delta = 0.f;
  
  memcpy(module->params, &tmp, sizeof(dt_iop_retouch2_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_retouch2_params_t));
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

void init_global(dt_iop_module_so_t *module)
{
  const int program = 21; // retouch_eh.cl, from programs.conf
  dt_iop_retouch2_global_data_t *gd
      = (dt_iop_retouch2_global_data_t *)malloc(sizeof(dt_iop_retouch2_global_data_t));
  module->data = gd;
  gd->kernel_retouch_clear_alpha = dt_opencl_create_kernel(program, "retouch_clear_alpha");
  gd->kernel_retouch_copy_alpha = dt_opencl_create_kernel(program, "retouch_copy_alpha");
  gd->kernel_retouch_copy_buffer_to_buffer = dt_opencl_create_kernel(program, "retouch_copy_buffer_to_buffer");
  gd->kernel_retouch_copy_buffer_to_image = dt_opencl_create_kernel(program, "retouch_copy_buffer_to_image");
  gd->kernel_retouch_fill = dt_opencl_create_kernel(program, "retouch_fill");
  gd->kernel_retouch_copy_image_to_buffer_masked = dt_opencl_create_kernel(program, "retouch_copy_image_to_buffer_masked");
  gd->kernel_retouch_copy_buffer_to_buffer_masked = dt_opencl_create_kernel(program, "retouch_copy_buffer_to_buffer_masked");
  
}

void cleanup_global(dt_iop_module_so_t *module)
{
	dt_iop_retouch2_global_data_t *gd = (dt_iop_retouch2_global_data_t *)module->data;
	
  dt_opencl_free_kernel(gd->kernel_retouch_clear_alpha);
  dt_opencl_free_kernel(gd->kernel_retouch_copy_alpha);
  dt_opencl_free_kernel(gd->kernel_retouch_copy_buffer_to_buffer);
  dt_opencl_free_kernel(gd->kernel_retouch_copy_buffer_to_image);
  dt_opencl_free_kernel(gd->kernel_retouch_fill);
  dt_opencl_free_kernel(gd->kernel_retouch_copy_image_to_buffer_masked);
  dt_opencl_free_kernel(gd->kernel_retouch_copy_buffer_to_buffer_masked);
  
  free(module->data);
  module->data = NULL;
}

void gui_focus(struct dt_iop_module_t *self, gboolean in)
{
  if(self->enabled && !darktable.develop->image_loading)
  {
    if(in)
    {
      dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
      if (bd)
      {
        // got focus, show all shapes
        if (bd->masks_shown == DT_MASKS_EDIT_OFF)
          dt_masks_set_edit_mode(self, DT_MASKS_EDIT_FULL);

        rt_show_forms_for_current_scale(self);

        dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), (bd->masks_shown != DT_MASKS_EDIT_OFF) && (darktable.develop->gui_module == self));
      }
    }
    else
    {
      // lost focus, hide all shapes and free if some are in creation
      if (darktable.develop->form_gui->creation && darktable.develop->form_gui->creation_module == self)
        dt_masks_change_form_gui(NULL);

      dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;
      
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), FALSE);
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_circle), FALSE);
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_ellipse), FALSE);
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), FALSE);
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), FALSE);
      
      dt_masks_set_edit_mode(self, DT_MASKS_EDIT_OFF);
    }
    
    dt_dev_reprocess_all(self->dev);
  }
}

/** commit is the synch point between core and gui, so it copies params to pipe data. */
void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *params, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  memcpy(piece->data, params, sizeof(dt_iop_retouch2_params_t));
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_retouch2_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;

  // check if there is new or deleted forms
  rt_resynch_params(self);
  
  // update clones count
  dt_masks_form_t *grp = dt_masks_get_from_id(self->dev, self->blend_params->mask_id);
  guint nb = 0;
  if(grp && (grp->type & DT_MASKS_GROUP)) nb = g_list_length(grp->points);
  gchar *str = g_strdup_printf("%d", nb);
  gtk_label_set_text(g->label_form, str);
  g_free(str);

  // update selected shape label
  rt_display_selected_shapes_lbl(g);

  // show the shapes for the current scale
  rt_show_forms_for_current_scale(self);
  
  // enable/disable algorithm toolbar
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_clone), p->algorithm == dt_iop_retouch2_clone);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_gaussian_blur), p->algorithm == dt_iop_retouch2_gaussian_blur);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_heal), p->algorithm == dt_iop_retouch2_heal);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_fill), p->algorithm == dt_iop_retouch2_fill);
  
  // enable/disable shapes toolbar
  int b1 = 0, b2 = 0, b3 = 0, b4 = 0;
  if(self->dev->form_gui && self->dev->form_visible && self->dev->form_gui->creation
     && self->dev->form_gui->creation_module == self)
  {
    if(self->dev->form_visible->type & DT_MASKS_CIRCLE)
      b1 = 1;
    else if(self->dev->form_visible->type & DT_MASKS_PATH)
      b2 = 1;
    else if(self->dev->form_visible->type & DT_MASKS_ELLIPSE)
      b3 = 1;
    else if(self->dev->form_visible->type & DT_MASKS_BRUSH)
      b4 = 1;
  }
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_circle), b1);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), b2);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_ellipse), b3);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), b4);

  // update the rest of the fields
  dt_bauhaus_slider_set(g->sl_num_scales, p->num_scales);
  dt_bauhaus_slider_set(g->sl_curr_scale, p->curr_scale);
  dt_bauhaus_slider_set(g->sl_blend_factor, p->blend_factor);

  dt_bauhaus_slider_set(g->sl_blur_radius, p->blur_radius);
  dt_bauhaus_slider_set(g->sl_fill_delta, p->fill_delta);
  dt_bauhaus_combobox_set(g->cmb_fill_mode, p->fill_mode);
  
  rt_display_selected_fill_color(g, p);

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_show_final_image), p->preview_type==dt_iop_rt_preview_final_image);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_show_current_scale), p->preview_type==dt_iop_rt_preview_current_scale);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_keep_current_scale), p->preview_type==dt_iop_rt_preview_keep_current_scale);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_copy_scale), g->copied_scale >= 0);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_paste_scale), g->copied_scale < 0);
  
  // show/hide some fields
  rt_show_hide_controls(g, p);
  
  // update edit shapes status
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  if (bd)
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), (bd->masks_shown != DT_MASKS_EDIT_OFF) && (darktable.develop->gui_module == self));
  }
  else
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), FALSE);
  }
  
}

void gui_init(dt_iop_module_t *self)
{
  const int bs = DT_PIXEL_APPLY_DPI(14);
  
  self->gui_data = malloc(sizeof(dt_iop_retouch2_gui_data_t));
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)self->params;

  dt_pthread_mutex_init(&g->lock, NULL);
  g->copied_scale = -1;
  
  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  
  // shapes toolbar
  GtkWidget *hbox_shapes = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  
  GtkWidget *label = gtk_label_new(_("# shapes:"));
  gtk_box_pack_start(GTK_BOX(hbox_shapes), label, FALSE, TRUE, 0);
  g->label_form = GTK_LABEL(gtk_label_new("-1"));
  g_object_set(G_OBJECT(hbox_shapes), "tooltip-text", _("to add a shape select an algorithm and a shape and click on the image.\nshapes are added to the current scale."), (char *)NULL);

  g->bt_brush = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_brush, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_brush), "button-press-event", G_CALLBACK(rt_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_brush), "tooltip-text", _("add brush"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_brush), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox_shapes), g->bt_brush, FALSE, FALSE, 0);

  g->bt_path = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_path, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_path), "button-press-event", G_CALLBACK(rt_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_path), "tooltip-text", _("add path"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_path), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox_shapes), g->bt_path, FALSE, FALSE, 0);

  g->bt_ellipse = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_ellipse, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_ellipse), "button-press-event", G_CALLBACK(rt_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_ellipse), "tooltip-text", _("add ellipse"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_ellipse), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_ellipse), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox_shapes), g->bt_ellipse, FALSE, FALSE, 0);

  g->bt_circle = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_circle, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_circle), "button-press-event", G_CALLBACK(rt_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_circle), "tooltip-text", _("add circle"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_circle), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_circle), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox_shapes), g->bt_circle, FALSE, FALSE, 0);

  g->bt_edit_masks = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_eye, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_edit_masks), "button-press-event", G_CALLBACK(rt_edit_masks_callback), self);
  g_object_set(G_OBJECT(g->bt_edit_masks), "tooltip-text", _("show and edit shapes on the current scale"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_edit_masks), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox_shapes), g->bt_edit_masks, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(hbox_shapes), GTK_WIDGET(g->label_form), FALSE, TRUE, 0);
  
  // algorithm toolbar
  GtkWidget *hbox_algo = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  
  GtkWidget *label2 = gtk_label_new(_("algorithms:"));
  gtk_box_pack_start(GTK_BOX(hbox_algo), label2, FALSE, TRUE, 0);
  
  g->bt_fill = dtgtk_togglebutton_new(_retouch2_cairo_paint_tool_fill, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_fill), "tooltip-text", _("activates fill tool"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_fill), "toggled", G_CALLBACK(rt_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_fill), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_fill), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_fill, FALSE, FALSE, 0);

  g->bt_gaussian_blur = dtgtk_togglebutton_new(_retouch2_cairo_paint_tool_blur, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_gaussian_blur), "tooltip-text", _("activates gaussian blur tool"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_gaussian_blur), "toggled", G_CALLBACK(rt_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_gaussian_blur), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_gaussian_blur), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_gaussian_blur, FALSE, FALSE, 0);

  g->bt_heal = dtgtk_togglebutton_new(_retouch2_cairo_paint_tool_heal, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_heal), "tooltip-text", _("activates healing tool"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_heal), "toggled", G_CALLBACK(rt_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_heal), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_heal), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_heal, FALSE, FALSE, 0);

  g->bt_clone = dtgtk_togglebutton_new(_retouch2_cairo_paint_tool_clone, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_clone), "tooltip-text", _("activates cloning tool"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_clone), "toggled", G_CALLBACK(rt_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_clone), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_clone), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_clone, FALSE, FALSE, 0);

  // shapes selected (label)
  GtkWidget *hbox_shape_sel = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  GtkWidget *label1 = gtk_label_new(_("shape selected:"));
  gtk_box_pack_start(GTK_BOX(hbox_shape_sel), label1, FALSE, TRUE, 0);
  g->label_form_selected = GTK_LABEL(gtk_label_new("-1"));
  g_object_set(G_OBJECT(hbox_shape_sel), "tooltip-text", _("click on a shape to select it\nto unselect click on the image."), (char *)NULL);
  gtk_box_pack_start(GTK_BOX(hbox_shape_sel), GTK_WIDGET(g->label_form_selected), FALSE, TRUE, 0);

  // suppress or show masks toolbar
  GtkWidget *hbox_show_hide = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  
  GtkWidget *label3 = gtk_label_new(_("masks display:"));
  gtk_box_pack_start(GTK_BOX(hbox_show_hide), label3, FALSE, TRUE, 0);
  
  g->bt_showmask = dtgtk_togglebutton_new(dtgtk_cairo_paint_showmask, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_showmask), "tooltip-text", _("display mask"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_showmask), "toggled", G_CALLBACK(rt_showmask_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_showmask), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_showmask), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_show_hide), g->bt_showmask, FALSE, FALSE, 0);

  g->bt_suppress = dtgtk_togglebutton_new(dtgtk_cairo_paint_eye_toggle, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_suppress), "tooltip-text", _("temporarily switch off shapes"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_suppress), "toggled", G_CALLBACK(rt_suppress_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_suppress), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_suppress), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_show_hide), g->bt_suppress, FALSE, FALSE, 0);

  // blur radius for blur algorithm
  g->hbox_blur = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);

  g->sl_blur_radius = dt_bauhaus_slider_new_with_range(self, 0.0, 200.0, 0.1, 0.1, 1);
  dt_bauhaus_widget_set_label(g->sl_blur_radius, _("blur radius"), _("blur radius"));
  dt_bauhaus_slider_set_format(g->sl_blur_radius, "%.01f");
  g_object_set(g->sl_blur_radius, "tooltip-text", _("radius of gaussian blur."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_blur_radius), "value-changed", G_CALLBACK(rt_blur_radius_callback), self);

  gtk_box_pack_start(GTK_BOX(g->hbox_blur), g->sl_blur_radius, TRUE, TRUE, 0);

  // number of scales and display final image/current scale
  GtkWidget *hbox_scale = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);

  g->sl_num_scales = dt_bauhaus_slider_new_with_range(self, 0.0, RETOUCH2_MAX_SCALES, 1, 0.0, 0);
  dt_bauhaus_widget_set_label(g->sl_num_scales, _("number of scales"), _("scale"));
  g_object_set(g->sl_num_scales, "tooltip-text", _("number of scales to decompose."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_num_scales), "value-changed", G_CALLBACK(rt_num_scales_callback), self);
  
  g->bt_show_final_image = dtgtk_togglebutton_new(_retouch2_cairo_paint_show_final_image, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_show_final_image), "tooltip-text", _("show final image"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_show_final_image), "toggled", G_CALLBACK(rt_preview_image_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_show_final_image), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_show_final_image), FALSE);

  g->bt_show_current_scale = dtgtk_togglebutton_new(_retouch2_cairo_paint_show_current_scale, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_show_current_scale), "tooltip-text", _("show current scale"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_show_current_scale), "toggled", G_CALLBACK(rt_preview_image_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_show_current_scale), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_show_current_scale), FALSE);
  
  g->bt_keep_current_scale = dtgtk_togglebutton_new(_retouch2_cairo_paint_keep_current_scale, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_keep_current_scale), "tooltip-text", _("keep current scale"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_keep_current_scale), "toggled", G_CALLBACK(rt_preview_image_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_keep_current_scale), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_keep_current_scale), FALSE);

  gtk_box_pack_end(GTK_BOX(hbox_scale), g->bt_keep_current_scale, FALSE, FALSE, 0);
  gtk_box_pack_end(GTK_BOX(hbox_scale), g->bt_show_current_scale, FALSE, FALSE, 0);
  gtk_box_pack_end(GTK_BOX(hbox_scale), g->bt_show_final_image, FALSE, FALSE, 0);
  gtk_box_pack_start(GTK_BOX(hbox_scale), g->sl_num_scales, TRUE, TRUE, 0);
  
  // current scale and copy/paste shapes
  GtkWidget *hbox_n_image = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  
  g->sl_curr_scale = dt_bauhaus_slider_new_with_range(self, 0.0, RETOUCH2_MAX_SCALES+1, 1, 0.0, 0);
  dt_bauhaus_widget_set_label(g->sl_curr_scale, _("current scale"), _("current scale"));
  g_object_set(g->sl_curr_scale, "tooltip-text", _("current decomposed scale, if zero, final image is displayed."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_curr_scale), "value-changed", G_CALLBACK(rt_curr_scale_callback), self);
  
  g->bt_copy_scale = dtgtk_togglebutton_new(_retouch2_cairo_paint_cut_forms, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_copy_scale), "tooltip-text", _("cut shapes from current scale."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_copy_scale), "toggled", G_CALLBACK(rt_copypaste_scale_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_copy_scale), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_copy_scale), FALSE);
  
  g->bt_paste_scale = dtgtk_togglebutton_new(_retouch2_cairo_paint_paste_forms, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_paste_scale), "tooltip-text", _("paste cutted shapes to current scale."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_paste_scale), "toggled", G_CALLBACK(rt_copypaste_scale_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_paste_scale), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_paste_scale), FALSE);
  
  gtk_box_pack_end(GTK_BOX(hbox_n_image), g->bt_paste_scale, FALSE, FALSE, 0);
  gtk_box_pack_end(GTK_BOX(hbox_n_image), g->bt_copy_scale, FALSE, FALSE, 0);
  gtk_box_pack_start(GTK_BOX(hbox_n_image), g->sl_curr_scale, TRUE, TRUE, 0);
  
  // blend factor
  GtkWidget *hbox_blend = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  
  g->sl_blend_factor = dt_bauhaus_slider_new_with_range(self, 0.0, 1.0, 0.05, 0.128, 3);
  dt_bauhaus_widget_set_label(g->sl_blend_factor, _("blend factor"), _("blend factor"));
  g_object_set(g->sl_blend_factor, "tooltip-text", _("blend factor, increse to lighten the image scale. works only for preview."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_blend_factor), "value-changed", G_CALLBACK(rt_blend_factor_callback), self);
  
  gtk_box_pack_start(GTK_BOX(hbox_blend), g->sl_blend_factor, TRUE, TRUE, 0);
  
  // color for fill algorithm
  GdkRGBA color = (GdkRGBA){.red = p->fill_color[0], 
    .green = p->fill_color[1], 
    .blue = p->fill_color[2], 
    .alpha = 1.0 };

  g->hbox_color = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  g->hbox_color_pick = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);

  g->colorpick = gtk_color_button_new_with_rgba(&color);
  gtk_color_chooser_set_use_alpha(GTK_COLOR_CHOOSER(g->colorpick), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->colorpick), bs, bs);
  gtk_color_button_set_title(GTK_COLOR_BUTTON(g->colorpick), _("select fill color"));

  g_signal_connect(G_OBJECT(g->colorpick), "color-set", G_CALLBACK(rt_colorpick_color_set_callback), self);

  g->color_picker = GTK_TOGGLE_BUTTON(dtgtk_togglebutton_new(dtgtk_cairo_paint_colorpicker, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER));
  g_object_set(G_OBJECT(g->color_picker), "tooltip-text", _("pick fill color from image"), (char *)NULL);
  gtk_widget_set_size_request(GTK_WIDGET(g->color_picker), bs, bs);
  g_signal_connect(G_OBJECT(g->color_picker), "toggled", G_CALLBACK(rt_request_pick_toggled_callback), self);

  g->cmb_fill_mode = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->cmb_fill_mode, NULL, _("fill mode"));
  dt_bauhaus_combobox_add(g->cmb_fill_mode, _("erase"));
  dt_bauhaus_combobox_add(g->cmb_fill_mode, _("color"));
  g_object_set(g->cmb_fill_mode, "tooltip-text", _("erase the detail or fills with choosen color."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->cmb_fill_mode), "value-changed", G_CALLBACK(rt_fill_mode_callback), self);

  g->sl_fill_delta = dt_bauhaus_slider_new_with_range(self, -1.0, 1.0, .0005, .0, 4);
  dt_bauhaus_widget_set_label(g->sl_fill_delta, _("delta"), _("delta"));
  g_object_set(g->sl_fill_delta, "tooltip-text", _("add delta to color to fine tune it. works with erase as well."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_fill_delta), "value-changed", G_CALLBACK(rt_fill_delta_callback), self);

  GtkWidget *label4 = gtk_label_new(_("fill color: "));

  gtk_box_pack_end(GTK_BOX(g->hbox_color_pick), GTK_WIDGET(g->color_picker), FALSE, FALSE, 0);
  gtk_box_pack_end(GTK_BOX(g->hbox_color_pick), GTK_WIDGET(g->colorpick), FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->hbox_color_pick), label4, FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->hbox_color), GTK_WIDGET(g->cmb_fill_mode), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->hbox_color), g->hbox_color_pick, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->hbox_color), g->sl_fill_delta, TRUE, TRUE, 0);

  
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_shapes, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_algo, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_shape_sel, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_scale, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_n_image, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_blend, TRUE, TRUE, 0);

  // mask opacity
  g->sl_mask_opacity = dt_bauhaus_slider_new_with_range(self, 0.0, 1.0, 0.05, 1., 3);
  dt_bauhaus_widget_set_label(g->sl_mask_opacity, _("mask opacity"), _("mask opacity"));
  g_object_set(g->sl_mask_opacity, "tooltip-text", _("mask opacity."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_mask_opacity), "value-changed", G_CALLBACK(rt_mask_opacity_callback), self);
  
  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_mask_opacity, TRUE, TRUE, 0);
  
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_show_hide, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), g->hbox_blur, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), g->hbox_color, TRUE, TRUE, 0);

  
  g_signal_connect(G_OBJECT(self->widget), "draw", G_CALLBACK(rt_draw_callback), self);

  gtk_widget_show_all(g->hbox_blur);
  gtk_widget_set_no_show_all(g->hbox_blur, TRUE);

  gtk_widget_show_all(g->hbox_color);
  gtk_widget_set_no_show_all(g->hbox_color, TRUE);

  rt_show_hide_controls(g, p);
}

void gui_reset(struct dt_iop_module_t *self)
{
  // hide the previous masks
  dt_masks_reset_form_gui();
}

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *)self->gui_data;
  if (g)
  {
    dt_pthread_mutex_destroy(&g->lock);
  }
  free(self->gui_data);
  self->gui_data = NULL;
}

void modify_roi_out(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece, dt_iop_roi_t *roi_out,
                    const dt_iop_roi_t *roi_in)
{
  *roi_out = *roi_in;
}

// needed if mask dest is in roi and mask src is not
void modify_roi_in(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece,
                   const dt_iop_roi_t *roi_out, dt_iop_roi_t *roi_in)
{
  *roi_in = *roi_out;

  // make two passes, a form's source may need the destination from a previous one
  for (int step = 0; step < 2; step++)
  {
    int roir = roi_in->width + roi_in->x;
    int roib = roi_in->height + roi_in->y;
    int roix = roi_in->x;
    int roiy = roi_in->y;
  
    dt_develop_blend_params_t *bp = self->blend_params;
    dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)piece->data;
  
    // We iterate through all forms
    dt_masks_form_t *grp = dt_masks_get_from_id(self->dev, bp->mask_id);
    if(grp && (grp->type & DT_MASKS_GROUP))
    {
      GList *forms = g_list_first(grp->points);
      while(forms)
      {
        dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;
        // we get the spot
        dt_masks_form_t *form = dt_masks_get_from_id(self->dev, grpt->formid);
        if(form)
        {
          // if the form is outside the roi, we just skip it
          // use just roi_in, so in the second pass that are outside roi_out but are needed by other forms
          // we may get more image than we need, but is a price to pay...
          if(!rt_masks_form_is_in_roi(self, piece, form, roi_in, roi_in))
          {
            forms = g_list_next(forms);
            continue;
          }
  
          // we get the area for the source
          int fl, ft, fw, fh;
  
          if(!dt_masks_get_source_area(self, piece, form, &fw, &fh, &fl, &ft))
          {
            forms = g_list_next(forms);
            continue;
          }
          fw *= roi_in->scale, fh *= roi_in->scale, fl *= roi_in->scale, ft *= roi_in->scale;
  
          // we enlarge the roi if needed
          roiy = fminf(ft, roiy);
          roix = fminf(fl, roix);
          roir = fmaxf(fl + fw, roir);
          roib = fmaxf(ft + fh, roib);
          
          // heal needs both source and destination areas
          const dt_iop_retouch2_algo_type_t algo = rt_get_algorithm_from_formid(p, grpt->formid);
          if (algo == dt_iop_retouch2_heal)
          {
            int dx = 0, dy = 0;
            if(rt_masks_get_delta(self, piece, roi_in, form, &dx, &dy))
            {
              roiy = fminf(ft + dy, roiy);
              roix = fminf(fl + dx, roix);
              roir = fmaxf(fl + fw + dx, roir);
              roib = fmaxf(ft + fh + dy, roib);
            }
          }
  
        }
        forms = g_list_next(forms);
      }
    }
  
    // now we set the values
    const float scwidth = piece->buf_in.width * roi_in->scale, scheight = piece->buf_in.height * roi_in->scale;
    roi_in->x = CLAMP(roix, 0, scwidth - 1);
    roi_in->y = CLAMP(roiy, 0, scheight - 1);
    roi_in->width = CLAMP(roir - roi_in->x, 1, scwidth + .5f - roi_in->x);
    roi_in->height = CLAMP(roib - roi_in->y, 1, scheight + .5f - roi_in->y);
  }
}


void init_key_accels(dt_iop_module_so_t *module)
{
  dt_accel_register_iop (module, TRUE, NC_("accel", "circle tool"),   0, 0);
  dt_accel_register_iop (module, TRUE, NC_("accel", "elipse tool"),   0, 0);
  dt_accel_register_iop (module, TRUE, NC_("accel", "path tool"),     0, 0);
  dt_accel_register_iop (module, TRUE, NC_("accel", "brush tool"),    0, 0);
}

static gboolean _add_circle_key_accel(GtkAccelGroup *accel_group, GObject *acceleratable, guint keyval,
                                      GdkModifierType modifier, gpointer data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)data;
  const dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *) module->gui_data;
  rt_add_shape_callback(GTK_WIDGET(g->bt_circle), NULL, module);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_circle), TRUE);
  return TRUE;
}

static gboolean _add_ellipse_key_accel(GtkAccelGroup *accel_group, GObject *acceleratable, guint keyval,
                                       GdkModifierType modifier, gpointer data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)data;
  const dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *) module->gui_data;
  rt_add_shape_callback(GTK_WIDGET(g->bt_ellipse), NULL, module);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_ellipse), TRUE);
  return TRUE;
}

static gboolean _add_brush_key_accel(GtkAccelGroup *accel_group, GObject *acceleratable, guint keyval,
                                       GdkModifierType modifier, gpointer data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)data;
  const dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *) module->gui_data;
  rt_add_shape_callback(GTK_WIDGET(g->bt_brush), NULL, module);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), TRUE);
  return TRUE;
}

static gboolean _add_path_key_accel(GtkAccelGroup *accel_group, GObject *acceleratable, guint keyval,
                                    GdkModifierType modifier, gpointer data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)data;
  const dt_iop_retouch2_gui_data_t *g = (dt_iop_retouch2_gui_data_t *) module->gui_data;
  rt_add_shape_callback(GTK_WIDGET(g->bt_path), NULL, module);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), TRUE);
  return TRUE;
}

void connect_key_accels(dt_iop_module_t *module)
{
  GClosure *closure;

  closure = g_cclosure_new(G_CALLBACK(_add_circle_key_accel), (gpointer)module, NULL);
  dt_accel_connect_iop (module, "circle tool", closure);

  closure = g_cclosure_new(G_CALLBACK(_add_ellipse_key_accel), (gpointer)module, NULL);
  dt_accel_connect_iop (module, "elipse tool", closure);

  closure = g_cclosure_new(G_CALLBACK(_add_brush_key_accel), (gpointer)module, NULL);
  dt_accel_connect_iop (module, "brush tool", closure);

  closure = g_cclosure_new(G_CALLBACK(_add_path_key_accel), (gpointer)module, NULL);
  dt_accel_connect_iop (module, "path tool", closure);
}


//--------------------------------------------------------------------------------------------------
// process
//--------------------------------------------------------------------------------------------------

static void rt_intersect_2_rois(dt_iop_roi_t *const roi_1, dt_iop_roi_t *const roi_2, 
                                const int dx, const int dy, const int padding, 
																dt_iop_roi_t * roi_dest)
{
  const int x_from = MAX(MAX((roi_1->x + 1 - padding), roi_2->x), (roi_2->x+dx));
  const int x_to = MIN(MIN((roi_1->x + roi_1->width + 1 + padding), roi_2->x + roi_2->width), (roi_2->x + roi_2->width+dx));
  
  const int y_from = MAX(MAX((roi_1->y + 1 - padding), roi_2->y), (roi_2->y+dy));
  const int y_to = MIN(MIN((roi_1->y + roi_1->height + 1 + padding), (roi_2->y + roi_2->height)), (roi_2->y + roi_2->height+dy));
  
  roi_dest->x = x_from;
  roi_dest->y = y_from;
  roi_dest->width = x_to - x_from;
  roi_dest->height = y_to - y_from;

}

static void rt_copy_in_to_out(const float *const in, const struct dt_iop_roi_t *const roi_in, 
															float *const out, const struct dt_iop_roi_t *const roi_out, 
															const int ch, const int dx, const int dy)
{
  const int rowsize = MIN(roi_out->width, roi_in->width) * ch * sizeof(float);
  const int xoffs = roi_out->x - roi_in->x - dx;
  const int yoffs = roi_out->y - roi_in->y - dy;
  const int y_to = MIN(roi_out->height, roi_in->height);

#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
  for (int y=0; y < y_to; y++)
  {
    size_t iindex = ((size_t)(y + yoffs) * roi_in->width + xoffs) * ch;
    size_t oindex = (size_t)y * roi_out->width * ch;
    float *in1 = (float *)in + iindex;
    float *out1 = (float *)out + oindex;

    memcpy(out1, in1, rowsize);
  }

}

static void rt_build_scaled_mask(float *const mask, dt_iop_roi_t *const roi_mask, 
																	float **mask_scaled, dt_iop_roi_t *roi_mask_scaled, dt_iop_roi_t *const roi_in, 
																	const int dx, const int dy, const int algo)
{
	float *mask_tmp = NULL;
	
	const int padding = (algo == dt_iop_retouch2_heal) ? 1: 0;
	
	*roi_mask_scaled = *roi_mask;
	
  roi_mask_scaled->x = roi_mask->x * roi_in->scale;
  roi_mask_scaled->y = roi_mask->y * roi_in->scale;
  roi_mask_scaled->width = ((roi_mask->width * roi_in->scale) + .5f);
  roi_mask_scaled->height = ((roi_mask->height * roi_in->scale) + .5f);
  roi_mask_scaled->scale = roi_in->scale;

	rt_intersect_2_rois(roi_mask_scaled, roi_in, dx, dy, padding, roi_mask_scaled);
	if (roi_mask_scaled->width < 1 || roi_mask_scaled->height < 1)
		goto cleanup;
	
  const int x_to = roi_mask_scaled->width + roi_mask_scaled->x;
  const int y_to = roi_mask_scaled->height + roi_mask_scaled->y;

  mask_tmp = calloc(roi_mask_scaled->width * roi_mask_scaled->height, sizeof(float));
  if (mask_tmp == NULL)
  {
    printf("retouch: error allocating memory\n");
    goto cleanup;
  }

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(mask_tmp, roi_mask_scaled) schedule(static)
#endif
  for(int yy = roi_mask_scaled->y; yy < y_to; yy++)
  {
    const int mask_index = ((int)(yy / roi_in->scale)) - roi_mask->y;
    if (mask_index < 0 || mask_index >= roi_mask->height) continue;
    
    const int mask_scaled_index = (yy - roi_mask_scaled->y) * roi_mask_scaled->width;

    float *m = mask + mask_index * roi_mask->width;
    float *ms = mask_tmp + mask_scaled_index;
    
    for(int xx = roi_mask_scaled->x; xx < x_to; xx++, ms++)
    {
      const int mx = ((int)(xx / roi_in->scale)) - roi_mask->x;
      if (mx < 0 || mx >= roi_mask->width) continue;
      
      *ms = m[mx];
    }
  }

cleanup:
	*mask_scaled = mask_tmp;
	
}

// img_src and mask_scaled must have the same roi
static void rt_copy_image_masked(float *const img_src, float *img_dest, dt_iop_roi_t *const roi_dest, const int ch, 
																	float *const mask_scaled, dt_iop_roi_t *const roi_mask_scaled, 
																	const float opacity, const int mask_display, const int use_sse)
{
#if defined(__SSE__)
  if (ch == 4 && use_sse)
  {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(img_dest) schedule(static)
#endif
  	for(int yy = 0; yy < roi_mask_scaled->height; yy++)
    {
      const int mask_index = yy * roi_mask_scaled->width;
      const int src_index = mask_index * ch;
      const int dest_index = (((yy + roi_mask_scaled->y - roi_dest->y) * roi_dest->width) + (roi_mask_scaled->x - roi_dest->x)) * ch;
      
      float *s = img_src + src_index;
      float *d = img_dest + dest_index;
      float *m = mask_scaled + mask_index;

      for(int xx = 0; xx < roi_mask_scaled->width; xx++, s+=ch, d+=ch, m++)
      {
        const float f = (*m) * opacity;
        
        const __m128 val1_f = _mm_set1_ps(1.0f - f);
        const __m128 valf = _mm_set1_ps(f);
        
        _mm_store_ps(d, _mm_add_ps(_mm_mul_ps(_mm_load_ps(d), val1_f), _mm_mul_ps(_mm_load_ps(s), valf)));
  
        if (mask_display && f)
          d[3] = f;
      }
    }
  }
  else
#endif
  {
  	const int ch1 = (ch==4) ? ch-1: ch;
  	
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(img_dest) schedule(static)
#endif
    for(int yy = 0; yy < roi_mask_scaled->height; yy++)
    {
      const int mask_index = yy * roi_mask_scaled->width;
      const int src_index = mask_index * ch;
      const int dest_index = (((yy + roi_mask_scaled->y - roi_dest->y) * roi_dest->width) + (roi_mask_scaled->x - roi_dest->x)) * ch;
      
      float *s = img_src + src_index;
      float *d = img_dest + dest_index;
      float *m = mask_scaled + mask_index;
     
      for(int xx = 0; xx < roi_mask_scaled->width; xx++, s+=ch, d+=ch, m++)
      {
        const float f = (*m) * opacity;
        
        for(int c = 0; c < ch1; c++)
        {
          d[c] = d[c] * (1.0f - f) + s[c] * f;
        }
        if (mask_display && f)
          d[3] = f;
      }
    }
  }

}

#if defined(__SSE__)
static void retouch2_fill_sse(float *const in, dt_iop_roi_t *const roi_in, 
                              float *const mask_scaled, dt_iop_roi_t *const roi_mask_scaled, const int mask_display, 
                              const float opacity, 
                              const float *const fill_color)
{
  const int ch = 4;
  
  const float valf4_fill[4] = { fill_color[0], fill_color[1], fill_color[2], 0.f };
  const __m128 val_fill = _mm_load_ps(valf4_fill);
  
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
	for(int yy = 0; yy < roi_mask_scaled->height; yy++)
  {
    const int mask_index = yy * roi_mask_scaled->width;
    const int dest_index = (((yy + roi_mask_scaled->y - roi_in->y) * roi_in->width) + (roi_mask_scaled->x - roi_in->x)) * ch;
    
    float *d = in + dest_index;
    float *m = mask_scaled + mask_index;

    for(int xx = 0; xx < roi_mask_scaled->width; xx++, d+=ch, m++)
    {
      const float f = (*m) * opacity;
      
      const __m128 val1_f = _mm_set1_ps(1.0f - f);
      const __m128 valf = _mm_set1_ps(f);
      
      _mm_store_ps(d, _mm_add_ps(_mm_mul_ps(_mm_load_ps(d), val1_f), _mm_mul_ps(val_fill, valf)));

      if (mask_display && f)
        d[3] = f;
    }
  }
}
#endif

static void retouch2_fill(float *const in, dt_iop_roi_t *const roi_in, const int ch, 
                          float *const mask_scaled, dt_iop_roi_t *const roi_mask_scaled, const int mask_display, 
                          const float opacity, 
                          const float *const fill_color, int use_sse)
{
#if defined(__SSE__)
  if (ch == 4 && use_sse)
  {
    retouch2_fill_sse(in, roi_in, mask_scaled, roi_mask_scaled, mask_display, opacity, fill_color);
    return;
  }
#endif
  const int ch1 = (ch==4) ? ch-1: ch;
  
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
	for(int yy = 0; yy < roi_mask_scaled->height; yy++)
  {
    const int mask_index = yy * roi_mask_scaled->width;
    const int dest_index = (((yy + roi_mask_scaled->y - roi_in->y) * roi_in->width) + (roi_mask_scaled->x - roi_in->x)) * ch;
    
    float *d = in + dest_index;
    float *m = mask_scaled + mask_index;

    for(int xx = 0; xx < roi_mask_scaled->width; xx++, d+=ch, m++)
    {
      const float f = (*m) * opacity;
      
      for(int c = 0; c < ch1; c++)
        d[c] = d[c] * (1.0f - f) + fill_color[c] * f;

      if (mask_display && f)
        d[3] = f;
    }
  }

}

static void retouch2_clone(float *const in, dt_iop_roi_t *const roi_in, const int ch, 
                          float *const mask_scaled, dt_iop_roi_t *const roi_mask_scaled, const int mask_display, 
                          const int dx, const int dy, const float opacity, int use_sse)
{
  // alloc temp image to avoid issues when areas self-intersects
  float *img_src = dt_alloc_align(64, roi_mask_scaled->width * roi_mask_scaled->height * ch * sizeof(float));
  if (img_src == NULL)
  {
    printf("error allocating memory for cloning\n");
    goto cleanup;
  }
  
  // copy source image to tmp
  rt_copy_in_to_out(in, roi_in, img_src, roi_mask_scaled, ch, dx, dy);
  
  // clone it
  rt_copy_image_masked(img_src, in, roi_in, ch, mask_scaled, roi_mask_scaled, opacity, mask_display, use_sse);

cleanup:
	if (img_src) dt_free_align(img_src);
}

static void retouch2_gaussian_blur(float *const in, dt_iop_roi_t *const roi_in, const int ch, 
                                    float *const mask_scaled, dt_iop_roi_t *const roi_mask_scaled, const int mask_display, 
                                    const float opacity, 
                                    const float blur_radius, dt_dev_pixelpipe_iop_t *piece, int use_sse)
{
  if (fabs(blur_radius) <= 0.1f && !mask_display) return;

  float *img_dest = NULL;
  
  // alloc temp image to blur
  img_dest = dt_alloc_align(64, roi_mask_scaled->width * roi_mask_scaled->height * ch * sizeof(float));
  if (img_dest == NULL)
  {
    printf("error allocating memory for blurring\n");
    goto cleanup;
  }
  
  // copy source image so we blur just the mask area (at least the smallest rect that covers it)
  rt_copy_in_to_out(in, roi_in, img_dest, roi_mask_scaled, ch, 0, 0);

  if (fabs(blur_radius) > 0.1f)
  {
    const float sigma = blur_radius * roi_in->scale / piece->iscale;
  
    float Labmax[] = { INFINITY, INFINITY, INFINITY, INFINITY };
    float Labmin[] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
  
    dt_gaussian_t *g = dt_gaussian_init(roi_mask_scaled->width, roi_mask_scaled->height, ch, Labmax, Labmin, sigma, DT_IOP_GAUSSIAN_ZERO);
    if(g)
    {
      if (ch == 4)
        dt_gaussian_blur_4c(g, img_dest, img_dest);
      else
        dt_gaussian_blur(g, img_dest, img_dest);
      dt_gaussian_free(g);
    }
  }
  
  // copy blurred (temp) image to destination image
  rt_copy_image_masked(img_dest, in, roi_in, ch, mask_scaled, roi_mask_scaled, opacity, mask_display, use_sse);

cleanup:
  if (img_dest) dt_free_align(img_dest);
}

static void retouch2_heal(float *const in, dt_iop_roi_t *const roi_in, const int ch, 
                          float *const mask_scaled, dt_iop_roi_t *const roi_mask_scaled, const int mask_display, 
                          const int dx, const int dy, const float opacity, int use_sse)
{
  float *img_src = NULL;
  float *img_dest = NULL;

  // alloc temp images for source and destination
  img_src = dt_alloc_align(64, roi_mask_scaled->width * roi_mask_scaled->height * ch * sizeof(float));
  img_dest = dt_alloc_align(64, roi_mask_scaled->width * roi_mask_scaled->height * ch * sizeof(float));
  if ((img_src == NULL) || (img_dest == NULL))
  {
    printf("error allocating memory for healing\n");
    goto cleanup;
  }

  // copy source and destination to temp images
  rt_copy_in_to_out(in, roi_in, img_src, roi_mask_scaled, ch, dx, dy);
  rt_copy_in_to_out(in, roi_in, img_dest, roi_mask_scaled, ch, 0, 0);

  // heal it
  dt_heal(img_src, img_dest, mask_scaled, roi_mask_scaled->width, roi_mask_scaled->height, ch, use_sse);

  // copy healed (temp) image to destination image
  rt_copy_image_masked(img_dest, in, roi_in, ch, mask_scaled, roi_mask_scaled, opacity, mask_display, use_sse);
  
cleanup:
  if (img_src) dt_free_align(img_src);
  if (img_dest) dt_free_align(img_dest);

}

static void rt_process_forms(float *layer, dwt_params_t *const wt_p, const int scale1)
{
  int scale = scale1;
  _rt_user_data_t *usr_d = (_rt_user_data_t*)wt_p->user_data;
  dt_iop_module_t *self = usr_d->self;
  dt_dev_pixelpipe_iop_t *piece = usr_d->piece;
  
  // if preview a single scale, just process that scale and original image
  if (wt_p->return_layer > 0 && scale != wt_p->return_layer && scale != 0) return;
  // do not process the reconstructed image
  if (scale > wt_p->scales+1) return;

  dt_develop_blend_params_t *bp = (dt_develop_blend_params_t *)piece->blendop_data;
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)piece->data;
  dt_iop_roi_t *roi_layer = &usr_d->roi;
  const int mask_display = usr_d->mask_display && (scale == usr_d->display_scale);

  // user requested to preview one scale > max scales, so we are returning a lower scale, 
  // but we will use the forms from the requested scale
  if (wt_p->scales < p->num_scales && wt_p->return_layer > 0 && p->curr_scale != scale)
  {
    scale = p->curr_scale;
  }
  // when the requested scales is grather than max scales the residual image index will be different from the one defined by the user,
  // so we need to adjust it here, otherwise we will be using the shapes from a scale on the residual image
  else if (wt_p->scales < p->num_scales && wt_p->return_layer == 0 && scale == wt_p->scales+1)
  {
    scale = p->num_scales+1;
  }
  
  // iterate through all forms
  dt_masks_form_t *grp = dt_masks_get_from_id(self->dev, bp->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    GList *forms = g_list_first(grp->points);
    while(forms)
    {
      dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;
      if(grpt == NULL)
      {
        printf("rt_process_forms invalid form\n");
        forms = g_list_next(forms);
        continue;
      }
      if(grpt->formid == 0)
      {
        printf("rt_process_forms form is null\n");
        forms = g_list_next(forms);
        continue;
      }
      const int index = rt_get_index_from_formid(p, grpt->formid);
      if(index == -1)
      {
        // FIXME: we get this error when adding a new form and the system is still processing a previous add
        // we should not report an error in this case (and have a better way of adding a form)
        printf("rt_process_forms missing form from array=%i\n", grpt->formid);
        forms = g_list_next(forms);
        continue;
      }
      
      // only process current scale
      if(p->rt_forms[index].scale != scale)
      {
        forms = g_list_next(forms);
        continue;
      }
      
      // get the spot
      dt_masks_form_t *form = dt_masks_get_from_id(self->dev, grpt->formid);
      if(form == NULL)
      {
        printf("rt_process_forms missing form from masks=%i\n", grpt->formid);
        forms = g_list_next(forms);
        continue;
      }

      // if the form is outside the roi, we just skip it
      if(!rt_masks_form_is_in_roi(self, piece, form, roi_layer, roi_layer))
      {
        forms = g_list_next(forms);
        continue;
      }

      // get the mask
      float *mask = NULL;
      dt_iop_roi_t roi_mask = {0};
      
      dt_masks_get_mask(self, piece, form, &mask, &roi_mask.width, &roi_mask.height, &roi_mask.x, &roi_mask.y);
      if(mask == NULL)
      {
        printf("rt_process_forms error retrieving mask\n");
        forms = g_list_next(forms);
        continue;
      }
      
      // search the delta with the source
      const dt_iop_retouch2_algo_type_t algo = p->rt_forms[index].algorithm;
      int dx = 0, dy = 0;
      
      if (algo != dt_iop_retouch2_gaussian_blur && algo != dt_iop_retouch2_fill)
      {
        if(!rt_masks_get_delta(self, piece, roi_layer, form, &dx, &dy))
        {
          forms = g_list_next(forms);
          if (mask) free(mask);
          continue;
        }
      }
      
      // scale the mask
      float *mask_scaled = NULL;
      dt_iop_roi_t roi_mask_scaled = {0};

      rt_build_scaled_mask(mask, &roi_mask, &mask_scaled, &roi_mask_scaled, roi_layer, dx, dy, algo);
      
      // we don't need the original mask anymore
      if (mask)
      {
      	free(mask);
      	mask = NULL;
      }
      
      if (mask_scaled == NULL)
      {
        forms = g_list_next(forms);
        continue;
      }
      
      if ((dx != 0 || dy != 0 || algo == dt_iop_retouch2_gaussian_blur || algo == dt_iop_retouch2_fill) && 
          ((roi_mask_scaled.width > 2) && (roi_mask_scaled.height > 2)))
      {
        double start = dt_get_wtime();
        
        if (algo == dt_iop_retouch2_clone)
        {
          retouch2_clone(layer, roi_layer, wt_p->ch, mask_scaled, &roi_mask_scaled, mask_display, dx, dy,
                          grpt->opacity, wt_p->use_sse);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch2_clone took %0.04f sec\n", dt_get_wtime() - start);
        }
        else if (algo == dt_iop_retouch2_heal)
        {
          retouch2_heal(layer, roi_layer, wt_p->ch, mask_scaled, &roi_mask_scaled, mask_display, dx, dy,
                          grpt->opacity, wt_p->use_sse);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch2_heal took %0.04f sec\n", dt_get_wtime() - start);
        }
        else if (algo == dt_iop_retouch2_gaussian_blur)
        {
          retouch2_gaussian_blur(layer, roi_layer, wt_p->ch, mask_scaled, &roi_mask_scaled, mask_display, 
                          grpt->opacity, p->rt_forms[index].blur_radius, piece, wt_p->use_sse);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch2_gaussian_blur took %0.04f sec\n", dt_get_wtime() - start);
        }
        else if (algo == dt_iop_retouch2_fill)
        {
          // add a delta to the color so it can be fine-adjusted by the user
          float fill_color[3];
          
          if (p->rt_forms[index].fill_mode == dt_iop_rt_fill_erase)
          {
            fill_color[0] = fill_color[1] = fill_color[2] = p->rt_forms[index].fill_delta;
          }
          else
          {
            fill_color[0] = p->rt_forms[index].fill_color[0]+p->rt_forms[index].fill_delta;
            fill_color[1] = p->rt_forms[index].fill_color[1]+p->rt_forms[index].fill_delta;
            fill_color[2] = p->rt_forms[index].fill_color[2]+p->rt_forms[index].fill_delta;
          }
          
          retouch2_fill(layer, roi_layer, wt_p->ch, mask_scaled, &roi_mask_scaled, mask_display, 
                          grpt->opacity, fill_color, wt_p->use_sse);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch2_fill took %0.04f sec\n", dt_get_wtime() - start);
        }
        else
          printf("rt_process_forms unknown algorithm %i\n", algo);
        
      }
      
      if (mask) free(mask);
      if (mask_scaled) free(mask_scaled);
      
      forms = g_list_next(forms);
    }
  }

}

static void rt_process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
                void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out, 
                int use_sse)
{
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)piece->data;
  
  float *in_retouch = NULL;
  
  dt_iop_roi_t roi_retouch = *roi_in;
  dt_iop_roi_t *roi_rt = &roi_retouch;
  
  const int ch = piece->colors;
  _rt_user_data_t usr_data = {0};
  dwt_params_t dwt_p = {0};
  dt_iop_retouch2_preview_types_t preview_type = p->preview_type;
  
  int gui_active = 0;
  if (self->dev) gui_active = (self == self->dev->gui_module);
  if (!gui_active && preview_type == dt_iop_rt_preview_current_scale) preview_type = dt_iop_rt_preview_final_image;
  
  // we will do all the clone, heal, etc on the input image, 
  // this way the source for one algorithm can be the destination from a previous one
  in_retouch = dt_alloc_align(64, roi_rt->width * roi_rt->height * ch * sizeof(float));
  memcpy(in_retouch, ivoid, roi_rt->width * roi_rt->height * ch * sizeof(float));

  // user data passed from the decompose routine to the one that process each scale
  usr_data.self = self;
  usr_data.piece = piece;
  usr_data.roi = *roi_rt;
  usr_data.mask_display = 0;
  usr_data.display_scale = p->curr_scale;

  // parameters for the decompose routine
  dwt_p.image = in_retouch;
  dwt_p.ch = ch;
  dwt_p.width = roi_rt->width;
  dwt_p.height = roi_rt->height;
  dwt_p.scales = p->num_scales;
  dwt_p.return_layer = (preview_type == dt_iop_rt_preview_final_image) ? 0: p->curr_scale;
  dwt_p.blend_factor = p->blend_factor;
  dwt_p.user_data = &usr_data;
  dwt_p.preview_scale = roi_in->scale / piece->iscale;
  dwt_p.use_sse = use_sse;

  // check if this module should expose mask. 
  if(self->request_mask_display && self->dev->gui_attached && (self == self->dev->gui_module)
     && (piece->pipe == self->dev->pipe) )
  {
    for(size_t j = 0; j < roi_rt->width*roi_rt->height*ch; j += ch) in_retouch[j + 3] = 0.f;
    
    piece->pipe->mask_display = 1;
    usr_data.mask_display = 1;
  }
  
  // check if the image support this number of scales
  if (piece->pipe->type == DT_DEV_PIXELPIPE_FULL && gui_active)
  {
    const int max_scales = dwt_get_max_scale(&dwt_p);
    if (dwt_p.scales > max_scales)
    {
      dt_control_log(_("max scale is %i for this image size"), max_scales);
    }
  }
  
  // decompose it
  if(self->suppress_mask && self->dev->gui_attached && (self == self->dev->gui_module)
     && (piece->pipe == self->dev->pipe))
  {
    dwt_decompose(&dwt_p, NULL);
  }
  else
  {
    dwt_decompose(&dwt_p, rt_process_forms);
  }

  // copy alpha channel if nedded
  if(piece->pipe->mask_display && !usr_data.mask_display) 
  {
    const float *const i = ivoid;
    for(size_t j = 0; j < roi_rt->width*roi_rt->height*ch; j += ch) in_retouch[j + 3] = i[j + 3];
  }
  
  // return final image
  rt_copy_in_to_out(in_retouch, roi_rt, ovoid, roi_out, ch, 0, 0);
  
  check_nan(in_retouch, roi_rt->width*roi_rt->height*ch);
  
  if (in_retouch) dt_free_align(in_retouch);

}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
                void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  rt_process(self, piece, ivoid, ovoid, roi_in, roi_out, 0);
}

#if defined(__SSE__x)
void process_sse2(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
                  void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  rt_process(self, piece, ivoid, ovoid, roi_in, roi_out, 1);
}
#endif

#ifdef HAVE_OPENCL

static cl_int rt_copy_in_to_out_cl(const int devid, cl_mem dev_in, const struct dt_iop_roi_t *const roi_in, 
																cl_mem dev_out, const struct dt_iop_roi_t *const roi_out, const int dx, const int dy, 
																const int kernel)
{
	cl_int err = CL_SUCCESS;
	
  const int xoffs = roi_out->x - roi_in->x - dx;
  const int yoffs = roi_out->y - roi_in->y - dy;

  cl_mem dev_roi_in = NULL;
	cl_mem dev_roi_out = NULL;
	
	size_t sizes[] = { ROUNDUPWD(MIN(roi_out->width, roi_in->width)), ROUNDUPHT(MIN(roi_out->height, roi_in->height)), 1 };

  dev_roi_in = dt_opencl_copy_host_to_device_constant(devid, sizeof (dt_iop_roi_t), (void *) roi_in);
  dev_roi_out = dt_opencl_copy_host_to_device_constant(devid, sizeof (dt_iop_roi_t), (void *) roi_out);
  if (dev_roi_in == NULL || dev_roi_out == NULL)
  {
  	printf("rt_copy_in_to_out_cl error 1\n");
  	err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    goto cleanup;
  }

  dt_opencl_set_kernel_arg(devid, kernel, 0, sizeof(cl_mem), (void *)&dev_in);
  dt_opencl_set_kernel_arg(devid, kernel, 1, sizeof(cl_mem), (void *)&dev_roi_in);
  dt_opencl_set_kernel_arg(devid, kernel, 2, sizeof(cl_mem), (void *)&dev_out);
  dt_opencl_set_kernel_arg(devid, kernel, 3, sizeof(cl_mem), (void *)&dev_roi_out);
  dt_opencl_set_kernel_arg(devid, kernel, 4, sizeof(int), (void *)&xoffs);
  dt_opencl_set_kernel_arg(devid, kernel, 5, sizeof(int), (void *)&yoffs);
  err = dt_opencl_enqueue_kernel_2d(devid, kernel, sizes);
	if (err != CL_SUCCESS)
	{
		printf("rt_copy_in_to_out_cl error 2\n");
		goto cleanup;
	}

cleanup:
	if (dev_roi_in) dt_opencl_release_mem_object(dev_roi_in);
	if (dev_roi_out) dt_opencl_release_mem_object(dev_roi_out);
	
	return err;
}

static cl_int rt_build_scaled_mask_cl(const int devid, float *const mask, dt_iop_roi_t *const roi_mask, 
																			float **mask_scaled, cl_mem *p_dev_mask_scaled, dt_iop_roi_t *roi_mask_scaled, dt_iop_roi_t *const roi_in, 
																			const int dx, const int dy, const int algo)
{
	cl_int err = CL_SUCCESS;
	
	rt_build_scaled_mask(mask, roi_mask, mask_scaled, roi_mask_scaled, roi_in, dx, dy, algo);
	if (*mask_scaled == NULL)
	{
//  	printf("rt_build_scaled_mask_cl error 1\n");
//  	err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    goto cleanup;
	}

	cl_mem dev_mask_scaled = dt_opencl_alloc_device_buffer(devid, roi_mask_scaled->width * roi_mask_scaled->height * sizeof(float));
  if (dev_mask_scaled == NULL)
  {
  	printf("rt_build_scaled_mask_cl error 2\n");
  	err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
  	goto cleanup;
  }

  err = dt_opencl_write_buffer_to_device(devid, *mask_scaled, dev_mask_scaled, 0,
  																				roi_mask_scaled->width * roi_mask_scaled->height * sizeof(float), TRUE);
	if (err != CL_SUCCESS)
	{
		printf("rt_build_scaled_mask_cl error 4\n");
		goto cleanup;
	}

	*p_dev_mask_scaled = dev_mask_scaled;
	
cleanup:

	if (err != CL_SUCCESS) printf("rt_build_scaled_mask_cl error\n");
		
	return err;
}

static cl_int rt_copy_image_masked_cl(const int devid, cl_mem dev_src, cl_mem dev_dest, dt_iop_roi_t *const roi_dest, 
																							cl_mem dev_mask_scaled, dt_iop_roi_t *const roi_mask_scaled, 
																							const float opacity, const int mask_display, const int kernel)
{
	cl_int err = CL_SUCCESS;

	cl_mem dev_roi_dest = NULL;
	cl_mem dev_roi_mask_scaled = NULL;

	size_t sizes[] = { ROUNDUPWD(roi_mask_scaled->width), ROUNDUPHT(roi_mask_scaled->height), 1 };

	dev_roi_dest = dt_opencl_copy_host_to_device_constant(devid, sizeof (dt_iop_roi_t), (void *) roi_dest);

  dev_roi_mask_scaled = dt_opencl_copy_host_to_device_constant(devid, sizeof (dt_iop_roi_t), (void *) roi_mask_scaled);

  if (dev_roi_dest == NULL || dev_roi_mask_scaled == NULL)
  {
  	err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    goto cleanup;
  }

  dt_opencl_set_kernel_arg(devid, kernel, 0, sizeof(cl_mem), (void *)&dev_src);
  dt_opencl_set_kernel_arg(devid, kernel, 1, sizeof(cl_mem), (void *)&dev_dest);
  dt_opencl_set_kernel_arg(devid, kernel, 2, sizeof(cl_mem), (void *)&dev_roi_dest);
  dt_opencl_set_kernel_arg(devid, kernel, 3, sizeof(cl_mem), (void *)&dev_mask_scaled);
  dt_opencl_set_kernel_arg(devid, kernel, 4, sizeof(cl_mem), (void *)&dev_roi_mask_scaled);
  dt_opencl_set_kernel_arg(devid, kernel, 5, sizeof(float), (void *)&opacity);
  dt_opencl_set_kernel_arg(devid, kernel, 6, sizeof(int), (void *)&mask_display);
  err = dt_opencl_enqueue_kernel_2d(devid, kernel, sizes);
	if (err != CL_SUCCESS) goto cleanup;

cleanup:
	if (dev_roi_dest) dt_opencl_release_mem_object(dev_roi_dest);
	if (dev_roi_mask_scaled) dt_opencl_release_mem_object(dev_roi_mask_scaled);
	
	return err;
}

static cl_int retouch2_clone_cl(const int devid, cl_mem dev_layer, dt_iop_roi_t *const roi_layer, 
																cl_mem dev_mask_scaled, dt_iop_roi_t *const roi_mask_scaled, const int mask_display, 
																const int dx, const int dy, const float opacity, 
																dt_iop_retouch2_global_data_t *gd)
{
	cl_int err = CL_SUCCESS;
	
	const int ch = 4;
	
  cl_mem dev_src = NULL;
  
  // alloc source temp image to avoid issues when areas self-intersects
  dev_src = dt_opencl_alloc_device_buffer(devid, roi_mask_scaled->width * roi_mask_scaled->height * ch * sizeof(float));
  if(dev_src == NULL)
  {
  	printf("retouch2_clone_cl error 2\n");
  	err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
  	goto cleanup;
  }

  // copy source image to tmp
	err = rt_copy_in_to_out_cl(devid, dev_layer, roi_layer, 
															dev_src, roi_mask_scaled, dx, dy, gd->kernel_retouch_copy_buffer_to_buffer);
	if (err != CL_SUCCESS)
	{
		printf("retouch2_clone_cl error 4\n");
		goto cleanup;
	}

  // clone it
	err = rt_copy_image_masked_cl(devid, dev_src, dev_layer, roi_layer, 
																					dev_mask_scaled, roi_mask_scaled, 
																					opacity, mask_display, gd->kernel_retouch_copy_buffer_to_buffer_masked);
	if (err != CL_SUCCESS)
	{
		printf("retouch2_clone_cl error 5\n");
		goto cleanup;
	}

cleanup:
  if (dev_src) dt_opencl_release_mem_object(dev_src);

  return err;
}

static cl_int retouch2_fill_cl(const int devid, cl_mem dev_layer, dt_iop_roi_t *const roi_layer, 
															cl_mem dev_mask_scaled, dt_iop_roi_t *const roi_mask_scaled, const int mask_display, 
															const float opacity, float *color, 
															dt_iop_retouch2_global_data_t *gd)
{
	cl_int err = CL_SUCCESS;
	
	cl_mem dev_roi_layer = NULL;
	cl_mem dev_roi_mask_scaled = NULL;

  // fill it
	const int kernel = gd->kernel_retouch_fill;
	size_t sizes[] = { ROUNDUPWD(roi_mask_scaled->width), ROUNDUPHT(roi_mask_scaled->height), 1 };

  dev_roi_layer = dt_opencl_copy_host_to_device_constant(devid, sizeof (dt_iop_roi_t), (void *) roi_layer);
  dev_roi_mask_scaled = dt_opencl_copy_host_to_device_constant(devid, sizeof (dt_iop_roi_t), (void *) roi_mask_scaled);
  if (dev_roi_layer == NULL || dev_roi_mask_scaled == NULL)
  {
  	err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    goto cleanup;
  }

  dt_opencl_set_kernel_arg(devid, kernel, 0, sizeof(cl_mem), (void *)&dev_layer);
  dt_opencl_set_kernel_arg(devid, kernel, 1, sizeof(cl_mem), (void *)&dev_roi_layer);
  dt_opencl_set_kernel_arg(devid, kernel, 2, sizeof(cl_mem), (void *)&dev_mask_scaled);
  dt_opencl_set_kernel_arg(devid, kernel, 3, sizeof(cl_mem), (void *)&dev_roi_mask_scaled);
  dt_opencl_set_kernel_arg(devid, kernel, 4, sizeof(int), (void *)&mask_display);
  dt_opencl_set_kernel_arg(devid, kernel, 5, sizeof(float), (void *)&opacity);
  dt_opencl_set_kernel_arg(devid, kernel, 6, sizeof(float), (void *)&(color[0]));
  dt_opencl_set_kernel_arg(devid, kernel, 7, sizeof(float), (void *)&(color[1]));
  dt_opencl_set_kernel_arg(devid, kernel, 8, sizeof(float), (void *)&(color[2]));
  err = dt_opencl_enqueue_kernel_2d(devid, kernel, sizes);
	if (err != CL_SUCCESS) goto cleanup;
  
	
cleanup:
	dt_opencl_release_mem_object(dev_roi_layer);
	dt_opencl_release_mem_object(dev_roi_mask_scaled);
	
  return err;
}

static cl_int retouch2_gaussian_blur_cl(const int devid, cl_mem dev_layer, dt_iop_roi_t *const roi_layer, 
																				cl_mem dev_mask_scaled, dt_iop_roi_t *const roi_mask_scaled, const int mask_display, 
																				const float opacity, const float blur_radius, dt_dev_pixelpipe_iop_t *piece,
																				dt_iop_retouch2_global_data_t *gd)
{
	cl_int err = CL_SUCCESS;
	
  if (fabs(blur_radius) <= 0.1f && !mask_display) return err;

  const int ch = 4;
  
  cl_mem dev_dest = NULL;
  
  dev_dest = dt_opencl_alloc_device(devid, roi_mask_scaled->width, roi_mask_scaled->height, ch * sizeof(float));
  if(dev_dest == NULL)
  {
  	printf("retouch2_gaussian_blur_cl error 2\n");
  	err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
  	goto cleanup;
  }

	err = rt_copy_in_to_out_cl(devid, dev_layer, roi_layer, 
															dev_dest, roi_mask_scaled, 0, 0, gd->kernel_retouch_copy_buffer_to_image);
	if (err != CL_SUCCESS)
	{
		printf("retouch2_gaussian_blur_cl error 4\n");
		goto cleanup;
	}

  if (fabs(blur_radius) > 0.1f)
  {
    const float sigma = blur_radius * roi_layer->scale / piece->iscale;
  
    float Labmax[] = { INFINITY, INFINITY, INFINITY, INFINITY };
    float Labmin[] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
  
    dt_gaussian_cl_t *g = dt_gaussian_init_cl(devid, roi_mask_scaled->width, roi_mask_scaled->height, ch, Labmax, Labmin, sigma, DT_IOP_GAUSSIAN_ZERO);
    if(g)
    {
    	err = dt_gaussian_blur_cl(g, dev_dest, dev_dest);
      dt_gaussian_free_cl(g);
      if (err != CL_SUCCESS) goto cleanup;
    }
  }
  
  // copy blurred (temp) image to destination image
	err = rt_copy_image_masked_cl(devid, dev_dest, dev_layer, roi_layer, 
																					dev_mask_scaled, roi_mask_scaled, 
																					opacity, mask_display, gd->kernel_retouch_copy_image_to_buffer_masked);
	if (err != CL_SUCCESS)
	{
		printf("retouch2_gaussian_blur_cl error 5\n");
		goto cleanup;
	}

cleanup:
  if (dev_dest) dt_opencl_release_mem_object(dev_dest);
  
  return err;
}

static cl_int retouch2_heal_cl(const int devid, cl_mem dev_layer, dt_iop_roi_t *const roi_layer, 
															float *mask_scaled, cl_mem dev_mask_scaled, dt_iop_roi_t *const roi_mask_scaled, 
															const int dx, const int dy, 
															const int mask_display, const float opacity, 
															dt_iop_retouch2_global_data_t *gd)
{
	cl_int err = CL_SUCCESS;
	
	const int ch = 4;
	
  cl_mem dev_src = NULL;
  cl_mem dev_dest = NULL;
  
  dev_src = dt_opencl_alloc_device_buffer(devid, roi_mask_scaled->width * roi_mask_scaled->height * ch * sizeof(float));
  if(dev_src == NULL)
  {
  	printf("retouch2_heal_cl: error allocating memory for healing\n");
    err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    goto cleanup;
  }

  dev_dest = dt_opencl_alloc_device_buffer(devid, roi_mask_scaled->width * roi_mask_scaled->height * ch * sizeof(float));
  if(dev_dest == NULL)
  {
  	printf("retouch2_heal_cl: error allocating memory for healing\n");
    err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    goto cleanup;
  }

	err = rt_copy_in_to_out_cl(devid, dev_layer, roi_layer, 
															dev_src, roi_mask_scaled, dx, dy, gd->kernel_retouch_copy_buffer_to_buffer);
	if (err != CL_SUCCESS)
	{
		printf("retouch2_heal_cl error 4\n");
		goto cleanup;
	}

	err = rt_copy_in_to_out_cl(devid, dev_layer, roi_layer, 
															dev_dest, roi_mask_scaled, 0, 0, gd->kernel_retouch_copy_buffer_to_buffer);
	if (err != CL_SUCCESS)
	{
		printf("retouch2_heal_cl error 4\n");
		goto cleanup;
	}

  // heal it
  heal_params_cl_t *hp = dt_heal_init_cl(devid);
  if (hp)
  {
		err = dt_heal_cl(hp, dev_src, dev_dest, mask_scaled, roi_mask_scaled->width, roi_mask_scaled->height);
		dt_heal_free_cl(hp);
		
		if (err != CL_SUCCESS) goto cleanup;
  }

  // copy healed (temp) image to destination image
	err = rt_copy_image_masked_cl(devid, dev_dest, dev_layer, roi_layer, 
																				dev_mask_scaled, roi_mask_scaled, 
																				opacity, mask_display, gd->kernel_retouch_copy_buffer_to_buffer_masked);
	if (err != CL_SUCCESS)
	{
		printf("retouch2_heal_cl error 6\n");
		goto cleanup;
	}

cleanup:
	if (dev_src) dt_opencl_release_mem_object(dev_src);
	if (dev_dest) dt_opencl_release_mem_object(dev_dest);
	
	return err;
}

static cl_int rt_process_forms_cl(cl_mem dev_layer, dwt_params_cl_t *const wt_p, const int scale1)
{
	cl_int err = CL_SUCCESS;
	
  int scale = scale1;
  _rt_user_data_t *usr_d = (_rt_user_data_t*)wt_p->user_data;
  dt_iop_module_t *self = usr_d->self;
  dt_dev_pixelpipe_iop_t *piece = usr_d->piece;
  
  // if preview a single scale, just process that scale and original image
  if (wt_p->return_layer > 0 && scale != wt_p->return_layer && scale != 0) return err;
  // do not process the reconstructed image
  if (scale > wt_p->scales+1) return err;

  dt_develop_blend_params_t *bp = (dt_develop_blend_params_t *)piece->blendop_data;
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)piece->data;
  dt_iop_retouch2_global_data_t *gd = (dt_iop_retouch2_global_data_t *)self->data;
  const int devid = piece->pipe->devid;
  dt_iop_roi_t *roi_layer = &usr_d->roi;
  const int mask_display = usr_d->mask_display && (scale == usr_d->display_scale);

  // user requested to preview one scale > max scales, so we are returning a lower scale, 
  // but we will use the forms from the requested scale
  if (wt_p->scales < p->num_scales && wt_p->return_layer > 0 && p->curr_scale != scale)
  {
    scale = p->curr_scale;
  }
  // when the requested scales is grather than max scales the residual image index will be different from the one defined by the user,
  // so we need to adjust it here, otherwise we will be using the shapes from a scale on the residual image
  else if (wt_p->scales < p->num_scales && wt_p->return_layer == 0 && scale == wt_p->scales+1)
  {
    scale = p->num_scales+1;
  }
  
  // iterate through all forms
  dt_masks_form_t *grp = dt_masks_get_from_id(self->dev, bp->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    GList *forms = g_list_first(grp->points);
    while(forms && err == CL_SUCCESS)
    {
      dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;
      if(grpt == NULL)
      {
        printf("rt_process_forms invalid form\n");
        forms = g_list_next(forms);
        continue;
      }
      if(grpt->formid == 0)
      {
        printf("rt_process_forms form is null\n");
        forms = g_list_next(forms);
        continue;
      }
      const int index = rt_get_index_from_formid(p, grpt->formid);
      if(index == -1)
      {
        // FIXME: we get this error when adding a new form and the system is still processing a previous add
        // we should not report an error in this case (and have a better way of adding a form)
        printf("rt_process_forms missing form from array=%i\n", grpt->formid);
        forms = g_list_next(forms);
        continue;
      }
      
      // only process current scale
      if(p->rt_forms[index].scale != scale)
      {
        forms = g_list_next(forms);
        continue;
      }
      
      // get the spot
      dt_masks_form_t *form = dt_masks_get_from_id(self->dev, grpt->formid);
      if(form == NULL)
      {
        printf("rt_process_forms missing form from masks=%i\n", grpt->formid);
        forms = g_list_next(forms);
        continue;
      }

      // if the form is outside the roi, we just skip it
      if(!rt_masks_form_is_in_roi(self, piece, form, roi_layer, roi_layer))
      {
        forms = g_list_next(forms);
        continue;
      }

      // get the mask
      float *mask = NULL;
      dt_iop_roi_t roi_mask = {0};
      
      dt_masks_get_mask(self, piece, form, &mask, &roi_mask.width, &roi_mask.height, &roi_mask.x, &roi_mask.y);
      if(mask == NULL)
      {
        printf("rt_process_forms error retrieving mask\n");
        forms = g_list_next(forms);
        continue;
      }
      
      int dx = 0, dy = 0;
      
      // search the delta with the source
      const dt_iop_retouch2_algo_type_t algo = p->rt_forms[index].algorithm;
      if (algo != dt_iop_retouch2_gaussian_blur && algo != dt_iop_retouch2_fill)
      {
        if(!rt_masks_get_delta(self, piece, roi_layer, form, &dx, &dy))
        {
          forms = g_list_next(forms);
          if (mask) free(mask);
          continue;
        }
      }
      
      // scale the mask
      cl_mem dev_mask_scaled = NULL;
      float *mask_scaled = NULL;
      dt_iop_roi_t roi_mask_scaled = {0};

     	err = rt_build_scaled_mask_cl(devid, mask, &roi_mask, &mask_scaled, &dev_mask_scaled, &roi_mask_scaled, roi_layer, dx, dy, algo);
      
     	if (algo != dt_iop_retouch2_heal && mask_scaled != NULL)
     	{
     		free(mask_scaled);
     		mask_scaled = NULL;
     	}
     	
      // we don't need the original mask anymore
      if (mask)
      {
      	free(mask);
      	mask = NULL;
      }
      
      if (mask_scaled == NULL && algo == dt_iop_retouch2_heal)
      {
        forms = g_list_next(forms);
        if (dev_mask_scaled) dt_opencl_release_mem_object(dev_mask_scaled);
        continue;
      }
      

      if ((err == CL_SUCCESS) && (dx != 0 || dy != 0 || algo == dt_iop_retouch2_gaussian_blur || algo == dt_iop_retouch2_fill) && 
          ((roi_mask_scaled.width > 2) && (roi_mask_scaled.height > 2)))
      {
        double start = dt_get_wtime();
        
        if (algo == dt_iop_retouch2_clone)
        {
          err = retouch2_clone_cl(devid, dev_layer, roi_layer, 
																	dev_mask_scaled, &roi_mask_scaled, mask_display, 
																	dx, dy, grpt->opacity, gd);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch2_clone took %0.04f sec\n", dt_get_wtime() - start);
        }
        else if (algo == dt_iop_retouch2_heal)
        {
        	err = retouch2_heal_cl(devid, dev_layer, roi_layer, 
																	mask_scaled, dev_mask_scaled, &roi_mask_scaled, 
																	dx, dy, mask_display, grpt->opacity, gd);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch2_heal took %0.04f sec\n", dt_get_wtime() - start);
        }
        else if (algo == dt_iop_retouch2_gaussian_blur)
        {
        	err = retouch2_gaussian_blur_cl(devid, dev_layer, roi_layer, 
																					dev_mask_scaled, &roi_mask_scaled, mask_display, 
																					grpt->opacity, p->rt_forms[index].blur_radius, piece, gd);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch2_gaussian_blur took %0.04f sec\n", dt_get_wtime() - start);
        }
        else if (algo == dt_iop_retouch2_fill)
        {
          // add a delta to the color so it can be fine-adjusted by the user
          float fill_color[3];
          
          if (p->rt_forms[index].fill_mode == dt_iop_rt_fill_erase)
          {
            fill_color[0] = fill_color[1] = fill_color[2] = p->rt_forms[index].fill_delta;
          }
          else
          {
            fill_color[0] = p->rt_forms[index].fill_color[0]+p->rt_forms[index].fill_delta;
            fill_color[1] = p->rt_forms[index].fill_color[1]+p->rt_forms[index].fill_delta;
            fill_color[2] = p->rt_forms[index].fill_color[2]+p->rt_forms[index].fill_delta;
          }
          
          err = retouch2_fill_cl(devid, dev_layer, roi_layer, 
																	dev_mask_scaled, &roi_mask_scaled, mask_display, 
																	grpt->opacity, fill_color, gd);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch2_fill took %0.04f sec\n", dt_get_wtime() - start);
        }
        else
          printf("rt_process_forms unknown algorithm %i\n", algo);
        
      }
      
      if (mask) free(mask);
      if (mask_scaled) free(mask_scaled);
      if (dev_mask_scaled) dt_opencl_release_mem_object(dev_mask_scaled);
      
      forms = g_list_next(forms);
    }
  }

  return err;
}

int process_cl(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_retouch2_params_t *p = (dt_iop_retouch2_params_t *)piece->data;
  dt_iop_retouch2_global_data_t *gd = (dt_iop_retouch2_global_data_t *)self->data;

  cl_int err = CL_SUCCESS;
  const int devid = piece->pipe->devid;

  cl_mem in_retouch = NULL;
  
  dt_iop_roi_t roi_retouch = *roi_in;
  dt_iop_roi_t *roi_rt = &roi_retouch;
  
  const int ch = piece->colors;
  _rt_user_data_t usr_data = {0};
  dwt_params_cl_t *dwt_p = NULL;
  dt_iop_retouch2_preview_types_t preview_type = p->preview_type;
  
  int gui_active = 0;
  if (self->dev) gui_active = (self == self->dev->gui_module);
  if (!gui_active && preview_type == dt_iop_rt_preview_current_scale) preview_type = dt_iop_rt_preview_final_image;
  
  // we will do all the clone, heal, etc on the input image, 
  // this way the source for one algorithm can be the destination from a previous one
  in_retouch = dt_opencl_alloc_device_buffer(devid, roi_rt->width * roi_rt->height * ch * sizeof(float));
  if (in_retouch == NULL)
  {
  	printf("rt_process: error allocating memory for wavelet decompose\n");
  	err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
  	goto cleanup;
  }
  
  // copy input image to the new buffer
  {
		size_t origin[] = { 0, 0, 0 };
		size_t region[] = { roi_rt->width, roi_rt->height, 1 };
		err = dt_opencl_enqueue_copy_image_to_buffer(devid, dev_in, in_retouch, origin, region, 0);
		if(err != CL_SUCCESS) goto cleanup;
  }

  // user data passed from the decompose routine to the one that process each scale
  usr_data.self = self;
  usr_data.piece = piece;
  usr_data.roi = *roi_rt;
  usr_data.mask_display = 0;
  usr_data.display_scale = p->curr_scale;

  // init the decompose routine
  dwt_p = dt_dwt_init_cl(devid, 
													in_retouch, 
													roi_rt->width, 
													roi_rt->height, 
													p->num_scales, 
													(preview_type == dt_iop_rt_preview_final_image) ? 0: p->curr_scale, 
													p->blend_factor, 
													&usr_data, 
													roi_in->scale / piece->iscale);
  if (dwt_p == NULL)
  {
  	printf("rt_process: error initializing wavelet decompose\n");
  	err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
  	goto cleanup;
  }

  // check if this module should expose mask. 
  if(self->request_mask_display && self->dev->gui_attached && (self == self->dev->gui_module)
     && (piece->pipe == self->dev->pipe) )
  {
  	const int kernel = gd->kernel_retouch_clear_alpha;
  	size_t sizes[] = { ROUNDUPWD(roi_rt->width), ROUNDUPHT(roi_rt->height), 1 };

    dt_opencl_set_kernel_arg(devid, kernel, 0, sizeof(cl_mem), (void *)&in_retouch);
    dt_opencl_set_kernel_arg(devid, kernel, 1, sizeof(int), (void *)&(roi_rt->width));
    dt_opencl_set_kernel_arg(devid, kernel, 2, sizeof(int), (void *)&(roi_rt->height));
    err = dt_opencl_enqueue_kernel_2d(devid, kernel, sizes);
  	if (err != CL_SUCCESS) goto cleanup;
  	
    piece->pipe->mask_display = 1;
    usr_data.mask_display = 1;
  }
  
  // check if the image support this number of scales
  if (piece->pipe->type == DT_DEV_PIXELPIPE_FULL && gui_active)
  {
    const int max_scales = dwt_get_max_scale_cl(dwt_p);
    if (dwt_p->scales > max_scales)
    {
      dt_control_log(_("max scale is %i for this image size"), max_scales);
    }
  }
  
  // decompose it
  if(self->suppress_mask && self->dev->gui_attached && (self == self->dev->gui_module)
     && (piece->pipe == self->dev->pipe))
  {
  	err = dwt_decompose_cl(dwt_p, NULL);
    if (err != CL_SUCCESS) goto cleanup;
  }
  else
  {
  	err = dwt_decompose_cl(dwt_p, rt_process_forms_cl);
    if (err != CL_SUCCESS) goto cleanup;
  }

  // copy alpha channel if nedded
  if(piece->pipe->mask_display && !usr_data.mask_display) 
  {
  	const int kernel = gd->kernel_retouch_copy_alpha;
  	size_t sizes[] = { ROUNDUPWD(roi_rt->width), ROUNDUPHT(roi_rt->height), 1 };

    dt_opencl_set_kernel_arg(devid, kernel, 0, sizeof(cl_mem), (void *)&dev_in);
    dt_opencl_set_kernel_arg(devid, kernel, 1, sizeof(cl_mem), (void *)&in_retouch);
    dt_opencl_set_kernel_arg(devid, kernel, 2, sizeof(int), (void *)&(roi_rt->width));
    dt_opencl_set_kernel_arg(devid, kernel, 3, sizeof(int), (void *)&(roi_rt->height));
    err = dt_opencl_enqueue_kernel_2d(devid, kernel, sizes);
  	if (err != CL_SUCCESS) goto cleanup;
  }
  
  // return final image
	err = rt_copy_in_to_out_cl(devid, in_retouch, roi_in, 
																	dev_out, roi_out, 0, 0,
																	gd->kernel_retouch_copy_buffer_to_image);

cleanup:
  if(dwt_p) dt_dwt_free_cl(dwt_p);

  if (in_retouch) dt_opencl_release_mem_object(in_retouch);
  
//  if (err != CL_SUCCESS) dt_print(DT_DEBUG_OPENCL, "[opencl_retouch] couldn't enqueue kernel! %d\n", err);
  if (err != CL_SUCCESS) printf("[opencl_retouch] couldn't enqueue kernel! %d\n", err);
  
  return (err == CL_SUCCESS) ? TRUE: FALSE;
}
#endif

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-space on;
