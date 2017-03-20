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
#include "bauhaus/bauhaus.h"
#include "develop/masks.h"
#include "develop/blend.h"
#include "develop/imageop_math.h"
#include "gui/accelerators.h"
#include <stdlib.h>
#include "common/gaussian.h"
#include "develop/heal.h"
#include "develop/dwt.h"

//#define _FFT_MULTFR_

// this is the version of the modules parameters,
// and includes version information about compile-time dt
DT_MODULE_INTROSPECTION(3, dt_iop_retouch_params_t)

#define RETOUCH_NO_FORMS 200
#define RETOUCH_MAX_SCALES 15
#define RETOUCH_NO_SCALES (RETOUCH_MAX_SCALES+2)

typedef enum dt_iop_retouch_preview_types_t
{
  dt_iop_rt_preview_final_image = 1,
  dt_iop_rt_preview_current_scale = 2,
  dt_iop_rt_preview_keep_current_scale = 3
} dt_iop_retouch_preview_types_t;

typedef enum dt_iop_retouch_fill_modes_t
{
  dt_iop_rt_fill_erase = 0,
  dt_iop_rt_fill_color = 1
} dt_iop_retouch_fill_modes_t;

typedef enum dt_iop_retouch_algo_type_t
{
  dt_iop_retouch_clone = 1, 
  dt_iop_retouch_heal = 2,
  dt_iop_retouch_gaussian_blur = 3, 
  dt_iop_retouch_fill = 4
} dt_iop_retouch_algo_type_t;

typedef struct dt_iop_retouch_form_data_t
{
  int id;
  dt_iop_retouch_algo_type_t algo;
} dt_iop_retouch_form_data_t;

typedef struct _rt_user_data_t
{
  dt_iop_module_t *self;
  dt_dev_pixelpipe_iop_t *piece;
  dt_iop_roi_t roi;
  int mask_display;
  int display_scale;
} _rt_user_data_t;


typedef struct dt_iop_retouch_params_v1_t
{
  dt_iop_retouch_form_data_t rt_forms[RETOUCH_NO_SCALES][RETOUCH_NO_FORMS]; // 0==original image; 1..RETOUCH_MAX_SCALES==scale; RETOUCH_MAX_SCALES+1==residual
  
  int algorithm; // clone, heal, blur, fill
  int heal_epsilon[RETOUCH_NO_SCALES]; // various parameters for the algorithm
  int xheal_max_iter[RETOUCH_NO_SCALES]; 
  int heal_w_factor[RETOUCH_NO_SCALES];
  float gblur_radius[RETOUCH_NO_SCALES];
  float fill_color[RETOUCH_NO_SCALES][3];

  int num_scales; // number of wavelets scales
  int curr_scale; // current wavelet scale
  
  int copied_scale; // scale to be copied to another scale
  
  dt_iop_retouch_preview_types_t preview_type; // final image, current scale...
  /*rt_show_channels_t*/dt_iop_retouch_preview_types_t rt_show_channel; // luma, chroma, red, green, blue...

  float rt_exposure; // exposure compensation for the scale
  int rt_normalize; // user asks to normalize the exposure for the scale
  float rt_bclip_percent; // % clipping for normalize
  float rt_wclip_percent;
} dt_iop_retouch_params_v1_t;

typedef struct dt_iop_retouch_params_v2_t
{
  dt_iop_retouch_form_data_t rt_forms[RETOUCH_NO_SCALES][RETOUCH_NO_FORMS]; // 0==original image; 1..RETOUCH_MAX_SCALES==scale; RETOUCH_MAX_SCALES+1==residual
  
  int algorithm; // clone, heal, blur, fill
  int heal_epsilon[RETOUCH_NO_SCALES]; // various parameters for the algorithm
  int xheal_max_iter[RETOUCH_NO_SCALES]; 
  int heal_w_factor[RETOUCH_NO_SCALES];
  float gblur_radius[RETOUCH_NO_SCALES];
  float fill_color[RETOUCH_NO_SCALES][3];

  int num_scales; // number of wavelets scales
  int curr_scale; // current wavelet scale
  float blend_factor;
  
  int copied_scale; // scale to be copied to another scale
  
  dt_iop_retouch_preview_types_t preview_type; // final image, current scale...
  /*rt_show_channels_t*/dt_iop_retouch_preview_types_t xrt_show_channel; // luma, chroma, red, green, blue...

  float xrt_exposure; // exposure compensation for the scale
  int xrt_normalize; // user asks to normalize the exposure for the scale
  float xrt_bclip_percent; // % clipping for normalize
  float xrt_wclip_percent;
  
  // G'MIC inpaint
  int xinp_scales[RETOUCH_NO_SCALES];
  int xinp_patch_size[RETOUCH_NO_SCALES];
  int xinp_iter_per_scale[RETOUCH_NO_SCALES];
  int xinp_blend_size[RETOUCH_NO_SCALES];
  int xinp_outer_blending[RETOUCH_NO_SCALES];
} dt_iop_retouch_params_v2_t;

typedef struct dt_iop_retouch_params_t
{
  dt_iop_retouch_form_data_t rt_forms[RETOUCH_NO_SCALES][RETOUCH_NO_FORMS]; // 0==original image; 1..RETOUCH_MAX_SCALES==scale; RETOUCH_MAX_SCALES+1==residual
  
  int algorithm; // clone, heal, blur, fill
  int xheal_epsilon[RETOUCH_NO_SCALES]; // various parameters for the algorithm
  int xheal_w_factor[RETOUCH_NO_SCALES];
  float gblur_radius[RETOUCH_NO_SCALES];
  float fill_color[RETOUCH_NO_SCALES][3];

  int num_scales; // number of wavelets scales
  int curr_scale; // current wavelet scale
  float blend_factor;
  
  int copied_scale; // scale to be copied to another scale
  
  dt_iop_retouch_preview_types_t preview_type; // final image, current scale...
  
  int fill_mode[RETOUCH_NO_SCALES];
  float blend_delta[RETOUCH_NO_SCALES];
} dt_iop_retouch_params_t;

typedef struct dt_iop_retouch_gui_data_t
{
  GtkLabel *label_form; // display number of forms
  GtkWidget *bt_edit_masks, *bt_path, *bt_circle, *bt_ellipse, *bt_brush; // shapes
  GtkWidget *bt_clone, *bt_heal, *bt_gaussian_blur, *bt_fill; // algorithms
  GtkWidget *bt_showmask, *bt_suppress; // supress & show masks
  
//  GtkWidget *sl_heal_epsilon; // parameters for some algorithms
//  GtkWidget *sl_heal_w_factor;
//  GtkWidget *hbox_heal;
  
  GtkWidget *sl_gblur_radius;
  GtkWidget *hbox_gblur;
 
  GtkWidget *sl_num_scales; // number of scales to decompose
  GtkWidget *sl_curr_scale; // current wavelet scale
  GtkWidget *sl_blend_factor; // blend factor
  
  GtkWidget *bt_show_final_image; // show recoposed image
  GtkWidget *bt_show_current_scale; // show decomposed scale
  GtkWidget *bt_keep_current_scale; // keep showing decomposed scale even if module is not active

  GtkWidget *bt_copy_scale; // copy all shapes from one scale to another
  GtkWidget *bt_paste_scale;

  GtkWidget *hbox_color;
  GtkWidget *hbox_color_pick;
  GtkWidget *colorpick; // select a specific color
  GtkToggleButton *color_picker; // pick a color from the picture

  GtkWidget *cmb_fill_mode;
  GtkWidget *sl_blend_delta;
} dt_iop_retouch_gui_data_t;

typedef struct dt_iop_retouch_params_t dt_iop_retouch_data_t;

// this returns a translatable name
const char *name()
{
  return _("retouch_eh");
}

int groups()
{
  return IOP_GROUP_CORRECT;
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_NO_MASKS;
}

int legacy_params(dt_iop_module_t *self, const void *const old_params, const int old_version,
                  void *new_params, const int new_version)
{
  if(old_version == 1 && new_version == 3)
  {
    dt_iop_retouch_params_v1_t *o = (dt_iop_retouch_params_v1_t *)old_params;
    dt_iop_retouch_params_t *n = (dt_iop_retouch_params_t *)new_params;

    memcpy(n, o, sizeof(*o));

    n->num_scales = o->num_scales;
    n->curr_scale = o->curr_scale;
    n->blend_factor = .128f;
    
    n->copied_scale = o->copied_scale;
    
    n->preview_type = o->preview_type;
    
    for (int i = 0; i < RETOUCH_NO_SCALES; i++)
    {
      n->fill_mode[i]=dt_iop_rt_fill_erase;
      n->blend_delta[i]=0;
    }
    return 0;
  }
  if(old_version == 2 && new_version == 3)
  {
    dt_iop_retouch_params_v2_t *o = (dt_iop_retouch_params_v2_t *)old_params;
    dt_iop_retouch_params_t *n = (dt_iop_retouch_params_t *)new_params;

    for (int i = 0; i < RETOUCH_NO_SCALES; i++)
    {
      for (int j = 0; j < RETOUCH_NO_FORMS; j++)
        n->rt_forms[i][j] = o->rt_forms[i][j];
    }
    
    n->algorithm = o->algorithm;
    for (int i = 0; i < RETOUCH_NO_SCALES; i++)
    {
//      n->heal_epsilon[i] = o->heal_epsilon[i];
//      n->heal_w_factor[i] = o->heal_w_factor[i];
      n->gblur_radius[i] = o->gblur_radius[i];
      n->fill_color[i][0] = o->fill_color[i][0];
      n->fill_color[i][1] = o->fill_color[i][1];
      n->fill_color[i][2] = o->fill_color[i][2];
    }
    
    n->num_scales = o->num_scales;
    n->curr_scale = o->curr_scale;
    n->blend_factor = o->blend_factor;
    
    n->copied_scale = o->copied_scale;
    
    n->preview_type = o->preview_type;

    for (int i = 0; i < RETOUCH_NO_SCALES; i++)
    {
      n->fill_mode[i]=dt_iop_rt_fill_erase;
      n->blend_delta[i]=0;
    }
    return 0;
  }
  return 1;
}


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

static void _retouch_cairo_paint_tool_clone(cairo_t *cr,
                                          const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  cairo_arc(cr, 0.65, 0.35, 0.35, 0, 2 * M_PI);
  cairo_stroke(cr);
  
  cairo_arc(cr, 0.35, 0.65, 0.35, 0, 2 * M_PI);
  cairo_stroke(cr);

  POSTAMBLE;
}

static void _retouch_cairo_paint_tool_heal(cairo_t *cr,
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

static void _retouch_cairo_paint_tool_fill(cairo_t *cr,
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

static void _retouch_cairo_paint_tool_blur(cairo_t *cr, gint x, gint y, gint w, gint h, gint flags)
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

static void _retouch_cairo_paint_paste_forms(cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags)
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

static void _retouch_cairo_paint_cut_forms(cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags)
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

static void _retouch_cairo_paint_show_final_image(cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags)
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

static void _retouch_cairo_paint_show_current_scale(cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags)
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

static void _retouch_cairo_paint_keep_current_scale(cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags)
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
// GUI callbacks
//---------------------------------------------------------------------------------

void rt_paste_forms_from_scale(dt_iop_retouch_params_t *p, const int source_scale, const int dest_scale)
{
  if (source_scale != dest_scale && source_scale >= 0 && dest_scale >= 0)
  {
    for (int i = 0; i < RETOUCH_NO_FORMS; i++)
    {
      p->rt_forms[dest_scale][i] = p->rt_forms[source_scale][i];
      
      p->rt_forms[source_scale][i].id = 0;
      p->rt_forms[source_scale][i].algo = 0;
    }
  }
}

static void rt_display_selected_fill_color(dt_iop_retouch_gui_data_t *g, dt_iop_retouch_params_t *p)
{
  GdkRGBA c = (GdkRGBA){.red = p->fill_color[p->curr_scale][0]+p->blend_factor, .green = p->fill_color[p->curr_scale][1]+p->blend_factor, 
                          .blue = p->fill_color[p->curr_scale][2]+p->blend_factor, .alpha = 1.0 };
  gtk_color_chooser_set_rgba(GTK_COLOR_CHOOSER(g->colorpick), &c);
}

static void rt_update_gui_scale_data(dt_iop_retouch_gui_data_t *g, dt_iop_retouch_params_t *p)
{
  
//  dt_bauhaus_slider_set(g->sl_heal_epsilon, p->heal_epsilon[p->curr_scale]);
//  dt_bauhaus_slider_set(g->sl_heal_w_factor, p->heal_w_factor[p->curr_scale]);
  dt_bauhaus_slider_set(g->sl_gblur_radius, p->gblur_radius[p->curr_scale]);
  dt_bauhaus_slider_set(g->sl_blend_delta, p->blend_delta[p->curr_scale]);
  dt_bauhaus_combobox_set(g->cmb_fill_mode, p->fill_mode[p->curr_scale]);
  
  rt_display_selected_fill_color(g, p);
}

static void rt_show_hide_controls(dt_iop_retouch_gui_data_t *d, dt_iop_retouch_params_t *p)
{
  switch (p->algorithm)
  {
    case dt_iop_retouch_heal:
//      gtk_widget_show(GTK_WIDGET(d->hbox_heal));
      gtk_widget_hide(GTK_WIDGET(d->hbox_gblur));
      gtk_widget_hide(GTK_WIDGET(d->hbox_color));
      break;
    case dt_iop_retouch_gaussian_blur:
//      gtk_widget_hide(GTK_WIDGET(d->hbox_heal));
      gtk_widget_show(GTK_WIDGET(d->hbox_gblur));
      gtk_widget_hide(GTK_WIDGET(d->hbox_color));
      break;
    case dt_iop_retouch_fill:
//      gtk_widget_hide(GTK_WIDGET(d->hbox_heal));
      gtk_widget_hide(GTK_WIDGET(d->hbox_gblur));
      gtk_widget_show(GTK_WIDGET(d->hbox_color));
      if (p->fill_mode[p->curr_scale] == dt_iop_rt_fill_color)
        gtk_widget_show(GTK_WIDGET(d->hbox_color_pick));
      else
        gtk_widget_hide(GTK_WIDGET(d->hbox_color_pick));
      break;
    case dt_iop_retouch_clone:
    default:
//      gtk_widget_hide(GTK_WIDGET(d->hbox_heal));
      gtk_widget_hide(GTK_WIDGET(d->hbox_gblur));
      gtk_widget_hide(GTK_WIDGET(d->hbox_color));
      break;
  }
}

static int rt_allow_create_form(dt_iop_module_t *self)
{
  int allow = 1;
  
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  if (p)
  {
    allow = (p->rt_forms[p->curr_scale][RETOUCH_NO_FORMS-1].id == 0);
  }
  return allow;
}

static void rt_reset_form_creation(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;
  if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_path)) ||
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_circle)) ||
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_ellipse)) ||
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_brush)))
  {
    // we unset the creation mode
/*    dt_masks_form_t *form = darktable.develop->form_visible;
    if(form) dt_masks_free_form(form); */
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

static void rt_masks_selection_change(dt_iop_module_t *self, dt_iop_retouch_params_t *p)
{
  if (!self->enabled) return;
  
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  if (bd == NULL) return;
  int scale = p->curr_scale;
  int count = 0;
  
  for (int i = 0; i < RETOUCH_NO_FORMS && count == 0; i++)
  {
    if (p->rt_forms[scale][i].id != 0) count++;
  }

  // if selection empty, we hide all
  if (bd->masks_shown == DT_MASKS_EDIT_OFF || count == 0)
  {
    dt_masks_change_form_gui(NULL);
    dt_control_queue_redraw_center();
    return;
  }
  
  // else, we create a new from group with the selection and display it
  dt_masks_form_t *grp = dt_masks_create(DT_MASKS_GROUP);
  for (int i = 0; i < RETOUCH_NO_FORMS; i++)
  {
    int grid = self->blend_params->mask_id;
    int id = p->rt_forms[scale][i].id;
    dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
    if(form)
    {
      dt_masks_point_group_t *fpt = (dt_masks_point_group_t *)malloc(sizeof(dt_masks_point_group_t));
      fpt->formid = id;
      fpt->parentid = grid;
      fpt->state = DT_MASKS_STATE_USE;
      fpt->opacity = 1.0f;
      grp->points = g_list_append(grp->points, fpt);
    }
  }

  dt_masks_form_t *grp2 = dt_masks_create(DT_MASKS_GROUP);
  grp2->formid = 0;
  dt_masks_group_ungroup(grp2, grp);
//  dt_masks_free_form(grp);
  dt_masks_change_form_gui(grp2);
  darktable.develop->form_gui->edit_mode = bd->masks_shown;
  dt_control_queue_redraw_center();
}


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
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;

  // turn off the other color picker
  gtk_toggle_button_set_active(g->color_picker, FALSE);

  GdkRGBA c = (GdkRGBA){.red = p->fill_color[p->curr_scale][0]+p->blend_factor, .green = p->fill_color[p->curr_scale][1]+p->blend_factor, 
                          .blue = p->fill_color[p->curr_scale][2]+p->blend_factor, .alpha = 1.0 };
  gtk_color_chooser_get_rgba(GTK_COLOR_CHOOSER(widget), &c);
  p->fill_color[p->curr_scale][0] = c.red-p->blend_factor;
  p->fill_color[p->curr_scale][1] = c.green-p->blend_factor;
  p->fill_color[p->curr_scale][2] = c.blue-p->blend_factor;

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static gboolean rt_draw_callback(GtkWidget *widget, cairo_t *cr, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return FALSE;
  if(self->picked_output_color_max[0] < 0) return FALSE;
  if(self->request_color_pick == DT_REQUEST_COLORPICK_OFF) return FALSE;
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;

  // interrupt if no valid color reading
  if(self->picked_output_color_min[0] == INFINITY) return FALSE;

  if(fabsf(p->fill_color[p->curr_scale][0]+p->blend_factor - self->picked_output_color[0]) < 0.0001f
     && fabsf(p->fill_color[p->curr_scale][1]+p->blend_factor - self->picked_output_color[1]) < 0.0001f
     && fabsf(p->fill_color[p->curr_scale][2]+p->blend_factor - self->picked_output_color[2]) < 0.0001f)
  {
    // interrupt infinite loops
    return FALSE;
  }

  p->fill_color[p->curr_scale][0] = self->picked_output_color[0]-p->blend_factor;
  p->fill_color[p->curr_scale][1] = self->picked_output_color[1]-p->blend_factor;
  p->fill_color[p->curr_scale][2] = self->picked_output_color[2]-p->blend_factor;

  rt_display_selected_fill_color(g, p);
    
  dt_dev_add_history_item(darktable.develop, self, TRUE);
  
  return FALSE;
}

static void rt_copypaste_scale_callback(GtkToggleButton *togglebutton, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  int active = gtk_toggle_button_get_active(togglebutton);
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;

  if (togglebutton == (GtkToggleButton *)g->bt_copy_scale)
  {
    p->copied_scale = (active) ? p->curr_scale: -1;
  }
  else if (togglebutton == (GtkToggleButton *)g->bt_paste_scale)
  {
    rt_paste_forms_from_scale(p, p->copied_scale, p->curr_scale);
    
    p->copied_scale = -1;
  }

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_copy_scale), p->copied_scale >= 0);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_paste_scale), p->copied_scale < 0);

  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_scale_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  p->num_scales = dt_bauhaus_slider_get(slider);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_preview_image_callback(GtkToggleButton *togglebutton, dt_iop_module_t *module)
{
  if(darktable.gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)module->params;
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)module->gui_data;

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

static void rt_n_image_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;
  p->curr_scale = dt_bauhaus_slider_get(slider);
  
  rt_masks_selection_change(self, p);

  rt_update_gui_scale_data(g, p);
  rt_show_hide_controls(g, p);
  
  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_blend_factor_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  p->blend_factor = dt_bauhaus_slider_get(slider);
  
  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}


static gboolean rt_edit_masks_callback(GtkWidget *widget, GdkEventButton *event, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return FALSE;
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;

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

    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), bd->masks_shown != DT_MASKS_EDIT_OFF);

    darktable.gui->reset = 0;

    dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
    rt_masks_selection_change(self, p);
    
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
    
    dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
    dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;

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
   
/*    if (type == DT_MASKS_BRUSH && p->algorithm != dt_iop_retouch_fill)
    {
    	return TRUE;
    }
    else*/
    {
		// we create the new form
		dt_masks_form_t *spot = NULL;
		if (p->algorithm == dt_iop_retouch_clone || p->algorithm == dt_iop_retouch_heal)
		  spot = dt_masks_create(type | DT_MASKS_CLONE);
		else
		  spot = dt_masks_create(type);
		
		dt_masks_change_form_gui(spot);
		darktable.develop->form_gui->creation = TRUE;
		darktable.develop->form_gui->creation_module = self;
		dt_control_queue_redraw_center();
    }
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

  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;

  if (togglebutton == (GtkToggleButton *)g->bt_gaussian_blur)
    p->algorithm = dt_iop_retouch_gaussian_blur;
  else if (togglebutton == (GtkToggleButton *)g->bt_clone)
    p->algorithm = dt_iop_retouch_clone;
  else if (togglebutton == (GtkToggleButton *)g->bt_heal)
    p->algorithm = dt_iop_retouch_heal;
  else if (togglebutton == (GtkToggleButton *)g->bt_fill)
    p->algorithm = dt_iop_retouch_fill;

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_clone), (p->algorithm == dt_iop_retouch_clone));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_heal), (p->algorithm == dt_iop_retouch_heal));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_gaussian_blur), (p->algorithm == dt_iop_retouch_gaussian_blur));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_fill), (p->algorithm == dt_iop_retouch_fill));

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
/*
static void rt_heal_epsilon_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  p->heal_epsilon[p->curr_scale] = dt_bauhaus_slider_get(slider);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_heal_w_factor_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  p->heal_w_factor[p->curr_scale] = dt_bauhaus_slider_get(slider);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}
*/
static void rt_gblur_radius_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  p->gblur_radius[p->curr_scale] = dt_bauhaus_slider_get(slider);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_fill_mode_callback(GtkComboBox *combo, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;
  
  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;

  p->fill_mode[p->curr_scale] = dt_bauhaus_combobox_get((GtkWidget *)combo);
  
  rt_show_hide_controls(g, p);
  
  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void rt_blend_delta_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  p->blend_delta[p->curr_scale] = dt_bauhaus_slider_get(slider);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

//--------------------------------------------------------------------------------------------------
// GUI
//--------------------------------------------------------------------------------------------------


dt_iop_retouch_algo_type_t rt_get_algo_from_id(dt_iop_retouch_form_data_t *rt_forms, const int formid)
{
  dt_iop_retouch_algo_type_t algo = 0;
  int i = 0;
  
  while (algo == 0 && i < RETOUCH_NO_FORMS)
  {
    if (rt_forms[i].id == formid) algo = rt_forms[i].algo;
    i++;
  }
  
  return algo;
}

dt_iop_retouch_algo_type_t rt_get_algo_from_forms_id(dt_iop_retouch_params_t *p, const int formid)
{
  dt_iop_retouch_algo_type_t algo = 0;
  int i = 0;
  
  while (algo == 0 && i < RETOUCH_NO_SCALES)
  {
    algo = rt_get_algo_from_id(p->rt_forms[i], formid);
    i++;
  }
  
  return algo;
}

static void rt_resynch_params_f(dt_iop_retouch_params_t *p, dt_develop_blend_params_t *bp, dt_iop_retouch_form_data_t *rt_forms, const int algorithm, const int add_new)
{
  dt_iop_retouch_form_data_t forms_d[RETOUCH_NO_FORMS] = { {0, 0} };
  
  // we go through all forms in blend params
  dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, bp->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    GList *forms = g_list_first(grp->points);
    int i = 0;
    while((i < RETOUCH_NO_FORMS) && forms)
    {
      dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;
      
      int j = 0;
      while( j < RETOUCH_NO_FORMS && rt_forms[j].id != grpt->formid )
      {
        j++;
      }
      
      if (j < RETOUCH_NO_FORMS)
      {
        forms_d[i].algo = rt_forms[j].algo;
        forms_d[i].id = grpt->formid;
        i++;
      }
      else if (add_new)
      {
        int algo_new = rt_get_algo_from_forms_id(p, grpt->formid);
        if (algo_new <= 0)
        {
          dt_masks_form_t *parent_form = dt_masks_get_from_id(darktable.develop, grpt->formid);
          if (parent_form)
          {
            forms_d[i].algo = algorithm;
            forms_d[i].id = grpt->formid;
            
            i++;
          }
        }
      }
        
      forms = g_list_next(forms);
    }
  }

  // we reaffect params
  for(int i = 0; i < RETOUCH_NO_FORMS; i++)
  {
    rt_forms[i].algo = forms_d[i].algo;
    rt_forms[i].id = forms_d[i].id;
  }
}

static void rt_resynch_params(struct dt_iop_module_t *self)
{
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
  dt_develop_blend_params_t *bp = self->blend_params;

  for(int i = 0; i < RETOUCH_NO_SCALES; i++)
  {
    rt_resynch_params_f(p, bp, p->rt_forms[i], p->algorithm, (p->curr_scale == i));
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
//    dx = dy = 0;
//    res = 1;
    dt_masks_point_brush_t *pt = (dt_masks_point_brush_t *)form->points->data;

    res = rt_masks_point_calc_delta(self, piece, roi, pt->corner, form->source, dx, dy);
  }

  return res;
}


void init(dt_iop_module_t *module)
{
  // we don't need global data:
  module->data = NULL; // malloc(sizeof(dt_iop_retouch_global_data_t));
  module->params = calloc(1, sizeof(dt_iop_retouch_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_retouch_params_t));
  // our module is disabled by default
  module->default_enabled = 0;
  module->priority = 179; // module order created by iop_dependencies.py, do not edit!
  module->params_size = sizeof(dt_iop_retouch_params_t);
  module->gui_data = NULL;
  // init defaults:
  dt_iop_retouch_params_t tmp = (dt_iop_retouch_params_t){ .rt_forms={ {{0, 0}, {0, 0}} }, 
                                    .algorithm=dt_iop_retouch_clone, 
                                    .num_scales=0, 
                                    .curr_scale=0, 
                                    .blend_factor=.128f,
                                    .copied_scale=-1,
                                    .preview_type=dt_iop_rt_preview_final_image};
  for (int i=0; i< RETOUCH_NO_SCALES; i++)
  {
    tmp.xheal_epsilon[i] = 255;
    tmp.xheal_w_factor[i] = 4;
    tmp.gblur_radius[i] = 0.1f;
    tmp.fill_color[i][0] = tmp.fill_color[i][1] = tmp.fill_color[i][2] = 0.f;

    tmp.fill_mode[i] = dt_iop_rt_fill_erase;
    tmp.blend_delta[i] = 0.f;
  }
  
  memcpy(module->params, &tmp, sizeof(dt_iop_retouch_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_retouch_params_t));
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
  free(module->data); // just to be sure
  module->data = NULL;
}

void gui_focus(struct dt_iop_module_t *self, gboolean in)
{
  if(self->enabled)
  {
    if(in)
    {
      // got focus, show all shapes
      dt_masks_set_edit_mode(self, DT_MASKS_EDIT_FULL);

      dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), TRUE);
      
      dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;
      rt_masks_selection_change(self, p);

    }
    else
    {
      // lost focus, hide all shapes and free if some are in creation
      if (darktable.develop->form_gui->creation && darktable.develop->form_gui->creation_module == self)
        dt_masks_change_form_gui(NULL);
/*      {
        dt_masks_form_t *form = darktable.develop->form_visible;
        if(form) dt_masks_free_form(form);
        dt_masks_change_form_gui(NULL);
      }*/
      dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;
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
  memcpy(piece->data, params, sizeof(dt_iop_retouch_params_t));
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_retouch_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;

  rt_resynch_params(self);
  
  // update clones count
  dt_masks_form_t *grp = dt_masks_get_from_id(self->dev, self->blend_params->mask_id);
  guint nb = 0;
  if(grp && (grp->type & DT_MASKS_GROUP)) nb = g_list_length(grp->points);
  gchar *str = g_strdup_printf("%d", nb);
  gtk_label_set_text(g->label_form, str);
  g_free(str);


  // update buttons status
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  if (bd)
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), bd->masks_shown != DT_MASKS_EDIT_OFF);
  }
  else
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), FALSE);
  }
  
  rt_masks_selection_change(self, p);
  
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_clone), p->algorithm == dt_iop_retouch_clone);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_gaussian_blur), p->algorithm == dt_iop_retouch_gaussian_blur);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_heal), p->algorithm == dt_iop_retouch_heal);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_fill), p->algorithm == dt_iop_retouch_fill);
  
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

  dt_bauhaus_slider_set(g->sl_num_scales, p->num_scales);
  dt_bauhaus_slider_set(g->sl_curr_scale, p->curr_scale);
  dt_bauhaus_slider_set(g->sl_blend_factor, p->blend_factor);

  rt_update_gui_scale_data(g, p);

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_show_final_image), p->preview_type==dt_iop_rt_preview_final_image);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_show_current_scale), p->preview_type==dt_iop_rt_preview_current_scale);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_keep_current_scale), p->preview_type==dt_iop_rt_preview_keep_current_scale);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_copy_scale), p->copied_scale >= 0);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_paste_scale), p->copied_scale < 0);
  
  rt_show_hide_controls(g, p);
  
}

void gui_init(dt_iop_module_t *self)
{
  const int bs = DT_PIXEL_APPLY_DPI(14);
  
  self->gui_data = malloc(sizeof(dt_iop_retouch_gui_data_t));
  dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *)self->gui_data;
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)self->params;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  GtkWidget *label = gtk_label_new(_("# strokes:"));
  gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, TRUE, 0);
  g->label_form = GTK_LABEL(gtk_label_new("-1"));
  g_object_set(G_OBJECT(hbox), "tooltip-text", _("click on a shape and drag on canvas.\nuse the mouse wheel "
                                                 "to adjust size.\nright click to remove a shape."), (char *)NULL);

  g->bt_brush = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_brush, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_brush), "button-press-event", G_CALLBACK(rt_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_brush), "tooltip-text", _("add brush"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_brush), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_brush, FALSE, FALSE, 0);

  g->bt_path = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_path, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_path), "button-press-event", G_CALLBACK(rt_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_path), "tooltip-text", _("add path"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_path), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_path, FALSE, FALSE, 0);

  g->bt_ellipse = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_ellipse, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_ellipse), "button-press-event", G_CALLBACK(rt_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_ellipse), "tooltip-text", _("add ellipse"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_ellipse), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_ellipse), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_ellipse, FALSE, FALSE, 0);

  g->bt_circle = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_circle, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_circle), "button-press-event", G_CALLBACK(rt_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_circle), "tooltip-text", _("add circle"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_circle), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_circle), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_circle, FALSE, FALSE, 0);

  g->bt_edit_masks = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_eye, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_edit_masks), "button-press-event", G_CALLBACK(rt_edit_masks_callback), self);
  g_object_set(G_OBJECT(g->bt_edit_masks), "tooltip-text", _("show and edit mask elements"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_edit_masks), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_edit_masks, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(hbox), GTK_WIDGET(g->label_form), FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox, TRUE, TRUE, 0);

  
  GtkWidget *hbox_algo = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  
  GtkWidget *label1 = gtk_label_new(_("algorithms:"));
  gtk_box_pack_start(GTK_BOX(hbox_algo), label1, FALSE, TRUE, 0);
  
  g->bt_fill = dtgtk_togglebutton_new(_retouch_cairo_paint_tool_fill, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_fill), "tooltip-text", _("activates fill tool"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_fill), "toggled", G_CALLBACK(rt_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_fill), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_fill), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_fill, FALSE, FALSE, 0);

  g->bt_gaussian_blur = dtgtk_togglebutton_new(_retouch_cairo_paint_tool_blur, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_gaussian_blur), "tooltip-text", _("activates gaussian blur tool"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_gaussian_blur), "toggled", G_CALLBACK(rt_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_gaussian_blur), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_gaussian_blur), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_gaussian_blur, FALSE, FALSE, 0);

  g->bt_heal = dtgtk_togglebutton_new(_retouch_cairo_paint_tool_heal, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_heal), "tooltip-text", _("activates healing tool"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_heal), "toggled", G_CALLBACK(rt_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_heal), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_heal), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_heal, FALSE, FALSE, 0);

  g->bt_clone = dtgtk_togglebutton_new(_retouch_cairo_paint_tool_clone, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_clone), "tooltip-text", _("activates cloning tool"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_clone), "toggled", G_CALLBACK(rt_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_clone), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_clone), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_clone, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(self->widget), hbox_algo, TRUE, TRUE, 0);
  
  GtkWidget *hbox_show_hide = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  
  GtkWidget *label2 = gtk_label_new(_("masks display:"));
  gtk_box_pack_start(GTK_BOX(hbox_show_hide), label2, FALSE, TRUE, 0);
  
  g->bt_showmask = dtgtk_togglebutton_new(dtgtk_cairo_paint_showmask, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_showmask), "tooltip-text", _("display mask"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_showmask), "toggled", G_CALLBACK(rt_showmask_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_showmask), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_showmask), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_show_hide), g->bt_showmask, FALSE, FALSE, 0);

  g->bt_suppress = dtgtk_togglebutton_new(dtgtk_cairo_paint_eye_toggle, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_suppress), "tooltip-text", _("temporarily switch off masks"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_suppress), "toggled", G_CALLBACK(rt_suppress_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_suppress), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_suppress), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_show_hide), g->bt_suppress, FALSE, FALSE, 0);

/*  g->hbox_heal = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);

  g->sl_heal_epsilon = dt_bauhaus_slider_new_with_range(self, 1.0, 500.0, 1, 255.0, 0);
  dt_bauhaus_widget_set_label(g->sl_heal_epsilon, _("epsilon"), _("epsilon"));
  g_object_set(g->sl_heal_epsilon, "tooltip-text", _("deviation-from-smoothness tolerance."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_heal_epsilon), "value-changed", G_CALLBACK(rt_heal_epsilon_callback), self);

  g->sl_heal_w_factor = dt_bauhaus_slider_new_with_range(self, 4.0, 255.0, 1, 4.0, 0);
  dt_bauhaus_widget_set_label(g->sl_heal_w_factor, _("w factor"), _("w factor"));
  g_object_set(g->sl_heal_w_factor, "tooltip-text", _("over-relaxation factor."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_heal_w_factor), "value-changed", G_CALLBACK(rt_heal_w_factor_callback), self);

  gtk_box_pack_start(GTK_BOX(g->hbox_heal), g->sl_heal_epsilon, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->hbox_heal), g->sl_heal_w_factor, TRUE, TRUE, 0);
*/
  g->hbox_gblur = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);

  g->sl_gblur_radius = dt_bauhaus_slider_new_with_range(self, 0.1, 200.0, 0.1, 0.1, 1);
  dt_bauhaus_widget_set_label(g->sl_gblur_radius, _("blur radius"), _("blur radius"));
  dt_bauhaus_slider_set_format(g->sl_gblur_radius, "%.01f");
  g_object_set(g->sl_gblur_radius, "tooltip-text", _("radius of gaussian blur."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_gblur_radius), "value-changed", G_CALLBACK(rt_gblur_radius_callback), self);

  gtk_box_pack_start(GTK_BOX(g->hbox_gblur), g->sl_gblur_radius, TRUE, TRUE, 0);

  GtkWidget *hbox_scale = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);

  g->sl_num_scales = dt_bauhaus_slider_new_with_range(self, 0.0, RETOUCH_MAX_SCALES, 1, 0.0, 0);
  dt_bauhaus_widget_set_label(g->sl_num_scales, _("number of scales"), _("scale"));
  g_object_set(g->sl_num_scales, "tooltip-text", _("number of scales to decompose."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_num_scales), "value-changed", G_CALLBACK(rt_scale_callback), self);
  
  g->bt_show_final_image = dtgtk_togglebutton_new(_retouch_cairo_paint_show_final_image, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_show_final_image), "tooltip-text", _("show final image"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_show_final_image), "toggled", G_CALLBACK(rt_preview_image_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_show_final_image), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_show_final_image), FALSE);
  
  g->bt_show_current_scale = dtgtk_togglebutton_new(_retouch_cairo_paint_show_current_scale, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_show_current_scale), "tooltip-text", _("show current scale"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_show_current_scale), "toggled", G_CALLBACK(rt_preview_image_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_show_current_scale), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_show_current_scale), FALSE);
  
  g->bt_keep_current_scale = dtgtk_togglebutton_new(_retouch_cairo_paint_keep_current_scale, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_keep_current_scale), "tooltip-text", _("keep current scale"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_keep_current_scale), "toggled", G_CALLBACK(rt_preview_image_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_keep_current_scale), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_keep_current_scale), FALSE);
  
  gtk_box_pack_end(GTK_BOX(hbox_scale), g->bt_keep_current_scale, FALSE, FALSE, 0);
  gtk_box_pack_end(GTK_BOX(hbox_scale), g->bt_show_current_scale, FALSE, FALSE, 0);
  gtk_box_pack_end(GTK_BOX(hbox_scale), g->bt_show_final_image, FALSE, FALSE, 0);
  gtk_box_pack_start(GTK_BOX(hbox_scale), g->sl_num_scales, TRUE, TRUE, 0);
  
  GtkWidget *hbox_n_image = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  
  g->sl_curr_scale = dt_bauhaus_slider_new_with_range(self, 0.0, RETOUCH_MAX_SCALES+1, 1, 0.0, 0);
  dt_bauhaus_widget_set_label(g->sl_curr_scale, _("current scale"), _("current scale"));
  g_object_set(g->sl_curr_scale, "tooltip-text", _("current decomposed scale, if zero, final image is displayed."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_curr_scale), "value-changed", G_CALLBACK(rt_n_image_callback), self);
  
  g->bt_copy_scale = dtgtk_togglebutton_new(_retouch_cairo_paint_cut_forms, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_copy_scale), "tooltip-text", _("cut shapes from current scale."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_copy_scale), "toggled", G_CALLBACK(rt_copypaste_scale_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_copy_scale), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_copy_scale), FALSE);
  
  g->bt_paste_scale = dtgtk_togglebutton_new(_retouch_cairo_paint_paste_forms, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_paste_scale), "tooltip-text", _("paste cutted shapes to current scale."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_paste_scale), "toggled", G_CALLBACK(rt_copypaste_scale_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_paste_scale), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_paste_scale), FALSE);
  
  GtkWidget *hbox_blend = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  
  g->sl_blend_factor = dt_bauhaus_slider_new_with_range(self, 0.0, 1.0, 0.05, 0.128, 3);
  dt_bauhaus_widget_set_label(g->sl_blend_factor, _("blend factor"), _("blend factor"));
  g_object_set(g->sl_blend_factor, "tooltip-text", _("blend factor, increse to lighten the image scale. works only for preview."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_blend_factor), "value-changed", G_CALLBACK(rt_blend_factor_callback), self);
  
  GdkRGBA color = (GdkRGBA){.red = p->fill_color[p->curr_scale][0]+p->blend_factor, 
    .green = p->fill_color[p->curr_scale][1]+p->blend_factor, 
    .blue = p->fill_color[p->curr_scale][2]+p->blend_factor, 
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

  g->sl_blend_delta = dt_bauhaus_slider_new_with_range(self, -1.0, 1.0, .0005, .0, 4);
  dt_bauhaus_widget_set_label(g->sl_blend_delta, _("delta"), _("delta"));
  g_object_set(g->sl_blend_delta, "tooltip-text", _("add delta to the image before compare if darken or lighten."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_blend_delta), "value-changed", G_CALLBACK(rt_blend_delta_callback), self);

  GtkWidget *label4 = gtk_label_new(_("fill color: "));

  gtk_box_pack_end(GTK_BOX(g->hbox_color_pick), GTK_WIDGET(g->color_picker), FALSE, FALSE, 0);
  gtk_box_pack_end(GTK_BOX(g->hbox_color_pick), GTK_WIDGET(g->colorpick), FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->hbox_color_pick), label4, FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->hbox_color), GTK_WIDGET(g->cmb_fill_mode), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->hbox_color), g->hbox_color_pick, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(g->hbox_color), g->sl_blend_delta, TRUE, TRUE, 0);

  gtk_box_pack_end(GTK_BOX(hbox_n_image), g->bt_paste_scale, FALSE, FALSE, 0);
  gtk_box_pack_end(GTK_BOX(hbox_n_image), g->bt_copy_scale, FALSE, FALSE, 0);
  gtk_box_pack_start(GTK_BOX(hbox_n_image), g->sl_curr_scale, TRUE, TRUE, 0);
  
  gtk_box_pack_start(GTK_BOX(hbox_blend), g->sl_blend_factor, TRUE, TRUE, 0);
  
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_scale, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_n_image, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_blend, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_show_hide, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), g->hbox_gblur, TRUE, TRUE, 0);
//  gtk_box_pack_start(GTK_BOX(self->widget), g->hbox_heal, TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), g->hbox_color, TRUE, TRUE, 0);

  
  g_signal_connect(G_OBJECT(self->widget), "draw", G_CALLBACK(rt_draw_callback), self);

//  gtk_widget_show_all(g->hbox_heal);
//  gtk_widget_set_no_show_all(g->hbox_heal, TRUE);

  gtk_widget_show_all(g->hbox_gblur);
  gtk_widget_set_no_show_all(g->hbox_gblur, TRUE);

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
 
  int roir = roi_in->width + roi_in->x;
  int roib = roi_in->height + roi_in->y;
  int roix = roi_in->x;
  int roiy = roi_in->y;

  dt_develop_blend_params_t *bp = self->blend_params;
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)piece->data;

  // We iterate through all retouch or polygons
//  dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, bp->mask_id);
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
        if(!rt_masks_form_is_in_roi(self, piece, form, roi_in, roi_out))
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
        const dt_iop_retouch_algo_type_t algo = rt_get_algo_from_forms_id(p, grpt->formid);
        if (algo == dt_iop_retouch_heal)
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


void init_key_accels(dt_iop_module_so_t *module)
{
  dt_accel_register_iop (module, TRUE, NC_("accel", "retouch circle tool"),   0, 0);
  dt_accel_register_iop (module, TRUE, NC_("accel", "retouch elipse tool"),   0, 0);
  dt_accel_register_iop (module, TRUE, NC_("accel", "retouch path tool"),     0, 0);
  dt_accel_register_iop (module, TRUE, NC_("accel", "retouch brush tool"),     0, 0);
}

static gboolean _add_circle_key_accel(GtkAccelGroup *accel_group, GObject *acceleratable, guint keyval,
                                      GdkModifierType modifier, gpointer data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)data;
  const dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *) module->gui_data;
  rt_add_shape_callback(GTK_WIDGET(g->bt_circle), NULL, module);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_circle), TRUE);
  return TRUE;
}

static gboolean _add_ellipse_key_accel(GtkAccelGroup *accel_group, GObject *acceleratable, guint keyval,
                                       GdkModifierType modifier, gpointer data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)data;
  const dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *) module->gui_data;
  rt_add_shape_callback(GTK_WIDGET(g->bt_ellipse), NULL, module);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_ellipse), TRUE);
  return TRUE;
}

static gboolean _add_brush_key_accel(GtkAccelGroup *accel_group, GObject *acceleratable, guint keyval,
                                       GdkModifierType modifier, gpointer data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)data;
  const dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *) module->gui_data;
  rt_add_shape_callback(GTK_WIDGET(g->bt_brush), NULL, module);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), TRUE);
  return TRUE;
}

static gboolean _add_path_key_accel(GtkAccelGroup *accel_group, GObject *acceleratable, guint keyval,
                                    GdkModifierType modifier, gpointer data)
{
  dt_iop_module_t *module = (dt_iop_module_t *)data;
  const dt_iop_retouch_gui_data_t *g = (dt_iop_retouch_gui_data_t *) module->gui_data;
  rt_add_shape_callback(GTK_WIDGET(g->bt_path), NULL, module);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), TRUE);
  return TRUE;
}

void connect_key_accels(dt_iop_module_t *module)
{
  GClosure *closure;

  closure = g_cclosure_new(G_CALLBACK(_add_circle_key_accel), (gpointer)module, NULL);
  dt_accel_connect_iop (module, "retouch circle tool", closure);

  closure = g_cclosure_new(G_CALLBACK(_add_ellipse_key_accel), (gpointer)module, NULL);
  dt_accel_connect_iop (module, "retouch elipse tool", closure);

  closure = g_cclosure_new(G_CALLBACK(_add_brush_key_accel), (gpointer)module, NULL);
  dt_accel_connect_iop (module, "retouch brush tool", closure);

  closure = g_cclosure_new(G_CALLBACK(_add_path_key_accel), (gpointer)module, NULL);
  dt_accel_connect_iop (module, "retouch path tool", closure);
}

//---------------------------------------------------------------------------------
// copy image routines
//---------------------------------------------------------------------------------

void rt_copy_in_to_out(const float *const in, const struct dt_iop_roi_t *const roi_in, float *const out, const struct dt_iop_roi_t *const roi_out, const int ch)
{
  const int rowsize = MIN(roi_out->width, roi_in->width) * ch * sizeof(float);
  const int xoffs = roi_out->x - roi_in->x;
  const int yoffs = roi_out->y - roi_in->y;
  const int y_to = MIN(roi_out->height, roi_in->height);

#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
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


//--------------------------------------------------------------------------------------------------
// process
//--------------------------------------------------------------------------------------------------

#if defined(__SSE__x)
void retouch_fill_sse(const int fts, const int fhs, const int fls, const int fws, const int dx, const int dy,
                    dt_iop_roi_t *const roi_in, dt_iop_roi_t *const roi_out,
                    float *const mask, const int mask_width, const int mask_height, float *const in, float *out, const int ch, const int mask_display, const float gopacity,
                    const float *fill_color)
{ 
  const int y_from = MAX(MAX((fts + 1), roi_out->y), (roi_in->y+dy));
  const int y_to = MIN(MIN((fts + fhs + 1), (roi_out->y + roi_out->height)), (roi_in->y + roi_in->height+dy));
  
  const int x_fom = MAX(MAX((fls + 1), roi_out->x), (roi_in->x+dx));
  const int x_to = MIN(MIN((fls + fws + 1), roi_out->x + roi_out->width), (roi_in->x + roi_in->width+dx));
  
  const float valf4_fill[4] = { fill_color[0], fill_color[1], fill_color[2], 0.f };
  const __m128 val_fill = _mm_load_ps(valf4_fill);
  
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(out) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int mindex = (int)((yy - fts) / roi_in->scale);
    if (mindex >= mask_height) continue;
    
    const int oindex = ch * (roi_out->width * (yy - roi_out->y) - roi_out->x + x_fom);
    const int iindex = ch * (roi_in->width * (yy - dy - roi_in->y) - dx - roi_in->x + x_fom);
    
    float *o = out + oindex;
    float *i = in + iindex;
    float *m = mask + mindex * mask_width;
    
    for(int xx = x_fom; xx < x_to; xx++, o+=ch, i+=ch)
    {
      const int mx = (int)((xx - fls) / roi_in->scale);
      if (mx >= mask_width) continue;
      
      const float f = m[mx] * gopacity;

      const float val1_f4[4] = { 1.0f - f, 1.0f - f, 1.0f - f, 1.0f };
      const float valf4[4] = { f, f, f, 0.f };

      const __m128 val1_f = _mm_load_ps(val1_f4);
      const __m128 valf = _mm_load_ps(valf4);
      
      _mm_store_ps(o, _mm_add_ps(_mm_mul_ps(_mm_load_ps(i), val1_f), _mm_mul_ps(val_fill, valf)));

      if (mask_display && f)
        o[3] = f;
    }
  }
}
#endif

void retouch_fill(int use_sse, const int fts, const int fhs, const int fls, const int fws, const int dx, const int dy,
                    dt_iop_roi_t *const roi_in, dt_iop_roi_t *const roi_out,
                    float *const mask, const int mask_width, const int mask_height, float *const in, float *out, const int ch, const int mask_display, const float gopacity, 
                    const float *fill_color)
{
#if defined(__SSE__x)
  if (ch == 4 && use_sse)
  {
    retouch_fill_sse(fts, fhs, fls, fws, dx, dy, roi_in, roi_out, mask, mask_width, mask_height, in, out, ch, mask_display, gopacity, fill_color);
    return;
  }
#endif
  const int ch1 = 3;
  
  const int y_from = MAX(MAX((fts + 1), roi_out->y), (roi_in->y+dy));
  const int y_to = MIN(MIN((fts + fhs + 1), (roi_out->y + roi_out->height)), (roi_in->y + roi_in->height+dy));
  
  const int x_fom = MAX(MAX((fls + 1), roi_out->x), (roi_in->x+dx));
  const int x_to = MIN(MIN((fls + fws + 1), roi_out->x + roi_out->width), (roi_in->x + roi_in->width+dx));

  for(int yy = y_from; yy < y_to; yy++)
  {
    const int mindex = (int)((yy - fts) / roi_in->scale);
    if (mindex >= mask_height) continue;
    
    const int oindex = ch * (roi_out->width * (yy - roi_out->y) - roi_out->x + x_fom);
    const int iindex = ch * (roi_in->width * (yy - dy - roi_in->y) - dx - roi_in->x + x_fom);
    
    float *o = out + oindex;
    float *i = in + iindex;
    float *m = mask + mindex * mask_width;
    
    for(int xx = x_fom; xx < x_to; xx++, o+=ch, i+=ch)
    {
      const int mx = (int)((xx - fls) / roi_in->scale);
      if (mx >= mask_width) continue;
      
      const float f = m[mx] * gopacity;
      
      for(int c = 0; c < ch1; c++)
        o[c] = i[c] * (1.0f - f) + fill_color[c] * f;
        
      if (mask_display && f)
        o[3] = f;
    }
  }
  
}

#if defined(__SSE__x)
void retouch_clone_sse(const int fts, const int fhs, const int fls, const int fws, const int dx, const int dy,
                    dt_iop_roi_t *const roi_in, dt_iop_roi_t *const roi_out,
                    float *const mask, const int mask_width, const int mask_height, float *const in, float *out, const int ch, const int mask_display, const float gopacity)
{ 
  const int y_from = MAX(MAX((fts + 1), roi_out->y), (roi_in->y+dy));
  const int y_to = MIN(MIN((fts + fhs + 1), (roi_out->y + roi_out->height)), (roi_in->y + roi_in->height+dy));
  
  const int x_fom = MAX(MAX((fls + 1), roi_out->x), (roi_in->x+dx));
  const int x_to = MIN(MIN((fls + fws + 1), roi_out->x + roi_out->width), (roi_in->x + roi_in->width+dx));
  
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(out) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int mindex = (int)((yy - fts) / roi_in->scale);
    if (mindex >= mask_height) continue;
    
    const int oindex = ch * (roi_out->width * (yy - roi_out->y) - roi_out->x + x_fom);
    const int iindex = ch * (roi_in->width * (yy - dy - roi_in->y) - dx - roi_in->x + x_fom);
    
    float *o = out + oindex;
    float *i = in + iindex;
    float *m = mask + mindex * mask_width;
    
    for(int xx = x_fom; xx < x_to; xx++, o+=ch, i+=ch)
    {
      const int mx = (int)((xx - fls) / roi_in->scale);
      if (mx >= mask_width) continue;
      
      const float f = m[mx] * gopacity;

      const float val1_f4[4] = { 1.0f - f, 1.0f - f, 1.0f - f, 1.0f };
      const float valf4[4] = { f, f, f, 0.f };

      const __m128 val1_f = _mm_load_ps(val1_f4);
      const __m128 valf = _mm_load_ps(valf4);
      
      _mm_store_ps(o, _mm_add_ps(_mm_mul_ps(_mm_load_ps(o), val1_f), _mm_mul_ps(_mm_load_ps(i), valf)));

      if (mask_display && f)
        o[3] = f;
    }
  }
  
}
#endif

void retouch_clone(int use_sse, const int fts, const int fhs, const int fls, const int fws, const int dx, const int dy,
                    dt_iop_roi_t *const roi_in, dt_iop_roi_t *const roi_out,
                    float *const mask, const int mask_width, const int mask_height, float *const in, float *out, const int ch, const int mask_display, const float gopacity)
{
#if defined(__SSE__x)
  if (ch == 4 && use_sse)
  {
    retouch_clone_sse(fts, fhs, fls, fws, dx, dy, roi_in, roi_out, mask, mask_width, mask_height, in, out, ch, mask_display, gopacity);
    return;
  }
#endif
  const int ch1 = 3;
  
  const int y_from = MAX(MAX((fts + 1), roi_out->y), (roi_in->y+dy));
  const int y_to = MIN(MIN((fts + fhs + 1), (roi_out->y + roi_out->height)), (roi_in->y + roi_in->height+dy));
  
  const int x_fom = MAX(MAX((fls + 1), roi_out->x), (roi_in->x+dx));
  const int x_to = MIN(MIN((fls + fws + 1), roi_out->x + roi_out->width), (roi_in->x + roi_in->width+dx));

  for(int yy = y_from; yy < y_to; yy++)
  {
    const int mindex = (int)((yy - fts) / roi_in->scale);
    if (mindex >= mask_height) continue;
    
    const int oindex = ch * (roi_out->width * (yy - roi_out->y) - roi_out->x + x_fom);
    const int iindex = ch * (roi_in->width * (yy - dy - roi_in->y) - dx - roi_in->x + x_fom);
    
    float *o = out + oindex;
    float *i = in + iindex;
    float *m = mask + mindex * mask_width;
    
    for(int xx = x_fom; xx < x_to; xx++, o+=ch, i+=ch)
    {
      const int mx = (int)((xx - fls) / roi_in->scale);
      if (mx >= mask_width) continue;
      
      const float f = m[mx] * gopacity;
      
      for(int c = 0; c < ch1; c++)
        o[c] = o[c] * (1.0f - f) + i[c] * f;
      
      if (mask_display && f)
        o[3] = f;
    }
  }
  
}

#if defined(__SSE__x)
void retouch_gaussian_blur_sse(dt_dev_pixelpipe_iop_t *piece, const int fts, const int fhs, const int fls, const int fws, const int dx, const int dy,
                          dt_iop_roi_t *const roi_in, dt_iop_roi_t *const roi_out, float *const mask, const int mask_width, const int mask_height, 
                          float *const in, float *out, const int ch, const int mask_display, const float gopacity, const float gblur_radius)
{
  float *dest = NULL;
  
  const int y_from = MAX(MAX((fts + 1), roi_out->y), (roi_in->y+dy));
  const int y_to = MIN(MIN((fts + fhs + 1), (roi_out->y + roi_out->height)), (roi_in->y + roi_in->height+dy));
  
  const int x_fom = MAX(MAX((fls + 1), roi_out->x), (roi_in->x+dx));
  const int x_to = MIN(MIN((fls + fws + 1), roi_out->x + roi_out->width), (roi_in->x + roi_in->width+dx));

  const int width_tmp = x_to - x_fom;
  const int height_tmp = y_to - y_from;
  
  if (width_tmp <= 0 || height_tmp <= 0) return;
  
  // alloc temp image to blur
  dest = dt_alloc_align(64, width_tmp * height_tmp * ch * sizeof(float));
  
  if (dest == NULL)
  {
    printf("error allocating memory for blurring\n");
    goto cleanup;
  }
  
  memset(dest, 0, width_tmp * height_tmp * ch * sizeof(float));
  
  // copy source image so we blur just the mask area (at least the smallest rect that covers it)
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(dest) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int dindex = ch * ((yy-y_from)*width_tmp);
    const int iindex = ch * (roi_in->width * (yy - dy - roi_in->y) - dx - roi_in->x + x_fom);
    float *d = dest + dindex;
    float *i = in + iindex;

    memcpy(d, i, width_tmp * ch * sizeof(float));
  }
  
  // blur it
  const float sigma = gblur_radius * roi_in->scale / piece->iscale;

  float Labmax[] = { INFINITY, INFINITY, INFINITY, INFINITY };
  float Labmin[] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };

  dt_gaussian_t *g = dt_gaussian_init(width_tmp, height_tmp, ch, Labmax, Labmin, sigma, DT_IOP_GAUSSIAN_ZERO);
  if(g)
  {
    dt_gaussian_blur_4c(g, dest, dest);
    dt_gaussian_free(g);
  }
  
  // copy blurred (temp) image to destination image
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(dest, out) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int mindex = (int)((yy - fts) / roi_in->scale);
    if (mindex >= mask_height) continue;
    
    const int dindex = ch * ((yy-y_from)*width_tmp);
    const int oindex = ch * (roi_out->width * (yy - roi_out->y) - roi_out->x + x_fom);
    
    float *d = dest + dindex;
    float *o = out + oindex;
    float *m = mask + mindex * mask_width;
   
    for(int xx = x_fom; xx < x_to; xx++, d+=ch, o+=ch)
    {
      const int mx = (int)((xx - fls) / roi_in->scale);
      if (mx >= mask_width) continue;
      
      const float f = (m[mx]) * gopacity;
      
      const __m128 val1_f = _mm_set1_ps(1.0f - f);
      const __m128 valf = _mm_set1_ps(f);
      
      _mm_store_ps(o, _mm_add_ps(_mm_mul_ps(_mm_load_ps(o), val1_f), _mm_mul_ps(_mm_load_ps(d), valf)));

      if (mask_display && f)
        o[3] = f;
    }
  }
  
cleanup:
  if (dest) dt_free_align(dest);
}
#endif

void retouch_gaussian_blur(int use_sse, dt_dev_pixelpipe_iop_t *piece, const int fts, const int fhs, const int fls, const int fws, const int dx, const int dy,
                          dt_iop_roi_t *const roi_in, dt_iop_roi_t *const roi_out, float *const mask, const int mask_width, const int mask_height, 
                          float *const in, float *out, const int ch, const int mask_display, const float gopacity, const float gblur_radius)
{
#if defined(__SSE__x)
  if (ch == 4 && use_sse)
  {
    retouch_gaussian_blur_sse(piece, fts, fhs, fls, fws, dx, dy, roi_in, roi_out, mask, mask_width, mask_height, in, out, ch, mask_display, gopacity, gblur_radius);
    return;
  }
#endif
  const int ch1 = 3;
  float *dest = NULL;
  
  const int y_from = MAX(MAX((fts + 1), roi_out->y), (roi_in->y+dy));
  const int y_to = MIN(MIN((fts + fhs + 1), (roi_out->y + roi_out->height)), (roi_in->y + roi_in->height+dy));
  
  const int x_fom = MAX(MAX((fls + 1), roi_out->x), (roi_in->x+dx));
  const int x_to = MIN(MIN((fls + fws + 1), roi_out->x + roi_out->width), (roi_in->x + roi_in->width+dx));

  const int width_tmp = x_to - x_fom;
  const int height_tmp = y_to - y_from;
  
  if (width_tmp <= 0 || height_tmp <= 0) return;
  
  // alloc temp image to blur
  dest = dt_alloc_align(64, width_tmp * height_tmp * ch * sizeof(float));
  
  if (dest == NULL)
  {
    printf("error allocating memory for blurring\n");
    goto cleanup;
  }
  
  memset(dest, 0, width_tmp * height_tmp * ch * sizeof(float));
  
  // copy source image so we blur just the mask area (at least the smallest rect that covers it)
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(dest) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int dindex = ch * ((yy-y_from)*width_tmp);
    const int iindex = ch * (roi_in->width * (yy - dy - roi_in->y) - dx - roi_in->x + x_fom);
    float *d = dest + dindex;
    float *i = in + iindex;

    memcpy(d, i, width_tmp * ch * sizeof(float));
  }
  
  const float sigma = gblur_radius * roi_in->scale / piece->iscale;

  float Labmax[] = { INFINITY, INFINITY, INFINITY, INFINITY };
  float Labmin[] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };

  dt_gaussian_t *g = dt_gaussian_init(width_tmp, height_tmp, ch, Labmax, Labmin, sigma, DT_IOP_GAUSSIAN_ZERO);
  if(g)
  {
    dt_gaussian_blur(g, dest, dest);
    dt_gaussian_free(g);
  }

  // copy blurred (temp) image to destination image
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(dest, out) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int mindex = (int)((yy - fts) / roi_in->scale);
    if (mindex >= mask_height) continue;
    
    const int dindex = ch * ((yy-y_from)*width_tmp);
    const int oindex = ch * (roi_out->width * (yy - roi_out->y) - roi_out->x + x_fom);
    
    float *d = dest + dindex;
    float *o = out + oindex;
    float *m = mask + mindex * mask_width;
   
    for(int xx = x_fom; xx < x_to; xx++, d+=ch, o+=ch)
    {
      const int mx = (int)((xx - fls) / roi_in->scale);
      if (mx >= mask_width) continue;
      
      const float f = (m[mx]) * gopacity;
      
      for(int c = 0; c < ch1; c++)
      {
        o[c] = o[c] * (1.0f - f) + d[c] * f;
      }
      if (mask_display && f)
        o[3] = f;
    }
  }
  
cleanup:
  if (dest) dt_free_align(dest);
}

#if defined(__SSE__x)
void retouch_heal_sse(const int fts, const int fhs, const int fls, const int fws, const int dx, const int dy,
                    dt_iop_roi_t *const roi_in, dt_iop_roi_t *const roi_out,
                    float *const mask, const int mask_width, const int mask_height, float *const in, float *out, const int ch, const int mask_display, const float gopacity)
{
  float *src = NULL;
  float *dest = NULL;
  float *mask_heal = NULL;
  
  const int y_from = MAX(MAX((fts + 1), roi_out->y), (roi_in->y+dy));
  const int y_to = MIN(MIN((fts + fhs + 1), (roi_out->y + roi_out->height)), (roi_in->y + roi_in->height+dy));
  
  const int x_fom = MAX(MAX((fls + 1), roi_out->x), (roi_in->x+dx));
  const int x_to = MIN(MIN((fls + fws + 1), roi_out->x + roi_out->width), (roi_in->x + roi_in->width+dx));

  const int width_tmp = x_to - x_fom;
  const int height_tmp = y_to - y_from;

  if (width_tmp <= 0 || height_tmp <= 0) return;
  
  // alloc temp images for source, destination and mask, so all share the same coordinates
  src = dt_alloc_align(64, width_tmp * height_tmp * ch * sizeof(float));
  dest = dt_alloc_align(64, width_tmp * height_tmp * ch * sizeof(float));
  mask_heal = dt_alloc_align(64, width_tmp * height_tmp * sizeof(float));
  
  if ((src == NULL) || (dest == NULL) || (mask_heal == NULL))
  {
    printf("error allocating memory for healing\n");
    goto cleanup;
  }

  memset(src, 0, width_tmp * height_tmp * ch * sizeof(float));
  memset(dest, 0, width_tmp * height_tmp * ch * sizeof(float));
  memset(mask_heal, 0, width_tmp * height_tmp * sizeof(float));
  
  // copy source and destination to temp images
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(dest) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int dindex = ch * ((yy-y_from)*width_tmp);
    const int oindex = ch * (roi_out->width * (yy - roi_out->y) - roi_out->x + x_fom);
    const int iindex = ch * (roi_in->width * (yy - dy - roi_in->y) - dx - roi_in->x + x_fom);
    
    float *o = out + oindex;
    float *i = in + iindex;
    float *d = dest + dindex;
    float *s = src + dindex;
    
    memcpy(d, o, width_tmp * ch * sizeof(float));
    memcpy(s, i, width_tmp * ch * sizeof(float));
  }

  // copy mask to temp image
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(dest) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int mindex = (int)((yy - fts) / roi_in->scale);
    if (mindex >= mask_height) continue;
    
    const int mhindex = ((yy-y_from)*width_tmp);

    float *m = mask + mindex * mask_width;
    float *mh = mask_heal + mhindex;
    
    for(int xx = x_fom; xx < x_to; xx++, mh++)
    {
      const int mx = (int)((xx - fls) / roi_in->scale);
      if (mx >= mask_width) continue;
      
      *mh = m[mx];
      
    }
  }

  // heal it
  dt_dev_heal(src, dest, mask_heal, width_tmp, height_tmp, ch);

  // copy healed (temp) image to destination image
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(dest) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int mindex = (int)((yy - fts) / roi_in->scale);
    if (mindex >= mask_height) continue;
    
    const int dindex = ch * ((yy-y_from)*width_tmp);
    const int mhindex = ((yy-y_from)*width_tmp);
    const int oindex = ch * (roi_out->width * (yy - roi_out->y) - roi_out->x + x_fom);
    
    float *o = out + oindex;
    float *d = dest + dindex;
    float *mh = mask_heal + mhindex;
    
    for(int xx = x_fom; xx < x_to; xx++, d+=ch, o+=ch, mh++)
    {
      const float f = (*mh) * gopacity;
      
      const __m128 val1_f = _mm_set1_ps(1.0f - f);
      const __m128 valf = _mm_set1_ps(f);
      
      _mm_store_ps(o, _mm_add_ps(_mm_mul_ps(_mm_load_ps(o), val1_f), _mm_mul_ps(_mm_load_ps(d), valf)));

      if (mask_display && f)
        o[3] = f;
    }
  }
  
  cleanup:
    if (src) dt_free_align(src);
    if (dest) dt_free_align(dest);
    if (mask_heal) dt_free_align(mask_heal);
}
#endif

void retouch_heal(int use_sse, const int fts, const int fhs, const int fls, const int fws, const int dx, const int dy,
                    dt_iop_roi_t *const roi_in, dt_iop_roi_t *const roi_out,
                    float *const mask, const int mask_width, const int mask_height, float *const in, float *out, const int ch, const int mask_display, const float gopacity)
{
#if defined(__SSE__x)
  if (ch == 4 && use_sse)
  {
    retouch_heal_sse(fts, fhs, fls, fws, dx, dy, roi_in, roi_out, mask, mask_width, mask_height, in, out, ch, mask_display, gopacity);
  return;
  }
#endif
  const int ch1 = 3;
  
  float *src = NULL;
  float *dest = NULL;
  float *mask_heal = NULL;
  
  const int y_from = MAX(MAX((fts + 1), roi_out->y), (roi_in->y+dy));
  const int y_to = MIN(MIN((fts + fhs + 1), (roi_out->y + roi_out->height)), (roi_in->y + roi_in->height+dy));
  
  const int x_fom = MAX(MAX((fls + 1), roi_out->x), (roi_in->x+dx));
  const int x_to = MIN(MIN((fls + fws + 1), roi_out->x + roi_out->width), (roi_in->x + roi_in->width+dx));

  const int width_tmp = x_to - x_fom;
  const int height_tmp = y_to - y_from;

  if (width_tmp <= 0 || height_tmp <= 0) return;
  
  // alloc temp images for source, destination and mask, so all share the same coordinates
  src = dt_alloc_align(64, width_tmp * height_tmp * ch * sizeof(float));
  dest = dt_alloc_align(64, width_tmp * height_tmp * ch * sizeof(float));
  mask_heal = dt_alloc_align(64, width_tmp * height_tmp * sizeof(float));
  
  if ((src == NULL) || (dest == NULL) || (mask_heal == NULL))
  {
    printf("error allocating memory for healing\n");
    goto cleanup;
  }

  memset(src, 0, width_tmp * height_tmp * ch * sizeof(float));
  memset(dest, 0, width_tmp * height_tmp * ch * sizeof(float));
  memset(mask_heal, 0, width_tmp * height_tmp * sizeof(float));
  
  // copy source and destination to temp images
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(out, dest, src) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int dindex = ch * ((yy-y_from)*width_tmp);
    const int oindex = ch * (roi_out->width * (yy - roi_out->y) - roi_out->x + x_fom);
    const int iindex = ch * (roi_in->width * (yy - dy - roi_in->y) - dx - roi_in->x + x_fom);
    
    float *o = out + oindex;
    float *i = in + iindex;
    float *d = dest + dindex;
    float *s = src + dindex;
    
    memcpy(d, o, width_tmp * ch * sizeof(float));
    memcpy(s, i, width_tmp * ch * sizeof(float));
  }

  // copy mask to temp image
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(mask_heal) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int mindex = (int)((yy - fts) / roi_in->scale);
    if (mindex >= mask_height) continue;
    
    const int mhindex = ((yy-y_from)*width_tmp);

    float *m = mask + mindex * mask_width;
    float *mh = mask_heal + mhindex;
    
    for(int xx = x_fom; xx < x_to; xx++, mh++)
    {
      const int mx = (int)((xx - fls) / roi_in->scale);
      if (mx >= mask_width) continue;
      
      *mh = m[mx];
      
    }
  }

  // heal it
  dt_dev_heal(src, dest, mask_heal, width_tmp, height_tmp, ch);

  // copy healed (temp) image to destination image
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(out, dest, mask_heal) schedule(static)
#endif
#endif
  for(int yy = y_from; yy < y_to; yy++)
  {
    const int mindex = (int)((yy - fts) / roi_in->scale);
    if (mindex >= mask_height) continue;
    
    const int dindex = ch * ((yy-y_from)*width_tmp);
    const int mhindex = ((yy-y_from)*width_tmp);
    const int oindex = ch * (roi_out->width * (yy - roi_out->y) - roi_out->x + x_fom);
    
    float *o = out + oindex;
    float *d = dest + dindex;
    float *mh = mask_heal + mhindex;
    
    for(int xx = x_fom; xx < x_to; xx++, d+=ch, o+=ch, mh++)
    {
      const float f = (*mh) * gopacity;
      
      for(int c = 0; c < ch1; c++)
        o[c] = o[c] * (1.0f - f) + d[c] * f;

      if (mask_display && f)
        o[3] = f;
    }
  }
  
cleanup:
  if (src) dt_free_align(src);
  if (dest) dt_free_align(dest);
  if (mask_heal) dt_free_align(mask_heal);
}

void rt_process_forms(float *layer, dwt_params_t *wt_p, const int scale1)
{
  int scale = scale1;
  _rt_user_data_t *usr_d = (_rt_user_data_t*)wt_p->user_data;
  dt_iop_module_t *self = usr_d->self;
  dt_dev_pixelpipe_iop_t *piece = usr_d->piece;
  
  if (wt_p->return_layer > 0 && scale != wt_p->return_layer) return;
  if (scale > wt_p->scales+1) return;

  dt_develop_blend_params_t *bp = (dt_develop_blend_params_t *)piece->blendop_data;
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)piece->data;
  float *in = layer;
  float *out = layer;
  dt_iop_roi_t *roi_in = &usr_d->roi;
  dt_iop_roi_t *roi_out = &usr_d->roi;
  const int mask_display = usr_d->mask_display && (scale == usr_d->display_scale);

  if (wt_p->scales != p->num_scales && wt_p->return_layer == scale && scale > 0) scale = p->curr_scale;
  
  float fill_color[3];
  
  if (p->fill_mode[scale] == dt_iop_rt_fill_erase)
  {
    fill_color[0] = fill_color[1] = fill_color[2] = p->blend_delta[scale];
  }
  else
  {
    fill_color[0] = p->fill_color[scale][0]+p->blend_delta[scale];
    fill_color[1] = p->fill_color[scale][1]+p->blend_delta[scale];
    fill_color[2] = p->fill_color[scale][2]+p->blend_delta[scale];
  }
  
  if (scale > 0 && scale < p->num_scales+1)
  {
    fill_color[0] += p->blend_factor;
    fill_color[1] += p->blend_factor;
    fill_color[2] += p->blend_factor;
  }
  
  // iterate through all forms
  dt_masks_form_t *grp = dt_masks_get_from_id(self->dev, bp->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    GList *forms = g_list_first(grp->points);
    while(forms)
    {
      dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;
      const dt_iop_retouch_algo_type_t algo = rt_get_algo_from_id(p->rt_forms[scale], grpt->formid);
      if (algo == 0)
      {
        // form must belong to another scale
        forms = g_list_next(forms);
        continue;
      }
      
      // get the spot
      dt_masks_form_t *form = dt_masks_get_from_id(self->dev, grpt->formid);
      if(!form)
      {
        printf("rt_process_forms missing form\n");
        forms = g_list_next(forms);
        continue;
      }

      // if the form is outside the roi, we just skip it
      if(!rt_masks_form_is_in_roi(self, piece, form, roi_in, roi_out))
      {
        forms = g_list_next(forms);
        continue;
      }

      // get the mask
      float *mask = NULL;
      int posx, posy, mask_width, mask_height;
      dt_masks_get_mask(self, piece, form, &mask, &mask_width, &mask_height, &posx, &posy);
      int fts = posy * roi_in->scale, fhs = mask_height * roi_in->scale, fls = posx * roi_in->scale,
          fws = mask_width * roi_in->scale;
      int dx = 0, dy = 0;

      // search the delta with the source
      if (algo != dt_iop_retouch_gaussian_blur && algo != dt_iop_retouch_fill)
      {
        if(!rt_masks_get_delta(self, piece, roi_in, form, &dx, &dy))
        {
          forms = g_list_next(forms);
          free(mask);
  
          continue;
        }
      }

      if ((dx != 0 || dy != 0 || algo == dt_iop_retouch_gaussian_blur || algo == dt_iop_retouch_fill) && 
          ((fws > 2) && (fhs > 2)))
      {
        double start = dt_get_wtime();
        
        if (algo == dt_iop_retouch_clone)
        {
          retouch_clone(wt_p->use_sse, fts, fhs, fls, fws, dx, dy, roi_in, roi_out, mask, mask_width, mask_height, in, out, wt_p->ch, mask_display, grpt->opacity);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch_clone took %0.04f sec\n", dt_get_wtime() - start);
        }
        else if (algo == dt_iop_retouch_heal)
        {
          retouch_heal(wt_p->use_sse, fts, fhs, fls, fws, dx, dy, roi_in, roi_out, mask, mask_width, mask_height, in, out, wt_p->ch, mask_display, grpt->opacity);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch_heal took %0.04f sec\n", dt_get_wtime() - start);
        }
        else if (algo == dt_iop_retouch_gaussian_blur)
        {
          retouch_gaussian_blur(wt_p->use_sse, piece, fts, fhs, fls, fws, dx, dy, roi_in, roi_out, mask, mask_width, mask_height, in, out, wt_p->ch, mask_display, grpt->opacity,
                              p->gblur_radius[scale]);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch_gaussian_blur took %0.04f sec\n", dt_get_wtime() - start);
        }
        else if (algo == dt_iop_retouch_fill)
        {
          retouch_fill(wt_p->use_sse, fts, fhs, fls, fws, dx, dy, roi_in, roi_out, mask, mask_width, mask_height, in, out, wt_p->ch, mask_display, grpt->opacity, 
                              fill_color);
          if(darktable.unmuted & DT_DEBUG_PERF) printf("rt_process_forms retouch_fill took %0.04f sec\n", dt_get_wtime() - start);
        }
        else
          printf("rt_process_forms unknown algorithm %i\n", algo);
        
      }
      
      free(mask);
      
      forms = g_list_next(forms);
    }
  }

}

void rt_process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
                void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out, 
                int use_sse)
{
  dt_iop_retouch_params_t *p = (dt_iop_retouch_params_t *)piece->data;

  float *in_retouch = NULL;
  
  dt_iop_roi_t roi_retouch = *roi_in;
  dt_iop_roi_t *roi_rt = &roi_retouch;
  
  const int ch = piece->colors;
  _rt_user_data_t usr_data = {0};
  dt_iop_retouch_preview_types_t preview_type = p->preview_type;
  
  int gui_active = 0;
  if (self->dev) gui_active = (self == self->dev->gui_module);
  if (!gui_active && preview_type == dt_iop_rt_preview_current_scale) preview_type = dt_iop_rt_preview_final_image;
  
  dwt_params_t dwt_p = {0};
  
  in_retouch = dt_alloc_align(64, roi_rt->width * roi_rt->height * ch * sizeof(float));
  memcpy(in_retouch, ivoid, roi_rt->width * roi_rt->height * ch * sizeof(float));

  usr_data.self = self;
  usr_data.piece = piece;
  usr_data.roi = *roi_rt;
  usr_data.mask_display = 0;
  usr_data.display_scale = p->curr_scale;

  dwt_p.image = in_retouch;
  dwt_p.ch = ch;
  dwt_p.width = roi_in->width;
  dwt_p.height = roi_in->height;
  dwt_p.scales = p->num_scales;
  dwt_p.return_layer = (preview_type == dt_iop_rt_preview_final_image) ? 0: p->curr_scale;
  dwt_p.blend_factor = p->blend_factor;
  dwt_p.user_data = &usr_data;
  dwt_p.preview_scale = roi_in->scale / piece->iscale;
  dwt_p.use_sse = use_sse;

  // check if _this_ module should expose mask. 
  if(self->request_mask_display && self->dev->gui_attached && (self == self->dev->gui_module)
     && (piece->pipe == self->dev->pipe) )
  {
    for(size_t j = 0; j < roi_rt->width*roi_rt->height*ch; j += ch) in_retouch[j + 3] = 0.f;
    
    piece->pipe->mask_display = 1;
    usr_data.mask_display = 1;
  }
  
  const int max_scales = dwt_get_max_scale(piece->buf_in.width, piece->buf_in.height, dwt_p.preview_scale);
  if (dwt_p.scales > max_scales && gui_active)
  {
    dt_control_log(_("max scale is %i for this image size"), max_scales);
  }
  else
  {
    const int max_scales2 = dwt_get_max_scale(roi_rt->width, roi_rt->height, dwt_p.preview_scale);
    if (dwt_p.scales > max_scales2 && gui_active)
    {
      printf("max scale is %i for this image preview size\n", max_scales2);
    }
  }
  
  if(self->suppress_mask && self->dev->gui_attached && (self == self->dev->gui_module)
     && (piece->pipe == self->dev->pipe))
  {
    dwt_decompose(&dwt_p, NULL);
  }
  else
  {
    dwt_decompose(&dwt_p, rt_process_forms);
  }

  if(piece->pipe->mask_display && !usr_data.mask_display) 
  {
    const float *const i = ivoid;
    for(size_t j = 0; j < roi_rt->width*roi_rt->height*ch; j += ch) in_retouch[j + 3] = i[j + 3];
  }
  rt_copy_in_to_out(in_retouch, roi_rt, ovoid, roi_out, ch);
  
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

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-space on;
