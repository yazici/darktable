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

extern "C" {
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "bauhaus/bauhaus.h"
#include "develop/masks.h"
#include "develop/blend.h"
#include "gui/accelerators.h"
#include <stdlib.h>
}

//#include "external/grabcut.hpp"

extern "C" {

DT_MODULE_INTROSPECTION(1, dt_iop_adjust_mask_params_t)

#define MASKADJ_NO_FORMS 200

typedef enum dt_iop_adjust_mask_masks_type_t
{
  dt_iop_adjust_mask_foreground = 1, 
  dt_iop_adjust_mask_include = 2,
  dt_iop_adjust_mask_exclude = 3
} dt_iop_adjust_mask_masks_type_t;

typedef struct dt_iop_adjust_mask_form_data_t
{
  int id;
  dt_iop_adjust_mask_masks_type_t mask_type;
} dt_iop_adjust_mask_form_data_t;

typedef struct dt_iop_adjust_mask_params_t
{
  dt_iop_adjust_mask_form_data_t maskadj_forms[MASKADJ_NO_FORMS];
  
  int mask_type; // foreground, include, exclude
  
} dt_iop_adjust_mask_params_t;

typedef struct dt_iop_adjust_mask_gui_data_t
{
  GtkLabel *label_form; // display number of forms
  GtkWidget *bt_edit_masks, *bt_path, *bt_brush; // shapes
  GtkWidget *bt_foreground_mask, *bt_include_mask, *bt_exclude_mask; // masks types
  GtkWidget *bt_execute;

} dt_iop_adjust_mask_gui_data_t;

typedef struct dt_iop_adjust_mask_params_t dt_iop_adjust_mask_data_t;

const char *name()
{
  return _("adjust_mask_eh");
}

int groups()
{
  return IOP_GROUP_CORRECT;
}
/*
int operation_tags()
{
  return IOP_TAG_DISTORT | IOP_TAG_DECORATION;
}

int operation_tags()
{
  return IOP_TAG_DISTORT;
}

int operation_tags_filter()
{
  // switch off clipping and decoration, we want to see the full image.
  return IOP_TAG_DECORATION | IOP_TAG_CLIPPING;
}
*/
/*
int flags()
{
  return IOP_FLAGS_ALLOW_TILING | IOP_FLAGS_TILING_FULL_ROI;
}
*/
int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_NO_MASKS;
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

static void _maskadj_cairo_paint_preserve_mask(cairo_t *cr,
                                          const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  cairo_set_source_rgba(cr, 0.0, 1.0, 0.0, 1.0);
  cairo_rectangle(cr, 0., 0., 1., 1.);
  cairo_fill(cr);

  POSTAMBLE;
}

static void _maskadj_cairo_paint_discard_mask(cairo_t *cr,
                                          const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;
  
  cairo_set_source_rgba(cr, 1.0, 0.0, 0.0, 1.0);
  cairo_rectangle(cr, 0., 0., 1., 1.);
  cairo_fill(cr);

  POSTAMBLE;
}

static void _maskadj_cairo_paint_rigidity_mask(cairo_t *cr, gint x, gint y, gint w, gint h, gint flags)
{
  PREAMBLE;
  
  cairo_set_source_rgba(cr, 0.0, 0.0, 1.0, 1.0);
  cairo_rectangle(cr, 0., 0., 1., 1.);
  cairo_fill(cr);

  POSTAMBLE;
}
/*
static void maskadj_show_hide_controls(dt_iop_adjust_mask_gui_data_t *d, dt_iop_adjust_mask_params_t *p)
{
  switch (p->scaling)
  {
    case dt_iop_adjust_mask_scaling_relative:
      gtk_widget_show(GTK_WIDGET(d->vbox_r_size));
      gtk_widget_hide(GTK_WIDGET(d->vbox_a_size));
      break;
    case dt_iop_adjust_mask_scaling_absolute:
    default:
      gtk_widget_hide(GTK_WIDGET(d->vbox_r_size));
      gtk_widget_show(GTK_WIDGET(d->vbox_a_size));
      break;
  }
}
*/

static int maskadj_allow_create_form(dt_iop_module_t *self)
{
  int allow = 1;
  
  dt_iop_adjust_mask_params_t *p = (dt_iop_adjust_mask_params_t *)self->params;
  if (p)
  {
    allow = (p->maskadj_forms[MASKADJ_NO_FORMS-1].id == 0);
  }
  return allow;
}

static void maskadj_reset_form_creation(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_adjust_mask_gui_data_t *g = (dt_iop_adjust_mask_gui_data_t *)self->gui_data;
  if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_path)) ||
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_brush)))
  {
    // we unset the creation mode
    dt_masks_form_t *form = darktable.develop->form_visible;
    if(form) dt_masks_free_form(form);
    dt_masks_change_form_gui(NULL);
  }
  if (widget != g->bt_path) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), FALSE);
  if (widget != g->bt_brush) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), FALSE);
}

static void maskadj_masks_selection_change(dt_iop_module_t *self, dt_iop_adjust_mask_params_t *p)
{
  if (!self->enabled) return;
  
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  if (bd == NULL) return;
//  int scale = p->curr_scale;
  int count = 0;
  
  for (int i = 0; i < MASKADJ_NO_FORMS && count == 0; i++)
  {
    if (p->maskadj_forms[i].id != 0) count++;
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
  for (int i = 0; i < MASKADJ_NO_FORMS; i++)
  {
    int grid = self->blend_params->mask_id;
    int id = p->maskadj_forms[i].id;
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
  dt_masks_free_form(grp);
  dt_masks_change_form_gui(grp2);
  darktable.develop->form_gui->edit_mode = (dt_masks_edit_mode_t)bd->masks_shown;
  dt_control_queue_redraw_center();
}
/*
static gboolean maskadj_masks_being_displayed(dt_iop_module_t *self)
{
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  if (bd == NULL) return FALSE;
  return (bd->masks_shown != DT_MASKS_EDIT_OFF);
}
*/
static gboolean maskadj_edit_masks_callback(GtkWidget *widget, GdkEventButton *event, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return FALSE;
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  dt_iop_adjust_mask_gui_data_t *g = (dt_iop_adjust_mask_gui_data_t *)self->gui_data;

  if (bd == NULL) return FALSE;
  
  if(event->button == 1)
  {
    darktable.gui->reset = 1;

    dt_iop_request_focus(self);
    self->request_color_pick = DT_REQUEST_COLORPICK_OFF;

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

    dt_iop_adjust_mask_params_t *p = (dt_iop_adjust_mask_params_t *)self->params;
    maskadj_masks_selection_change(self, p);
    dt_dev_reprocess_all(self->dev);
    
    return TRUE;
  }

  return FALSE;
}

static gboolean maskadj_execute_callback(GtkWidget *widget, GdkEventButton *event, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return FALSE;
/*  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  dt_iop_adjust_mask_gui_data_t *g = (dt_iop_adjust_mask_gui_data_t *)self->gui_data;

  if (bd == NULL) return FALSE;
*/  
  
  
  return FALSE;
}

static gboolean maskadj_add_shape_callback(GtkWidget *widget, GdkEventButton *e, dt_iop_module_t *self)
{
  const int allow = maskadj_allow_create_form(self);
  if (allow)
  {
    maskadj_reset_form_creation(widget, self);

    if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget))) return FALSE;

    dt_iop_adjust_mask_gui_data_t *g = (dt_iop_adjust_mask_gui_data_t *)self->gui_data;

    // we want to be sure that the iop has focus
    dt_iop_request_focus(self);

    dt_masks_type_t type = DT_MASKS_CIRCLE;
    if (widget == g->bt_path)
      type = DT_MASKS_PATH;
    else if (widget == g->bt_brush)
      type = DT_MASKS_BRUSH;

    // we create the new form
    dt_masks_form_t *spot = NULL;

    spot = dt_masks_create(type);

    dt_masks_change_form_gui(spot);
    darktable.develop->form_gui->creation = TRUE;
    darktable.develop->form_gui->creation_module = self;
    dt_control_queue_redraw_center();

  }
  else
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget), FALSE);
  }
  
  return !allow;
}

static void maskadj_select_algorithm_callback(GtkToggleButton *togglebutton, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_adjust_mask_params_t *p = (dt_iop_adjust_mask_params_t *)self->params;
  dt_iop_adjust_mask_gui_data_t *g = (dt_iop_adjust_mask_gui_data_t *)self->gui_data;

  if (togglebutton == (GtkToggleButton *)g->bt_exclude_mask)
    p->mask_type = dt_iop_adjust_mask_exclude;
  else if (togglebutton == (GtkToggleButton *)g->bt_foreground_mask)
    p->mask_type = dt_iop_adjust_mask_foreground;
  else if (togglebutton == (GtkToggleButton *)g->bt_include_mask)
    p->mask_type = dt_iop_adjust_mask_include;

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_foreground_mask), (p->mask_type == dt_iop_adjust_mask_foreground));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_include_mask), (p->mask_type == dt_iop_adjust_mask_include));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_exclude_mask), (p->mask_type == dt_iop_adjust_mask_exclude));

//  maskadj_show_hide_controls(g, p);
  
  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}



dt_iop_adjust_mask_masks_type_t maskadj_get_mask_type_from_id(dt_iop_adjust_mask_form_data_t *maskadj_forms, const int formid)
{
  dt_iop_adjust_mask_masks_type_t mask_type = (dt_iop_adjust_mask_masks_type_t)0;
  int i = 0;
  
  while (mask_type == 0 && i < MASKADJ_NO_FORMS)
  {
    if (maskadj_forms[i].id == formid) mask_type = maskadj_forms[i].mask_type;
    i++;
  }
  
  return mask_type;
}

static void maskadj_resynch_params_f(dt_iop_adjust_mask_params_t *p, dt_develop_blend_params_t *bp, dt_iop_adjust_mask_form_data_t *maskadj_forms, const int mask_type, const int add_new)
{
  dt_iop_adjust_mask_form_data_t forms_d[MASKADJ_NO_FORMS] = { {0, (dt_iop_adjust_mask_masks_type_t)0} };
  
  // we go through all forms in blend params
  dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, bp->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    GList *forms = g_list_first(grp->points);
    int i = 0;
    while((i < MASKADJ_NO_FORMS) && forms)
    {
      dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;
      
      int j = 0;
      while( j < MASKADJ_NO_FORMS && maskadj_forms[j].id != grpt->formid )
      {
        j++;
      }
      
      if (j < MASKADJ_NO_FORMS)
      {
        forms_d[i].mask_type = maskadj_forms[j].mask_type;
        forms_d[i].id = grpt->formid;
        i++;
      }
      else /*if (add_new)*/
      {
        int mask_type_new = maskadj_get_mask_type_from_id(p->maskadj_forms, grpt->formid);
        if (mask_type_new <= 0)
        {
          dt_masks_form_t *parent_form = dt_masks_get_from_id(darktable.develop, grpt->formid);
          if (parent_form)
          {
            forms_d[i].mask_type = (dt_iop_adjust_mask_masks_type_t)mask_type;
            forms_d[i].id = grpt->formid;
            
            i++;
          }
        }
      }
        
      forms = g_list_next(forms);
    }
  }

  // we reaffect params
  for(int i = 0; i < MASKADJ_NO_FORMS; i++)
  {
    maskadj_forms[i].mask_type = forms_d[i].mask_type;
    maskadj_forms[i].id = forms_d[i].id;
  }
}

static void maskadj_resynch_params(struct dt_iop_module_t *self)
{
  dt_iop_adjust_mask_params_t *p = (dt_iop_adjust_mask_params_t *)self->params;
  dt_develop_blend_params_t *bp = self->blend_params;

  maskadj_resynch_params_f(p, bp, p->maskadj_forms, p->mask_type, 1);

}

static gboolean maskadj_masks_form_is_in_roi(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
                                     dt_masks_form_t *form, const dt_iop_roi_t *roi_out)
{
  // we get the area for the form
  int fl, ft, fw, fh;

  if(!dt_masks_get_area(self, piece, form, &fw, &fh, &fl, &ft)) return FALSE;

  // is the form outside of the roi?
  fw *= roi_out->scale, fh *= roi_out->scale, fl *= roi_out->scale, ft *= roi_out->scale;
  if(ft >= roi_out->y + roi_out->height || ft + fh <= roi_out->y || fl >= roi_out->x + roi_out->width
     || fl + fw <= roi_out->x)
    return FALSE;

  return TRUE;
}


/*
void init_key_accels(dt_iop_module_so_t *self)
{
  dt_accel_register_slider_iop(self, FALSE, NC_("accel", "border size"));
  dt_accel_register_iop(self, FALSE, NC_("accel", "pick border color from image"), 0, 0);
  dt_accel_register_slider_iop(self, FALSE, NC_("accel", "frame line size"));
  dt_accel_register_iop(self, FALSE, NC_("accel", "pick frame line color from image"), 0, 0);
}

void connect_key_accels(dt_iop_module_t *self)
{
  dt_iop_adjust_mask_gui_data_t *g = (dt_iop_adjust_mask_gui_data_t *)self->gui_data;
  dt_accel_connect_button_iop(self, "pick border color from image", GTK_WIDGET(g->colorpick));
  dt_accel_connect_slider_iop(self, "border size", GTK_WIDGET(g->size));
  dt_accel_connect_button_iop(self, "pick frame line color from image", GTK_WIDGET(g->frame_colorpick));
  dt_accel_connect_slider_iop(self, "frame line size", GTK_WIDGET(g->frame_size));
}
*/

/*
int distort_backtransform(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, float *points, size_t points_count)
{
  dt_iop_adjust_mask_data_t *d = (dt_iop_adjust_mask_data_t *)piece->data;

  for(size_t i = 0; i < points_count * 2; i += 2)
  {
    points[i] /= d->r_width;
    points[i + 1] /= d->r_height;
  }

  return 1;
}

int distort_transform(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, float *points, size_t points_count)
{
  dt_iop_adjust_mask_data_t *d = (dt_iop_adjust_mask_data_t *)piece->data;

  for(size_t i = 0; i < points_count * 2; i += 2)
  {
    points[i] *= d->r_width;
    points[i + 1] *= d->r_height;
  }

  return 1;
}
*/

// 1st pass: how large would the output be, given this input roi?
// this is always called with the full buffer before processing.
void modify_roi_out(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece, dt_iop_roi_t *roi_out,
                    const dt_iop_roi_t *roi_in)
{
  *roi_out = *roi_in;
}

// 2nd pass: which roi would this operation need as input to fill the given output region?
void modify_roi_in(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece,
                   const dt_iop_roi_t *roi_out, dt_iop_roi_t *roi_in)
{
/*  *roi_in = *roi_out;
 
  int roir = roi_in->width + roi_in->x;
  int roib = roi_in->height + roi_in->y;
  int roix = roi_in->x;
  int roiy = roi_in->y;

  dt_develop_blend_params_t *bp = self->blend_params;

  // We iterate through all retouch or polygons
  dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, bp->mask_id);
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
        if(!maskadj_masks_form_is_in_roi(self, piece, form, roi_in, roi_out))
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
      }
      forms = g_list_next(forms);
    }
  }

  // now we set the values
  const float scwidth = piece->buf_in.width * roi_in->scale, scheight = piece->buf_in.height * roi_in->scale;
  roi_in->x = CLAMP(roix, 0, scwidth - 1);
  roi_in->y = CLAMP(roiy, 0, scheight - 1);
  roi_in->width = CLAMP(roir - roi_in->x, 1, scwidth + .5f - roi_in->x);
  roi_in->height = CLAMP(roib - roi_in->y, 1, scheight + .5f - roi_in->y);*/
  
//  const float scale = roi_in->scale / piece->iscale;
  const float scale = roi_out->scale;
  
  roi_in->scale = roi_out->scale;

  roi_in->x = 0;
  roi_in->y = 0;
  roi_in->width = round(piece->buf_in.width * scale);
  roi_in->height = round(piece->buf_in.height * scale);
  roi_in->width = MAX(1, roi_in->width);
  roi_in->height = MAX(1, roi_in->height);

}


void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_adjust_mask_params_t *p = (dt_iop_adjust_mask_params_t *)p1;
  dt_iop_adjust_mask_data_t *d = (dt_iop_adjust_mask_data_t *)piece->data;
  memcpy(d, p, sizeof(dt_iop_adjust_mask_params_t));
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_adjust_mask_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(struct dt_iop_module_t *self)
{
  dt_iop_adjust_mask_gui_data_t *g = (dt_iop_adjust_mask_gui_data_t *)self->gui_data;
  dt_iop_adjust_mask_params_t *p = (dt_iop_adjust_mask_params_t *)self->params;

  maskadj_resynch_params(self);

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
  
//  maskadj_masks_selection_change(self, p);
  
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_foreground_mask), p->mask_type == dt_iop_adjust_mask_foreground);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_exclude_mask), p->mask_type == dt_iop_adjust_mask_exclude);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_include_mask), p->mask_type == dt_iop_adjust_mask_include);
  
  int b2 = 0, b4 = 0;
  if(self->dev->form_gui && self->dev->form_visible && self->dev->form_gui->creation
     && self->dev->form_gui->creation_module == self)
  {
    if(self->dev->form_visible->type & DT_MASKS_PATH)
      b2 = 1;
    else if(self->dev->form_visible->type & DT_MASKS_BRUSH)
      b4 = 1;
  }
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), b2);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), b4);



}

void init(dt_iop_module_t *module)
{
  module->params = calloc(1, sizeof(dt_iop_adjust_mask_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_adjust_mask_params_t));
  module->default_enabled = 0;
  module->params_size = sizeof(dt_iop_adjust_mask_params_t);
  module->gui_data = NULL;
  module->priority = 955; // module order created by iop_dependencies.py, do not edit! // from frames
//  module->priority = 184; // module order created by iop_dependencies.py, do not edit! // from spots
//  module->priority = 169; // module order created by iop_dependencies.py, do not edit! // from exposure

}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

void gui_focus(struct dt_iop_module_t *self, gboolean in)
{
  if(self->enabled)
  {
    if(in)
    {
      // got focus, show all shapes
/*      dt_masks_set_edit_mode(self, DT_MASKS_EDIT_FULL);

      dt_iop_adjust_mask_gui_data_t *g = (dt_iop_adjust_mask_gui_data_t *)self->gui_data;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), TRUE);
      
      dt_iop_adjust_mask_params_t *p = (dt_iop_adjust_mask_params_t *)self->params;
      maskadj_masks_selection_change(self, p);
*/
    }
    else
    {
      // lost focus, hide all shapes and free if some are in creation
      if (darktable.develop->form_gui->creation && darktable.develop->form_gui->creation_module == self)
      {
        dt_masks_form_t *form = darktable.develop->form_visible;
        if(form) dt_masks_free_form(form);
        dt_masks_change_form_gui(NULL);
      }
      dt_iop_adjust_mask_gui_data_t *g = (dt_iop_adjust_mask_gui_data_t *)self->gui_data;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), FALSE);
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), FALSE);
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), FALSE);
      dt_masks_set_edit_mode(self, DT_MASKS_EDIT_OFF);
    }
    
    dt_dev_reprocess_all(self->dev);
  }
}


void gui_init(struct dt_iop_module_t *self)
{
  const int bs = DT_PIXEL_APPLY_DPI(14);
  
  self->gui_data = malloc(sizeof(dt_iop_adjust_mask_gui_data_t));
  dt_iop_adjust_mask_gui_data_t *g = (dt_iop_adjust_mask_gui_data_t *)self->gui_data;
//  dt_iop_adjust_mask_params_t *p = (dt_iop_adjust_mask_params_t *)self->params;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  // masks
  GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  GtkWidget *label = gtk_label_new(_("# strokes:"));
  gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, TRUE, 0);
  g->label_form = GTK_LABEL(gtk_label_new("-1"));
  g_object_set(G_OBJECT(hbox), "tooltip-text", _("click on a shape and drag on canvas.\nuse the mouse wheel "
                                                 "to adjust size.\nright click to remove a shape."), (char *)NULL);

  g->bt_brush = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_brush, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_brush), "button-press-event", G_CALLBACK(maskadj_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_brush), "tooltip-text", _("add brush"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_brush), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_brush, FALSE, FALSE, 0);

  g->bt_path = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_path, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_path), "button-press-event", G_CALLBACK(maskadj_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_path), "tooltip-text", _("add path"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_path), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_path, FALSE, FALSE, 0);

  g->bt_edit_masks = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_eye, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_edit_masks), "button-press-event", G_CALLBACK(maskadj_edit_masks_callback), self);
  g_object_set(G_OBJECT(g->bt_edit_masks), "tooltip-text", _("show and edit mask elements"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_edit_masks), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_edit_masks, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(hbox), GTK_WIDGET(g->label_form), FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox, TRUE, TRUE, 0);

  GtkWidget *hbox_algo = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  
  GtkWidget *label1 = gtk_label_new(_("masks:"));
  gtk_box_pack_start(GTK_BOX(hbox_algo), label1, FALSE, TRUE, 0);
  
  g->bt_exclude_mask = dtgtk_togglebutton_new(_maskadj_cairo_paint_rigidity_mask, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_exclude_mask), "tooltip-text", _("rigidity mask"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_exclude_mask), "toggled", G_CALLBACK(maskadj_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_exclude_mask), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_exclude_mask), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_exclude_mask, FALSE, FALSE, 0);

  g->bt_include_mask = dtgtk_togglebutton_new(_maskadj_cairo_paint_discard_mask, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_include_mask), "tooltip-text", _("discard mask"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_include_mask), "toggled", G_CALLBACK(maskadj_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_include_mask), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_include_mask), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_include_mask, FALSE, FALSE, 0);

  g->bt_foreground_mask = dtgtk_togglebutton_new(_maskadj_cairo_paint_preserve_mask, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_foreground_mask), "tooltip-text", _("preserve mask"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_foreground_mask), "toggled", G_CALLBACK(maskadj_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_foreground_mask), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_foreground_mask), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_foreground_mask, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(self->widget), hbox_algo, TRUE, TRUE, 0);
  
  g->bt_execute = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_circle, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_execute), "button-press-event", G_CALLBACK(maskadj_execute_callback), self);
  g_object_set(G_OBJECT(g->bt_execute), "tooltip-text", _("add circle"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_execute), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_execute), bs, bs);
  gtk_box_pack_end(GTK_BOX(self->widget), g->bt_execute, FALSE, FALSE, 0);



//  maskadj_show_hide_controls(g, p);

}


void reload_defaults(dt_iop_module_t *self)
{
  dt_iop_adjust_mask_params_t tmp = (dt_iop_adjust_mask_params_t){ .maskadj_forms={ {0, (dt_iop_adjust_mask_masks_type_t)0} }, 
                                                                .mask_type=dt_iop_adjust_mask_foreground/*, 
                                                                .scaling = dt_iop_adjust_mask_scaling_relative,
                                                                .r_width = 1.f,
                                                                .r_height = 1.f,
                                                                .a_width = 0,
                                                                .a_height = 0,
                                                                .rigidity = 0.f,
                                                                .delta_x = 1,
                                                                .vertical_first = 0,
                                                                .enlarge_step = 1.5f,
                                                                .side_switch_frequency = 0,
                                                                .energy_function = dt_iop_adjust_mask_energy_xabs,
                                                                .preserve_mask_strength = 1000,
                                                                .discard_mask_strength = 1000*/
                                                                };
  
  memcpy(self->params, &tmp, sizeof(dt_iop_adjust_mask_params_t));
  memcpy(self->default_params, &tmp, sizeof(dt_iop_adjust_mask_params_t));
  self->default_enabled = 0;
}

void gui_cleanup(struct dt_iop_module_t *self)
{
  free(self->gui_data);
  self->gui_data = NULL;
}

void gui_reset(struct dt_iop_module_t *self)
{
  // hide the previous masks
  dt_masks_reset_form_gui();
}

void maskadj_copy_in_to_out(const float *const in, const struct dt_iop_roi_t *const roi_in, float *const out, const struct dt_iop_roi_t *const roi_out, const int ch)
{
  const int rowsize = roi_out->width*ch*sizeof(float);
  const int xoffs = roi_out->x - roi_in->x;
  const int yoffs = roi_out->y - roi_in->y;
  const int iwidth = roi_in->width;

#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
#endif
  for (int y=0; y < roi_out->height; y++) // copy each row
  {
    size_t iindex = ((size_t)(y + yoffs) * iwidth + xoffs) * ch;
    size_t oindex = (size_t)y * roi_out->width * ch;
    float *in1 = (float *)in + iindex;
    float *out1 = (float *)out + oindex;

    memcpy(out1, in1, rowsize);
  }
}

void maskadj_merge_mask_to_out(const float *const mask, const struct dt_iop_roi_t *const roi_mask, float *const out, const struct dt_iop_roi_t *const roi_out, const int ch, float* rgb)
{
  const int xoffs = roi_out->x - roi_mask->x;
  const int yoffs = roi_out->y - roi_mask->y;
  const int iwidth = roi_mask->width;

#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static)
#endif
#endif
  for (int y=0; y < roi_out->height; y++) // copy each row
  {
    size_t iindex = ((size_t)(y + yoffs) * iwidth + xoffs);
    size_t oindex = (size_t)y * roi_out->width * ch;
    float *in1 = (float *)mask + iindex;
    float *out1 = (float *)out + oindex;

    for (int x=0; x < roi_out->width; x++, in1++, out1 += ch)
    {
//      for (int c = 0; c < 4; c++) out1[c] = in1[0]*rgb[c] + out1[c]*(1.f-in1[0]);
      if (in1[0] > 0.f)
      {
        float gray = 0.3f * out1[0] + 0.59f * out1[1] + 0.11f * out1[2];
        float f = in1[0] * .25f;
        for (int c = 0; c < 4; c++) out1[c] = rgb[c]*f + gray*(1.f-f);
      }
    }
  }
}

int maskadj_get_masks(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, 
                        float *out, const dt_iop_roi_t *const roi_out, const int p_mask_type)
{
  int forms_count = 0;
  dt_develop_blend_params_t *bp = (dt_develop_blend_params_t *)piece->blendop_data;
  dt_iop_adjust_mask_data_t *p = (dt_iop_adjust_mask_data_t *)piece->data;
  
  // iterate through all forms
  dt_masks_form_t *grp = dt_masks_get_from_id(self->dev, bp->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    double start = dt_get_wtime();
    
    GList *forms = g_list_first(grp->points);
    while(forms)
    {
      dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;
      const dt_iop_adjust_mask_masks_type_t mask_type = maskadj_get_mask_type_from_id(p->maskadj_forms, grpt->formid);
      if (mask_type == 0)
      {
        printf("maskadj_get_masks unknown mask\n");
        forms = g_list_next(forms);
        continue;
      }
      else if (mask_type != p_mask_type)
      {
        forms = g_list_next(forms);
        continue;
      }

      // we get the spot
      dt_masks_form_t *form = dt_masks_get_from_id(self->dev, grpt->formid);
      if(!form)
      {
        printf("maskadj_get_masks missing form\n");
        forms = g_list_next(forms);
        continue;
      }

      // if the form is outside the roi, we just skip it
      if(!maskadj_masks_form_is_in_roi(self, piece, form, roi_out))
      {
        forms = g_list_next(forms);
        continue;
      }

      // we get the mask
      float *mask = NULL;
      int posx, posy, mask_width, mask_height;
      dt_masks_get_mask(self, piece, form, &mask, &mask_width, &mask_height, &posx, &posy);
      int fts = posy * roi_out->scale, fhs = mask_height * roi_out->scale, fls = posx * roi_out->scale,
          fws = mask_width * roi_out->scale;
      int dx = 0, dy = 0;

      if ((fws > 2) && (fhs > 2))
      {
        // now we do the pixel clone
        if (mask_type == dt_iop_adjust_mask_foreground)
        {
//          if (*pres_buffer == NULL) *pres_buffer = (float*)dt_alloc_align(64, roi_out->width * roi_out->height * sizeof(float));
//          out = *pres_buffer;
        }
        else if (mask_type == dt_iop_adjust_mask_include)
        {
//          if (*disc_buffer == NULL) *disc_buffer = (float*)dt_alloc_align(64, roi_out->width * roi_out->height * sizeof(float));
//          out = *disc_buffer;
        }
        else if (mask_type == dt_iop_adjust_mask_exclude)
        {
//          if (*rig_buffer == NULL) *rig_buffer = (float*)dt_alloc_align(64, roi_out->width * roi_out->height * sizeof(float));
//          out = *rig_buffer;
        }
        else
        {
//          out = NULL;
          printf("maskadj_get_masks unknown mask_type %i\n", mask_type);
        }
        
//        if (out)
        {
          forms_count++;
          
          for(int yy = fts + 1; yy < fts + fhs - 1; yy++)
          {
            // we test if we are inside roi_out
            if(yy < roi_out->y || yy >= roi_out->y + roi_out->height) continue;
            // we test if the source point is inside roi_out
            if(yy - dy < roi_out->y || yy - dy >= roi_out->y + roi_out->height) continue;
            
            for(int xx = fls + 1; xx < fls + fws - 1; xx++)
            {
              // we test if we are inside roi_out
              if(xx < roi_out->x || xx >= roi_out->x + roi_out->width) continue;
              // we test if the source point is inside roi_out
              if(xx - dx < roi_out->x || xx - dx >= roi_out->x + roi_out->width) continue;

              float f = (mask[((int)((yy - fts) / roi_out->scale)) * mask_width
                             + (int)((xx - fls) / roi_out->scale)]);
              
                out[((size_t)roi_out->width * (yy - roi_out->y) + xx - roi_out->x)] = f;
            }
          }
        }     
      }
      
      free(mask);
      
      forms = g_list_next(forms);
    }
    
    if(darktable.unmuted & DT_DEBUG_PERF) printf("maskadj_get_masks took %0.04f sec\n", dt_get_wtime() - start);
  }

  return forms_count;
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
//  dt_iop_adjust_mask_data_t *d = (dt_iop_adjust_mask_data_t *)piece->data;
  const dt_iop_colorspace_type_t cst = dt_iop_module_colorspace(self);

  if (cst != iop_cs_rgb) return;

  float *mask = NULL;
  int forms_count;
  const int ch = piece->colors;
  float rgb[3];

  maskadj_copy_in_to_out((float*)ivoid, roi_in, (float*)ovoid, roi_out, ch);
  
  mask = (float *)dt_alloc_align(64, roi_out->width * roi_out->height * sizeof(float));
  
  memset(mask, 0, roi_out->width * roi_out->height * sizeof(float));
  forms_count = maskadj_get_masks(self, piece, mask, roi_out, dt_iop_adjust_mask_foreground);
  if (forms_count > 0)
  {
    rgb[0] = 0.f; rgb[1] = 1.f; rgb[2] = 0.f;
    maskadj_merge_mask_to_out(mask, roi_out, (float *)ovoid, roi_out, ch, rgb);
  }
  
  memset(mask, 0, roi_out->width * roi_out->height * sizeof(float));
  forms_count = maskadj_get_masks(self, piece, mask, roi_out, dt_iop_adjust_mask_include);
  if (forms_count > 0)
  {
    rgb[0] = 1.f; rgb[1] = 0.f; rgb[2] = 0.f;
    maskadj_merge_mask_to_out(mask, roi_out, (float *)ovoid, roi_out, ch, rgb);
  }
  
  memset(mask, 0, roi_out->width * roi_out->height * sizeof(float));
  forms_count = maskadj_get_masks(self, piece, mask, roi_out, dt_iop_adjust_mask_exclude);
  if (forms_count > 0)
  {
    rgb[0] = 0.f; rgb[1] = 0.f; rgb[2] = 1.f;
    maskadj_merge_mask_to_out(mask, roi_out, (float *)ovoid, roi_out, ch, rgb);
  }
  
  
/*  if (self->dev->gui_attached && g && piece->pipe->type == DT_DEV_PIXELPIPE_PREVIEW)
  {
    const Mat* image;
        Mat mask;
    Mat bgdModel, fgdModel;
    Rect rect;
    bool isInitialized = false;
    
    if (isInitialized)
    {
           grabCut( *image, mask, rect, bgdModel, fgdModel, 1 );
    }
   else
   {
     grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK );
     isInitialized = true;
   }
  }*/
  
  if (mask) dt_free_align(mask);
    

}

} // extern "C"

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
