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

/*
 * This iop is based on the Liquid Rescale library by Carlo Baldassi
 * http://liblqr.wikidot.com/
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

#include "external/lqr/lqr.h"

extern "C" {

DT_MODULE_INTROSPECTION(1, dt_iop_liqrescale_params_t)

#define LQR_MAX_IMAGE_SIZE UINT16_MAX
#define LQR_NO_FORMS 200

typedef enum dt_iop_liqrescale_masks_type_t
{
  dt_iop_liqrescale_preserve_mask = 1, 
  dt_iop_liqrescale_discard_mask = 2,
  dt_iop_liqrescale_rigidity_mask = 3
} dt_iop_liqrescale_masks_type_t;

typedef struct dt_iop_liqrescale_form_data_t
{
  int id;
  dt_iop_liqrescale_masks_type_t mask_type;
} dt_iop_liqrescale_form_data_t;

typedef enum dt_iop_liqrescale_scaling_t
{
  dt_iop_liqrescale_scaling_relative = 0, 
  dt_iop_liqrescale_scaling_absolute = 1
} dt_iop_liqrescale_scaling_t;

typedef enum dt_iop_liqrescale_energy_functions_t
{
  dt_iop_liqrescale_energy_xabs = 0, 
  dt_iop_liqrescale_energy_sumabs = 1, 
  dt_iop_liqrescale_energy_norm = 2, 
  dt_iop_liqrescale_energy_luma_xabs = 3, 
  dt_iop_liqrescale_energy_luma_sumabs = 4, 
  dt_iop_liqrescale_energy_luma_norm = 5, 
  dt_iop_liqrescale_energy_sobelx = 6, 
  dt_iop_liqrescale_energy_sobel = 7, 
} dt_iop_liqrescale_energy_functions_t;

typedef struct dt_iop_liqrescale_params_t
{
  dt_iop_liqrescale_form_data_t lqr_forms[LQR_NO_FORMS];

  int mask_type; // preserve, discard, rigidity

  int scaling; // absolute, relative
  float r_width;
  float r_height; // new (relative) width & height, if final size<=2 keep the original size
  int a_width;
  int a_height; // new (absolute) width & height, if <=2 keep the original size
  float rigidity;
  int delta_x; // not used in UI yet
  int vertical_first;
  float enlarge_step;
  int side_switch_frequency;
  int energy_function;
  int preserve_mask_strength;
  int discard_mask_strength;
} dt_iop_liqrescale_params_t;

typedef struct dt_iop_liqrescale_gui_data_t
{
  GtkLabel *label_form; // display number of forms
  GtkWidget *bt_edit_masks, *bt_path, *bt_circle, *bt_ellipse, *bt_brush; // shapes
  GtkWidget *bt_preserve_mask, *bt_discard_mask, *bt_rigidity_mask; // masks types

  GtkWidget *cmb_scaling;
  GtkWidget *vbox_r_size;
  GtkWidget *sl_r_width, *sl_r_height;
  GtkWidget *vbox_a_size;
  GtkWidget *sl_a_width, *sl_a_height;
  GtkWidget *sl_rigidity;
  GtkWidget *cmb_vertical_first;
  GtkWidget *sl_enlarge_step;
  GtkWidget *sl_side_switch_frequency;
  GtkWidget *cmb_energy_function;
  GList *lst_energy_function;
  GtkWidget *sl_preserve_mask_strength;
  GtkWidget *sl_discard_mask_strength;
} dt_iop_liqrescale_gui_data_t;

typedef struct dt_iop_liqrescale_params_t dt_iop_liqrescale_data_t;

const char *name()
{
  return _("liqres_eh");
}

int groups()
{
  return IOP_GROUP_EFFECT;
}

int operation_tags()
{
  return IOP_TAG_DISTORT;
}

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

static void _lqr_cairo_paint_preserve_mask(cairo_t *cr,
    const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;

  cairo_set_source_rgba(cr, 0.0, 1.0, 0.0, 1.0);
  cairo_rectangle(cr, 0., 0., 1., 1.);
  cairo_fill(cr);

  POSTAMBLE;
}

static void _lqr_cairo_paint_discard_mask(cairo_t *cr,
    const gint x, const gint y, const gint w, const gint h, const gint flags)
{
  PREAMBLE;

  cairo_set_source_rgba(cr, 1.0, 0.0, 0.0, 1.0);
  cairo_rectangle(cr, 0., 0., 1., 1.);
  cairo_fill(cr);

  POSTAMBLE;
}

static void _lqr_cairo_paint_rigidity_mask(cairo_t *cr, gint x, gint y, gint w, gint h, gint flags)
{
  PREAMBLE;

  cairo_set_source_rgba(cr, 0.0, 0.0, 1.0, 1.0);
  cairo_rectangle(cr, 0., 0., 1., 1.);
  cairo_fill(cr);

  POSTAMBLE;
}

static void lqr_show_hide_controls(dt_iop_liqrescale_gui_data_t *d, dt_iop_liqrescale_params_t *p)
{
  switch (p->scaling)
  {
  case dt_iop_liqrescale_scaling_relative:
    gtk_widget_show(GTK_WIDGET(d->vbox_r_size));
    gtk_widget_hide(GTK_WIDGET(d->vbox_a_size));
    break;
  case dt_iop_liqrescale_scaling_absolute:
  default:
    gtk_widget_hide(GTK_WIDGET(d->vbox_r_size));
    gtk_widget_show(GTK_WIDGET(d->vbox_a_size));
    break;
  }
}


static int lqr_allow_create_form(dt_iop_module_t *self)
{
  int allow = 1;

  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  if (p)
  {
    allow = (p->lqr_forms[LQR_NO_FORMS-1].id == 0);
  }
  return allow;
}

static void lqr_reset_form_creation(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;
  if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_path)) ||
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_circle)) ||
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_ellipse)) ||
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(g->bt_brush)))
  {
    // we unset the creation mode
    dt_masks_form_t *form = darktable.develop->form_visible;
    if(form) dt_masks_free_form(form);
    dt_masks_change_form_gui(NULL);
  }
  if (widget != g->bt_path) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), FALSE);
  if (widget != g->bt_circle) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_circle), FALSE);
  if (widget != g->bt_ellipse) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_ellipse), FALSE);
  if (widget != g->bt_brush) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), FALSE);
}

static void lqr_masks_selection_change(dt_iop_module_t *self, dt_iop_liqrescale_params_t *p)
{
  if (!self->enabled) return;

  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  if (bd == NULL) return;

  dt_masks_set_edit_mode(self, (dt_masks_edit_mode_t)bd->masks_shown);
  return;

  /*  int count = 0;
  for (int i = 0; i < LQR_NO_FORMS && count == 0; i++)
  {
    if (p->lqr_forms[i].id != 0) count++;
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
  for (int i = 0; i < LQR_NO_FORMS; i++)
  {
    int grid = self->blend_params->mask_id;
    int id = p->lqr_forms[i].id;
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
  dt_control_queue_redraw_center();*/
}

static gboolean lqr_masks_being_displayed(dt_iop_module_t *self)
{
  /*  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  if (bd == NULL) return FALSE;
  return (bd->masks_shown != DT_MASKS_EDIT_OFF);*/
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  if (bd == NULL) return FALSE;
  return ((bd->masks_shown != DT_MASKS_EDIT_OFF) || (darktable.develop->form_gui->creation && darktable.develop->form_gui->creation_module == self));
}

static gboolean lqr_edit_masks_callback(GtkWidget *widget, GdkEventButton *event, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return FALSE;
  dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
  dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;

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

    dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
    lqr_masks_selection_change(self, p);
    dt_dev_reprocess_all(self->dev);

    return TRUE;
  }

  return FALSE;
}

static gboolean lqr_add_shape_callback(GtkWidget *widget, GdkEventButton *e, dt_iop_module_t *self)
{
  const int allow = lqr_allow_create_form(self);
  if (allow)
  {
    lqr_reset_form_creation(widget, self);

    if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget))) return FALSE;

    dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;

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

    spot = dt_masks_create(type);

    dt_masks_change_form_gui(spot);
    darktable.develop->form_gui->creation = TRUE;
    darktable.develop->form_gui->creation_module = self;
    dt_control_queue_redraw_center();

    dt_iop_gui_blend_data_t *bd = (dt_iop_gui_blend_data_t *)self->blend_data;
    if (!bd->masks_shown) dt_dev_reprocess_all(self->dev);
  }
  else
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget), FALSE);
  }

  return !allow;
}

static void lqr_select_algorithm_callback(GtkToggleButton *togglebutton, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;

  if (togglebutton == (GtkToggleButton *)g->bt_rigidity_mask)
    p->mask_type = dt_iop_liqrescale_rigidity_mask;
  else if (togglebutton == (GtkToggleButton *)g->bt_preserve_mask)
    p->mask_type = dt_iop_liqrescale_preserve_mask;
  else if (togglebutton == (GtkToggleButton *)g->bt_discard_mask)
    p->mask_type = dt_iop_liqrescale_discard_mask;

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_preserve_mask), (p->mask_type == dt_iop_liqrescale_preserve_mask));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_discard_mask), (p->mask_type == dt_iop_liqrescale_discard_mask));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_rigidity_mask), (p->mask_type == dt_iop_liqrescale_rigidity_mask));

  //  lqr_show_hide_controls(g, p);

  darktable.gui->reset = reset;

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}


static void _scaling_callback(GtkComboBox *combo, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;

  p->scaling = dt_bauhaus_combobox_get((GtkWidget *)combo);

  lqr_show_hide_controls(g, p);

  darktable.gui->reset = reset;

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _vertical_first_callback(GtkComboBox *combo, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;

  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;

  p->vertical_first = dt_bauhaus_combobox_get((GtkWidget *)combo);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _r_width_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  p->r_width = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _r_height_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  p->r_height = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _a_width_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_liqrescale_data_t *p = (dt_iop_liqrescale_data_t *)self->params;

  p->a_width = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _a_height_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_liqrescale_data_t *p = (dt_iop_liqrescale_data_t *)self->params;

  p->a_height = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _rigidity_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_liqrescale_data_t *p = (dt_iop_liqrescale_data_t *)self->params;

  p->rigidity = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _enlarge_step_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  p->enlarge_step = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _side_switch_frequency_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  p->side_switch_frequency = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _energy_function_callback(GtkComboBox *combo, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;

  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;

  p->energy_function = GPOINTER_TO_UINT(
      g_list_nth_data(g->lst_energy_function, dt_bauhaus_combobox_get((GtkWidget *)combo)));

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _preserve_mask_strength_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  p->preserve_mask_strength = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _discard_mask_strength_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  p->discard_mask_strength = dt_bauhaus_slider_get(slider);

  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

dt_iop_liqrescale_masks_type_t lqr_get_mask_type_from_id(dt_iop_liqrescale_form_data_t *lqr_forms, const int formid)
{
  dt_iop_liqrescale_masks_type_t mask_type = (dt_iop_liqrescale_masks_type_t)0;
  int i = 0;

  while (mask_type == 0 && i < LQR_NO_FORMS)
  {
    if (lqr_forms[i].id == formid) mask_type = lqr_forms[i].mask_type;
    i++;
  }

  return mask_type;
}

static void lqr_resynch_params(struct dt_iop_module_t *self)
{
  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
  dt_develop_blend_params_t *bp = self->blend_params;

  dt_iop_liqrescale_form_data_t *lqr_forms = p->lqr_forms;
  const int mask_type = p->mask_type;

  dt_iop_liqrescale_form_data_t forms_d[LQR_NO_FORMS] = { {0, (dt_iop_liqrescale_masks_type_t)0} };

  // we go through all forms in blend params
  dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, bp->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    GList *forms = g_list_first(grp->points);
    int i = 0;
    while((i < LQR_NO_FORMS) && forms)
    {
      dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;

      int j = 0;
      while( j < LQR_NO_FORMS && lqr_forms[j].id != grpt->formid )
      {
        j++;
      }

      if (j < LQR_NO_FORMS)
      {
        forms_d[i].mask_type = lqr_forms[j].mask_type;
        forms_d[i].id = grpt->formid;
        i++;
      }
      else
      {
        int mask_type_new = lqr_get_mask_type_from_id(p->lqr_forms, grpt->formid);
        if (mask_type_new <= 0)
        {
          dt_masks_form_t *parent_form = dt_masks_get_from_id(darktable.develop, grpt->formid);
          if (parent_form)
          {
            forms_d[i].mask_type = (dt_iop_liqrescale_masks_type_t)mask_type;
            forms_d[i].id = grpt->formid;

            i++;
          }
        }
      }

      forms = g_list_next(forms);
    }
  }

  // we reaffect params
  for(int i = 0; i < LQR_NO_FORMS; i++)
  {
    lqr_forms[i].mask_type = forms_d[i].mask_type;
    lqr_forms[i].id = forms_d[i].id;
  }
}

static gboolean lqr_masks_form_is_in_roi(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
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

/*
int distort_backtransform(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, float *points, size_t points_count)
{
  dt_iop_liqrescale_data_t *d = (dt_iop_liqrescale_data_t *)piece->data;

  for(size_t i = 0; i < points_count * 2; i += 2)
  {
    points[i] /= d->r_width;
    points[i + 1] /= d->r_height;
  }

  return 1;
}

int distort_transform(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, float *points, size_t points_count)
{
  dt_iop_liqrescale_data_t *d = (dt_iop_liqrescale_data_t *)piece->data;

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
  if (lqr_masks_being_displayed(self)) return;

  dt_iop_liqrescale_data_t *d = (dt_iop_liqrescale_data_t *)piece->data;

  const float scale = roi_in->scale / piece->iscale;
  int new_width;
  int new_height;

  if (d->scaling == dt_iop_liqrescale_scaling_relative)
  {
    new_width = roi_out->width * d->r_width;
    new_height = roi_out->height * d->r_height;
  }
  else
  {
    new_width = d->a_width*scale;
    new_height = d->a_height*scale;
  }

  if (new_width <= 2) new_width = roi_out->width;
  if (new_height <= 2) new_height = roi_out->height;

  roi_out->width = new_width;
  roi_out->height = new_height;

  // sanity check.
  roi_out->width = CLAMP(roi_out->width, 1, 3 * roi_in->width);
  roi_out->height = CLAMP(roi_out->height, 1, 3 * roi_in->height);
}

// 2nd pass: which roi would this operation need as input to fill the given output region?
void modify_roi_in(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece,
    const dt_iop_roi_t *roi_out, dt_iop_roi_t *roi_in)
{
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
  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)p1;
  dt_iop_liqrescale_data_t *d = (dt_iop_liqrescale_data_t *)piece->data;
  memcpy(d, p, sizeof(dt_iop_liqrescale_params_t));
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_liqrescale_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(struct dt_iop_module_t *self)
{
  dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;
  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;

  lqr_resynch_params(self);

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

  //  lqr_masks_selection_change(self, p);

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_preserve_mask), p->mask_type == dt_iop_liqrescale_preserve_mask);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_rigidity_mask), p->mask_type == dt_iop_liqrescale_rigidity_mask);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_discard_mask), p->mask_type == dt_iop_liqrescale_discard_mask);

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


  dt_bauhaus_combobox_set(g->cmb_scaling, p->scaling);
  dt_bauhaus_slider_set(g->sl_r_width, p->r_width);
  dt_bauhaus_slider_set(g->sl_r_height, p->r_height);
  dt_bauhaus_slider_set(g->sl_a_width, p->a_width);
  dt_bauhaus_slider_set(g->sl_a_height, p->a_height);
  dt_bauhaus_slider_set(g->sl_rigidity, p->rigidity);
  dt_bauhaus_combobox_set(g->cmb_vertical_first, p->vertical_first);
  dt_bauhaus_slider_set(g->sl_enlarge_step, p->enlarge_step);
  dt_bauhaus_slider_set(g->sl_side_switch_frequency, p->side_switch_frequency);
  dt_bauhaus_combobox_set(g->cmb_energy_function, g_list_index(g->lst_energy_function, GUINT_TO_POINTER(p->energy_function)));
  dt_bauhaus_slider_set(g->sl_preserve_mask_strength, p->preserve_mask_strength);
  dt_bauhaus_slider_set(g->sl_discard_mask_strength, p->discard_mask_strength);

}

void init(dt_iop_module_t *module)
{
  module->params = calloc(1, sizeof(dt_iop_liqrescale_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_liqrescale_params_t));
  module->default_enabled = 0;
  module->params_size = sizeof(dt_iop_liqrescale_params_t);
  module->gui_data = NULL;
  module->priority = 955; // module order created by iop_dependencies.py, do not edit! // from borders.c

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

      dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), TRUE);

      dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;
      lqr_masks_selection_change(self, p);
       */
    }
    else
    {
      // lost focus, hide all shapes and free if some are in creation
      if (darktable.develop->form_gui->creation && darktable.develop->form_gui->creation_module == self)
      {
        //        dt_masks_form_t *form = darktable.develop->form_visible;
        //        if(form) dt_masks_free_form(form);
        dt_masks_change_form_gui(NULL);
      }
      dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;
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


void gui_init(struct dt_iop_module_t *self)
{
  const int bs = DT_PIXEL_APPLY_DPI(14);

  self->gui_data = malloc(sizeof(dt_iop_liqrescale_gui_data_t));
  dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;
  dt_iop_liqrescale_params_t *p = (dt_iop_liqrescale_params_t *)self->params;

  g->lst_energy_function = NULL;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  // masks
  GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  GtkWidget *label = gtk_label_new(_("# strokes:"));
  gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, TRUE, 0);
  g->label_form = GTK_LABEL(gtk_label_new("-1"));
  g_object_set(G_OBJECT(hbox), "tooltip-text", _("click on a shape and drag on canvas.\nuse the mouse wheel "
      "to adjust size.\nright click to remove a shape."), (char *)NULL);

  g->bt_brush = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_brush, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_brush), "button-press-event", G_CALLBACK(lqr_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_brush), "tooltip-text", _("add brush"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_brush), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_brush), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_brush, FALSE, FALSE, 0);

  g->bt_path = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_path, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_path), "button-press-event", G_CALLBACK(lqr_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_path), "tooltip-text", _("add path"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_path), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_path), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_path, FALSE, FALSE, 0);

  g->bt_ellipse = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_ellipse, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_ellipse), "button-press-event", G_CALLBACK(lqr_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_ellipse), "tooltip-text", _("add ellipse"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_ellipse), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_ellipse), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_ellipse, FALSE, FALSE, 0);

  g->bt_circle = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_circle, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_circle), "button-press-event", G_CALLBACK(lqr_add_shape_callback), self);
  g_object_set(G_OBJECT(g->bt_circle), "tooltip-text", _("add circle"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_circle), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_circle), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_circle, FALSE, FALSE, 0);

  g->bt_edit_masks = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_eye, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_signal_connect(G_OBJECT(g->bt_edit_masks), "button-press-event", G_CALLBACK(lqr_edit_masks_callback), self);
  g_object_set(G_OBJECT(g->bt_edit_masks), "tooltip-text", _("show and edit mask elements"), (char *)NULL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_edit_masks), FALSE);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_edit_masks), bs, bs);
  gtk_box_pack_end(GTK_BOX(hbox), g->bt_edit_masks, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(hbox), GTK_WIDGET(g->label_form), FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox, TRUE, TRUE, 0);

  GtkWidget *hbox_algo = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);

  GtkWidget *label1 = gtk_label_new(_("masks:"));
  gtk_box_pack_start(GTK_BOX(hbox_algo), label1, FALSE, TRUE, 0);

  g->bt_rigidity_mask = dtgtk_togglebutton_new(_lqr_cairo_paint_rigidity_mask, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_rigidity_mask), "tooltip-text", _("rigidity mask"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_rigidity_mask), "toggled", G_CALLBACK(lqr_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_rigidity_mask), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_rigidity_mask), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_rigidity_mask, FALSE, FALSE, 0);

  g->bt_discard_mask = dtgtk_togglebutton_new(_lqr_cairo_paint_discard_mask, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_discard_mask), "tooltip-text", _("discard mask"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_discard_mask), "toggled", G_CALLBACK(lqr_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_discard_mask), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_discard_mask), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_discard_mask, FALSE, FALSE, 0);

  g->bt_preserve_mask = dtgtk_togglebutton_new(_lqr_cairo_paint_preserve_mask, CPF_STYLE_FLAT | CPF_DO_NOT_USE_BORDER);
  g_object_set(G_OBJECT(g->bt_preserve_mask), "tooltip-text", _("preserve mask"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->bt_preserve_mask), "toggled", G_CALLBACK(lqr_select_algorithm_callback), self);
  gtk_widget_set_size_request(GTK_WIDGET(g->bt_preserve_mask), bs, bs);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->bt_preserve_mask), FALSE);
  gtk_box_pack_end(GTK_BOX(hbox_algo), g->bt_preserve_mask, FALSE, FALSE, 0);

  gtk_box_pack_start(GTK_BOX(self->widget), hbox_algo, TRUE, TRUE, 0);


  // lqr controls
  g->cmb_scaling = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->cmb_scaling, NULL, _("scaling"));
  dt_bauhaus_combobox_add(g->cmb_scaling, _("relative"));
  dt_bauhaus_combobox_add(g->cmb_scaling, _("absolute"));
  g_object_set(g->cmb_scaling, "tooltip-text", _("scaling mode"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->cmb_scaling), "value-changed", G_CALLBACK(_scaling_callback), self);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->cmb_scaling), TRUE, TRUE, 0);

  g->vbox_r_size = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);

  g->sl_r_width = dt_bauhaus_slider_new_with_range(self, 0.0, 3.0, .005, 1.0, 3);
  dt_bauhaus_widget_set_label(g->sl_r_width, _("width"), _("width"));
  g_object_set(g->sl_r_width, "tooltip-text", _("relative output width\nset to 0 for no scaling"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_r_width), "value-changed", G_CALLBACK(_r_width_callback), self);
  gtk_box_pack_start(GTK_BOX(g->vbox_r_size), g->sl_r_width, TRUE, TRUE, 0);

  g->sl_r_height = dt_bauhaus_slider_new_with_range(self, 0.0, 3.0, .005, 1.0, 3);
  dt_bauhaus_widget_set_label(g->sl_r_height, _("height"), _("height"));
  g_object_set(g->sl_r_height, "tooltip-text", _("relative output height\nset to 0 for no scaling"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_r_height), "value-changed", G_CALLBACK(_r_height_callback), self);
  gtk_box_pack_start(GTK_BOX(g->vbox_r_size), g->sl_r_height, TRUE, TRUE, 0);

  gtk_box_pack_start(GTK_BOX(self->widget), g->vbox_r_size, TRUE, TRUE, 0);

  g->vbox_a_size = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);

  g->sl_a_width = dt_bauhaus_slider_new_with_range(self, 0, LQR_MAX_IMAGE_SIZE, 1, 0, 0);
  dt_bauhaus_widget_set_label(g->sl_a_width, _("width"), _("width"));
  g_object_set(g->sl_a_width, "tooltip-text", _("maximum output width\nset to 0 for no scaling"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_a_width), "value-changed", G_CALLBACK(_a_width_callback), self);
  gtk_box_pack_start(GTK_BOX(g->vbox_a_size), GTK_WIDGET(g->sl_a_width), TRUE, TRUE, 0);

  g->sl_a_height = dt_bauhaus_slider_new_with_range(self, 0, LQR_MAX_IMAGE_SIZE, 1, 0, 0);
  dt_bauhaus_widget_set_label(g->sl_a_height, _("height"), _("height"));
  g_object_set(g->sl_a_height, "tooltip-text", _("maximum output height\nset to 0 for no scaling"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_a_height), "value-changed", G_CALLBACK(_a_height_callback), self);
  gtk_box_pack_start(GTK_BOX(g->vbox_a_size), GTK_WIDGET(g->sl_a_height), TRUE, TRUE, 0);

  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->vbox_a_size), TRUE, TRUE, 0);

  g->cmb_vertical_first = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->cmb_vertical_first, NULL, _("vertical first"));
  dt_bauhaus_combobox_add(g->cmb_vertical_first, _("no"));
  dt_bauhaus_combobox_add(g->cmb_vertical_first, _("yes"));
  g_object_set(g->cmb_vertical_first, "tooltip-text", _("rescale vertically first (instead of horizontally)"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->cmb_vertical_first), "value-changed", G_CALLBACK(_vertical_first_callback), self);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->cmb_vertical_first), TRUE, TRUE, 0);

  g->sl_enlarge_step = dt_bauhaus_slider_new_with_range(self, 1.0, 2.0, .005, 1.5, 3);
  dt_bauhaus_widget_set_label(g->sl_enlarge_step, _("enlarge step"), _("enlarge step"));
  g_object_set(g->sl_enlarge_step, "tooltip-text", _("set the maximum enlargement in a single step"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_enlarge_step), "value-changed", G_CALLBACK(_enlarge_step_callback), self);
  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_enlarge_step, TRUE, TRUE, 0);

  g->sl_side_switch_frequency = dt_bauhaus_slider_new_with_range(self, 0, 40, 1, 0, 0);
  dt_bauhaus_widget_set_label(g->sl_side_switch_frequency, _("side switch frequency"), _("side switch frequency"));
  g_object_set(g->sl_side_switch_frequency, "tooltip-text", _("set the number of switches of the side choice for each size modification"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_side_switch_frequency), "value-changed", G_CALLBACK(_side_switch_frequency_callback), self);
  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_side_switch_frequency, TRUE, TRUE, 0);

  g->cmb_energy_function = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->cmb_energy_function, NULL, _("energy function"));

  dt_bauhaus_combobox_add(g->cmb_energy_function, _("xabs"));
  g->lst_energy_function = g_list_append(g->lst_energy_function, GUINT_TO_POINTER(dt_iop_liqrescale_energy_xabs));
  dt_bauhaus_combobox_add(g->cmb_energy_function, _("sumabs"));
  g->lst_energy_function = g_list_append(g->lst_energy_function, GUINT_TO_POINTER(dt_iop_liqrescale_energy_sumabs));
  dt_bauhaus_combobox_add(g->cmb_energy_function, _("norm"));
  g->lst_energy_function = g_list_append(g->lst_energy_function, GUINT_TO_POINTER(dt_iop_liqrescale_energy_norm));
  dt_bauhaus_combobox_add(g->cmb_energy_function, _("luma xabs"));
  g->lst_energy_function = g_list_append(g->lst_energy_function, GUINT_TO_POINTER(dt_iop_liqrescale_energy_luma_xabs));
  dt_bauhaus_combobox_add(g->cmb_energy_function, _("luma sumabs"));
  g->lst_energy_function = g_list_append(g->lst_energy_function, GUINT_TO_POINTER(dt_iop_liqrescale_energy_luma_sumabs));
  dt_bauhaus_combobox_add(g->cmb_energy_function, _("luma norm"));
  g->lst_energy_function = g_list_append(g->lst_energy_function, GUINT_TO_POINTER(dt_iop_liqrescale_energy_luma_norm));
  dt_bauhaus_combobox_add(g->cmb_energy_function, _("sobelx"));
  g->lst_energy_function = g_list_append(g->lst_energy_function, GUINT_TO_POINTER(dt_iop_liqrescale_energy_sobelx));
  dt_bauhaus_combobox_add(g->cmb_energy_function, _("sobel"));
  g->lst_energy_function = g_list_append(g->lst_energy_function, GUINT_TO_POINTER(dt_iop_liqrescale_energy_sobel));

  g_object_set(g->cmb_energy_function, "tooltip-text", _("energy evaluation function"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->cmb_energy_function), "value-changed", G_CALLBACK(_energy_function_callback), self);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->cmb_energy_function), TRUE, TRUE, 0);

  g->sl_preserve_mask_strength = dt_bauhaus_slider_new_with_range(self, 100, 10000, 1, 1000, 0);
  dt_bauhaus_widget_set_label(g->sl_preserve_mask_strength, _("preserve mask strength"), _("preserve mask strength"));
  g_object_set(g->sl_preserve_mask_strength, "tooltip-text", _("preserve mask strength"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_preserve_mask_strength), "value-changed", G_CALLBACK(_preserve_mask_strength_callback), self);
  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_preserve_mask_strength, TRUE, TRUE, 0);

  g->sl_discard_mask_strength = dt_bauhaus_slider_new_with_range(self, 100, 10000, 1, 1000, 0);
  dt_bauhaus_widget_set_label(g->sl_discard_mask_strength, _("discard mask strength"), _("discard mask strength"));
  g_object_set(g->sl_discard_mask_strength, "tooltip-text", _("discard mask strength"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_discard_mask_strength), "value-changed", G_CALLBACK(_discard_mask_strength_callback), self);
  gtk_box_pack_start(GTK_BOX(self->widget), g->sl_discard_mask_strength, TRUE, TRUE, 0);

  g->sl_rigidity = dt_bauhaus_slider_new_with_range(self, 0, 10000, 1, 0, 0);
  dt_bauhaus_widget_set_label(g->sl_rigidity, _("rigidity strength"), _("rigidity strength"));
  g_object_set(g->sl_rigidity, "tooltip-text", _("rigidity strength"), (char *)NULL);
  g_signal_connect(G_OBJECT(g->sl_rigidity), "value-changed", G_CALLBACK(_rigidity_callback), self);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->sl_rigidity), TRUE, TRUE, 0);


  gtk_widget_show_all(g->vbox_a_size);
  gtk_widget_set_no_show_all(g->vbox_a_size, TRUE);

  gtk_widget_show_all(g->vbox_r_size);
  gtk_widget_set_no_show_all(g->vbox_r_size, TRUE);

  lqr_show_hide_controls(g, p);

}


void reload_defaults(dt_iop_module_t *self)
{
  dt_iop_liqrescale_params_t tmp = (dt_iop_liqrescale_params_t){ .lqr_forms={ {0, (dt_iop_liqrescale_masks_type_t)0} }, 
    .mask_type=dt_iop_liqrescale_preserve_mask, 
    .scaling = dt_iop_liqrescale_scaling_relative,
    .r_width = 1.f,
    .r_height = 1.f,
    .a_width = 0,
    .a_height = 0,
    .rigidity = 0.f,
    .delta_x = 1,
    .vertical_first = 0,
    .enlarge_step = 1.5f,
    .side_switch_frequency = 0,
    .energy_function = dt_iop_liqrescale_energy_xabs,
    .preserve_mask_strength = 1000,
    .discard_mask_strength = 1000
  };

  memcpy(self->params, &tmp, sizeof(dt_iop_liqrescale_params_t));
  memcpy(self->default_params, &tmp, sizeof(dt_iop_liqrescale_params_t));
  self->default_enabled = 0;
}

void gui_cleanup(struct dt_iop_module_t *self)
{
  dt_iop_liqrescale_gui_data_t *g = (dt_iop_liqrescale_gui_data_t *)self->gui_data;
  g_list_free(g->lst_energy_function);

  free(self->gui_data);
  self->gui_data = NULL;
}

void gui_reset(struct dt_iop_module_t *self)
{
  // hide the previous masks
  dt_masks_reset_form_gui();
}


static void lqr_write_carver_to_image(LqrCarver *r, float *out, const dt_iop_roi_t *const roi_out, const int ch)
{
  gint x, y;
  void *v_rgb;

  /* make sure the image is RGB */
  if(lqr_carver_get_channels(r) == 3)
  {
    /* initialize image reading */
    lqr_carver_scan_reset(r);

    /* readout (no need to init rgb) */
    while (lqr_carver_scan_ext(r, &x, &y, &v_rgb))
    {
      gfloat *rgb = (gfloat *)v_rgb;

      const int y1 = y - roi_out->y;
      const int x1 = x - roi_out->x;
      if (y1 >= 0 && x1 >= 0 && y1 < roi_out->height && x1 < roi_out->width)
      {
        for (int k = 0; k < 3; k++) out[y1 * roi_out->width * ch + x1 * ch + k] = (float) rgb[k];
      }
    }
  }
  else
    printf("liqres wrong number of channels\n");
}

void lqr_copy_in_to_out(const float *const in, const struct dt_iop_roi_t *const roi_in, float *const out, const struct dt_iop_roi_t *const roi_out, const int ch)
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

void lqr_merge_mask_to_out(const gdouble *const mask, const struct dt_iop_roi_t *const roi_mask, float *const out, const struct dt_iop_roi_t *const roi_out, const int ch, float* rgb)
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
    gdouble *in1 = (gdouble *)mask + iindex;
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

/* define custom energy function: sobelx */
gfloat lqr_sobelx(gint x, gint y, gint w, gint h, LqrReadingWindow *rw, gpointer extra_data)
{
  gint i, j;
  gdouble e = 0;
  gdouble k[3][3] = { {0.125, 0.25, 0.125}, {0, 0, 0}, {-0.125, -0.25, -0.125} };

  for (i = -1; i <= 1; i++) {
    for (j = -1; j <= 1; j++) {
      e += k[i + 1][j + 1] * lqr_rwindow_read(rw, i, j, 0);
    }
  }
  return (gfloat) fabs(e);
}

/* define custom energy function: sobel */
gfloat lqr_sobel(gint x, gint y, gint w, gint h, LqrReadingWindow *rw, gpointer extra_data)
{
  gint i, j;
  gdouble ex = 0;
  gdouble ey = 0;
  gdouble k[3][3] = { {0.125, 0.25, 0.125}, {0, 0, 0}, {-0.125, -0.25, -0.125} };

  for (i = -1; i <= 1; i++) {
    for (j = -1; j <= 1; j++) {
      ex += k[i + 1][j + 1] * lqr_rwindow_read(rw, i, j, 0);
      ey += k[j + 1][i + 1] * lqr_rwindow_read(rw, i, j, 0);
    }
  }
  return (gfloat) (sqrt(ex * ex + ey * ey));
}

/* set the energy function */
void lqr_set_energy(LqrCarver *carver, int energy_function)
{
  if (energy_function == dt_iop_liqrescale_energy_xabs) {
    lqr_carver_set_energy_function_builtin(carver, LQR_EF_GRAD_XABS);
  } else if (energy_function == dt_iop_liqrescale_energy_sumabs) {
    lqr_carver_set_energy_function_builtin(carver, LQR_EF_GRAD_SUMABS);
  } else if (energy_function == dt_iop_liqrescale_energy_norm) {
    lqr_carver_set_energy_function_builtin(carver, LQR_EF_GRAD_NORM);
  } else if (energy_function == dt_iop_liqrescale_energy_luma_xabs) {
    lqr_carver_set_energy_function_builtin(carver, LQR_EF_LUMA_GRAD_XABS);
  } else if (energy_function == dt_iop_liqrescale_energy_luma_sumabs) {
    lqr_carver_set_energy_function_builtin(carver, LQR_EF_LUMA_GRAD_SUMABS);
  } else if (energy_function == dt_iop_liqrescale_energy_luma_norm) {
    lqr_carver_set_energy_function_builtin(carver, LQR_EF_LUMA_GRAD_NORM);
  } else if (energy_function == dt_iop_liqrescale_energy_sobelx) {
    lqr_carver_set_energy_function(carver, lqr_sobelx, 1, LQR_ER_BRIGHTNESS, NULL);
  } else if (energy_function == dt_iop_liqrescale_energy_sobel) {
    lqr_carver_set_energy_function(carver, lqr_sobel, 1, LQR_ER_BRIGHTNESS, NULL);
  }

}

void lqr_get_masks(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, 
    const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out, 
    gdouble **pres_buffer, gdouble **disc_buffer, gdouble **rig_buffer)
{
  dt_develop_blend_params_t *bp = (dt_develop_blend_params_t *)piece->blendop_data;
  dt_iop_liqrescale_data_t *p = (dt_iop_liqrescale_data_t *)piece->data;

  gdouble *out = NULL;

  // iterate through all forms
  dt_masks_form_t *grp = dt_masks_get_from_id(self->dev, bp->mask_id);
  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    double start = dt_get_wtime();

    GList *forms = g_list_first(grp->points);
    while(forms)
    {
      dt_masks_point_group_t *grpt = (dt_masks_point_group_t *)forms->data;
      const dt_iop_liqrescale_masks_type_t mask_type = lqr_get_mask_type_from_id(p->lqr_forms, grpt->formid);
      if (mask_type == 0)
      {
        printf("lqr_get_masks unknown mask\n");
        forms = g_list_next(forms);
        continue;
      }

      // we get the spot
      dt_masks_form_t *form = dt_masks_get_from_id(self->dev, grpt->formid);
      if(!form)
      {
        printf("lqr_get_masks missing form\n");
        forms = g_list_next(forms);
        continue;
      }

      // if the form is outside the roi, we just skip it
      if(!lqr_masks_form_is_in_roi(self, piece, form, roi_in, roi_out))
      {
        forms = g_list_next(forms);
        continue;
      }

      // we get the mask
      float *mask = NULL;
      int posx, posy, mask_width, mask_height;
      dt_masks_get_mask(self, piece, form, &mask, &mask_width, &mask_height, &posx, &posy);
      int fts = posy * roi_in->scale, fhs = mask_height * roi_in->scale, fls = posx * roi_in->scale,
          fws = mask_width * roi_in->scale;
      int dx = 0, dy = 0;

      if ((fws > 2) && (fhs > 2))
      {
        if (mask_type == dt_iop_liqrescale_preserve_mask)
        {
          if (*pres_buffer == NULL) *pres_buffer = (gdouble*)dt_alloc_align(64, roi_in->width * roi_in->height * sizeof(gdouble));
          out = *pres_buffer;
        }
        else if (mask_type == dt_iop_liqrescale_discard_mask)
        {
          if (*disc_buffer == NULL) *disc_buffer = (gdouble*)dt_alloc_align(64, roi_in->width * roi_in->height * sizeof(gdouble));
          out = *disc_buffer;
        }
        else if (mask_type == dt_iop_liqrescale_rigidity_mask)
        {
          if (*rig_buffer == NULL) *rig_buffer = (gdouble*)dt_alloc_align(64, roi_in->width * roi_in->height * sizeof(gdouble));
          out = *rig_buffer;
        }
        else
        {
          out = NULL;
          printf("lqr_get_masks unknown mask_type %i\n", mask_type);
        }

        if (out)
        {
          for(int yy = fts + 1; yy < fts + fhs - 1; yy++)
          {
            // we test if we are inside roi_out
            if(yy < roi_in->y || yy >= roi_in->y + roi_in->height) continue;
            // we test if the source point is inside roi_in
            if(yy - dy < roi_in->y || yy - dy >= roi_in->y + roi_in->height) continue;

            for(int xx = fls + 1; xx < fls + fws - 1; xx++)
            {
              // we test if we are inside roi_in
              if(xx < roi_in->x || xx >= roi_in->x + roi_in->width) continue;
              // we test if the source point is inside roi_in
              if(xx - dx < roi_in->x || xx - dx >= roi_in->x + roi_in->width) continue;

              const float f = (mask[((int)((yy - fts) / roi_in->scale)) * mask_width
                                    + (int)((xx - fls) / roi_in->scale)]);

              out[((size_t)roi_in->width * (yy - roi_in->y) + xx - roi_in->x)] = f;
            }
          }
        }     
      }

      free(mask);

      forms = g_list_next(forms);
    }

    if(darktable.unmuted & DT_DEBUG_PERF) printf("lqr_get_masks took %0.04f sec\n", dt_get_wtime() - start);
  }

}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
    void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_liqrescale_data_t *d = (dt_iop_liqrescale_data_t *)piece->data;
  const dt_iop_colorspace_type_t cst = dt_iop_module_colorspace(self);

  // just to be sure...
  if (cst != iop_cs_rgb) return;

  gdouble *pres_buffer = NULL;
  gdouble *disc_buffer = NULL;
  gdouble *rig_buffer = NULL;

  const int ch = piece->colors;

  // when editing masks the image is not resized
  if (lqr_masks_being_displayed(self))
  {
    float rgb[3];

    lqr_get_masks(self, piece, roi_in, roi_out, &pres_buffer, &disc_buffer, &rig_buffer);

    lqr_copy_in_to_out((float*)ivoid, roi_in, (float*)ovoid, roi_out, ch);

    if (pres_buffer)
    {
      rgb[0] = 0.f; rgb[1] = 1.f; rgb[2] = 0.f;
      lqr_merge_mask_to_out(pres_buffer, roi_in, (float *)ovoid, roi_out, ch, rgb);
    }
    if (disc_buffer)
    {
      rgb[0] = 1.f; rgb[1] = 0.f; rgb[2] = 0.f;
      lqr_merge_mask_to_out(disc_buffer, roi_in, (float *)ovoid, roi_out, ch, rgb);
    }
    if (rig_buffer)
    {
      rgb[0] = 0.f; rgb[1] = 0.f; rgb[2] = 1.f;
      lqr_merge_mask_to_out(rig_buffer, roi_in, (float *)ovoid, roi_out, ch, rgb);
    }

    if (pres_buffer) dt_free_align(pres_buffer);
    if (disc_buffer) dt_free_align(disc_buffer);
    if (rig_buffer) dt_free_align(rig_buffer);

    return;
  }

  const int old_width = roi_in->width;
  const int old_height = roi_in->height;
  // scale has been done in modify_roi_out()
  // now deal with the size we get
  const int new_width = roi_out->width;
  const int new_height = roi_out->height;

  LqrResizeOrder res_order = (d->vertical_first) ? LQR_RES_ORDER_VERT: LQR_RES_ORDER_HOR;

  float rigidity = d->rigidity;
  int delta_x = d->delta_x;

  // create new buffer for the carver object
  gfloat *rgb_buffer = g_try_new(gfloat, 3 * old_width * old_height);

  // fill it with input image
  for (int y = 0; y < roi_in->height; y++)
  {
    float *in = ((float*)ivoid) + y * roi_in->width * ch;

    for (int x = 0; x < roi_in->width; x++, in += ch)
    {
      for (int k = 0; k < 3; k++) rgb_buffer[(y * old_width + x) * 3 + k] = in[k];
    }
  }

  // swallow the buffer in a (minimal) LqrCarver object (arguments are width, height and number of colour channels)
  LqrCarver *carver;
  carver = lqr_carver_new_ext(rgb_buffer, old_width, old_height, 3, LQR_COLDEPTH_32F);

  // initialize the carver (with default values), so that we can do the resizing
  lqr_carver_init(carver, delta_x, rigidity);

  // add the bias (positive to preserve, negative to discard) and the rigidity mask
  lqr_get_masks(self, piece, roi_in, roi_out, &pres_buffer, &disc_buffer, &rig_buffer);

  if (pres_buffer) {
    lqr_carver_bias_add(carver, pres_buffer, d->preserve_mask_strength);
  }

  if (disc_buffer) {
    lqr_carver_bias_add(carver, disc_buffer, -d->discard_mask_strength);
  }

  if (rig_buffer) {
    lqr_carver_rigmask_add(carver, rig_buffer);
  }

  // set the energy function
  lqr_set_energy(carver, d->energy_function);

  // set the side switch frequency
  lqr_carver_set_side_switch_frequency(carver, d->side_switch_frequency);

  // set the enlargement step
  lqr_carver_set_enl_step(carver, d->enlarge_step);

  // set the rescaling order
  lqr_carver_set_resize_order(carver, res_order);

  // actual liquid rescale
  lqr_carver_resize(carver, new_width, new_height);

  // copy resized image to out
  lqr_write_carver_to_image(carver, (float*)ovoid, roi_out, ch);

  // destroy the carver object
  lqr_carver_destroy(carver);

  if (pres_buffer) dt_free_align(pres_buffer);
  if (disc_buffer) dt_free_align(disc_buffer);
  if (rig_buffer) dt_free_align(rig_buffer);
}

} // extern "C"


// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
