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
#include "gui/accelerators.h"
#include "develop/imageop_math.h"
#include <stdlib.h>
}

#include <gmic_qt_lib.h>

extern "C" {

DT_MODULE_INTROSPECTION(1, dt_iop_dt_gmic_lab_params_t)


typedef enum dt_iop_gmic_control_type_t
{
	dt_iop_gmic_control_sl = 0,
	dt_iop_gmic_control_cmb = 1,
	dt_iop_gmic_control_lbl = 2,
	dt_iop_gmic_control_btn = 3,
	dt_iop_gmic_control_colorbtn = 4,
	dt_iop_gmic_control_sep = 5,
	dt_iop_gmic_control_txt = 6,
	dt_iop_gmic_control_link = 7,
	dt_iop_gmic_control_box = 8,
	dt_iop_gmic_control_chk = 9,
} dt_iop_gmic_control_type_t;

typedef enum dt_iop_dt_gmic_lab_pixelpipepos_t
{
	dt_iop_gmic_pipepos_rgb = 0,
	dt_iop_gmic_pipepos_lab = 1,
	dt_iop_gmic_pipepos_linear_rgb = 2,
} dt_iop_dt_gmic_lab_pixelpipepos_t;

typedef struct dt_iop_dt_gmic_lab_params_t
{
	int num;
	int keep_image_loaded;
  char filter_data[451];
  int filter_data_size;
} dt_iop_dt_gmic_lab_params_t;

typedef struct dt_iop_dt_gmic_lab_gui_data_t
{
	int image_requested;
	int image_locked;
	
	float *image;
	int width;
	int height;
	int ch;
	float imageScale;
	
	GtkLabel *label_name;
	GtkLabel *label_command;
	GtkLabel *label_preview_command;
	
	GtkButton *bt_call_gmic;
	GtkWidget *cmb_keep_image_loaded;
	
	GtkWidget *vbox_gmic;
	
	GtkWidget **actual_params_widg;
	int actual_params_size;
	
	GtkWidget **box_widget;
	int box_size;
	
	GtkWidget **sl_widget;
	int sl_size;
	
	GtkWidget **cmb_widget;
	int cmb_size;
	
	GtkWidget **lbl_widget;
	int lbl_size;
	
	GtkWidget **btn_widget;
	int btn_size;
	
	GtkWidget **colorbtn_widget;
	int colorbtn_size;
	
	GtkWidget **sep_widget;
	int sep_size;
	
	GtkWidget **txt_widget;
	int txt_size;
	
	GtkWidget **link_widget;
	int link_size;
	
	GtkWidget **chk_widget;
	int chk_size;
	
	dt_pthread_mutex_t lock;
} dt_iop_dt_gmic_lab_gui_data_t;

typedef struct dt_iop_dt_gmic_lab_params_t dt_iop_dt_gmic_lab_data_t;

typedef struct dt_iop_dt_gmic_lab_global_data_t
{
	gmic_qt_lib_t * gmic_gd;
} dt_iop_dt_gmic_lab_global_data_t;


const char *name()
{
  return _("dt_gmic_lab_eh");
}

int groups()
{
  return IOP_GROUP_BASIC;
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING;
}

static void ellipsize_button(GtkWidget *button)
{
	gtk_label_set_ellipsize(GTK_LABEL(gtk_bin_get_child(GTK_BIN(button))), PANGO_ELLIPSIZE_MIDDLE);
}

static void update_param_value_from_gui(gmic_filter_param_t *params, GtkWidget *widg)
{
	
	if (params->param_type == gmic_param_int || params->param_type == gmic_param_float)
	{
		params->n_current_value = (float)dt_bauhaus_slider_get(widg);
	}
	else if (params->param_type == gmic_param_bool)
	{
		params->n_current_value = (float)gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widg));
	}
	else if (params->param_type == gmic_param_choise)
	{
		params->n_current_value = (float)dt_bauhaus_combobox_get(widg);
	}
	else if (params->param_type == gmic_param_color)
	{
		GdkRGBA color;
		gtk_color_chooser_get_rgba(GTK_COLOR_CHOOSER(widg), &color);
		
		params->rgb_current_value[0] = color.red * 255.f;
		params->rgb_current_value[1] = color.green * 255.f;
		params->rgb_current_value[2] = color.blue * 255.f;
		params->rgb_current_value[3] = color.alpha;
	}
	else if (params->param_type == gmic_param_separator)
	{
	}
	else if (params->param_type == gmic_param_note || params->param_type == gmic_param_link)
	{
	}
	else if (params->param_type == gmic_param_file || params->param_type == gmic_param_folder)
	{
	  GtkWidget *win = dt_ui_main_window(darktable.gui->ui);
	  GtkWidget *filechooser;
	  if (params->param_type == gmic_param_folder)
	  {
			filechooser = gtk_file_chooser_dialog_new(
					_("select directory"), GTK_WINDOW(win), GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, _("_cancel"),
					GTK_RESPONSE_CANCEL, _("_select"), GTK_RESPONSE_ACCEPT, (char *)NULL);
	  }
	  else
	  {
			filechooser = gtk_file_chooser_dialog_new(
					_("select file"), GTK_WINDOW(win), GTK_FILE_CHOOSER_ACTION_OPEN, _("_cancel"),
					GTK_RESPONSE_CANCEL, _("_select"), GTK_RESPONSE_ACCEPT, (char *)NULL);
	  }
	  
	  gtk_file_chooser_set_select_multiple(GTK_FILE_CHOOSER(filechooser), FALSE);
	  gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(filechooser), g_get_home_dir());
	  if(gtk_dialog_run(GTK_DIALOG(filechooser)) == GTK_RESPONSE_ACCEPT)
	  {
	    gchar *dir = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(filechooser));
	    
			if (params->str_current_value) free(params->str_current_value);
			params->str_current_value = (char*)calloc(1, strlen(dir) + 1);
			strcpy(params->str_current_value, dir);

			gtk_button_set_label(GTK_BUTTON(widg), params->str_current_value);
			ellipsize_button(widg);
			
	    g_free(dir);
	  }
	  gtk_widget_destroy(filechooser);
	}
	else if (params->param_type == gmic_param_text)
	{
		const char * text = gtk_entry_get_text(GTK_ENTRY(widg));
		
		if (params->str_current_value) free(params->str_current_value);
		params->str_current_value = (char*)calloc(1, strlen(text) + 1);
		sprintf(params->str_current_value, "\"%s\"", text);
	}
	else if (params->param_type == gmic_param_button)
	{
		params->n_current_value = 0.f;
	}

}

static void update_output_params_from_gui(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_dt_gmic_lab_gui_data_t *g = (dt_iop_dt_gmic_lab_gui_data_t *)self->gui_data;
  dt_iop_dt_gmic_lab_params_t *p = (dt_iop_dt_gmic_lab_params_t *)self->params;
  dt_iop_dt_gmic_lab_global_data_t *gd = (dt_iop_dt_gmic_lab_global_data_t *)self->data;
  
  if (g == NULL || p == NULL || gd == NULL) return;
  
  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  gmic_filter_execution_data_t * filter_data = NULL;
  
  dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
	gmic_filter_definition_t *filter_definition = gmic_getFilterDefinitionFromFilterData(gd->gmic_gd, p->filter_data, p->filter_data_size);
	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
	if (filter_definition)
	{
		// do we have the same definition?
		if (filter_definition->num_params == g->actual_params_size)
		{
			if (filter_definition->params)
			{
				int widg_index = -1;
				for (int i = 0; i < g->actual_params_size; i++)
				{
					if (g->actual_params_widg[i] == widget)
					{
						widg_index = i;
						break;
					}
				}
				
				if (widg_index >= 0)
				{
					update_param_value_from_gui(filter_definition->params+widg_index, g->actual_params_widg[widg_index]);
				}
				
				dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
				filter_data = gmic_getFilterExecDataFromFilteDefinition(filter_definition);
				dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
			}
		}
		else
		{
			printf("update_output_params_from_gui() different params size\n");
		}
		
		dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
		gmic_freeDefinition(filter_definition);
		dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
	}
	
	if (filter_data)
	{
		if (filter_data->filter_data_size < (int)sizeof(p->filter_data))
		{
			memcpy(p->filter_data, filter_data->filter_data, sizeof(p->filter_data));
			p->filter_data_size = filter_data->filter_data_size;
		}
		
		dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
		gmic_freeFilterExecData(filter_data);
		dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
	}
	
  darktable.gui->reset = reset;

}

static void sl_widget_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;

  update_output_params_from_gui(slider, self);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void cmb_widget_callback(GtkComboBox *combo, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;
  
  update_output_params_from_gui(GTK_WIDGET(combo), self);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void btn_widget_callback(GtkWidget *widget, dt_iop_module_t *self)
{
	if(self->dt->gui->reset) return;
	
	update_output_params_from_gui(widget, self);
	
	dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void colorbtn_widget_callback(GtkColorButton *widget, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;

  update_output_params_from_gui(GTK_WIDGET(widget), self);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void txt_widget_callback(GtkWidget *entry, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;
  
  update_output_params_from_gui(entry, self);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void chk_widget_callback(GtkWidget *widget, dt_iop_module_t *self)
{
	if(self->dt->gui->reset) return;
	
	update_output_params_from_gui(widget, self);
	
	dt_dev_add_history_item(darktable.develop, self, TRUE);
}


// return the index of the first empty entry on a widget array
// alloc/realloc memory if no room left
static int get_single_control_index(GtkWidget ***widg_array, int *array_size)
{
	int index = -1;
	GtkWidget **w_array = *widg_array;
	int arr_size = *array_size;
	
	for (int i = 0; i < arr_size; i++)
	{
		if (w_array[i] == NULL)
		{
			index = i;
			break;
		}
	}
	
	if (index == -1)
	{
		const int inc = 50;
		void *ptr_tmp = realloc(w_array, (arr_size + inc)*sizeof(GtkWidget *));
		if (ptr_tmp)
		{
			w_array = (GtkWidget **)ptr_tmp;
			for (int i = arr_size; i < arr_size + inc; i++) w_array[i] = NULL;
			*widg_array = w_array;
			index = arr_size;
			arr_size += inc;
			*array_size = arr_size;
		}
	}

	return index;
}

static int get_control_index(dt_iop_dt_gmic_lab_gui_data_t *g, int control_type)
{
	int index = -1;
	if (control_type == dt_iop_gmic_control_sl)
	{
		index = get_single_control_index(&g->sl_widget, &g->sl_size);
	}
	else if (control_type == dt_iop_gmic_control_cmb)
	{
		index = get_single_control_index(&g->cmb_widget, &g->cmb_size);
	}
	else if (control_type == dt_iop_gmic_control_lbl)
	{
		index = get_single_control_index(&g->lbl_widget, &g->lbl_size);
	}
	else if (control_type == dt_iop_gmic_control_btn)
	{
		index = get_single_control_index(&g->btn_widget, &g->btn_size);
	}
	else if (control_type == dt_iop_gmic_control_colorbtn)
	{
		index = get_single_control_index(&g->colorbtn_widget, &g->colorbtn_size);
	}
	else if (control_type == dt_iop_gmic_control_sep)
	{
		index = get_single_control_index(&g->sep_widget, &g->sep_size);
	}
	else if (control_type == dt_iop_gmic_control_txt)
	{
		index = get_single_control_index(&g->txt_widget, &g->txt_size);
	}
	else if (control_type == dt_iop_gmic_control_link)
	{
		index = get_single_control_index(&g->link_widget, &g->link_size);
	}
	else if (control_type == dt_iop_gmic_control_box)
	{
		index = get_single_control_index(&g->box_widget, &g->box_size);
	}
	else if (control_type == dt_iop_gmic_control_chk)
	{
		index = get_single_control_index(&g->chk_widget, &g->chk_size);
	}

	return index;
}

static void destroy_single_widget_array(GtkWidget **widg_arr, const int arr_size)
{
	if (widg_arr)
	{
		for (int i = 0; i < arr_size; i++)
		{
			if (widg_arr[i])
			{
				gtk_widget_destroy(widg_arr[i]);
				widg_arr[i] = NULL;
			}
		}
	}

}

static void destroy_gui_controls(dt_iop_module_t *self)
{
	dt_iop_dt_gmic_lab_gui_data_t *g = (dt_iop_dt_gmic_lab_gui_data_t *)self->gui_data;
	if (g == NULL) return;
	
	if (g->actual_params_widg)
	{
		for (int i = 0; i < g->actual_params_size; i++) 
			g->actual_params_widg[i] = NULL;
	}

	destroy_single_widget_array(g->sl_widget, g->sl_size);
	destroy_single_widget_array(g->cmb_widget, g->cmb_size);
	destroy_single_widget_array(g->lbl_widget, g->lbl_size);
	destroy_single_widget_array(g->btn_widget, g->btn_size);
	destroy_single_widget_array(g->colorbtn_widget, g->colorbtn_size);
	destroy_single_widget_array(g->sep_widget, g->sep_size);
	destroy_single_widget_array(g->txt_widget, g->txt_size);
	destroy_single_widget_array(g->link_widget, g->link_size);
	destroy_single_widget_array(g->box_widget, g->box_size);
	destroy_single_widget_array(g->chk_widget, g->chk_size);
	
}

static void update_filter_labels(gmic_filter_definition_t *filter_definition, dt_iop_dt_gmic_lab_gui_data_t *g)
{
	const char *blank = "";
	const char *filter_name = blank;
	const char *filter_command = blank;
	const char *filter_preview_command = blank;
	
	if (filter_definition)
	{
		if (filter_definition->filter_name) filter_name = filter_definition->filter_name;
		if (filter_definition->filter_command) filter_command = filter_definition->filter_command;
		if (filter_definition->filter_preview_command) filter_preview_command = filter_definition->filter_preview_command;
	}

	gtk_label_set_text(g->label_name, filter_name);
	gtk_label_set_use_markup(GTK_LABEL(g->label_name), true);

	gtk_label_set_text(g->label_command, filter_command);

	gtk_label_set_text(g->label_preview_command, filter_preview_command);

}

// update values on already created gui controls from values in dt_iop_dt_gmic_lab_params_t->filter_data
static void update_gui_controls(dt_iop_module_t *self)
{
  dt_iop_dt_gmic_lab_gui_data_t *g = (dt_iop_dt_gmic_lab_gui_data_t *)self->gui_data;
  dt_iop_dt_gmic_lab_params_t *p = (dt_iop_dt_gmic_lab_params_t *)self->params;
  dt_iop_dt_gmic_lab_global_data_t *gd = (dt_iop_dt_gmic_lab_global_data_t *)self->data;
  
  if (g == NULL || p == NULL || gd == NULL) return;
  if (g->actual_params_widg == NULL) return;
  
  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  // get g'mic filter definition
  dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
	gmic_filter_definition_t *filter_definition = gmic_getFilterDefinitionFromFilterData(gd->gmic_gd, p->filter_data, p->filter_data_size);
	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
	if (filter_definition)
	{
		update_filter_labels(filter_definition, g);
		
		if (filter_definition->params)
		{
			// go through all parameters and set the new value
			gmic_filter_param_t *params = filter_definition->params;
			for (int num_p = 0; num_p < filter_definition->num_params; num_p++, params++)
			{
				GtkWidget *widg = g->actual_params_widg[num_p];
				
				if (widg == NULL)
				{
					continue;
				}
				
				if (params->param_type == gmic_param_int || params->param_type == gmic_param_float)
				{
						dt_bauhaus_slider_set(widg, params->n_current_value);
				}
				else if (params->param_type == gmic_param_bool)
				{
						gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widg), params->n_current_value);
				}
				else if (params->param_type == gmic_param_choise)
				{
						dt_bauhaus_combobox_set(widg, params->n_current_value);
				}
				else if (params->param_type == gmic_param_color)
				{
					  GdkRGBA color = (GdkRGBA){.red = params->rgb_current_value[0] / 255.f, 
					    .green = params->rgb_current_value[1] / 255.f, 
					    .blue = params->rgb_current_value[2] / 255.f, 
					    .alpha = 1.0 };
					  gtk_color_chooser_set_rgba(GTK_COLOR_CHOOSER(widg), &color);

				}
				else if (params->param_type == gmic_param_separator)
				{
				}
				else if (params->param_type == gmic_param_note || params->param_type == gmic_param_link)
				{
				}
				else if (params->param_type == gmic_param_text)
				{
				  char *str = (params->str_current_value) ? params->str_current_value: params->str_default_value;
				  
				  if (str)
				  {
				  	if (str[0] == '\"') str++;
				  	int len = strlen(str);
				  	if (str[len-1] =='\"') str[len-1] = 0;
				  }
				  gtk_entry_set_text(GTK_ENTRY(widg), str);
				}
				else if (params->param_type == gmic_param_button || params->param_type == gmic_param_file || params->param_type == gmic_param_folder)
				{
					char *str = (params->str_current_value) ? params->str_current_value: 
																	(params->str_default_value) ? params->str_default_value: params->param_name;
					
					gtk_button_set_label(GTK_BUTTON(widg), str);
					
					ellipsize_button(widg);
				}
			}
		}

		dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
		gmic_freeDefinition(filter_definition);
		dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
	}
	else
	{
		update_filter_labels(NULL, g);
	}
	
  darktable.gui->reset = reset;
  
}

// You must free the result if result is non-NULL.
static char *str_replace(char *orig, const char *rep, const char *with)
{
	char *result; // the return string
	char *ins;    // the next insert point
	char *tmp;    // varies
	int len_rep;  // length of rep (the string to remove)
	int len_with; // length of with (the string to replace rep with)
	int len_front; // distance between rep and end of last rep
	int count;    // number of replacements

	// sanity checks and initialization
	if (!orig || !rep)
		return NULL;
	len_rep = strlen(rep);
	if (len_rep == 0)
		return NULL; // empty rep causes infinite loop during count
	if (!with)
		with = "";
	len_with = strlen(with);

	// count the number of replacements needed
	ins = orig;
	for (count = 0; (tmp = strstr(ins, rep)); ++count)
	{
		ins = tmp + len_rep;
	}

	tmp = result = (char*)malloc(strlen(orig) + (len_with - len_rep) * count + 1);

	if (!result)
		return NULL;

	// first time through the loop, all the variable are set correctly
	// from here on,
	//    tmp points to the end of the result string
	//    ins points to the next occurrence of rep in orig
	//    orig points to the remainder of orig after "end of rep"
	while (count--)
	{
		ins = strstr(orig, rep);
		len_front = ins - orig;
		tmp = strncpy(tmp, orig, len_front) + len_front;
		tmp = strcpy(tmp, with) + len_with;
		orig += len_front + len_rep; // move to next "end of rep"
	}
	strcpy(tmp, orig);
	return result;
}

// return true if command changed or there's no command
static int has_gmic_command_changed(dt_iop_dt_gmic_lab_gui_data_t *g, dt_iop_dt_gmic_lab_params_t *p, dt_iop_dt_gmic_lab_global_data_t *gd)
{
	int changed = 1;
	
  if (g->actual_params_widg != NULL && p->filter_data_size > 0)
  {
  	if (g->actual_params_widg[0] != NULL)
  	{
  		gmic_filter_definition_t *filter_definition = NULL;
  	  
			dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
			filter_definition = gmic_getFilterDefinitionFromFilterData(gd->gmic_gd, p->filter_data, p->filter_data_size);
			dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);

			if (filter_definition)
			{
				if (filter_definition->filter_name && filter_definition->filter_command && filter_definition->filter_preview_command)
				{
					changed = strcmp(filter_definition->filter_name, gtk_label_get_text(g->label_name)) != 0 ||
							strcmp(filter_definition->filter_command, gtk_label_get_text(g->label_command)) != 0 ||
							strcmp(filter_definition->filter_preview_command, gtk_label_get_text(g->label_preview_command)) != 0;
				}
				
				dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
				gmic_freeDefinition(filter_definition);
				dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
			}
  	}
  }
  
	return changed;
}

// create/update gui controls from the g'mic command in dt_iop_dt_gmic_lab_params_t->filter_data
static void create_gui_controls(dt_iop_module_t *self)
{
  dt_iop_dt_gmic_lab_gui_data_t *g = (dt_iop_dt_gmic_lab_gui_data_t *)self->gui_data;
  dt_iop_dt_gmic_lab_params_t *p = (dt_iop_dt_gmic_lab_params_t *)self->params;
  dt_iop_dt_gmic_lab_global_data_t *gd = (dt_iop_dt_gmic_lab_global_data_t *)self->data;
  
  if (g == NULL || p == NULL || gd == NULL) return;
  
  // if command not changed controls are already created, so just update values
/*  if (g->actual_params_widg != NULL && p->filter_data_size > 0)
	{
		if (g->actual_params_widg[0] != NULL)
		{
			update_gui_controls(self);
			return;
		}
	}
*/
	if (!has_gmic_command_changed(g, p, gd))
	{
		update_gui_controls(self);
		return;
	}

  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  // new command, destroy old controls before create new ones
	destroy_gui_controls(self);
	
	gmic_filter_definition_t *filter_definition = NULL;
	
	// get filter definition from g'mic
	if (p->filter_data_size > 0)
	{
		dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
		filter_definition = gmic_getFilterDefinitionFromFilterData(gd->gmic_gd, p->filter_data, p->filter_data_size);
		dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
	}
	if (filter_definition)
	{
		update_filter_labels(filter_definition, g);
		
		// init array that hold widgets on the same order as the filter definition
		g->actual_params_size = filter_definition->num_params;
		g->actual_params_widg = (GtkWidget **)realloc(g->actual_params_widg, (g->actual_params_size)*sizeof(GtkWidget *));
		for (int i = 0; i < g->actual_params_size; i++) g->actual_params_widg[i] = NULL;

		// create gui controls
		if (filter_definition->params)
		{
			GtkWidget *lbl_markup = gtk_label_new(" ");
			
			// go through all parameters 
			gmic_filter_param_t *params = filter_definition->params;
			for (int num_p = 0; num_p < filter_definition->num_params; num_p++, params++)
			{
				GtkWidget *widg = NULL;
				
				// create a different widget depending on parametes type
				if (params->param_type == gmic_param_int || params->param_type == gmic_param_float)
				{
					float sl_increment = (params->param_type == gmic_param_float) ? ((params->n_max_value-params->n_min_value)/100.f): 1.f;
					int sl_num_decimals = (params->param_type == gmic_param_float) ? 2: 0;
					
					int w_index = get_control_index(g, dt_iop_gmic_control_sl);
					if (w_index >= 0)
					{
						g->sl_widget[w_index] = 
								dt_bauhaus_slider_new_with_range(self, params->n_min_value, params->n_max_value, sl_increment, params->n_default_value, sl_num_decimals);
						
						widg = g->sl_widget[w_index];
						
						dt_bauhaus_widget_set_label(widg, params->param_name, params->param_name);
						g_signal_connect(G_OBJECT(widg), "value-changed", G_CALLBACK(sl_widget_callback), self);
	
						gtk_box_pack_start(GTK_BOX(g->vbox_gmic), widg, TRUE, TRUE, 0);
						
						dt_bauhaus_slider_set(widg, params->n_current_value);
						
						gtk_widget_show(GTK_WIDGET(widg));
					}
				  
				}
				else if (params->param_type == gmic_param_bool)
				{
					int w_index = get_control_index(g, dt_iop_gmic_control_chk);
					if (w_index >= 0)
					{
						g->chk_widget[w_index] = gtk_check_button_new_with_label(params->param_name);
						widg = g->chk_widget[w_index];
						
						g_signal_connect(G_OBJECT(widg), "toggled", G_CALLBACK(chk_widget_callback), self);
						gtk_box_pack_start(GTK_BOX(g->vbox_gmic), widg, TRUE, TRUE, 0);
						
						gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widg), params->n_current_value);
						
						gtk_widget_show(GTK_WIDGET(widg));
					}
				}
				else if (params->param_type == gmic_param_choise)
				{
					int w_index = get_control_index(g, dt_iop_gmic_control_cmb);
					if (w_index >= 0)
					{
						g->cmb_widget[w_index] = dt_bauhaus_combobox_new(self);
						widg = g->cmb_widget[w_index];
						
						dt_bauhaus_widget_set_label(widg, NULL, params->param_name);
						
						char *str_choise = params->str_default_value;
						for (int i = 0; i < params->list_size; i++)
						{
							gtk_label_set_text(GTK_LABEL(lbl_markup), str_choise);
							gtk_label_set_use_markup(GTK_LABEL(lbl_markup), true);
							
							dt_bauhaus_combobox_add(widg, gtk_label_get_text(GTK_LABEL(lbl_markup)));
							str_choise += strlen(str_choise) + 1;
						}
						
						g_signal_connect(G_OBJECT(widg), "value-changed", G_CALLBACK(cmb_widget_callback), self);
						gtk_box_pack_start(GTK_BOX(g->vbox_gmic), widg, TRUE, TRUE, 0);
						
						dt_bauhaus_combobox_set(widg, params->n_current_value);
						
						gtk_widget_show(GTK_WIDGET(widg));
					}
				}
				else if (params->param_type == gmic_param_color)
				{
					int w_index = get_control_index(g, dt_iop_gmic_control_colorbtn);
					int w_box_index = get_control_index(g, dt_iop_gmic_control_box);
					int w_lbl_index = get_control_index(g, dt_iop_gmic_control_lbl);
					if (w_index >= 0)
					{
						const int bs = DT_PIXEL_APPLY_DPI(14);
						
					  GdkRGBA color = (GdkRGBA){.red = params->rgb_current_value[0] / 255.f, 
					    .green = params->rgb_current_value[1] / 255.f, 
					    .blue = params->rgb_current_value[2] / 255.f, 
					    .alpha = 1.0 };

						g->colorbtn_widget[w_index] = gtk_color_button_new_with_rgba(&color);
						widg = g->colorbtn_widget[w_index];
						
						g->box_widget[w_box_index] = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
						GtkWidget *box_widg = g->box_widget[w_box_index];
						
						g->lbl_widget[w_lbl_index] = gtk_label_new(params->param_name);
						GtkWidget *lbl_widg = g->lbl_widget[w_lbl_index];
						
						gtk_color_chooser_set_use_alpha(GTK_COLOR_CHOOSER(widg), FALSE);
						gtk_widget_set_size_request(GTK_WIDGET(widg), bs, bs);
						
						g_signal_connect(G_OBJECT(widg), "color-set", G_CALLBACK(colorbtn_widget_callback), self);
						
						gtk_box_pack_end(GTK_BOX(box_widg), widg, FALSE, TRUE, 0);
						gtk_box_pack_start(GTK_BOX(box_widg), lbl_widg, FALSE, TRUE, 0);
						gtk_box_pack_start(GTK_BOX(g->vbox_gmic), box_widg, FALSE, TRUE, 0);
						gtk_widget_show(GTK_WIDGET(box_widg));
						gtk_widget_show(GTK_WIDGET(lbl_widg));
						gtk_widget_show(GTK_WIDGET(widg));
					}
				}
				else if (params->param_type == gmic_param_separator)
				{
					int w_index = get_control_index(g, dt_iop_gmic_control_sep);
					if (w_index >= 0)
					{
						g->sep_widget[w_index] = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
						widg = g->sep_widget[w_index];
						
					  gtk_widget_set_margin_start(widg, DT_PIXEL_APPLY_DPI(30)); // gtk+ css doesn't support margins :(
					  gtk_widget_set_name(widg, "section_label"); // make sure that we can style these easily

						gtk_box_pack_start(GTK_BOX(g->vbox_gmic), widg, TRUE, TRUE, 0);
						gtk_widget_show(GTK_WIDGET(widg));
					}
				}
				else if (params->param_type == gmic_param_note)
				{
					int w_index = get_control_index(g, dt_iop_gmic_control_lbl);
					if (w_index >= 0)
					{
						char *new_value = str_replace(params->str_default_value, "color:orangered", "normal");
						char *new_value2 = NULL;
						
						if (new_value)
							new_value2 = str_replace(new_value, "<br/>", "");
						else
							new_value2 = str_replace(params->str_default_value, "<br/>", "");
						
						if (new_value) free(new_value);
						new_value = new_value2;
						
						if (new_value)
							g->lbl_widget[w_index] = gtk_label_new(new_value);
						else 
							g->lbl_widget[w_index] = gtk_label_new(params->str_default_value);
						widg = g->lbl_widget[w_index];
						
						gtk_label_set_use_markup(GTK_LABEL(widg), true);
						gtk_label_set_single_line_mode(GTK_LABEL(widg), false);
						gtk_label_set_line_wrap(GTK_LABEL(widg), true);
						gtk_label_set_line_wrap_mode(GTK_LABEL(widg), PANGO_WRAP_WORD_CHAR);
						gtk_label_set_xalign(GTK_LABEL(widg), 0);
						
						gtk_box_pack_start(GTK_BOX(g->vbox_gmic), widg, TRUE, TRUE, 0);
						gtk_widget_show(GTK_WIDGET(widg));
						
						if (new_value) free(new_value);
					}
				}
				else if (params->param_type == gmic_param_link)
				{
					int w_index = get_control_index(g, dt_iop_gmic_control_link);
					if (w_index >= 0)
					{
						g->link_widget[w_index] = gtk_link_button_new_with_label(params->str_url, params->str_default_value);
						widg = g->link_widget[w_index];
						gtk_widget_set_halign(widg, GTK_ALIGN_START);
						
						gtk_box_pack_start(GTK_BOX(g->vbox_gmic), widg, TRUE, TRUE, 0);
						gtk_widget_show(GTK_WIDGET(widg));
					}
				}
				else if (params->param_type == gmic_param_text)
				{
					int w_index = get_control_index(g, dt_iop_gmic_control_txt);
					if (w_index >= 0)
					{
						g->txt_widget[w_index] = gtk_entry_new();
						widg = g->txt_widget[w_index];
					  gtk_entry_set_width_chars(GTK_ENTRY(widg), 1);
					  dt_gui_key_accel_block_on_focus_connect(widg);

					  char *str = (params->str_current_value) ? params->str_current_value: params->str_default_value;
					  
					  if (str)
					  {
					  	if (str[0] == '\"') str++;
					  	int len = strlen(str);
					  	if (str[len-1] =='\"') str[len-1] = 0;
					  }
					  gtk_entry_set_text(GTK_ENTRY(widg), str);
					  
					  g_signal_connect(G_OBJECT(widg), "changed", G_CALLBACK(txt_widget_callback), self);

						gtk_box_pack_start(GTK_BOX(g->vbox_gmic), widg, TRUE, TRUE, 0);
						gtk_widget_show(GTK_WIDGET(widg));
					}
				}
				else if (params->param_type == gmic_param_value)
				{
				}
				else if (params->param_type == gmic_param_button || params->param_type == gmic_param_file || params->param_type == gmic_param_folder)
				{
					int w_index = get_control_index(g, dt_iop_gmic_control_btn);
					if (w_index >= 0)
					{
						char *str = (params->str_current_value) ? params->str_current_value: 
												(params->str_default_value) ? params->str_default_value: params->param_name;
						g->btn_widget[w_index] = gtk_button_new_with_label(str);
						widg = g->btn_widget[w_index];
						
						ellipsize_button(widg);
						
						g_signal_connect(G_OBJECT(widg), "clicked", G_CALLBACK(btn_widget_callback), self);
						gtk_box_pack_start(GTK_BOX(g->vbox_gmic), widg, FALSE, TRUE, 0);
						gtk_widget_show(GTK_WIDGET(widg));
					}
				}
				
				// save the widget in an array with the same order as the filter definition
				if (widg && params->is_actual_parameter)
				{
					g->actual_params_widg[num_p] = widg;
				}
			}
			
			gtk_widget_destroy(lbl_markup);
		}

		dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
		gmic_freeDefinition(filter_definition);
		dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
	}
	else
	{
		update_filter_labels(NULL, g);
	}
	
  darktable.gui->reset = reset;
  
}

static void call_dt_gmic_lab_qt_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_dt_gmic_lab_gui_data_t *g = (dt_iop_dt_gmic_lab_gui_data_t *)self->gui_data;
  dt_iop_dt_gmic_lab_params_t *p = (dt_iop_dt_gmic_lab_params_t *)self->params;
  dt_iop_dt_gmic_lab_global_data_t *gd = (dt_iop_dt_gmic_lab_global_data_t *)self->data;
  
  if (g == NULL || p == NULL || gd == NULL) return;

  if(darktable.gui->reset) return;
  int reset = darktable.gui->reset;
  darktable.gui->reset = 1;

  dt_pthread_mutex_lock(&g->lock);
  if (g->image_locked == 0 && g->image != NULL && g->width > 0 && g->height > 0)
  {
  	g->image_locked = 1;
  	dt_pthread_mutex_unlock(&g->lock);
  	
  	// disable dt window while the plugin is running
  	GtkWidget *mainw = dt_ui_main_window(darktable.gui->ui);
  	gtk_widget_set_sensitive (mainw, FALSE);
  	
  	// call the plugin
  	dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
    gmic_filter_execution_data_t *filter_data = gmic_launchPluginCommand(p->filter_data, p->filter_data_size, g->image, 
    		g->width, g->height, g->ch, g->imageScale);
  	dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
    
    // enable dt window and process the result
    gtk_widget_set_sensitive (mainw, TRUE);
  	if (filter_data)
  	{
  		if (filter_data->return_value == 1)
  		{
  			int data_valid = (filter_data->filter_name != NULL && 
  					filter_data->filter_command != NULL && 
						filter_data->filter_preview_command != NULL && 
						filter_data->filter_params != NULL && 
						filter_data->filter_data != NULL);
  			
  			// check if we can hold the filter
  			if (filter_data->filter_data)
  			{
  				if (filter_data->filter_data_size > (int)sizeof(p->filter_data))
  				{
  					data_valid = 0;
  					const char *text = (filter_data->filter_name) ? filter_data->filter_name: "";
  					
  					dt_control_log(_("G'MIC filter %s not supported"), text);
  					printf("G'MIC filter (%s) params lenght is %i, max supported is %i\n", text, 
  							filter_data->filter_data_size, (int)sizeof(p->filter_data));
  				}
  			}
  			
  			// if returned size is ok, copy values and create/update gui controls
  			if (data_valid)
  			{
  				int new_filter = 1;
  				gmic_filter_definition_t *filter_definition = NULL;
  				
  				if (p->filter_data_size > 0)
  				{
						dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
						filter_definition = gmic_getFilterDefinitionFromFilterData(gd->gmic_gd, p->filter_data, p->filter_data_size);
						dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
  				}
  				
  				if (filter_definition && filter_definition->filter_name && filter_definition->filter_command && filter_definition->filter_preview_command)
  				{
						new_filter = (strcmp(filter_data->filter_name, filter_definition->filter_name) != 0 || 
								strcmp(filter_data->filter_command, filter_definition->filter_command) != 0 || 
								strcmp(filter_data->filter_preview_command, filter_definition->filter_preview_command) != 0);
  				}
  				
					memset(p->filter_data, 0, sizeof(p->filter_data));
					memcpy(p->filter_data, filter_data->filter_data, filter_data->filter_data_size);
					p->filter_data_size = filter_data->filter_data_size;
					
					if (new_filter) destroy_gui_controls(self);
					create_gui_controls(self);
					
					if (filter_definition)
					{
						dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
						gmic_freeDefinition(filter_definition);
						dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
					}
  			}
  		}
  		
  		dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
  		gmic_freeFilterExecData(filter_data);
  		dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
  	}

  	// free image to save some memory
  	dt_pthread_mutex_lock(&g->lock);
  	if (!p->keep_image_loaded)
  	{
			if (g->image) dt_free_align(g->image);
			g->image = NULL;
  	}
  	
  	g->image_locked = 0;
  	
  	dt_pthread_mutex_unlock(&g->lock);
  }
  else
  {
  	// we don't have an image to process, so request one
  	g->image_requested = 1;
  	dt_pthread_mutex_unlock(&g->lock);
  	
  	p->num = (p->num) ? 0: 1; // hack so pipe full is processed
  	dt_dev_invalidate_all(self->dev);
    dt_control_log(_("G'MIC: Image not ready to process"));
  }
  
  darktable.gui->reset = reset;
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
  
  return;
}

static void keep_image_loaded_callback(GtkComboBox *combo, dt_iop_module_t *self)
{
  if(self->dt->gui->reset) return;
  dt_iop_dt_gmic_lab_params_t *p = (dt_iop_dt_gmic_lab_params_t *)self->params;

  p->keep_image_loaded = dt_bauhaus_combobox_get((GtkWidget *)combo);
  
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_dt_gmic_lab_params_t *p = (dt_iop_dt_gmic_lab_params_t *)p1;
  dt_iop_dt_gmic_lab_data_t *d = (dt_iop_dt_gmic_lab_data_t *)piece->data;
  memcpy(d, p, sizeof(dt_iop_dt_gmic_lab_params_t));
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_dt_gmic_lab_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_reset(struct dt_iop_module_t *self)
{
	destroy_gui_controls(self);
}

void gui_update(struct dt_iop_module_t *self)
{
  dt_iop_dt_gmic_lab_gui_data_t *g = (dt_iop_dt_gmic_lab_gui_data_t *)self->gui_data;
  dt_iop_dt_gmic_lab_params_t *p = (dt_iop_dt_gmic_lab_params_t *)self->params;

	create_gui_controls(self);
  
  dt_bauhaus_combobox_set(g->cmb_keep_image_loaded, p->keep_image_loaded);
  
  gtk_widget_show(GTK_WIDGET(g->vbox_gmic));
  
}

void init(dt_iop_module_t *module)
{
  module->params = calloc(1, sizeof(dt_iop_dt_gmic_lab_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_dt_gmic_lab_params_t));
  module->default_enabled = 0;
  module->params_size = sizeof(dt_iop_dt_gmic_lab_params_t);
  module->gui_data = NULL;
  module->priority = 358; // module order created by iop_dependencies.py, do not edit! // from colorreconstruction
//  module->priorityx = 955; // module order created by iop_dependencies.py, do not edit! // from frames
//  module->priorityx = 179; // module order created by iop_dependencies.py, do not edit!
//  module->priorityx = 164; // module order created by iop_dependencies.py, do not edit! // from exposure
//  module->priorityx = 184; // module order created by iop_dependencies.py, do not edit! // from spots
//  module->priorityx = 169; // module order created by iop_dependencies.py, do not edit!
//  module->priorityx = 343; // module order created by iop_dependencies.py, do not edit! // from colorin
//  module->priorityx = 358; // module order created by iop_dependencies.py, do not edit! // from colorreconstruction
//  module->priorityx = 149; // module order created by iop_dependencies.py, do not edit! // from tonemap (before exposure)
//  module->priorityx = 671; // module order created by iop_dependencies.py, do not edit! // from tonecurve

}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

void init_global(dt_iop_module_so_t *module)
{
  dt_iop_dt_gmic_lab_global_data_t *gd = (dt_iop_dt_gmic_lab_global_data_t *)malloc(sizeof(dt_iop_dt_gmic_lab_global_data_t));
  module->data = gd;
  
  dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
  gd->gmic_gd = gmic_qt_init();
  dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
}

void cleanup_global(dt_iop_module_so_t *module)
{
  dt_iop_dt_gmic_lab_global_data_t *gd = (dt_iop_dt_gmic_lab_global_data_t *)module->data;
  
  dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
  gmic_qt_free(gd->gmic_gd);
  dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
  
  free(module->data);
  module->data = NULL;
}


void gui_init(struct dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_dt_gmic_lab_gui_data_t));
  dt_iop_dt_gmic_lab_gui_data_t *g = (dt_iop_dt_gmic_lab_gui_data_t *)self->gui_data;

  dt_pthread_mutex_init(&g->lock, NULL);
  g->image = NULL;
  g->width = 0;
  g->height = 0;
  g->ch = 0;
  g->imageScale = 1.f;
  g->image_requested = 0;
  g->image_locked = 0;
  
  g->actual_params_widg = NULL;
  g->actual_params_size = 0;
  
  g->sl_widget = NULL;
  g->sl_size = 0;

  g->cmb_widget = NULL;
  g->cmb_size = 0;

  g->lbl_widget = NULL;
  g->lbl_size = 0;

  g->btn_widget = NULL;
  g->btn_size = 0;

  g->colorbtn_widget = NULL;
  g->colorbtn_size = 0;

  g->sep_widget = NULL;
  g->sep_size = 0;

  g->txt_widget = NULL;
  g->txt_size = 0;

  g->link_widget = NULL;
  g->link_size = 0;

  g->box_widget = NULL;
  g->box_size = 0;

  g->chk_widget = NULL;
  g->chk_size = 0;

	
  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  GtkWidget *widget = gtk_button_new_with_label(_("call G'MIC Qt plugin"));
  g->bt_call_gmic = GTK_BUTTON(widget);
  g_signal_connect(G_OBJECT(widget), "clicked", G_CALLBACK(call_dt_gmic_lab_qt_callback), self);
  gtk_box_pack_start(GTK_BOX(self->widget), widget, FALSE, TRUE, 0);

  GtkWidget *hbox_lbl_filter = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
  GtkWidget *label_filter = gtk_label_new(_("G'MIC filter:"));
  g->label_name = GTK_LABEL(gtk_label_new(""));
  gtk_box_pack_start(GTK_BOX(hbox_lbl_filter), GTK_WIDGET(label_filter), FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(hbox_lbl_filter), GTK_WIDGET(g->label_name), FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), hbox_lbl_filter, FALSE, TRUE, 0);

  g->vbox_gmic = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  gtk_box_pack_start(GTK_BOX(self->widget), g->vbox_gmic, TRUE, TRUE, 0);
  
  g->cmb_keep_image_loaded = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->cmb_keep_image_loaded, NULL, _("keep image loaded"));
  dt_bauhaus_combobox_add(g->cmb_keep_image_loaded, _("no"));
  dt_bauhaus_combobox_add(g->cmb_keep_image_loaded, _("yes"));
  g_object_set(g->cmb_keep_image_loaded, "tooltip-text", _("keeps the image used by G'MIC loaded in memory."), (char *)NULL);
  g_signal_connect(G_OBJECT(g->cmb_keep_image_loaded), "value-changed", G_CALLBACK(keep_image_loaded_callback), self);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->cmb_keep_image_loaded), TRUE, TRUE, 0);
  
  g->label_command = GTK_LABEL(gtk_label_new(""));
  g->label_preview_command = GTK_LABEL(gtk_label_new(""));

  GtkWidget *widg;
  
  widg = GTK_WIDGET(g->label_name);
	gtk_label_set_single_line_mode(GTK_LABEL(widg), false);
	gtk_label_set_line_wrap(GTK_LABEL(widg), true);
	gtk_label_set_line_wrap_mode(GTK_LABEL(widg), PANGO_WRAP_WORD_CHAR);
	gtk_label_set_xalign(GTK_LABEL(widg), 0);
  gtk_label_set_use_markup(GTK_LABEL(widg), true);

  widg = GTK_WIDGET(g->label_command);
	gtk_label_set_single_line_mode(GTK_LABEL(widg), false);
	gtk_label_set_line_wrap(GTK_LABEL(widg), true);
	gtk_label_set_line_wrap_mode(GTK_LABEL(widg), PANGO_WRAP_WORD_CHAR);
	gtk_label_set_xalign(GTK_LABEL(widg), 0);

  widg = GTK_WIDGET(g->label_preview_command);
	gtk_label_set_single_line_mode(GTK_LABEL(widg), false);
	gtk_label_set_line_wrap(GTK_LABEL(widg), true);
	gtk_label_set_line_wrap_mode(GTK_LABEL(widg), PANGO_WRAP_WORD_CHAR);
	gtk_label_set_xalign(GTK_LABEL(widg), 0);

  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->label_command), FALSE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->label_preview_command), FALSE, TRUE, 0);

  gtk_widget_show_all(g->vbox_gmic);
  gtk_widget_set_no_show_all(g->vbox_gmic, TRUE);

}


void reload_defaults(dt_iop_module_t *self)
{
  dt_iop_dt_gmic_lab_params_t tmp = { 0 };
  
  memcpy(self->params, &tmp, sizeof(dt_iop_dt_gmic_lab_params_t));
  memcpy(self->default_params, &tmp, sizeof(dt_iop_dt_gmic_lab_params_t));
  self->default_enabled = 0;
  
  destroy_gui_controls(self);
}

void gui_cleanup(struct dt_iop_module_t *self)
{
  dt_iop_dt_gmic_lab_gui_data_t *g = (dt_iop_dt_gmic_lab_gui_data_t *)self->gui_data;
  
  destroy_gui_controls(self);
  
  if (g->image) dt_free_align(g->image);
  dt_pthread_mutex_destroy(&g->lock);
  
  if (g->actual_params_widg) free(g->actual_params_widg);

  if (g->sl_widget) free(g->sl_widget);
  if (g->cmb_widget) free(g->cmb_widget);
  if (g->lbl_widget) free(g->lbl_widget);
  if (g->btn_widget) free(g->btn_widget);
  if (g->colorbtn_widget) free(g->colorbtn_widget);
  if (g->sep_widget) free(g->sep_widget);
  if (g->txt_widget) free(g->txt_widget);
  if (g->link_widget) free(g->link_widget);
  if (g->box_widget) free(g->box_widget);
  if (g->chk_widget) free(g->chk_widget);
  
  free(self->gui_data);
  self->gui_data = NULL;
}

// convert darktable image format to g'mic
// both images have the same size and channels
static void copy_image_to_gmic(float *image, float *gmic_image, const dt_iop_roi_t *const roi_in, const int ch_in, const int ch_gmic, const dt_iop_colorspace_type_t cst, const int pixelpipe_pos)
{
  // XYZ -> sRGB matrix, D65
  const float xyz_to_srgb[3][3] = {
    { 3.1338561, -1.6168667, -0.4906146 },
    { -0.9787684, 1.9161415, 0.0334540 },
    { 0.0719453, -0.2289914, 1.4052427 }
  };

  if (pixelpipe_pos == dt_iop_gmic_pipepos_rgb /*|| pixelpipe_pos == dt_iop_gmic_pipepos_linear_rgb*/) // after colorout
  {
    if (cst != iop_cs_rgb) return;
    
    float *in = image;
    float *out = gmic_image;
    if (ch_in == ch_gmic)
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, in += ch_in, out += ch_gmic)
      {
        out[0] = in[0] * 255.f;
        out[1] = in[1] * 255.f;
        out[2] = in[2] * 255.f;
        out[3] = 1.f;
      }
    }
    else
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, in += ch_in, out += ch_gmic)
      {
        out[0] = in[0] * 255.f;
        out[1] = in[1] * 255.f;
        out[2] = in[2] * 255.f;
      }
    }
  }
  else if (pixelpipe_pos == dt_iop_gmic_pipepos_lab) // after colorin
  {
    if (cst != iop_cs_Lab) return;

    float *in = image;
    float *out = gmic_image;
    if (ch_in == ch_gmic)
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, in += ch_in, out += ch_gmic)
      {
        // transform the pixel to sRGB:
        // Lab -> XYZ
        float XYZ[3];
        dt_Lab_to_XYZ(in, XYZ);
        // XYZ -> sRGB
        float rgb[3] = { 0, 0, 0 };
        for(int r = 0; r < 3; r++)
          for(int c = 0; c < 3; c++) rgb[r] += xyz_to_srgb[r][c] * XYZ[c];
        // linear sRGB -> gamma corrected sRGB
//        for(int c = 0; c < 3; c++)
//          rgb[c] = rgb[c] <= 0.0031308 ? 12.92 * rgb[c] : (1.0 + 0.055) * powf(rgb[c], 1.0 / 2.4) - 0.055;

        out[0] = rgb[0] * 255.f;
        out[1] = rgb[1] * 255.f;
        out[2] = rgb[2] * 255.f;
        out[3] = 1.f;
      }
    }
    else
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, in += ch_in, out += ch_gmic)
      {
        // transform the pixel to sRGB:
        // Lab -> XYZ
        float XYZ[3];
        dt_Lab_to_XYZ(in, XYZ);
        // XYZ -> sRGB
        float rgb[3] = { 0, 0, 0 };
        for(int r = 0; r < 3; r++)
          for(int c = 0; c < 3; c++) rgb[r] += xyz_to_srgb[r][c] * XYZ[c];
        // linear sRGB -> gamma corrected sRGB
//        for(int c = 0; c < 3; c++)
//          rgb[c] = rgb[c] <= 0.0031308 ? 12.92 * rgb[c] : (1.0 + 0.055) * powf(rgb[c], 1.0 / 2.4) - 0.055;

        out[0] = rgb[0] * 255.f;
        out[1] = rgb[1] * 255.f;
        out[2] = rgb[2] * 255.f;
      }
    }
  }
  else if (pixelpipe_pos == dt_iop_gmic_pipepos_linear_rgb) // after demosaic
  {
    if (cst != iop_cs_rgb) return;
 
    float *in = image;
    float *out = gmic_image;
    if (ch_in == ch_gmic)
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, in += ch_in, out += ch_gmic)
      {
      	float rgb[3] = { 0, 0, 0 };
      	for(int c = 0; c < 3; c++)
      		rgb[c] = in[c];
      	
        // linear sRGB -> gamma corrected sRGB
        for(int c = 0; c < 3; c++)
          rgb[c] = rgb[c] <= 0.0031308 ? 12.92 * rgb[c] : (1.0 + 0.055) * powf(rgb[c], 1.0 / 2.4) - 0.055;
        
        out[0] = rgb[0] * 255.f;
        out[1] = rgb[1] * 255.f;
        out[2] = rgb[2] * 255.f;
        out[3] = 1.f;
      }
    }
    else
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, in += ch_in, out += ch_gmic)
      {
      	float rgb[3] = { 0, 0, 0 };
      	for(int c = 0; c < 3; c++)
      		rgb[c] = in[c];

        // linear sRGB -> gamma corrected sRGB
        for(int c = 0; c < 3; c++)
          rgb[c] = rgb[c] <= 0.0031308 ? 12.92 * rgb[c] : (1.0 + 0.055) * powf(rgb[c], 1.0 / 2.4) - 0.055;
        
        out[0] = rgb[0] * 255.f;
        out[1] = rgb[1] * 255.f;
        out[2] = rgb[2] * 255.f;
      }
    }
  }
}

// convert g'mic image format to darktable
// g'mic can return different number of channels, but the image size is the same
static void copy_image_from_gmic(float *image, float *gmic_image, const dt_iop_roi_t *const roi_in, const int ch_in, const int ch_gmic, const dt_iop_colorspace_type_t cst, const int pixelpipe_pos)
{
  // sRGB -> XYZ matrix, D65
  const float srgb_to_xyz[3][3] = {
    { 0.4360747, 0.3850649, 0.1430804 },
    { 0.2225045, 0.7168786, 0.0606169 },
    { 0.0139322, 0.0971045, 0.7141733 }
  };

  if (pixelpipe_pos == dt_iop_gmic_pipepos_rgb /*|| pixelpipe_pos == dt_iop_gmic_pipepos_linear_rgb*/) // after colorout
  {
    if (cst != iop_cs_rgb) return;
    
    float *out = image;
    float *in = gmic_image;
    if (ch_gmic == 1 || ch_gmic == 2)
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, out += ch_in, in += ch_in)
      {
        out[0] = out[1] = out[2] = in[0] / 255.f;
      }
    }
    else
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, out += ch_in, in += ch_in)
      {
        out[0] = in[0] / 255.f;
        out[1] = in[1] / 255.f;
        out[2] = in[2] / 255.f;
      }
    }
  }
  else if (pixelpipe_pos == dt_iop_gmic_pipepos_lab) // after colorin
  {
    if (cst != iop_cs_Lab) return;

    float *out = image;
    float *in = gmic_image;
    if (ch_gmic == 1 || ch_gmic == 2)
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, out += ch_in, in += ch_in)
      {
        // transform the result back to Lab
        // sRGB -> XYZ
      	float rgb[3] = { 0, 0, 0 };
      	float XYZ[3] = { 0, 0, 0 };
        // gamma corrected sRGB -> linear sRGB
/*        for(int c = 0; c < 1; c++)
        {
        	rgb[c] = in[c] / 255.f;
          rgb[c] = rgb[c] <= 0.04045 ? rgb[c] / 12.92 : powf((rgb[c] + 0.055) / (1 + 0.055), 2.4);
        }
        rgb[1] = rgb[2] = rgb[0];
        */
      	rgb[1] = rgb[2] = rgb[0] = in[0] / 255.f;
        for(int r = 0; r < 3; r++)
          for(int c = 0; c < 3; c++) XYZ[r] += srgb_to_xyz[r][c] * rgb[c];
        // XYZ -> Lab
        dt_XYZ_to_Lab(XYZ, out);
      }
    }
    else
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, out += ch_in, in += ch_in)
      {
        // transform the result back to Lab
        // sRGB -> XYZ
      	float rgb[3] = { 0, 0, 0 };
      	float XYZ[3] = { 0, 0, 0 };
        // gamma corrected sRGB -> linear sRGB
/*        for(int c = 0; c < 3; c++)
        {
        	rgb[c] = in[c] / 255.f;
          rgb[c] = rgb[c] <= 0.04045 ? rgb[c] / 12.92 : powf((rgb[c] + 0.055) / (1 + 0.055), 2.4);
        }*/
      	for(int c = 0; c < 3; c++)
      		rgb[c] = in[c] / 255.f;
        for(int r = 0; r < 3; r++)
          for(int c = 0; c < 3; c++) XYZ[r] += srgb_to_xyz[r][c] * rgb[c];
        // XYZ -> Lab
        dt_XYZ_to_Lab(XYZ, out);
      }
    }
  }
  else if (pixelpipe_pos == dt_iop_gmic_pipepos_linear_rgb) // after demosaic
  {
    if (cst != iop_cs_rgb) return;

    float *out = image;
    float *in = gmic_image;
    if (ch_gmic == 1 || ch_gmic == 2)
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, out += ch_in, in += ch_in)
      {
      	float rgb[3] = { 0, 0, 0 };
      	
      	// gamma corrected sRGB -> linear sRGB
      	int c = 0;
      	rgb[c] = in[c] / 255.f;
        rgb[c] = rgb[c] <= 0.04045 ? rgb[c] / 12.92 : powf((rgb[c] + 0.055) / (1 + 0.055), 2.4);
        
        out[0] = out[1] = out[2] = rgb[0];
      }
    }
    else
    {
      for(int j = 0; j < roi_in->width*roi_in->height; j++, out += ch_in, in += ch_in)
      {
      	float rgb[3] = { 0, 0, 0 };
      	
        // gamma corrected sRGB -> linear sRGB
        for(int c = 0; c < 3; c++)
        {
        	rgb[c] = in[c] / 255.f;
          rgb[c] = rgb[c] <= 0.04045 ? rgb[c] / 12.92 : powf((rgb[c] + 0.055) / (1 + 0.055), 2.4);
        }
        
        out[0] = rgb[0];
        out[1] = rgb[1];
        out[2] = rgb[2];
      }
    }
  }
}

static int gmic_iop_module_demosaic = 0, gmic_iop_module_colorout = 0, gmic_iop_module_colorin = 0;
static int get_module_colorspace(const dt_iop_module_t *module)
{
  /* check if we do know what priority the color* plugins have */
  if(gmic_iop_module_colorout == 0 && gmic_iop_module_colorin == 0)
  {
    /* lets find out which priority colorin and colorout have */
    GList *iop = module->dev->iop;
    while(iop)
    {
      dt_iop_module_t *m = (dt_iop_module_t *)iop->data;
      if(m != module)
      {
        if(!strcmp(m->op, "colorin"))
        	gmic_iop_module_colorin = m->priority;
        else if(!strcmp(m->op, "colorout"))
        	gmic_iop_module_colorout = m->priority;
        else if(!strcmp(m->op, "demosaic"))
        	gmic_iop_module_demosaic = m->priority;
      }

      /* do we have both priorities, lets break out... */
      if(gmic_iop_module_colorout && gmic_iop_module_colorin && gmic_iop_module_demosaic) break;
      iop = g_list_next(iop);
    }
  }

  /* let check which colorspace module is within */
  if(module->priority > gmic_iop_module_colorout)
    return dt_iop_gmic_pipepos_rgb;
  else if(module->priority > gmic_iop_module_colorin)
    return dt_iop_gmic_pipepos_lab;
  else if(module->priority > gmic_iop_module_demosaic)
    return dt_iop_gmic_pipepos_linear_rgb;
  else
  	printf("gmic_eh.c: invalid colorspace\n");
  
  return 0;
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_dt_gmic_lab_data_t *p = (dt_iop_dt_gmic_lab_data_t *)piece->data;
  dt_iop_dt_gmic_lab_gui_data_t *g = (dt_iop_dt_gmic_lab_gui_data_t *)self->gui_data;
  dt_iop_dt_gmic_lab_global_data_t *gd = (dt_iop_dt_gmic_lab_global_data_t *)self->data;
  const dt_iop_colorspace_type_t cst = dt_iop_module_colorspace(self);
  const int ch = piece->colors;
  const int colorspace = get_module_colorspace(self);
  
  // save the full image to send it to plugin preview
  if(self->dev->gui_attached && g && piece->pipe->type == DT_DEV_PIXELPIPE_PREVIEW)
  {
  	if (g->image_requested == 1 || p->keep_image_loaded)
  	{
			dt_pthread_mutex_lock(&g->lock);
			if (g->image_locked == 0)
			{
				g->image_locked = 1;
				g->image_requested = 0;
				
				if (g->image) dt_free_align(g->image);
				g->width = roi_in->width;
				g->height = roi_in->height;
				g->ch = ch-1;
				g->imageScale = roi_in->scale / piece->iscale;
				g->image = (float*)dt_alloc_align(64, g->width * g->height * g->ch * sizeof(float));
				copy_image_to_gmic((float*)ivoid, g->image, roi_in, ch, ch-1, cst, colorspace);
				
				g->image_locked = 0;
			}
			dt_pthread_mutex_unlock(&g->lock);
  	}
  	else
  	{
  		dt_pthread_mutex_lock(&g->lock);
  		if (g->image)
  		{
  			dt_free_align(g->image);
  			g->image = NULL;
  		}
  		dt_pthread_mutex_unlock(&g->lock);
  	}
  }
  
  // if no filter defined just return the image
  if (p->filter_data_size == 0)
  {
  	memcpy(ovoid, ivoid, roi_out->width * roi_out->height * ch * sizeof(float));
  }
  else
  {
  	// change the image format to g'mic
    copy_image_to_gmic((float*)ivoid, (float*)ovoid, roi_in, ch, ch, cst, colorspace);
    
		// call g'mic
		dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
		gmic_filter_execution_data_t *filter_data = gmic_launchPluginHeadless(gd->gmic_gd, p->filter_data, p->filter_data_size, (float*)ovoid, 
				roi_out->width, roi_out->height, ch, roi_in->scale / piece->iscale);
		dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
		if (filter_data)
		{
			// change image format back
			copy_image_from_gmic((float*)ovoid, (float*)ovoid, roi_out, ch, filter_data->return_spectrum, cst, colorspace);
	
			dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
			gmic_freeFilterExecData(filter_data);
			dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
		}
		
		if(piece->pipe->mask_display) dt_iop_alpha_copy(ivoid, ovoid, roi_out->width, roi_out->height);
  }
  
}

} // extern "C"


// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
