/*
    This file is part of darktable,
    copyright (c) 2011 henrik andersson.

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

#include "common/darktable.h"
#include "develop/develop.h"
#include "control/control.h"
#include "libs/lib.h"
#include "gui/draw.h"

DT_MODULE(1)

#define DT_LIB_MASKS_CURVE_RES 64
#define DT_LIB_MASKS_MAX_CURVE_POINTS 128


typedef struct _curve_t
{
  float anchors[2][DT_LIB_MASKS_MAX_CURVE_POINTS];
  float feather_unit_vectors[2][DT_LIB_MASKS_MAX_CURVE_POINTS];
  float feather_weights[DT_LIB_MASKS_MAX_CURVE_POINTS];
  uint32_t anchor_count;
} _curve_t;

typedef struct dt_lib_masks_t
{
  uint32_t viewport_width, viewport_height;

  _curve_t curve;
  gboolean is_curve_edit_mode;
  gboolean is_dragging_anchor;
  int32_t drag_anchor;
  int32_t current_anchor;

  int32_t highlight_anchor;
  int32_t highlight_segment;
}
dt_lib_masks_t;

const char* name()
{
  return _("masks");
}

uint32_t views()
{
  return DT_VIEW_DARKROOM;
}

uint32_t container()
{
  return DT_UI_CONTAINER_PANEL_LEFT_CENTER;
}

void gui_reset(dt_lib_module_t *self)
{
}

int position()
{
  return 999;
}

static inline void _curve_calculate(float *px, float *py, float *dx,float *dy, uint32_t segments)
{
  // Catmull rom
  for (int k = 0; k < segments; k++)
  {
    float t = k / (float)segments;
    dx[k] = 0.5 * ((2*px[1]) + 
		   (-px[0]+px[2]) * t +
		   (2*px[0]-5*px[1]+4*px[2]-px[3]) * (t*t) +
		   (-px[0]+3*px[1]-3*px[2]+px[3]) * (t*t*t));

    dy[k] = 0.5 * ((2*py[1]) + 
		   (-py[0]+py[2]) * t +
		   (2*py[0]-5*py[1]+4*py[2]-py[3]) * (t*t) +
		   (-py[0]+3*py[1]-3*py[2]+py[3]) * (t*t*t));
  }
}

static void _curve_recalculate_feather_unit_vectors(_curve_t *curve)
{
  if(curve->anchor_count > 2)
    for(int k = 0; k < curve->anchor_count; k++)
    {
      float n[2][2], length;
      int p1 = (k==0)?(curve->anchor_count-1):k-1;
      int p2 = (k==curve->anchor_count-1)?0:k+1;
      n[0][0] = curve->anchors[1][k] - curve->anchors[1][p1];
      n[0][1] = -(curve->anchors[0][k] - curve->anchors[0][p1]);
      n[1][0] = curve->anchors[1][k] - curve->anchors[1][p2];
      n[1][1] = -(curve->anchors[0][k] - curve->anchors[0][p2]);

      length = sqrtf((n[0][0]*n[0][0])+(n[0][1]*n[0][1]));
      n[0][0] /= length;
      n[0][1] /= length;
      length = sqrtf((n[1][0]*n[1][0])+(n[1][1]*n[1][1]));
      n[1][0] /= length;
      n[1][1] /= length;

      float mx = (n[0][0]-n[1][0]) / 2.0f;
      float my = (n[0][1]-n[1][1]) / 2.0f;
      length = sqrtf((mx*mx)+(my*my));
      float ux = mx/length;
      float uy = my/length;

      curve->feather_unit_vectors[0][k] = ux;
      curve->feather_unit_vectors[1][k] = uy;

    }
}

void gui_init(dt_lib_module_t *self)
{
  dt_lib_masks_t *d = (dt_lib_masks_t *)g_malloc(sizeof(dt_lib_masks_t));
  self->data = (void *)d;
  memset(d, 0, sizeof(dt_lib_masks_t));
  
  self->widget = gtk_label_new("The masks...");
}

void gui_cleanup(dt_lib_module_t *self)
{
  g_free(self->data);
  self->data = NULL;
}

#define CURVE_SEGMENT_RES 32

void gui_post_expose(dt_lib_module_t *self, cairo_t *cri, int32_t width, int32_t height, int32_t pointerx, int32_t pointery)
{
  dt_lib_masks_t *d = (dt_lib_masks_t *)self->data;
  d->viewport_width = width;
  d->viewport_height = height;

  /* get zoomed pointer coords*/
  float pzx,pzy;
  dt_dev_get_pointer_zoom_pos(darktable.develop, pointerx, pointery, &pzx, &pzy);
  pzx += 0.5f;
  pzy += 0.5f;
  
  /* get darkroom zoom and scale expose */
  int32_t zoom, closeup;
  float zoom_x, zoom_y;
  float wd = darktable.develop->preview_pipe->backbuf_width;
  float ht = darktable.develop->preview_pipe->backbuf_height;
  
  /* get zoom scale */
  DT_CTL_GET_GLOBAL(zoom_y, dev_zoom_y);
  DT_CTL_GET_GLOBAL(zoom_x, dev_zoom_x);
  DT_CTL_GET_GLOBAL(closeup, dev_closeup);
  DT_CTL_GET_GLOBAL(zoom, dev_zoom);
  float zoom_scale = dt_dev_get_zoom_scale(darktable.develop, zoom, closeup ? 2 : 1, 1);
 
  float dx[CURVE_SEGMENT_RES];
  float dy[CURVE_SEGMENT_RES];

  /* scale and translate cairo */

  cairo_translate(cri, width/2.0, height/2.0f);
  cairo_scale(cri, zoom_scale, zoom_scale);
  cairo_translate(cri, -.5f*wd-zoom_x*wd, -.5f*ht-zoom_y*ht);

  double dashed[] = {4.0, 2.0};
  dashed[0] /= zoom_scale;
  dashed[1] /= zoom_scale;
  int len  = sizeof(dashed) / sizeof(dashed[0]);
  
  cairo_set_line_width(cri, 1.25/zoom_scale);
  if (d->curve.anchor_count)
  {

    /*
     * Draw the curve
     */

    /* add new mouse anchor */
    if (d->is_curve_edit_mode)
    {
      d->curve.anchors[0][d->curve.anchor_count] = pzx;
      d->curve.anchors[1][d->curve.anchor_count] = pzy;
      d->curve.anchor_count++;
    }

    /* draw each curve segment */
    float ptx[4],pty[4];
    float *px,*py;

    for (int k=0; k < d->curve.anchor_count; k++)
    {
      int k1,k2,k3,k4;
    
      k1 = (k-1)<0?d->curve.anchor_count-1:k-1;
      k2 = k;
      k3 = (k+1)%d->curve.anchor_count;
      k4 = (k+2)%d->curve.anchor_count;
    	
      ptx[0] = d->curve.anchors[0][k1];
      pty[0] = d->curve.anchors[1][k1];
      ptx[1] = d->curve.anchors[0][k2];
      pty[1] = d->curve.anchors[1][k2];
      ptx[2] = d->curve.anchors[0][k3];
      pty[2] = d->curve.anchors[1][k3];
      ptx[3] = d->curve.anchors[0][k4];
      pty[3] = d->curve.anchors[1][k4];
      
      px = ptx;
      py = pty;

      /* calculate sub segments from curve segment */
      _curve_calculate(px, py, dx, dy, CURVE_SEGMENT_RES);

      /* highlight segment */
      if (d->highlight_segment == k)
	cairo_set_source_rgba(cri, 1.0, 1.0, 1.0, 0.9);
      else
	cairo_set_source_rgba(cri, 1.0, 1.0, 1.0, 0.5);  

      /* draw the segment line */
      cairo_set_dash(cri, dashed, len, 0);
      cairo_move_to(cri, dx[0]*wd, dy[0]*ht);    
      for (int l=1;l<CURVE_SEGMENT_RES;l++)
	cairo_line_to(cri, dx[l]*wd, dy[l]*ht);
      cairo_line_to(cri, px[2]*wd,py[2]*ht);
      cairo_stroke_preserve(cri);

      /* draw the segment shadow */
      cairo_set_dash(cri, dashed, len, 4);
      if (d->highlight_segment == k)
	cairo_set_source_rgba(cri, 0.0, 0.0, 0.0, 0.9);
      else
	cairo_set_source_rgba(cri, 0.0, 0.0, 0.0, 0.5);  
      
      cairo_stroke(cri);

    }

    /* remove mouse pointer */
    if (d->is_curve_edit_mode)
      d->curve.anchor_count--;

    
    /* draw the anchors */
    cairo_set_dash(cri, dashed, 0, 0);
    for(int k = 0; k < d->curve.anchor_count; k++)
    {
      float anchor_size = 5.0f / zoom_scale;
      if (k == d->highlight_anchor)
      {
	anchor_size = 6.0f / zoom_scale;
	cairo_set_source_rgba(cri, 1.0, 1.0, 1.0, 0.9);
      }
      else
	cairo_set_source_rgba(cri, 1.0, 1.0, 1.0, 0.5);
      
      cairo_rectangle(cri, 
		      (d->curve.anchors[0][k]*wd) - (anchor_size*0.5), 
		      (d->curve.anchors[1][k]*ht) - (anchor_size*0.5), 
		      anchor_size, anchor_size);
      cairo_fill_preserve(cri);

      if (k == d->highlight_anchor)
	cairo_set_source_rgba(cri, 0.0, 0.0, 0.0, 0.9);
      else
	cairo_set_source_rgba(cri, 0.0, 0.0, 0.0, 0.5);
      cairo_stroke(cri);
    }
    
    /* draw anchors and feather normals if not dragging */
    if (!d->is_dragging_anchor || !d->is_curve_edit_mode)
    {
      cairo_set_dash(cri, dashed, 0, 0);
      /* draw the feather weights normals */
      cairo_set_source_rgba(cri, 1.0, 1.0, 1.0, 0.5);
      for(int k = 0; k < d->curve.anchor_count; k++)
      {
	cairo_move_to(cri, d->curve.anchors[0][k]*wd,d->curve.anchors[1][k]*ht);
	cairo_line_to(cri, 
		      (d->curve.anchors[0][k] + (d->curve.feather_unit_vectors[0][k] / 20.0f)) * wd, 
		      (d->curve.anchors[1][k] + (d->curve.feather_unit_vectors[1][k] / 20.0f)) * ht);
	cairo_stroke(cri);
      }
    }
  }
}

int mouse_moved(dt_lib_module_t *self, double x, double y, int which)
{
  dt_lib_masks_t *d = (dt_lib_masks_t *)self->data;
  /* get zoomed pointer coords*/
  float pzx,pzy;
  dt_dev_get_pointer_zoom_pos(darktable.develop, x, y, &pzx, &pzy); 
  pzx += 0.5f;
  pzy += 0.5f;

  /* reset any higlights */
  d->highlight_segment = -1;
  d->highlight_anchor = -1;

  /* check if we are in edit mode */
  if (d->is_curve_edit_mode)
    goto done;

  /* if dragging handle update */
  if (d->is_dragging_anchor)
  {
    d->curve.anchors[0][d->drag_anchor] = pzx;
    d->curve.anchors[1][d->drag_anchor] = pzy;
    goto done;
  }

  /* is mouse over anchor */
  double as = (2.0 / d->viewport_width) * 1.5; 
  for (int k=0;k<d->curve.anchor_count;k++)
    if ( (pzx > d->curve.anchors[0][k] - as && pzx < d->curve.anchors[0][k] + as) &&
	 (pzy > d->curve.anchors[1][k] - as && pzy < d->curve.anchors[1][k] + as))
    {
      d->highlight_anchor = k;
      goto done;
    }

  /* is mouse near segment */
  float cd = 100.0;
  for (int k=0; k < d->curve.anchor_count; k++)
  {
    float x3,x4,y3,y4;
    x3 = d->curve.anchors[0][k];
    x4 = (k < d->curve.anchor_count-1) ? d->curve.anchors[0][k+1] : d->curve.anchors[0][0];
    y3 = d->curve.anchors[1][k];
    y4 = (k < d->curve.anchor_count-1) ? d->curve.anchors[1][k+1] : d->curve.anchors[1][0];
    float m = (y3-y4)/(x3-x4);

    /* check y bounding*/
    if( pzy > fmin(y3,y4) && pzy < fmax(y3,y4))
    {
      float ix = x3 + ((pzy-y3) / m);
      float hd = pzx-ix;
      
      float a = atan(m);
      float distance = fabs(hd*sin(a));
      if (cd > distance)
      {
	cd = distance;
	d->highlight_segment = k;
      }
    }
  }

  /* check if close distance is within threshold length */
  if (cd > (as*5.0))
  {
    d->highlight_segment = -1;
    goto done;
  }

  return 0;

done:

  _curve_recalculate_feather_unit_vectors(&d->curve);

  dt_control_queue_redraw_center();
  return 1;
}

int button_released(struct dt_lib_module_t *self, double x, double y, int which, uint32_t state)
{
  dt_lib_masks_t *d = (dt_lib_masks_t *)self->data;

  if(which == 3)
  {
    if (d->curve.anchor_count < 3)
    {
      /* clear curve */
      d->curve.anchor_count = 0;
      goto done;
    }
    else
    {
      /* remove anchor */
      if (d->current_anchor != -1)
      {
	int k = d->current_anchor;
	memcpy(&d->curve.anchors[0][k],&d->curve.anchors[0][k+1], sizeof(float) * d->curve.anchor_count-k);
	memcpy(&d->curve.anchors[1][k],&d->curve.anchors[1][k+1], sizeof(float) * d->curve.anchor_count-k);
	d->curve.anchor_count--;
	goto done;
      }
    }
  }

  /* end dragging */
  if (d->is_dragging_anchor)
  {
    d->is_dragging_anchor = FALSE;
    goto done;
  }

  return 0;

done:
  return 1;
}

int button_pressed (struct dt_lib_module_t *self, double x, double y, int which, int type, uint32_t state)
{
  dt_lib_masks_t *d = (dt_lib_masks_t *)self->data;

  
  int32_t zoom, closeup;
  DT_CTL_GET_GLOBAL(closeup, dev_closeup);
  DT_CTL_GET_GLOBAL(zoom, dev_zoom);
  float zoom_scale = dt_dev_get_zoom_scale(darktable.develop, zoom, closeup ? 2 : 1, 1);

  /* get zoomed pointer coords*/
  float pzx,pzy;
  dt_dev_get_pointer_zoom_pos(darktable.develop, x, y, &pzx, &pzy);
  pzx += 0.5f;
  pzy += 0.5f;

  if (d->curve.anchor_count < 3)
    d->is_curve_edit_mode = TRUE;

  if (d->is_curve_edit_mode && which == 1)
  {
    /* adding new anchor to curve */
    d->curve.anchors[0][d->curve.anchor_count] = pzx;
    d->curve.anchors[1][d->curve.anchor_count] = pzy;
    d->curve.anchor_count++;
    dt_control_queue_redraw_center();
    return 1;
  } 
  else
  {

    /* check if we should end editmode */
    if (which == 3 && d->curve.anchor_count >= 3)
      d->is_curve_edit_mode = FALSE;

    /* is mouse over anchor */
    d->current_anchor = -1;
    double as = ((2.0 * 4.0) / d->viewport_width / zoom_scale); 
    for (int k=0;k<d->curve.anchor_count;k++)
    {
      if ( (pzx > d->curve.anchors[0][k] - as && pzx < d->curve.anchors[0][k] + as) &&
	   (pzy > d->curve.anchors[1][k] - as && pzy < d->curve.anchors[1][k] + as))
      {
	d->current_anchor = k;
	/* begin draw anchor */
	if (which == 1)
	{
	  d->drag_anchor = k;
	  d->is_dragging_anchor = TRUE;
	  dt_control_queue_redraw_center();
	  return 1;
	}
      }
    }

    /* is mouse near segment lets insert anchor and begin drag */
    if (which ==1 && d->highlight_segment != -1)
    {
      int k = d->highlight_segment;
      
      /* insert anchor on segment at px,py */
      d->curve.anchor_count++;
      for (int l = d->curve.anchor_count-1; l > k; l--)
      {
	d->curve.anchors[0][l] = d->curve.anchors[0][l-1]; 
	d->curve.anchors[1][l] = d->curve.anchors[1][l-1]; 
      }
      d->curve.anchors[0][k+1] = pzx;
      d->curve.anchors[1][k+1] = pzy;

      d->drag_anchor = k+1;
      d->is_dragging_anchor = TRUE;

      dt_control_queue_redraw_center();
      return 1;
    }
  }

  dt_control_queue_redraw_center();

  return 0;
}
