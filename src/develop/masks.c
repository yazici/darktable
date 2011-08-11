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
#include "control/control.h"
#include "masks.h"
#include "blend.h"

#include <librsvg/rsvg.h>
#include <librsvg/rsvg-cairo.h>

static char *_masks_load_svg(char *filename) 
{
  gchar configdir[1024],datadir[1024], *ffilename;
  dt_util_get_datadir(datadir, 1024);
  dt_util_get_user_config_dir(configdir, 1024);
  g_strlcat(datadir,"/watermarks/",1024);
  g_strlcat(configdir,"/watermarks/",1024);
  g_strlcat(datadir,filename,1024);
  g_strlcat(configdir,filename,1024);

  if (g_file_test(configdir,G_FILE_TEST_EXISTS))
    ffilename=configdir;
  else if (g_file_test(datadir,G_FILE_TEST_EXISTS))
    ffilename=datadir;
  else return NULL;  
  
  gchar *svgdata=NULL;
  gsize length;
  if( g_file_get_contents( ffilename, &svgdata, &length, NULL))
    return svgdata;
  
  return NULL;
}

void dt_develop_masks_process (struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece, void *i, void *o, const struct dt_iop_roi_t *roi_in, const struct dt_iop_roi_t *roi_out)
{
  /* get blend data and check if a mask is used */
  dt_develop_blend_params_t *d = (dt_develop_blend_params_t *)piece->blendop_data;
  if (d->mode && d->mask_id) 
  {
    /* we got an mask id, look up in masks data */

    /* Load svg if not loaded */
    gchar *svgdoc = _masks_load_svg ("stars.svg");
  
    /* if found render into 4th channel */
    GError *error = NULL;
    RsvgHandle *svg = rsvg_handle_new_from_data ((const guint8 *)svgdoc,strlen (svgdoc),&error);
    g_free (svgdoc);
    if (!svg || error)
      return;
    
    RsvgDimensionData dimension;
    rsvg_handle_get_dimensions (svg,&dimension);
    const float iw=piece->buf_in.width*roi_out->scale;
    const float ih=piece->buf_in.height*roi_out->scale;
    
    float scale=1.0;
    if ((dimension.width/dimension.height)>1.0)
      scale = iw/dimension.width;
    else
      scale = ih/dimension.height;

    int stride = cairo_format_stride_for_width (CAIRO_FORMAT_A8,roi_out->width);
    guint8 *image= (guint8 *)g_malloc (stride*roi_out->height);
    memset (image,0,stride*roi_out->height);
    cairo_surface_t *surface = cairo_image_surface_create_for_data (image,CAIRO_FORMAT_A8,roi_out->width,roi_out->height,stride);
    
    fprintf(stderr,"rendering mask...\n");
 
    /* render svg */
    dt_pthread_mutex_lock(&darktable.plugin_threadsafe);
    cairo_t *cr = cairo_create (surface);
    cairo_scale (cr,scale,scale);
    rsvg_handle_render_cairo (svg,cr);
    dt_pthread_mutex_unlock(&darktable.plugin_threadsafe);
    
    /* ensure that all operations on surface finishing up */
    cairo_surface_flush (surface);
    
    /* copy rendered mask into 4th channel */
    float *out=(float *)o;
    guint8 *sd = image;
#ifdef _OPENMP
  #pragma omp parallel for default(none) shared(roi_out,out,sd) schedule(static)
#endif
    for (int j=0; j<(roi_out->height*roi_out->width); j++)
      out[4*j+3] = ((float)sd[j]/0xff);
    

    /* clean up */
    cairo_surface_destroy (surface);
    g_object_unref (svg);
    g_free (image);
  }
}
