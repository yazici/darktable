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


void dt_develop_masks_process (struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece, void *i, void *o, const struct dt_iop_roi_t *roi_in, const struct dt_iop_roi_t *roi_out)
{
  /* get blend data and check if a mask is used */
  dt_develop_blend_params_t *d = (dt_develop_blend_params_t *)piece->blendop_data;
  if (d->mask_id) 
  {
    /* we got an mask id, look up in masks data */
    
    /* if found render into 4th channel */
    
    dt_control_log("Rendering masks is not yet implemented.");
  }
}