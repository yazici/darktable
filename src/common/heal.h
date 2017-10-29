/*
    This file is part of darktable,
    copyright (c) 2017 edgardo hoszowski.

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

#ifndef DT_DEVELOP_HEAL_H
#define DT_DEVELOP_HEAL_H

void dt_heal(const float *const src_buffer, float *dest_buffer, const float *const mask_buffer, 
    const int width, const int height, const int ch, const int use_sse);

#ifdef HAVE_OPENCL

typedef struct dt_heal_cl_global_t
{
  int kernel_dummy;
} dt_heal_cl_global_t;

typedef struct heal_params_cl_t
{
  dt_heal_cl_global_t *global;
  int devid;
} heal_params_cl_t;

dt_heal_cl_global_t *dt_heal_init_cl_global(void);
void dt_heal_free_cl_global(dt_heal_cl_global_t *g);

heal_params_cl_t *dt_heal_init_cl(const int devid);
void dt_heal_free_cl(heal_params_cl_t *p);

cl_int dt_heal_cl(heal_params_cl_t *p, cl_mem dev_src, cl_mem dev_dest, const float *const mask_buffer, 
    const int width, const int height);

#endif
#endif

