
#ifndef DT_DEVELOP_HEAL_H
#define DT_DEVELOP_HEAL_H

void dt_dev_heal(const float *const src_buffer, float *dest_buffer, const float *const mask_buffer, const int width, const int height, const int ch, 
    const float preview_scale, const int use_sse);

#endif

