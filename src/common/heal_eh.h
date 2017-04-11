
#ifndef DT_DEVELOP_HEAL_H
#define DT_DEVELOP_HEAL_H

void dt_heal(const float *const src_buffer, float *dest_buffer, const float *const mask_buffer, 
              const int width, const int height, const int ch, const int use_sse);

#endif

