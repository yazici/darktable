
#include <stdlib.h>
#include <cblas.h>

#include "develop/imageop.h"
#include "common/interpolation.h"
#include "common/convolution.h"

#include "rlucy_deblur.h"

//-----------------------------------------------------------------------------------------

/*
Created on 30 avr. 2017

@author: aurelien

This script shows an implementation of the Richardson-Lucy deconvolution.

In theory, blurred and noisy pictures can be perfectly sharpened if we perfectly
know the [*Point spread function*](https://en.wikipedia.org/wiki/Point_spread_function)
of their maker. In practice, we can only estimate it.
One of the means to do so is the [Richardson-Lucy deconvolution](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution).

The Richardson-Lucy algorithm used here has a damping coefficient wich allows to remove from
the iterations the pixels which deviate to much (x times the standard deviation of the difference
source image - deconvoluted image) from the original image. This pixels are considered
noise and would be amplificated from iteration to iteration otherwise.

*/

//-----------------------------------------------------------------------------------------

//#define RLUCY_NAN

#ifdef RLUCY_NAN

static void rlucy_check_nan(rlucy_image_t* im, const char *str)
{
  int i_nan = 0;
  
  for (int i = 0; i < im->w*im->h*im->h; i++) if ( isnan(im->im[i]) ) i_nan++;
  
  if (i_nan > 0) printf("%s nan: %i\n", str, i_nan);
}

#else

#define rlucy_check_nan(im, str) {}

#endif


//-----------------------------------------------------------------------------------------

#define TIME_CONV_FFT_LOCK 1
#define TIME_CONV_FFT_EXEC 2
#define TIME_CONV_FFT_COPY 3
#define TIME_CONV_FFT_MULT 4
#define TIME_CONV_FFT_PLAN 5

#define TIME_CONVOLVE 6
#define TIME_SUM 0
#define TIME_SUBTRACT 0
#define TIME_MULSUB 0
#define TIME_ADD 0
#define TIME_PADIMAGE 0
#define TIME_UNPADIMAGE 0
#define TIME_TRIMMASK 0
#define TIME_DIVTV 0
#define TIME_ROTATE 0
#define TIME_GRADIENT 0
#define TIME_DIVIDE 0
#define TIME_PROCESS 18
#define TIME_BUILD_KERNEL 19

#define MAX_TIME 50

const char *rlucy_time_desc[MAX_TIME] = {
        "dummy         ", "fft lock    ", "fft execute ", "fft copy    ", "fft mult    ", "fft plan    ",
        "convolve      ", "sum           ", "subtract      ", "mulsub        ", "add           ", "pad image     ",
        "unpad image   ", "trim mask     ", "divTV         ", "rotate        ", "gradient      ", "divide        ",
        "process       ", "build kernel  "
};

#define RLUCY_TIME

#ifdef RLUCY_TIME

#define RLUCY_TIME_DECL double start = 0;

#define RLUCY_TIME_BEGIN start = dt_get_wtime();

#define RLUCY_TIME_END(time_index) time[time_index] += dt_get_wtime() - start;

#else

#define RLUCY_TIME_DECL ;

#define RLUCY_TIME_BEGIN ;

#define RLUCY_TIME_END(time_index) ;

#endif

//-----------------------------------------------------------------------------------------

typedef struct {
    float *im;
    int w;
    int h;
    int c;
    size_t alloc_size;
} rlucy_image_t;


static void rlucy_free_image(rlucy_image_t *image) {
    if (image->im) {
        dt_free_align(image->im);
        image->im = NULL;
    }
    image->w = image->h = image->c = 0;
    image->alloc_size = 0;
}

static void rlucy_alloc_image(rlucy_image_t *image, const int w, const int h, const int c) {
    // re-alloc
    if (image->im != NULL) {
        if (image->alloc_size >= w * h * c) {
            image->w = w;
            image->h = h;
            image->c = c;
        } else {
            rlucy_free_image(image);
        }
    }

    if (image->im == NULL) {
        image->w = w;
        image->h = h;
        image->c = c;
        image->alloc_size = w * h * c;
        image->im = dt_alloc_align(64, image->alloc_size * sizeof(float));
    }
}

static void rlucy_copy_image(rlucy_image_t *img_dest, rlucy_image_t *img_src) {
    if (img_dest->im == NULL || img_src->im == NULL) {
        printf("rlucy_copy_image: NULL image\n");
        return;
    }

    if (img_src->w == img_dest->w && img_src->h == img_dest->h && img_src->c == img_dest->c) {
        memcpy(img_dest->im, img_src->im, img_src->w * img_src->h * img_src->c * sizeof(float));
        return;
    }

    if (img_src->c == img_dest->c) {
        const int h = MIN(img_src->h, img_dest->h);
        const int size = MIN(img_src->w * img_src->c, img_dest->w * img_dest->c) * sizeof(float);

        for (int y = 0; y < h; y++) {
            float *dest = img_dest->im + y * img_dest->w * img_dest->c;
            float *src = img_src->im + y * img_src->w * img_src->c;

            memcpy(dest, src, size);
        }
    } else
        printf("rlucy_copy_image: different channels not implemented\n");

}

static void rlucy_resize_1c(rlucy_image_t *image_src, rlucy_image_t *image_dest, const float scale, const int new_w,
                            const int new_h,
                            rlucy_image_t *im_tmp_in, rlucy_image_t *im_tmp_out) {
    float *out = NULL;
    float *in = NULL;
    dt_iop_roi_t roi_out = {0};
    dt_iop_roi_t roi_in = {0};
    int32_t out_stride = 0;
    int32_t in_stride = 0;

    // in-place
    if (image_src->im == image_dest->im) {
        if (scale == 1.f) return;
    } else {
        if (scale == 1.f) {
            rlucy_alloc_image(image_dest, image_src->w, image_src->h, image_src->c);
            memcpy(image_dest->im, image_src->im, image_dest->w * image_dest->h * image_dest->c * sizeof(float));
            return;
        }
    }

    roi_in.width = image_src->w;
    roi_in.height = image_src->h;
    roi_in.scale = 1.0f;

    roi_out.width = new_w;
    roi_out.height = new_h;
    roi_out.scale = scale;

    out_stride = roi_out.width * 4 * sizeof(float);
    in_stride = roi_in.width * 4 * sizeof(float);

    rlucy_alloc_image(im_tmp_out, roi_out.width, roi_out.height, 4);
    rlucy_alloc_image(im_tmp_in, roi_in.width, roi_in.height, 4);
    memset(im_tmp_in->im, 0, im_tmp_in->alloc_size * sizeof(float));

    out = im_tmp_out->im;
    in = im_tmp_in->im;

    int size = image_src->w * image_src->h;
    for (int i = 0; i < size; i++) {
        in[i * 4] = image_src->im[i];
    }

    const struct dt_interpolation *itor = dt_interpolation_new(DT_INTERPOLATION_USERPREF);

    dt_interpolation_resample(itor, out, &roi_out, out_stride, in, &roi_in, in_stride);

    rlucy_alloc_image(image_dest, roi_out.width, roi_out.height, image_src->c);

    size = image_dest->w * image_dest->h;
    for (int i = 0; i < size; i++) {
        image_dest->im[i] = out[i * 4];
    }

}

static void
rlucy_resize(rlucy_image_t *image_src, rlucy_image_t *image_dest, const float scale, const int new_w, const int new_h,
             rlucy_image_t *im_tmp_in, rlucy_image_t *im_tmp_out) {
    if (image_src->c == 1) {
        rlucy_resize_1c(image_src, image_dest, scale, new_w, new_h, im_tmp_in, im_tmp_out);
        return;
    }

    float *out = NULL;
    float *in = NULL;
    dt_iop_roi_t roi_out = {0};
    dt_iop_roi_t roi_in = {0};
    int32_t out_stride = 0;
    int32_t in_stride = 0;

    rlucy_image_t im_dest = {0};

    // in-place
    if (image_src->im == image_dest->im) {
        if (scale == 1.f) return;

        rlucy_alloc_image(&im_dest, new_w, new_h, image_src->c);
    } else {
        rlucy_alloc_image(image_dest, new_w, new_h, image_src->c);

        if (scale == 1.f) {
            memcpy(image_dest->im, image_src->im, image_dest->w * image_dest->h * image_dest->c * sizeof(float));
            return;
        }

        im_dest = *image_dest;
    }

    roi_in.width = image_src->w;
    roi_in.height = image_src->h;
    roi_in.scale = 1.0f;

    roi_out.width = im_dest.w;
    roi_out.height = im_dest.h;
    roi_out.scale = scale;

    out_stride = roi_out.width * image_src->c * sizeof(float);
    in_stride = roi_in.width * image_src->c * sizeof(float);

    out = im_dest.im;
    in = image_src->im;

    const struct dt_interpolation *itor = dt_interpolation_new(DT_INTERPOLATION_USERPREF);

    dt_interpolation_resample(itor, out, &roi_out, out_stride, in, &roi_in, in_stride);

    // in-place
    if (image_src->im == image_dest->im) {
        rlucy_alloc_image(image_dest, roi_out.width, roi_out.height, image_src->c);
        memcpy(image_dest->im, im_dest.im, image_dest->w * image_dest->h * image_dest->c * sizeof(float));

        rlucy_free_image(&im_dest);
    } else {
        image_dest->w = roi_out.width;
        image_dest->h = roi_out.height;
    }

}

static void rlucy_img_sum(const rlucy_image_t *const image, float *sum, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    // FIXME use OpenBLAS

    const int size = image->w * image->h * image->c;

    for (int c = 0; c < image->c; c++)
        sum[c] = 0.f;

    for (int i = 0; i < size; i += image->c) {
        for (int c = 0; c < image->c; c++)
            sum[c] += image->im[i + c];
    }

    RLUCY_TIME_END(TIME_SUM)

}

static void rlucy_img_abs(const rlucy_image_t *const image, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    const int size = image->w * image->h * image->c;

#ifdef _OPENMPx
#pragma omp parallel for default(none)  schedule(static)
#endif
    for (int i = 0; i < size; i += image->c) {
        for (int c = 0; c < image->c; c++)
            image->im[i + c] = fabs(image->im[i + c]);
    }

    RLUCY_TIME_END(TIME_SUM)

}

static void rlucy_img_divide(rlucy_image_t *image, float *div, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    const int size = image->w * image->h * image->c;
    float mult[4] = {0};

    for (int c = 0; c < image->c; c++) {
        if (div[c] == 0.f) {
            mult[c] = 0.f;
            printf("rlucy_img_divide: division by zero\n");
        } else
            mult[c] = 1.f / div[c];
    }

    float *im = image->im;
    const int ch = image->c;

#ifdef _OPENMPx
#pragma omp parallel for default(none) shared(im, mult) schedule(static)
#endif
    for (int i = 0; i < size; i += ch) {
        for (int c = 0; c < ch; c++) {
            im[i + c] *= mult[c];
        }
    }

    RLUCY_TIME_END(TIME_DIVIDE)

}

#if 0
static void rlucy_img_multiply(rlucy_image_t *image, float *mult, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
  
  const int size = image->w * image->h * image->c;
  
  float *im = image->im;
  const int ch = image->c;
  
#ifdef _OPENMPx
#pragma omp parallel for default(none) shared(im, mult) schedule(static)
#endif
  for (int i = 0; i < size; i+=ch)
  {
    for (int c = 0; c < ch; c++)
    {
      im[i+c] *= mult[c];
    }
  }
  
  RLUCY_TIME_END(TIME_DIVIDE)
  
}
#endif

static void rlucy_img_subtract_1(rlucy_image_t *image, float *sub, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    const int size = image->w * image->h * image->c;

    float *im = image->im;
    const int ch = image->c;

#ifdef _OPENMPx
#pragma omp parallel for default(none) shared(im, sub) schedule(static)
#endif
    for (int i = 0; i < size; i += ch) {
        for (int c = 0; c < ch; c++) {
            im[i + c] -= sub[c];
        }
    }

    RLUCY_TIME_END(TIME_DIVIDE)

}

static void rlucy_img_subtract(rlucy_image_t *img_in1,
                               const rlucy_image_t *const img_in2, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    const int size = img_in1->w * img_in1->h * img_in1->c;

    if (img_in1->w != img_in2->w || img_in1->h != img_in2->h) {
        printf("rlucy_img_subtract: invalid image size\n");
        return;
    }

    float *img_in1_im = img_in1->im;
    const float *const img_in2_im = img_in2->im;

#ifdef _OPENMPx
#pragma omp parallel for default(none) shared(img_in1_im) schedule(static)
#endif
    for (int i = 0; i < size; i++) {
        img_in1_im[i] -= img_in2_im[i];
    }

    RLUCY_TIME_END(TIME_SUBTRACT)

}

#if 0
static void rlucy_img_add(rlucy_image_t *img_in1,
    const rlucy_image_t *const img_in2, double *time)
{
  RLUCY_TIME_DECL
  RLUCY_TIME_BEGIN
  
  const int size = img_in1->w * img_in1->h * img_in1->c;
  
  if (img_in1->w != img_in2->w || img_in1->h != img_in2->h)
  {
    printf("rlucy_img_add: invalid image size\n");
    return;
  }

  float *img_in1_im = img_in1->im;
  const float *const img_in2_im = img_in2->im;
  
#ifdef _OPENMPx
#pragma omp parallel for default(none) shared(img_in1_im) schedule(static)
#endif
  for (int i = 0; i < size; i++)
  {
    img_in1_im[i] += img_in2_im[i];
  }
  
  RLUCY_TIME_END(TIME_SUBTRACT)
  
}
#endif

static void rlucy_img_add3(rlucy_image_t *img_in1,
                           const rlucy_image_t *const img_in2, float *add, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    const int size = img_in1->w * img_in1->h * img_in1->c;

    if (img_in1->w != img_in2->w || img_in1->h != img_in2->h) {
        printf("rlucy_img_add3: invalid image size\n");
        return;
    }

    float *img_in1_im = img_in1->im;
    const float *const img_in2_im = img_in2->im;

#ifdef _OPENMPx
#pragma omp parallel for default(none) shared(img_in1_im) schedule(static)
#endif
    for (int i = 0; i < size; i += img_in1->c) {
        for (int c = 0; c < img_in1->c; c++)
            img_in1_im[i + c] += img_in2_im[i + c] + add[c];
    }

    RLUCY_TIME_END(TIME_SUBTRACT)

}

static void rlucy_img_mulsub(rlucy_image_t *img_in1,
                             const rlucy_image_t *const img_in2,
                             const float *const mult, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    const int size = img_in1->w * img_in1->h * img_in1->c;

    if (img_in1->w != img_in2->w || img_in1->h != img_in2->h) {
        printf("rlucy_img_mulsub: invalid image size\n");
        return;
    }

    float *img_in1_im = img_in1->im;
    const float *const img_in2_im = img_in2->im;
    const int ch = img_in1->c;

#ifdef _OPENMPx
#pragma omp parallel for default(none) shared(img_in1_im) schedule(static)
#endif
    for (int i = 0; i < size; i += ch) {
        for (int c = 0; c < ch; c++) {
            img_in1_im[i + c] -= img_in2_im[i + c] * mult[c];
        }
    }

    RLUCY_TIME_END(TIME_MULSUB)

}

static void rlucy_img_divadd(rlucy_image_t *img_in1,
                             const rlucy_image_t *const img_in2,
                             const float *const add, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    const int size = img_in1->w * img_in1->h * img_in1->c;

    if (img_in1->w != img_in2->w || img_in1->h != img_in2->h) {
        printf("rlucy_img_divadd: invalid image size\n");
        return;
    }

    float *img_in1_im = img_in1->im;
    const float *const img_in2_im = img_in2->im;
    const int ch = img_in1->c;

#ifdef _OPENMPx
#pragma omp parallel for default(none) shared(img_in1_im) schedule(static)
#endif
    for (int i = 0; i < size; i += ch) {
        for (int c = 0; c < ch; c++) {
            float div = img_in2_im[i + c] + add[c];
            if (div != 0.f && div != -0.f)
                img_in1_im[i + c] /= div;
            else
                printf("rlucy_img_divadd: division by zero\n");
        }
    }

    RLUCY_TIME_END(TIME_MULSUB)

}

static void pad_image(const rlucy_image_t *const img_src,
                      const int pad_h, const int pad_v,
                      rlucy_image_t *img_dest, rlucy_image_t *img_tmp1, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    if (img_src->im == img_dest->im) {
        pad_image(img_src,
                  pad_h, pad_v,
                  img_tmp1, img_tmp1, time);
        rlucy_alloc_image(img_dest, img_tmp1->w, img_tmp1->h, img_tmp1->c);
        rlucy_copy_image(img_dest, img_tmp1);
    } else {
        rlucy_alloc_image(img_dest, img_src->w + pad_h * 2, img_src->h + pad_v * 2, img_src->c);

        convolve_pad_image(img_src->im, img_src->w, img_src->h, img_src->c,
                           pad_h, pad_v,
                           img_dest->im, img_dest->w, img_dest->h, time);
    }

    RLUCY_TIME_END(TIME_PADIMAGE)

}

static void unpad_image(rlucy_image_t *img_dest,
                        const int pad_h, const int pad_v,
                        rlucy_image_t *img_src, rlucy_image_t *img_tmp1, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    if (img_src->im == img_dest->im) {
        unpad_image(img_tmp1,
                    pad_h, pad_v,
                    img_src, img_tmp1, time);
        rlucy_alloc_image(img_dest, img_tmp1->w, img_tmp1->h, img_tmp1->c);
        rlucy_copy_image(img_dest, img_tmp1);
    } else {
        rlucy_alloc_image(img_dest, img_src->w - pad_h * 2, img_src->h - pad_v * 2, img_src->c);

        // copy image
        for (int y = 0; y < img_dest->h; y++) {
            float *dest = img_dest->im + y * img_dest->w * img_dest->c;
            float *src = img_src->im + (y + pad_v) * img_src->w * img_src->c + pad_h * img_src->c;

            memcpy(dest, src, img_dest->w * img_dest->c * sizeof(float));
        }
    }

    RLUCY_TIME_END(TIME_UNPADIMAGE)
}


//-----------------------------------------------------------------------------------------

static void rlucy_convolve(const rlucy_image_t *const image,
                           const rlucy_image_t *const kernel,
                           const int mode, rlucy_image_t *img_dest,
                           convolve_data_t *fft_conv_data,
                           double *time) {

    // FIXME ensure correct padding

    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    int w = 0, h = 0;

    convolve_get_dest_size(image->w, image->h, kernel->w, kernel->h,
                           &w, &h, mode);

    rlucy_alloc_image(img_dest, w, h, image->c);

    memset(img_dest->im, 0, img_dest->w * img_dest->h * img_dest->c * sizeof(float));

    convolve(fft_conv_data, image->im, image->w, image->h, image->c,
             kernel->im, kernel->w, kernel->h,
             img_dest->im, img_dest->w, img_dest->h,
             mode, &(time[TIME_CONV_FFT_LOCK]));

    RLUCY_TIME_END(TIME_CONVOLVE)

    rlucy_check_nan(img_dest, "rlucy_convolve fft");

}

static void rlucy_img_max(const rlucy_image_t *const image, float *max, double *time) {

    // FIXME use BLAS amax instead

    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    const int size = image->w * image->h * image->c;

    for (int c = 0; c < image->c; c++) {
        max[c] = -INFINITY;
    }

    for (int i = 0; i < size; i += image->c) {
        for (int c = 0; c < image->c; c++) {
            max[c] = MAX(max[c], image->im[i + c]);
        }
    }

    RLUCY_TIME_END(0)

}

// from:
// https://github.com/numpy/numpy/blob/v1.13.0/numpy/lib/function_base.py#L1502-L1840
//

static void rlucy_gradient(const rlucy_image_t *const f, rlucy_image_t *outvals, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    // FIXME use separable filters instead

    int edge_order = 2;
    float *yy = NULL;
    int len_axes = 2;
    const float dx[2] = {1.0f, 1.0f};

    yy = dt_alloc_align(64, f->w * f->h * f->c * sizeof(float));
    if (yy == NULL) goto cleanup;

    memcpy(yy, f->im, f->w * f->h * f->c * sizeof(float));

    for (int i = 0, axis = 0; i < len_axes; i++, axis++) {
        rlucy_image_t out = {0};

        rlucy_alloc_image(&out, f->w, f->h, f->c);

        int uniform_spacing = 1;

        if (uniform_spacing) {
            const float dx_2 = 1.f / (2.f * dx[i]);

            if (axis == 0) {
#ifdef _OPENMPx
#pragma omp parallel for default(none) shared(out, i) schedule(static)
#endif
                for (int y = 0; y < f->h; y++) {
                    for (int x = 1; x < f->w - 1; x++) {
                        const int idx_out = y * f->w * f->c + x * f->c;
                        const int idx_f2 = y * f->w * f->c + (x - 1) * f->c;
                        const int idx_f4 = y * f->w * f->c + (x + 1) * f->c;

                        for (int c = 0; c < f->c; c++) {
                            out.im[idx_out + c] = (f->im[idx_f4 + c] - f->im[idx_f2 + c]) * dx_2;
                        }
                    }
                }
            } else {
#ifdef _OPENMPx
#pragma omp parallel for default(none) shared(out, i) schedule(static)
#endif
                for (int y = 1; y < f->h - 1; y++) {
                    for (int x = 0; x < f->w; x++) {
                        const int idx_out = y * f->w * f->c + x * f->c;
                        const int idx_f2 = (y - 1) * f->w * f->c + x * f->c;
                        const int idx_f4 = (y + 1) * f->w * f->c + x * f->c;

                        for (int c = 0; c < f->c; c++) {
                            out.im[idx_out + c] = (f->im[idx_f4 + c] - f->im[idx_f2 + c]) * dx_2;
                        }
                    }
                }
            }
        }

        if (edge_order == 1) {
        } else {
            if (uniform_spacing) {
                const float ia = -1.5f / dx[i];
                const float ib = 2.f / dx[i];
                const float ic = -0.5f / dx[i];

                if (axis == 0) {
                    for (int y = 0; y < f->h; y++) {
                        const int idx_out = y * f->w * f->c;
                        const int idx_y2 = y * f->w * f->c;
                        const int idx_y3 = y * f->w * f->c + 1 * f->c;
                        const int idx_y4 = y * f->w * f->c + 2 * f->c;

                        for (int c = 0; c < f->c; c++) {
                            out.im[idx_out + c] = ia * yy[idx_y2 + c] + ib * yy[idx_y3 + c] + ic * yy[idx_y4 + c];
                        }
                    }
                } else {
                    for (int x = 0; x < f->w; x++) {
                        const int idx_out = x * f->c;
                        const int idx_y2 = x * f->c;
                        const int idx_y3 = x * f->c + 1 * f->w * f->c;
                        const int idx_y4 = x * f->c + 2 * f->w * f->c;

                        for (int c = 0; c < f->c; c++) {
                            out.im[idx_out + c] = ia * yy[idx_y2 + c] + ib * yy[idx_y3 + c] + ic * yy[idx_y4 + c];
                        }
                    }
                }
            }
        }

        if (uniform_spacing) {
            const float ia = 0.5f / dx[i];
            const float ib = -2.f / dx[i];
            const float ic = 1.5f / dx[i];

            if (axis == 0) {
                for (int y = 0; y < f->h; y++) {
                    const int idx_out = y * f->w * f->c + (f->w - 1) * f->c;
                    const int idx_y2 = y * f->w * f->c + (f->w - 3) * f->c;
                    const int idx_y3 = y * f->w * f->c + (f->w - 2) * f->c;
                    const int idx_y4 = y * f->w * f->c + (f->w - 1) * f->c;

                    for (int c = 0; c < f->c; c++) {
                        out.im[idx_out + c] = ia * yy[idx_y2 + c] + ib * yy[idx_y3 + c] + ic * yy[idx_y4 + c];
                    }
                }
            } else {
                for (int x = 0; x < f->w; x++) {
                    const int idx_out = (f->h - 1) * f->w * f->c + x * f->c;
                    const int idx_y2 = (f->h - 3) * f->w * f->c + x * f->c;
                    const int idx_y3 = (f->h - 2) * f->w * f->c + x * f->c;
                    const int idx_y4 = (f->h - 1) * f->w * f->c + x * f->c;

                    for (int c = 0; c < f->c; c++) {
                        out.im[idx_out + c] = ia * yy[idx_y2 + c] + ib * yy[idx_y3 + c] + ic * yy[idx_y4 + c];
                    }
                }
            }
        }

        outvals[axis] = out;
    }

    cleanup:

    if (yy) dt_free_align(yy);

    RLUCY_TIME_END(TIME_GRADIENT)

}

static void rlucy_img_mean(const rlucy_image_t *const image, float *mean, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    const int size = image->w * image->h * image->c;

    for (int c = 0; c < image->c; c++) {
        mean[c] = 0.f;
    }

    for (int i = 0; i < size; i += image->c) {
        for (int c = 0; c < image->c; c++) {
            mean[c] += image->im[i + c];
        }
    }

    for (int c = 0; c < image->c; c++) {
        mean[c] /= (float) (image->w * image->h);
    }

    RLUCY_TIME_END(0)

}

static void rlucy_img_absmax(const rlucy_image_t *const image, float *max, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    const int size = image->w * image->h * image->c;

    for (int c = 0; c < image->c; c++) {
        max[c] = 0.f;
    }

    for (int i = 0; i < size; i += image->c) {
        for (int c = 0; c < image->c; c++) {
            max[c] = MAX(max[c], fabs(image->im[i + c]));
        }
    }

    RLUCY_TIME_END(0)

}

static void rlucy_img_rotate90(const rlucy_image_t *const img_in,
                               rlucy_image_t *img_dest,
                               const int times, double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    if (times == 1) {
        rlucy_alloc_image(img_dest, img_in->h, img_in->w, img_in->c);

        if (img_in->w != img_dest->h || img_in->h != img_dest->w) {
            printf("rlucy_img_rotate90: invalid image size\n");
            return;
        }

        for (int y = 0; y < img_in->h; y++) {
            const int x1 = (img_dest->w - y - 1) * img_dest->c;
            const int y2 = y * img_in->w * img_in->c;

            for (int x = 0; x < img_in->w; x++) {
                const int y1 = (x * img_dest->w * img_dest->c) + x1;
                const int x2 = y2 + (x * img_in->c);

                for (int c = 0; c < img_in->c; c++) {
                    img_dest->im[y1 + c] = img_in->im[x2 + c];
                }
            }
        }
    } else if (times == 2) {
        rlucy_alloc_image(img_dest, img_in->w, img_in->h, img_in->c);

        if (img_in->w != img_dest->w || img_in->h != img_dest->h) {
            printf("rlucy_img_rotate90: invalid image size\n");
            return;
        }

        for (int y = 0; y < img_in->h; y++) {
            const int y1 = (img_dest->h - y - 1) * img_dest->w * img_dest->c;
            const int y2 = (y * img_in->w * img_in->c);

            for (int x = 0; x < img_in->w; x++) {
                const int x1 = y1 + (img_dest->w - x - 1) * img_dest->c;
                const int x2 = y2 + (x * img_in->c);

                for (int c = 0; c < img_in->c; c++) {
                    img_dest->im[x1 + c] = img_in->im[x2 + c];
                }
            }
        }
    } else
        printf("rlucy_img_rotate90: invalid argument times\n");

    RLUCY_TIME_END(TIME_ROTATE)

}

// mask[left, top, width, height]
static void rlucy_trim_mask(const rlucy_image_t *const img_src, rlucy_image_t *img_dest,
                            const int im_x, const int im_y, const int im_w, const int im_h, rlucy_image_t *im_tmp1,
                            double *time) {
    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN

    if (img_src->im == img_dest->im) {
        rlucy_image_t tmp = {0};

        rlucy_trim_mask(img_src, im_tmp1, im_x, im_y, im_w, im_h, &tmp, time);
        rlucy_alloc_image(img_dest, im_tmp1->w, im_tmp1->h, im_tmp1->c);
        rlucy_copy_image(img_dest, im_tmp1);

        rlucy_free_image(&tmp);
    } else {
        rlucy_alloc_image(img_dest, im_w, im_h, img_src->c);

        if (img_dest->w > img_src->w || img_dest->h > img_src->h)
            printf("rlucy_trim_mask: roi_dest > roi_src\n");
        if (img_dest->w + im_x > img_src->w || img_dest->h + im_y > img_src->h)
            printf("rlucy_trim_mask: roi_dest+x > roi_src\n");

        // copy image
        for (int y = 0; y < img_dest->h; y++) {
            const float *const src = img_src->im + (y + im_y) * img_src->w * img_src->c + im_x * img_src->c;
            float *dest = img_dest->im + y * img_dest->w * img_dest->c;

            memcpy(dest, src, img_dest->w * img_dest->c * sizeof(float));
        }
    }

    RLUCY_TIME_END(TIME_TRIMMASK)

}

/*
@jit(float32[:](float32[:]), cache=True, nogil=True)
def _normalize_kernel(kern):
*/
static void _normalize_kernel(rlucy_image_t *kern, double *time) {
/*    kern[kern < 0] = 0
    kern /= np.sum(kern, axis=(0, 1))
    return kern
*/
    // FIXME: axis=(0, 1) ???
    for (int i = 0; i < kern->w * kern->h * kern->c; i++) {
        if (kern->im[i] < 0.f) kern->im[i] = 0.f;
    }

    float kern_sum[4] = {0};
    rlucy_img_sum(kern, kern_sum, time);
    rlucy_img_divide(kern, kern_sum, time);
}

static void rlucy_img_dot(rlucy_image_t *image, float *dot) {
    for (int c = 0; c < image->c; c++) {
        dot[c] = 0.f;
    }

    const int size = image->w * image->h * image->c;
    for (int i = 0; i < size; i += image->c) {
        for (int c = 0; c < image->c; c++) {
            dot[c] += image->im[i + c] * image->im[i + c];
        }
    }

}

// from: https://github.com/numpy/numpy/blob/v1.13.0/numpy/linalg/linalg.py#L2014-L2257

// def norm(x, ord=None, axis=None, keepdims=False):
static void rlucy_img_norm(rlucy_image_t *image, float *norm, double *time) {

    rlucy_img_dot(image, norm);
    for (int c = 0; c < image->c; c++)
        norm[c] = sqrtf(norm[c]);

}

static void rlucy_img_norm2(rlucy_image_t *image1, rlucy_image_t *image2, float *norm, double *time) {

    float norm1[4] = {0};
    float norm2[4] = {0};

    rlucy_img_dot(image1, norm1);
    rlucy_img_dot(image2, norm2);
    for (int c = 0; c < image1->c; c++)
        norm[c] = sqrtf(norm1[c] + norm2[c]);

}

/*
@jit(float32[:](float32[:], float32, float32), cache=True, nogil=True)
def best_param(image, lambd, p=1):
*/
static void best_param(rlucy_image_t *image, float *lambd, int p, float *epsilon, double *time) {
/*
    Determine by a statistical method the best lambda parameter. [1]

    Find the minimum acceptable epsilon to avoid a degenerate constant solution [2]

    Reference :  
    .. [1] https://pdfs.semanticscholar.org/1d84/0981a4623bcd26985300e6aa922266dab7d5.pdf
    .. [2] http://www.cvg.unibe.ch/publications/perroneIJCV2015.pdf

    :param image: 
    :return: 
*/

    rlucy_image_t grad[2] = {0};
    rlucy_image_t img_tmp1 = {0};
    float grad_std[4] = {0};
    float grad_mean[4] = {0};
    float norm[4] = {0};
    float omega[4] = {0};

    float mean[4] = {0};

    /* grad = np.gradient(image) */
    rlucy_gradient(image, grad, time);
    if (grad[0].im == NULL || grad[1].im == NULL) goto cleanup;

    /* grad_std = np.linalg.norm(image - image.mean()) / image.size */
    rlucy_img_mean(image, mean, time);
    rlucy_alloc_image(&img_tmp1, image->w, image->h, image->c);
    rlucy_copy_image(&img_tmp1, image);
    rlucy_img_subtract_1(&img_tmp1, mean, time);
    rlucy_img_norm(&img_tmp1, norm, time);
    for (int c = 0; c < image->c; c++)
        grad_std[c] = norm[c] / (float) (image->w * image->h);

    /* grad_mean = np.linalg.norm(grad) / image.size */
    rlucy_img_norm2(&(grad[0]), &(grad[1]), grad_mean, time);
    for (int c = 0; c < image->c; c++)
        grad_mean[c] = grad_mean[c] / (float) (image->w * image->h);

    // # lambd = noise_reduction_factor * np.sum(np.sqrt(divTV(image, p=1)))**2 / (-np.log(np.std(image)**2) * * 2 * np.pi)

    /* omega = 2 * lambd * grad_std / p */
    for (int c = 0; c < image->c; c++)
        omega[c] = 2.f * lambd[c] * grad_std[c] / (float) p;

    /* epsilon = np.sqrt(grad_mean / (np.exp(omega) - 1)) */
    for (int c = 0; c < image->c; c++) {
        if (grad_mean[c] == 0.f || grad_mean[c] == -0.f)
            printf("best_param: grad_mean is zero\n");

        float div = (expf(omega[c]) - 1.f);
        epsilon[c] = sqrtf(grad_mean[c] / div);

//    double div = (exp((double)omega[c]) - 1.);
//    epsilon[c] = sqrt(((double)grad_mean[c]) / div);

        if (epsilon[c] == 0.f || epsilon[c] == -0.f) {
            printf("best_param: epsilon is zero, grad_mean[c]=%f, div=%f, omega[c]=%f, grad_std[c]=%f, lambd[c]=%f\n",
                   grad_mean[c], div, omega[c], grad_std[c], lambd[c]);
        }
    }
//    print(lambd, epsilon, p)
    /* return epsilon * 1.001 */
    for (int c = 0; c < image->c; c++) {
        if (epsilon[c] == 0.f || epsilon[c] == -0.f)
            printf("best_param: epsilon is zero\n");

        epsilon[c] *= 1.001f;
    }

    cleanup:

    rlucy_free_image(&img_tmp1);
    rlucy_free_image(&(grad[0]));
    rlucy_free_image(&(grad[1]));
}

/*
@jit(float32[:](float32[:], float32, float32), cache=True, nogil=True)
def divTV(u, epsilon=0, p=1):
*/
static void divTV(rlucy_image_t *u, float *epsilon, int p, rlucy_image_t *divtv, double *time) {
    rlucy_image_t grad[2] = {0};

    /* grad = np.gradient(u, edge_order=2) */
    rlucy_gradient(u, grad, time);
    if (grad[0].im == NULL || grad[1].im == NULL) goto cleanup;

    // # For Darktable implementation, don't bother to implement the p parameter, just use the p = 1 case to optimize the computations
    if (p == 1) {
        /* return np.abs(grad[0]) + np.abs(grad[1]) + epsilon */
        rlucy_img_abs(&(grad[0]), time);
        rlucy_img_abs(&(grad[1]), time);
        rlucy_img_add3(&(grad[0]), &(grad[1]), epsilon, time);
        rlucy_alloc_image(divtv, u->w, u->h, u->c);
        rlucy_copy_image(divtv, &(grad[0]));
    }
/*    else:
        return (np.abs(grad[0]) ** p + np.abs(grad[1]) ** p + epsilon ** p) ** (1 / p)
*/
    cleanup:

    rlucy_free_image(&(grad[0]));
    rlucy_free_image(&(grad[1]));
}

/*
@jit(float32[:](float32[:], float32[:], float32, float32, float32), cache=True, nogil=True)
def gradTVEM(u, ut, epsilon=1e-3, tau=1e-1, p=0.5):
*/
static void
gradTVEM(rlucy_image_t *u, rlucy_image_t *ut, float *epsilon /* = 1e-3 */, float *tau /* = 1e-1 */, float p /* = 0.5 */,
         rlucy_image_t *img_dest,
         double *time) {
/*  TVt = divTV(ut, epsilon=epsilon, p=p)
  TV = divTV(u, epsilon=epsilon, p=p)
  return TV / (tau + TVt)
*/
    rlucy_image_t *TV = img_dest;
    rlucy_image_t TVt = {0};

    divTV(ut, epsilon, p, &TVt, time);
    divTV(u, epsilon, p, TV, time);
    rlucy_img_divadd(TV, &TVt, tau, time);

    rlucy_free_image(&TVt);
}

/*
@jit(float32[:](float32[:], float32[:], float32[:]), cache=True, nogil=True)
def _convolve_image(u, image, psf):
*/
static void
_convolve_image(const rlucy_image_t *const u, const rlucy_image_t *const image, const rlucy_image_t *const psf,
                rlucy_image_t *dest,
                rlucy_image_t *tmp1, rlucy_image_t *tmp2,
                convolve_data_t *fft_conv_data, double *time) {
    /*    error = np.ascontiguousarray(convolve(u, psf, "valid"), np.float32)
        error -= image
        return convolve(error, np.rot90(psf, 2), "full")
    */
    rlucy_convolve(u, psf, convolve_mode_valid, tmp1, fft_conv_data, time);
    rlucy_img_subtract(tmp1, image, time);
    rlucy_img_rotate90(psf, tmp2, 2, time);
    rlucy_convolve(tmp1, tmp2, convolve_mode_full, dest, fft_conv_data, time);

}

/*
@jit(float32[:](float32[:], float32[:], float32[:]), cache=True, nogil=True)
def _convolve_kernel(u, image, psf):
*/
static void
_convolve_kernel(const rlucy_image_t *const u, const rlucy_image_t *const image, const rlucy_image_t *const psf,
                 rlucy_image_t *dest,
                 rlucy_image_t *tmp1, rlucy_image_t *tmp2,
                 convolve_data_t *fft_conv_data, double *time) {
    /*    error = np.ascontiguousarray(convolve(u, psf, "valid"), np.float32)
        error -= image
        return convolve(np.rot90(u, 2), error, "valid")
    */
    rlucy_convolve(u, psf, convolve_mode_valid, tmp1, fft_conv_data, time);
    rlucy_img_subtract(tmp1, image, time);

    rlucy_img_rotate90(u, tmp2, 2, time);
    rlucy_convolve(tmp2, tmp1, convolve_mode_valid, dest, fft_conv_data, time);

}

/*
@utils.timeit
def richardson_lucy_PAM(image: np.ndarray,
                        u: np.ndarray,
                        psf: np.ndarray,
                        lambd: float,
                        iterations: int,
                        epsilon=1e-3,
                        mask=None,
                        blind=True) -> np.ndarray:
*/
static void richardson_lucy_PAM(rlucy_image_t *image,
                                rlucy_image_t *u,
                                rlucy_image_t *psf,
                                float *lambd,
                                int iterations,
                                float *epsilon /* = 1e-3 */,
                                int *mask /* = None */,
                                int blind /* = True */,
                                convolve_data_t *fft_conv_data,
                                double *time) {
    printf("richardson_lucy_PAM not implemented\n");
}

/*
@jit(float32[:](float32[:], float32[:], float32[:], float32, int16, float32[:], float32[:], float32), cache=True,
     nogil=True)
def _update_both_MM(u, image, psf, lambd, iterations, mask_u, mask_i, epsilon):
*/
static void _update_both_MM(rlucy_image_t *u, rlucy_image_t *image, rlucy_image_t *psf, float *lambd,
                            int iterations, float *mask_u, float *mask_i, float *epsilon, int blind,
                            convolve_data_t *fft_conv_data, double *time) {
/*
    Utility function to launch the actual deconvolution in parallel
    :param u:
    :param image:
    :param psf:
    :param lambd:
    :param iterations:
    :param mask_u:
    :param mask_i:
    :param epsilon:
    :return:
*/

    rlucy_image_t ut = {0};
    rlucy_image_t u_masked = {0};
    rlucy_image_t image_masked = {0};
    rlucy_image_t gradu = {0};
    rlucy_image_t gradk = {0};
    rlucy_image_t img_tmp1 = {0};
    rlucy_image_t img_tmp2 = {0};
    rlucy_image_t img_tmp3 = {0};

//    float tau = 0.f;
    float k_step[4] = {epsilon[0], epsilon[1], epsilon[2], epsilon[3]};
    float u_step[4] = {epsilon[0], epsilon[1], epsilon[2], epsilon[3]};
    float eps[4] = {0};
    // # For Darktable implementation, don't bother to implement the p parameter
    const int p = 1;

    rlucy_alloc_image(&ut, u->w, u->h, u->c);
    if (ut.im == NULL) goto cleanup;

    for (int it = 0; it < iterations; it++) {
/*        ut = u
*/
        rlucy_copy_image(&ut, u);

        /* lambd = min([lambd, 50000]) */
        for (int c = 0; c < image->c; c++)
            lambd[c] = MIN(lambd[c], 50000.f);

        best_param(u, lambd, p, eps, time);

        for (int itt = 0; itt < 5; itt++) {
            // # Image update
            /* lambd = min([lambd, 50000]) */
            for (int c = 0; c < image->c; c++)
                lambd[c] = MIN(lambd[c], 50000.f);

            /* gradu = lambd * _convolve_image(u, image, psf) + gradTVEM(u, ut, eps, eps, p=p) */
            gradTVEM(u, &ut, eps, eps, p, &gradu, time);
            _convolve_image(u, image, psf, &img_tmp1, &img_tmp2, &img_tmp3, fft_conv_data, time);

            float lambd_a[4] = {-lambd[0], -lambd[1], -lambd[2], -lambd[3]};
            rlucy_img_mulsub(&gradu, &img_tmp1, lambd_a, time);

            /* dt = u_step * (np.amax(u) + 1 / u.size) / np.amax(np.abs(gradu) + 1e-31) */
            float max_u[4] = {0};
            float max_gradu[4] = {0};

            rlucy_img_max(u, max_u, time);
            rlucy_img_absmax(&gradu, max_gradu, time);
            float dt[4] = {0};
            for (int c = 0; c < image->c; c++) {
                dt[c] = u_step[c] * (max_u[c] + 1.f / (float) (u->w * u->h)) / MAX(max_gradu[c], 1e-31);
            }

            /* u -= dt * gradu */
            rlucy_img_mulsub(u, &gradu, dt, time);

            /* np.clip(u, 0, 1, out=u) */

            if (blind) {
                // # PSF update
/*            gradk = _convolve_kernel(u[mask_u[0]:mask_u[1], mask_u[2]:mask_u[3]],
                                               image[mask_i[0]:mask_i[1], mask_i[2]:mask_i[3]], psf)
*/
                if (mask_u) {
                    rlucy_trim_mask(u, &u_masked, mask_u[0], mask_u[1], mask_u[2], mask_u[3], &img_tmp1, time);
                } else {
                    rlucy_alloc_image(&u_masked, u->w, u->h, u->c);
                    rlucy_copy_image(&u_masked, u);
                }

                if (mask_i) {
                    rlucy_trim_mask(image, &image_masked, mask_i[0], mask_i[1], mask_i[2], mask_i[3], &img_tmp1, time);
                } else {
                    rlucy_alloc_image(&image_masked, image->w, image->h, image->c);
                    rlucy_copy_image(&image_masked, image);
                }

                _convolve_kernel(&u_masked, &image_masked, psf, &gradk, &img_tmp2, &img_tmp3, fft_conv_data, time);

                /* alpha = k_step * (np.amax(psf) + 1 / psf.size) / np.amax(np.abs(gradk) + 1e-31) */
                float max_psf[4] = {0};
                float max_gradk[4] = {0};

                rlucy_img_max(psf, max_psf, time);
                rlucy_img_absmax(&gradk, max_gradk, time);
                float alpha[4] = {0};
                for (int c = 0; c < image->c; c++) {
                    alpha[c] = k_step[c] * (max_psf[c] + 1.f / (float) (psf->w * psf->h)) / MAX(max_gradk[c], 1e-31);
                }

                /* psf -= alpha * gradk */
                rlucy_img_mulsub(psf, &gradk, alpha, time);

                /* psf = _normalize_kernel(psf) */
                _normalize_kernel(psf, time);
            }

            for (int c = 0; c < image->c; c++)
                lambd[c] *= 1.001;
        }
    }

    printf("_update_both_MM: iterations=%i\n", iterations * 5);

/*    return u.astype(np.float32), psf
*/

    cleanup:

    rlucy_free_image(&ut);
    rlucy_free_image(&u_masked);
    rlucy_free_image(&image_masked);
    rlucy_free_image(&gradu);
    rlucy_free_image(&gradk);
    rlucy_free_image(&img_tmp1);
    rlucy_free_image(&img_tmp2);
    rlucy_free_image(&img_tmp3);

}


static void richardson_lucy_MM(rlucy_image_t *image,
                               rlucy_image_t *u,
                               rlucy_image_t *psf,
                               float *lambd,
                               int iterations,
                               float *epsilon /* = 5e-3 */,
                               int *mask /* = None */,
                               int blind /* = True */,
                               convolve_data_t *fft_conv_data,
                               double *time) {
/*          
    """Richardson-Lucy Blind and non Blind Deconvolution with Total Variation Regularization by the Minimization-Maximization
    algorithm. This is known to give the sharp image in more than 50 % of the cases.

    Based on Matlab sourcecode of :

    Copyright (C) Daniele Perrone, perrone@iam.unibe.ch
	      Remo Diethelm, remo.diethelm@outlook.com
	      Paolo Favaro, paolo.favaro@iam.unibe.ch
	      2014, All rights reserved.

    :param ndarray image : Input 3 channels image.
    :param ndarray psf : The point spread function.
    :param int iterations : Number of iterations.
    :param float lambd : Lambda parameter of the total Variation regularization
    :param bool blind : Determine if it is a blind deconvolution is launched, thus if the PSF is updated
        between two iterations
    :returns ndarray: deconvoluted image

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] http://www.cvg.unibe.ch/dperrone/tvdb/perrone2014tv.pdf
    .. [3] http://hal.archives-ouvertes.fr/docs/00/43/75/81/PDF/preprint.pdf
    .. [4] http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html
    .. [5] http://www.cs.sfu.ca/~pingtan/Papers/pami10_deblur.pdf
    """
*/

    rlucy_image_t u_padded = {0};
    rlucy_image_t img_tmp1 = {0};

    // # image dimensions
/*    MK, NK, C = psf.shape
    M, N, C = image.shape
    pad = np.floor(MK / 2).astype(int)
*/
    int MK = psf->w;
//  int NK = psf->h;

    int M = image->w;
    int N = image->h;
//  int C = image->c;

    int pad = floor(MK / 2);

    printf("richardson_lucy_MM: working on image :%i, %i, %i\n", u->w, u->h, u->c);

/*    u = pad_image(u, (pad, pad))
*/
    pad_image(u, pad, pad, &u_padded, &img_tmp1, time);

    float mask_i[4] = {0};
    float mask_u[4] = {0};

    if (mask == NULL) {
        mask_i[0] = 0;
        mask_i[2] = M + 2 * pad;
        mask_i[1] = 0;
        mask_i[3] = N + 2 * pad;
        mask_u[0] = 0;
        mask_u[2] = M + 2 * pad;
        mask_u[1] = 0;
        mask_u[3] = N + 2 * pad;
    } else {
        mask_u[0] = mask[0] - pad;
        mask_u[2] = mask[2] + 2 * pad;
        mask_u[1] = mask[1] - pad;
        mask_u[3] = mask[3] + 2 * pad;
        mask_i[0] = mask[0];
        mask_i[2] = mask[2];
        mask_i[1] = mask[1];
        mask_i[3] = mask[3];
    }

    if (mask == NULL)
        _update_both_MM(&u_padded, image, psf, lambd, iterations, NULL, NULL, epsilon, blind, fft_conv_data, time);
    else
        _update_both_MM(&u_padded, image, psf, lambd, iterations, mask_u, mask_i, epsilon, blind, fft_conv_data, time);

    unpad_image(u, pad, pad, &u_padded, &img_tmp1, time);

    rlucy_free_image(&u_padded);
    rlucy_free_image(&img_tmp1);

}


static void build_pyramid(const int psf_size, const float lambd, int method,
                          int **kernels_size, float **images_scaling, float **lambdas, int *pyramid_len) {
/*
    To speed-up the deconvolution, the PSF is estimated successively on smaller images of increasing sizes. This function
    computes the intermediates sizes and regularization factors

    :param image_size:
    :param psf_size:
    :param lambd:
    :param method:
    :return:
*/

    int len = 1;
//  float lambda_tmp = lambd;
    int kernel_tmp = psf_size;

    int *arr_kernels = NULL;
    float *arr_images_scaling = NULL;
    float *arr_lambdas = NULL;

    float image_multiplier = .0f;
    float lambda_multiplier = .0f;
    float lambda_max = .0f;

    if (method == rlucy_method_fast) {
        image_multiplier = 1.1f;
        lambda_multiplier = 1.9f;
        lambda_max = 0.5f;
    } else if (method == rlucy_method_best) {
        image_multiplier = sqrtf(2.f);
        lambda_multiplier = 1.f / 2.1f;
        lambda_max = 999999.f;
    }

    while ((int) (kernel_tmp / image_multiplier) >= 3) {
        kernel_tmp = (int) floor(kernel_tmp / image_multiplier);

        len++;
    }

    arr_kernels = calloc(len, sizeof(int));
    arr_images_scaling = calloc(len, sizeof(float));
    arr_lambdas = calloc(len, sizeof(float));
    if (arr_kernels == NULL || arr_images_scaling == NULL || arr_lambdas == NULL)
        goto cleanup;

    arr_kernels[0] = psf_size;
    arr_images_scaling[0] = 1.f;
    arr_lambdas[0] = lambd;

    for (int i = 1; i < len; i++) {
        arr_lambdas[i] = MIN(arr_lambdas[i - 1] * lambda_multiplier, lambda_max);
        arr_kernels[i] = (int) (floor(arr_kernels[i - 1] / image_multiplier));
        arr_images_scaling[i] = arr_images_scaling[i - 1] / image_multiplier;

        if (arr_kernels[i] % 2 == 0) {
            arr_kernels[i] -= 1;
        }
        if (arr_kernels[i] < 3) {
            arr_kernels[i] = 3;
        }
    }

    if (!(arr_lambdas[len - 1] < lambda_max && arr_kernels[len - 1] >= 3) && len > 1)
        printf("rlucy_build_pyramid: error in construction, len=%i\n", len);

    *kernels_size = arr_kernels;
    *images_scaling = arr_images_scaling;
    *lambdas = arr_lambdas;
    *pyramid_len = len;

    return;

    cleanup:
    if (arr_kernels) free(arr_kernels);
    if (arr_images_scaling) free(arr_images_scaling);
    if (arr_lambdas) free(arr_lambdas);

    *pyramid_len = 0;

    return;
}


/*
def process_pyramid(pic, u, psf, lambd, method, epsilon=1e-3, quality=1):
*/
static void process_pyramid(rlucy_image_t *pic, rlucy_image_t *u, rlucy_image_t *psf, float lambd, int method,
                            float epsilon /* = 1e-3 */, int quality /* = 1 */,
                            convolve_data_t *fft_conv_data, double *time) {
/*
    To speed-up the deconvolution, the PSF is estimated successively on smaller images of increasing sizes. This function
    computes the intermediates deblured pictures and PSF.

    :param pic:
    :param u:
    :param psf:
    :param lambd:
    :param method:
    :param epsilon:
    :param quality: the number of iterations performed at each pyramid step will be adjusted by this factor.
    :return:
*/

    rlucy_image_t im = {0};
    rlucy_image_t img_tmp1 = {0};
    rlucy_image_t img_tmp2 = {0};

    int pyramid_len = 0;
    int *arr_kernels = NULL;
    float *arr_images_scaling = NULL;
    float *arr_lambdas = NULL;

    /* Mk, Nk, C = psf.shape */
    int Mk = psf->w;
//  int Nk = psf->h;
//  int C = psf->c;

    /* images, kernels, lambdas = build_pyramid(Mk, lambd, method) */
    build_pyramid(Mk, lambd, method, &arr_kernels, &arr_images_scaling, &arr_lambdas, &pyramid_len);

/*  rlucy_alloc_image(&u_zoom, u->w, u->h, u->c);
  if (u_zoom.im == NULL) goto cleanup;
  rlucy_copy_image(&u_zoom, u);
*/
    rlucy_alloc_image(&im, pic->w, pic->h, pic->c);
    if (im.im == NULL) goto cleanup;

    /* u = ndimage.zoom(u, (images[-1], images[-1], 1)) */
//  rlucy_resize(u, u, arr_images_scaling[pyramid_len-1], u->w * arr_images_scaling[pyramid_len-1], u->h * arr_images_scaling[pyramid_len-1], &img_tmp1, &img_tmp2);

    printf("\n== Processing Pyramid steps=%i ==\n", pyramid_len);

    int k_prec = Mk;
    int iterations = quality;
    float scale = 1.f;

    /* for i, k, l in zip(reversed(images), reversed(kernels), reversed(lambdas)): */
    for (int i = pyramid_len - 1; i >= 0; i--) {
        printf("== Pyramid step=%i ==\n", i);

        // # Resize blured, deblured images and PSF from previous step
        if (arr_images_scaling[i] != 1.f) {
            // # TODO : pad the picture to make its dimensions odd
            /* im = ndimage.zoom(pic, (i, i, 1)) */
            rlucy_resize(pic, &im, arr_images_scaling[i], pic->w * arr_images_scaling[i],
                         pic->h * arr_images_scaling[i], &img_tmp1, &img_tmp2);
        } else {
            /* im = pic.copy() */
            rlucy_alloc_image(&im, pic->w, pic->h, pic->c);
            rlucy_copy_image(&im, pic);
        }

        /* psf = ndimage.zoom(psf, (k / k_prec, k / k_prec, 1)) */
        scale = (float) arr_kernels[i] / (float) k_prec;
        rlucy_resize(psf, psf, scale, psf->w * scale, psf->h * scale, &img_tmp1, &img_tmp2);

        /* psf = _normalize_kernel(psf) */
        _normalize_kernel(psf, time);

        /* u = ndimage.zoom(u, (im.shape[0] / u.shape[0], im.shape[1] / u.shape[1], 1)) */
        scale = (float) (im.w) / (float) (u->w);
        rlucy_resize(u, u, scale, im.w, im.h, &img_tmp1, &img_tmp2);

        // # Make a blind Richardson-Lucy deconvolution on the RGB signal
        /* u, psf = method(im, u, psf, l, iterations, epsilon=epsilon) */
        float noise_damping_a[4] = {arr_lambdas[i], arr_lambdas[i], arr_lambdas[i], arr_lambdas[i]};
        float epsilon_a[4] = {epsilon, epsilon, epsilon, epsilon};

        if (method == rlucy_method_fast)
            richardson_lucy_PAM(&im, u, psf, noise_damping_a, iterations, epsilon_a, NULL, 1, fft_conv_data, time);
        else if (method == rlucy_method_best)
            richardson_lucy_MM(&im, u, psf, noise_damping_a, iterations, epsilon_a, NULL, 1, fft_conv_data, time);

        // # TODO : unpad the picture

        /* k_prec = k */
        k_prec = arr_kernels[i];
    }
    /* return u, psf */

    printf("== Pyramid end ==\n");

    cleanup:

    rlucy_free_image(&im);
    rlucy_free_image(&img_tmp1);
    rlucy_free_image(&img_tmp2);
}

static void rlucy_img_outer(float *vector, const int size, float *outer, const int ch) {
    const int size_c = size * ch;

    for (int i = 0; i < size; i++) {
        const int idx_outer = i * size_c;

        for (int j = 0; j < size; j++) {
            const float mult = vector[i] * vector[j];

            for (int c = 0; c < ch; c++) {
                outer[idx_outer + j * ch + c] = mult;
            }
        }
    }

}

// from:
// https://github.com/scipy/scipy/blob/v0.19.1/scipy/signal/windows.py#L1159-L1219
//

static void rlucy_buildGaussianWindow(float *win, const int win_size, const float shape) {
    if (win_size == 1) {
        win[0] = 1.f;
    } else {
        for (int i = 0; i < win_size; i++) {
            win[i] = i - (win_size - 1.f) / 2.f;
        }

        float sig2 = 2.f * shape * shape;

        for (int i = 0; i < win_size; i++) {
            win[i] = expf(-(win[i] * win[i]) / sig2);
        }
    }

}

static void rlucy_gaussian_kernel(rlucy_image_t *kern, const float beta, double *time) {
    float *win = NULL;
    const int radius = kern->w;

    if (kern->w != kern->h)
        printf("rlucy_gaussian_kernel: invalid kernel size\n");

//  window = np.kaiser(radius, beta)
    win = calloc(radius, sizeof(float));
    if (win) {
        rlucy_buildGaussianWindow(win, radius, beta);

//  kern = np.outer(window, window)
        rlucy_img_outer(win, radius, kern->im, kern->c);

        float kern_sum[4] = {0};

//  kern = kern / kern.sum()
        rlucy_img_sum(kern, kern_sum, time);
        rlucy_img_divide(kern, kern_sum, time);
    }

    if (win) free(win);
}

static void rlucy_uniform_kernel(rlucy_image_t *kern) {
    const int radius = kern->w;
    const float radius2 = 1.f / (radius * radius);
    const int size = kern->w * kern->h * kern->c;

    if (kern->w != kern->h)
        printf("rlucy_uniform_kernel: invalid kernel size\n");

    for (int i = 0; i < size; i++) {
        kern->im[i] = radius2;
    }
}

static void build_kernel(rlucy_image_t *psf, int blur_type, int blur_width, int ch, double *time) {
    rlucy_alloc_image(psf, blur_width, blur_width, ch);

    if (blur_type == rlucy_blur_type_gaussian) {
        rlucy_gaussian_kernel(psf, blur_width, time);
    } else if (blur_type == rlucy_blur_type_auto) {
        rlucy_uniform_kernel(psf);
    } else {
        printf("richardson_lucy: invalid blur type\n");
    }

}

/*
    This mimics a Darktable module inputs where the technical specs are hidden and translated into human-readable parameters.

    It's an interface between the regular user and the geeky actual deconvolution parameters

    :param pic: Blured image in RGB 8 bits as a 3D array where the last dimension is the RGB channel
    :param filename: File name to use to save the deblured picture
    :param destpath: The destination path to save the picture
    :param blur_type: kind of blur or "auto" to perform a blind deconvolution. Use "auto" for motion blur or composite blur.
    Other parameters : `poisson`, `gaussian`, `kaiser`
    :param blur_width: the width of the blur in px - must be an odd integer
    :param blur_strength: the strength of the blur, thus the standard deviation of the blur kernel
    :param auto_quality: when the `blur_type` is `auto`, the number of iterations of the initial blur estimation is half the
    square of the PSF size. The `auto_quality` parameter is a factor that allows you to reduce the number of iteration to speed up the process.
    Default : 1. Recommended values : between 0.25 and 2.
    :param noise_reduction_factor: the noise reduction factor lambda. For the `best` method, default is 1000, use 12000 to 30000 to speed up the convergence. Lower values don't help to reduce the noise, decrease the `ringing_facter` instead.
        for the `fast` method, default is 0.0006, increase up to 0.05 to reduce the noise. This unconsistency must be corrected soon.
    :param ringing_factor: the iterations factor. Typically 1e-3, reduce it to 5e-4 or ever 1e-4 if you see ringing or periodic edges appear.
    :param refine: True or False, decide if the blur kernel should be refined through myopic deconvolution
    :param refine_quality: the number of iterations to perform during the refinement step
    :param mask: the coordinates of the rectangular mask to apply on the image to refine the blur kernel from the top-left corner of the image
        in list [y_top, y_bottom, x_left, x_right]
    :param preview: If you want to fast preview your setting on a smaller picture size, set the downsampling ratio in `previem`. Default : 1
    :param psf: if you already know the PSF kernel, enter it here as an 3D array where the last dimension is the RGB component
    :param denoise: True or False. Perform an initial denoising by Total Variation - Chambolle algorithm before deconvoluting
    :param method: `fast` or `best`. Set the method to deconvolute.
    :return:
    """
    # TODO : refocus http://web.media.mit.edu/~bandy/refocus/PG07refocus.pdf
    # TODO : extract foreground only https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html#grabcut
*/
void rlucy_deblur_module(float *pic_in,
                         int blur_type,
                         int blur_width,
                         float noise_reduction_factor,
                         float quality,
                         float ringing_factor,
                         int refine,
                         int *mask,
                         float *psf_in,
                         int denoise,
                         int method,
                         float *pic_out,
                         int width,
                         int height,
                         int ch) {
    double time[MAX_TIME] = {0};

    RLUCY_TIME_DECL
    RLUCY_TIME_BEGIN


    // # Verify the input and scream like a virgin
    if (blur_width < 3) {
        printf("The PSF kernel is too small !\n");
        return;
    }
    if (blur_width % 2 == 0) {
        printf("The dimensions of the PSF must be odd !\n");
        return;
    }

    convolve_data_t fft_conv_data = {0};

    rlucy_image_t pic = {0};
    rlucy_image_t u = {0};
    rlucy_image_t psf = {0};
    rlucy_image_t img_tmp = {0};

    float noise_reduction_factor_a[4] = {noise_reduction_factor, noise_reduction_factor, noise_reduction_factor,
                                         noise_reduction_factor};
    float ringing_factor_a[4] = {ringing_factor, ringing_factor, ringing_factor, ringing_factor};

    rlucy_alloc_image(&pic, width, height, ch);
    if (pic.im == NULL) goto cleanup;
    memcpy(pic.im, pic_in, pic.w * pic.h * pic.c * sizeof(float));


    // # Make the picture dimensions odd to avoid ringing on the border of even pictures. We just replicate the last row/column
    int odd_vert = 0;
    int odd_hor = 0;

    if (pic.w % 2 == 0) {
        pad_image(&pic, 1, 0, &pic, &img_tmp, time);
        odd_hor = 1;
    }
    if (pic.h % 2 == 0) {
        pad_image(&pic, 0, 1, &pic, &img_tmp, time);
        odd_vert = 1;
    }

    if (psf_in == NULL) {
        build_kernel(&psf, blur_type, blur_width, ch, time);
    } else {
        rlucy_alloc_image(&psf, blur_width, blur_width, ch);
        if (psf.im == NULL) goto cleanup;
        memcpy(psf.im, psf_in, psf.w * psf.h * psf.c * sizeof(float));
    }


    rlucy_alloc_image(&u, pic.w, pic.h, pic.c);
    if (u.im == NULL) goto cleanup;
    rlucy_copy_image(&u, &pic);

    if (blur_type == rlucy_blur_type_auto) {
        printf("\n===== BLIND ESTIMATION OF BLUR =====\n");

        process_pyramid(&pic, &u, &psf, noise_reduction_factor, method, ringing_factor, quality, &fft_conv_data,
                        time);


    }

    if (refine) {
        if (mask) {
            printf("\n===== BLIND MASKED REFINEMENT =====\n");

            if (method == rlucy_method_fast)
                richardson_lucy_PAM(&pic, &u, &psf, noise_reduction_factor_a, quality, ringing_factor_a, mask, 1,
                                    &fft_conv_data, time);
            else if (method == rlucy_method_best)
                richardson_lucy_MM(&pic, &u, &psf, noise_reduction_factor_a, quality, ringing_factor_a, mask, 1,
                                   &fft_conv_data, time);

        } else {
            printf("\n===== BLIND UNMASKED REFINEMENT =====\n");

            if (method == rlucy_method_fast)
                richardson_lucy_PAM(&pic, &u, &psf, noise_reduction_factor_a, quality, ringing_factor_a, NULL, 1,
                                    &fft_conv_data, time);
            else if (method == rlucy_method_best)
                richardson_lucy_MM(&pic, &u, &psf, noise_reduction_factor_a, quality, ringing_factor_a, NULL, 1,
                                   &fft_conv_data, time);
        }
    }

    printf("\n===== REGULAR DECONVOLUTION =====\n");

    if (method == rlucy_method_fast)
        richardson_lucy_PAM(&pic, &u, &psf, noise_reduction_factor_a, quality, ringing_factor_a, NULL, 0,
                            &fft_conv_data, time);
    else if (method == rlucy_method_best)
        richardson_lucy_MM(&pic, &u, &psf, noise_reduction_factor_a, quality, ringing_factor_a, NULL, 0,
                           &fft_conv_data, time);


    // # if the picture has been padded to make it odd, unpad it to get the original size
    if (odd_hor) {
        /* u = u[:, :-1, ...] */
        unpad_image(&u, 1, 0, &u, &img_tmp, time);
    }
    if (odd_vert) {
        /* u = u[:-1, :, ...] */
        unpad_image(&u, 0, 1, &u, &img_tmp, time);
    }

    if (u.w != width || u.h != height || u.c != ch) {
        printf("rlucy_deblur_module: result with diffrent size than output image\n");
        printf("rlucy_deblur_module: u.w=%i, u.h=%i, u.c=%i\n", u.w, u.h, u.c);
        printf("rlucy_deblur_module: width=%i, height=%i, ch=%i\n", width, height, ch);

        rlucy_trim_mask(&u, &u, 0, 0, width, height, &img_tmp, time);
    }
    memcpy(pic_out, u.im, u.w * u.h * u.c * sizeof(float));

    if (psf_in) {
        if (psf.w == blur_width && psf.h == blur_width && psf.c == ch) {
            memcpy(psf_in, psf.im, psf.w * psf.h * psf.c * sizeof(float));
        } else {
            printf("rlucy_deblur_module: psf result with diffrent size than output psf\n");
            printf("rlucy_deblur_module: psf.w=%i, psf.h=%i, psf.c=%i\n", psf.w, psf.h, psf.c);
            printf("rlucy_deblur_module: blur_width=%i, ch=%i\n", blur_width, ch);
        }
    }

    RLUCY_TIME_END(TIME_PROCESS)

#ifdef RLUCY_TIME
    printf("\n");
    for (int i = 1; i < MAX_TIME; i++)
        if (time[i] > 0.0) printf("rlucy %i:%s took %0.06f sec\n", i, rlucy_time_desc[i], time[i]);
    printf("\n");
#endif

    cleanup:

    convolve_free(&fft_conv_data, time);

    rlucy_free_image(&u);
    rlucy_free_image(&psf);
    rlucy_free_image(&pic);
    rlucy_free_image(&img_tmp);

    printf("rlucy_deblur_module end\n");
}

