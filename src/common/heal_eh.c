
#include "control/control.h"
#include "develop/imageop.h"

/* Based on the original source code of GIMP's Healing Tool, by Jean-Yves Couleaud 
 * 
 * http://www.gimp.org/
 * 
 * */

/* NOTES
 *
 * The method used here is similar to the lighting invariant correction
 * method but slightly different: we do not divide the RGB components,
 * but subtract them I2 = I0 - I1, where I0 is the sample image to be
 * corrected, I1 is the reference pattern. Then we solve DeltaI=0
 * (Laplace) with I2 Dirichlet conditions at the borders of the
 * mask. The solver is a red/black checker Gauss-Seidel with over-relaxation.
 * It could benefit from a multi-grid evaluation of an initial solution
 * before the main iteration loop.
 *
 * I reduced the convergence criteria to 0.1% (0.001) as we are
 * dealing here with RGB integer components, more is overkill.
 *
 * Jean-Yves Couleaud cjyves@free.fr
 */

#define _FFT_MULTFR_


// Subtract bottom from top and store in result as a float
static void dt_dev_heal_sub(const float *const top_buffer, const float *const bottom_buffer, float *result_buffer, const int width,  const int height, const int ch)
{
  
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(result_buffer) schedule(static)
#endif
#endif
  for (int y=0; y < height; y++)
  {
    size_t index = (size_t)y * width * ch;
    float *t = (float *)top_buffer + index;
    float *b = (float *)bottom_buffer + index;
    float *r = (float *)result_buffer + index;
    for (int x=0; x < width * ch; x++)
    {
      *r++ = *t++ - *b++;
    }
  }

}

// Add first to second and store in result
static void dt_dev_heal_add(const float *const first_buffer, const float *const second_buffer, float *result_buffer, const int width,  const int height, const int ch)
{
  
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(result_buffer) schedule(static)
#endif
#endif
  for (int y=0; y < height; y++)
  {
    size_t index = (size_t)y * width * ch;
    float *f = (float *)first_buffer + index;
    float *s = (float *)second_buffer + index;
    float *r = (float *)result_buffer + index;
    for (int x=0; x < width * ch; x++)
    {
      *r++ = *f++ + *s++;
    }
  }
}

//#if defined(__SSE__) && defined(__GNUC__) && __GNUC__ >= 4
#if defined(__SSE__)
static float
dt_dev_heal_laplace_iteration_sse(float *pixels, const float *const Adiag, const int *const Aidx, const float w, const int nmask)
{
  typedef float v4sf __attribute__((vector_size(16)));
  int i;
  v4sf wv  = { w, w, w, w };
  v4sf err = { 0, 0, 0, 0 };
  union { v4sf v; float f[4]; } erru;

#define Xv(j) (*(v4sf*)&pixels[Aidx[i * 5 + j]])

#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(pixels, wv, err) schedule(static)
#endif
#endif
  for (i = 0; i < nmask; i++)
    {
      v4sf a    = { Adiag[i], Adiag[i], Adiag[i], Adiag[i] };
      v4sf diff = a * Xv(0) - wv * (Xv(1) + Xv(2) + Xv(3) + Xv(4));

      Xv(0) -= diff;
      err += diff * diff;
    }

  erru.v = err;

  return erru.f[0] + erru.f[1] + erru.f[2] + erru.f[3];
}
#endif

// Perform one iteration of Gauss-Seidel, and return the sum squared residual.
static float dt_dev_heal_laplace_iteration(float *pixels, const float *const Adiag, const int *const Aidx, const float w, const int nmask, 
    const int depth, const float preview_scale, const int use_sse)
{
  float err = 0;

//#if defined(__SSE__) && defined(__GNUC__) && __GNUC__ >= 4
#if defined(__SSE__)
  if (depth == 4 && use_sse)
    return dt_dev_heal_laplace_iteration_sse (pixels, Adiag, Aidx, w, nmask);
#endif

#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(pixels, err) schedule(static)
#endif
#endif
  for (int i = 0; i < nmask; i++)
  {
    int   j0 = Aidx[i * 5 + 0];
    int   j1 = Aidx[i * 5 + 1];
    int   j2 = Aidx[i * 5 + 2];
    int   j3 = Aidx[i * 5 + 3];
    int   j4 = Aidx[i * 5 + 4];
    float a  = Adiag[i];

    for (int k = 0; k < depth; k++)
    {
      float diff = (a * pixels[j0 + k] -
                    w * (pixels[j1 + k] +
                         pixels[j2 + k] +
                         pixels[j3 + k] +
                         pixels[j4 + k]));

      pixels[j0 + k] -= diff;
      err += diff * diff;
    }
  }

  return err;
}

// Solve the laplace equation for pixels and store the result in-place.
static void dt_dev_heal_laplace_loop(float *pixels,
                                    const int    height,
                                    const int    depth,
                                    const int    width,
                                    const float *const mask, 
                                    const float preview_scale, 
                                    const int use_sse)
{
  /* Tolerate a total deviation-from-smoothness of 0.1 LSBs at 8bit depth. */
  const int max_iter = 500;
  const float epsilon = .1f/255.f;
  int nmask;
  float *Adiag;
  int   *Aidx;

  Adiag = dt_alloc_align(64, sizeof(float) * width * height);
  Aidx  = dt_alloc_align(64, sizeof(int) * 5 * width * height);

  /* All off-diagonal elements of A are either -1 or 0. We could store it as a
   * general-purpose sparse matrix, but that adds some unnecessary overhead to
   * the inner loop. Instead, assume exactly 4 off-diagonal elements in each
   * row, all of which have value -1. Any row that in fact wants less than 4
   * coefs can put them in a dummy column to be multiplied by an empty pixel.
   */
  const int zero = depth * width * height;
  memset (pixels + zero, 0, depth * sizeof (float));

  /* Construct the system of equations.
   * Arrange Aidx in checkerboard order, so that a single linear pass over that
   * array results updating all of the red cells and then all of the black cells.
   */
  nmask = 0;
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(nmask, Adiag, Aidx) schedule(static)
#endif
#endif
  for (int parity = 0; parity < 2; parity++)
    for (int i = 0; i < height; i++)
      for (int j = (i&1)^parity; j < width; j+=2)
        if (mask[j + i * width])
          {
#define A_NEIGHBOR(o,di,dj) \
            if ((dj<0 && j==0) || (dj>0 && j==width-1) || (di<0 && i==0) || (di>0 && i==height-1)) \
              Aidx[o + nmask * 5] = zero; \
            else                                               \
              Aidx[o + nmask * 5] = ((i + di) * width + (j + dj)) * depth;

            /* Omit Dirichlet conditions for any neighbors off the
             * edge of the canvas.
             */
            Adiag[nmask] = 4 - (i==0) - (j==0) - (i==height-1) - (j==width-1);
            A_NEIGHBOR (0,  0,  0);
            A_NEIGHBOR (1,  0,  1);
            A_NEIGHBOR (2,  1,  0);
            A_NEIGHBOR (3,  0, -1);
            A_NEIGHBOR (4, -1,  0);
            nmask++;
          }

  /* Empirically optimal over-relaxation factor. (Benchmarked on
   * round brushes, at least. I don't know whether aspect ratio
   * affects it.)
   */
  const float w = (2.0 - 1.0 / (0.1575 * sqrt (nmask) + 0.8)) * .25f;
  const float err_exit = epsilon * epsilon * w * w;
  
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(nmask, Adiag) schedule(static)
#endif
#endif
  for (int i = 0; i < nmask; i++)
    Adiag[i] *= w;

  /* Gauss-Seidel with successive over-relaxation */
  for (int iter = 0; iter < max_iter; iter++)
  {
    float err = dt_dev_heal_laplace_iteration (pixels, Adiag, Aidx, w, nmask, depth, preview_scale, use_sse);
    if (err < err_exit)
      break;
  }

  dt_free_align(Adiag);
  dt_free_align(Aidx);
}

/* Original Algorithm Design:
 *
 * T. Georgiev, "Photoshop Healing Brush: a Tool for Seamless Cloning
 * http://www.tgeorgiev.net/Photoshop_Healing.pdf
 */

void dt_dev_heal(const float *const src_buffer, float *dest_buffer, const float *const mask_buffer, const int width,  const int height, const int ch, 
    const float preview_scale, const int use_sse)
{
  float *diff_alloc;
  float *diff_buffer;

  diff_alloc = dt_alloc_align(64, sizeof(float) * (4 + (width * height + 1) * ch));
  diff_buffer = (float*)(((uintptr_t)diff_alloc + 15) & ~15);

  /* subtract pattern from image and store the result as a float in diff */
  dt_dev_heal_sub(dest_buffer, src_buffer, diff_buffer, width, height, ch);

  dt_dev_heal_laplace_loop(diff_buffer, height, ch, width, mask_buffer, preview_scale, use_sse);

  /* add solution to original image and store in dest */
  dt_dev_heal_add(diff_buffer, src_buffer, dest_buffer, width, height, ch);

//  dt_free_align(diff_buffer);
  dt_free_align(diff_alloc);
}

