
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


// Subtract bottom from top and store in result as a float
static void dt_heal_sub(const float *const top_buffer, const float *const bottom_buffer, float *result_buffer, const int width, const int height, const int ch)
{
  const int i_size = width * height * ch;
  
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(result_buffer) schedule(static)
#endif
  for (int i = 0; i < i_size; i++) result_buffer[i] = top_buffer[i] - bottom_buffer[i];
}

// Add first to second and store in result
static void dt_heal_add(const float *const first_buffer, const float *const second_buffer, float *result_buffer, const int width, const int height, const int ch)
{
  const int i_size = width * height * ch;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(result_buffer) schedule(static)
#endif
  for (int i = 0; i < i_size; i++) result_buffer[i] = first_buffer[i] + second_buffer[i];
}

#if defined(__SSE__)
static float dt_heal_laplace_iteration_sse(float *pixels, const float *const Adiag, const int *const Aidx, const float w, const int nmask)
{
  union { __m128 v; float f[4]; } valb_err = {0};

  for (int i = 0; i < nmask; i++)
  {
    const int ii = i * 5;
    const int j0 = Aidx[ii + 0];
    const int j1 = Aidx[ii + 1];
    const int j2 = Aidx[ii + 2];
    const int j3 = Aidx[ii + 3];
    const int j4 = Aidx[ii + 4];
    
    __m128 valb_a = _mm_set1_ps(Adiag[i]);
    __m128 valb_w= { w, w, w, w };

    __m128 valb_j0 = _mm_load_ps(&(pixels[j0])); // center
    __m128 valb_j1 = _mm_load_ps(&(pixels[j1])); // E
    __m128 valb_j2 = _mm_load_ps(&(pixels[j2])); // S
    __m128 valb_j3 = _mm_load_ps(&(pixels[j3])); // W
    __m128 valb_j4 = _mm_load_ps(&(pixels[j4])); // N

/*  float diff = w * (a * pixels[j0 + k] -
                        (pixels[j1 + k] +
                         pixels[j2 + k] +
                         pixels[j3 + k] +
                         pixels[j4 + k]));*/
    __m128 valb_diff = _mm_mul_ps(valb_w, 
                                  _mm_sub_ps(_mm_mul_ps(valb_a, valb_j0), 
                                      _mm_add_ps(valb_j1, 
                                          _mm_add_ps(valb_j2, 
                                              _mm_add_ps(valb_j3, valb_j4)))));

/*  pixels[j0 + k] -= diff;*/
    _mm_store_ps(&(pixels[j0]), _mm_sub_ps(valb_j0, valb_diff));
/*  err += diff * diff;*/
    valb_err.v = _mm_add_ps(valb_err.v, _mm_mul_ps(valb_diff, valb_diff));
  }

  return valb_err.f[0] + valb_err.f[1] + valb_err.f[2];
}
#endif

// Perform one iteration of Gauss-Seidel, and return the sum squared residual.
static float dt_heal_laplace_iteration(float *pixels, const float *const Adiag, const int *const Aidx, const float w, const int nmask, 
                                        const int ch, const int use_sse)
{
#if defined(__SSE__)
  if (ch == 4 && use_sse)
    return dt_heal_laplace_iteration_sse(pixels, Adiag, Aidx, w, nmask);
#endif

  float err = 0;
  const int ch1 = (ch==4) ? ch-1: ch;

  for (int i = 0; i < nmask; i++)
  {
    int   j0 = Aidx[i * 5 + 0];
    int   j1 = Aidx[i * 5 + 1];
    int   j2 = Aidx[i * 5 + 2];
    int   j3 = Aidx[i * 5 + 3];
    int   j4 = Aidx[i * 5 + 4];
    float a  = Adiag[i];

    for (int k = 0; k < ch1; k++)
    {
      float diff = w * (a * pixels[j0 + k] -
                        (pixels[j1 + k] +
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
static void dt_heal_laplace_loop(float *pixels, const int width, const int height, const int ch, const float *const mask, const int use_sse)
{
  const int max_iter = 1000;
  int nmask = 0;

  float *Adiag = dt_alloc_align(64, sizeof(float) * width * height);
  int *Aidx = dt_alloc_align(64, sizeof(int) * 5 * width * height);

  if ((Adiag == NULL) || (Aidx == NULL))
  {
    printf("dt_heal_laplace_loop: error allocating memory for healing\n");
    goto cleanup;
  }

  /* All off-diagonal elements of A are either -1 or 0. We could store it as a
   * general-purpose sparse matrix, but that adds some unnecessary overhead to
   * the inner loop. Instead, assume exactly 4 off-diagonal elements in each
   * row, all of which have value -1. Any row that in fact wants less than 4
   * coefs can put them in a dummy column to be multiplied by an empty pixel.
   */
  const int zero = ch * width * height;
  memset (pixels + zero, 0, ch * sizeof (float));

  /* Construct the system of equations.
   * Arrange Aidx in checkerboard order, so that a single linear pass over that
   * array results updating all of the red cells and then all of the black cells.
   */
  for (int parity = 0; parity < 2; parity++)
  {
    for (int i = 0; i < height; i++)
    {
      for (int j = (i&1)^parity; j < width; j+=2)
      {
        if (mask[j + i * width])
				{
#define A_NEIGHBOR(o,di,dj) \
					if ((dj<0 && j==0) || (dj>0 && j==width-1) || (di<0 && i==0) || (di>0 && i==height-1)) \
						Aidx[o + nmask * 5] = zero; \
					else                                               \
						Aidx[o + nmask * 5] = ((i + di) * width + (j + dj)) * ch;
			
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
      }
    }
  }
  
  /* Empirically optimal over-relaxation factor. (Benchmarked on
   * round brushes, at least. I don't know whether aspect ratio
   * affects it.)
   */
  float w = ((2.0f - 1.0f / (0.1575f * sqrtf(nmask) + 0.8f)) * .25f);
  const float err_exit = .001f;
  
  /* Gauss-Seidel with successive over-relaxation */
  for (int iter = 0; iter < max_iter; iter++)
  {
    float err = dt_heal_laplace_iteration(pixels, Adiag, Aidx, w, nmask, ch, use_sse);
    if (err <= err_exit)
      break;
  }

cleanup:
  if (Adiag) dt_free_align(Adiag);
  if (Aidx) dt_free_align(Aidx);
}


/* Original Algorithm Design:
 *
 * T. Georgiev, "Photoshop Healing Brush: a Tool for Seamless Cloning
 * http://www.tgeorgiev.net/Photoshop_Healing.pdf
 */
void dt_heal(const float *const src_buffer, float *dest_buffer, const float *const mask_buffer, 
              const int width, const int height, const int ch, const int use_sse)
{
  float *diff_buffer = dt_alloc_align(64, width * (height + 1) * ch * sizeof(float));

  if (diff_buffer == NULL)
  {
    printf("dt_heal: error allocating memory for healing\n");
    goto cleanup;
  }

  /* subtract pattern from image and store the result in diff */
  dt_heal_sub(dest_buffer, src_buffer, diff_buffer, width, height, ch);

  dt_heal_laplace_loop(diff_buffer, width, height, ch, mask_buffer, use_sse);

  /* add solution to original image and store in dest */
  dt_heal_add(diff_buffer, src_buffer, dest_buffer, width, height, ch);

cleanup:
  if (diff_buffer) dt_free_align(diff_buffer);
}


