
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
static void dt_heal_sub(const float *const top_buffer, const float *const bottom_buffer, float *result_buffer, const int width, const int height, const int ch)
{
  const int i_size = width * height * ch;
  
#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(result_buffer) schedule(static)
#endif
#endif
  for (int i = 0; i < i_size; i++) result_buffer[i] = top_buffer[i] - bottom_buffer[i];
}

// Add first to second and store in result
static void dt_heal_add(const float *const first_buffer, const float *const second_buffer, float *result_buffer, const int width, const int height, const int ch)
{
  const int i_size = width * height * ch;

#ifdef _FFT_MULTFR_
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(result_buffer) schedule(static)
#endif
#endif
  for (int i = 0; i < i_size; i++) result_buffer[i] = first_buffer[i] + second_buffer[i];
}

//#if defined(__SSE__) && defined(__GNUC__) && __GNUC__ >= 4
#if defined(__SSE__x)
static float
dt_heal_laplace_iteration_sse(float *pixels, const float *const Adiag, const int *const Aidx, const float w, const int nmask)
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
static float dt_heal_laplace_iteration(float *pixels, const float *const Adiag, const int *const Aidx, const float w, const int nmask, 
    const int ch, const float preview_scale, const int use_sse)
{
//#if defined(__SSE__) && defined(__GNUC__) && __GNUC__ >= 4
#if defined(__SSE__x)
  if (ch == 4 && use_sse)
    return dt_heal_laplace_iteration_sse (pixels, Adiag, Aidx, w, nmask);
#endif

  float err = 0;
  const int ch1 = (ch==4) ? ch-1: ch;

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

    for (int k = 0; k < ch1; k++)
    {
/*      float diff = (a * pixels[j0 + k] -
                     w * (pixels[j1 + k] +
                          pixels[j2 + k] +
                          pixels[j3 + k] +
                          pixels[j4 + k]));*/
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
static void dt_heal_laplace_loop(float *pixels,
                                    const int    width,
                                    const int    height,
                                    const int    ch,
                                    const float *const mask, 
                                    const float softness, 
                                    const float preview_scale, 
                                    const int use_sse)
{
  /* Tolerate a total deviation-from-smoothness of 0.1 LSBs at 8bit depth. */
  const int max_iter = 1500;
  int nmask = 0;
//  const float epsilon = .1f/255.f;
//  const int ch1 = (ch==4) ? ch-1: ch;

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

  /* Empirically optimal over-relaxation factor. (Benchmarked on
   * round brushes, at least. I don't know whether aspect ratio
   * affects it.)
   */
  float w = ((2.0 - 1.0 / (0.1575 * sqrtf(nmask) * softness + 0.8)) * .25f);
//  float w = (.43f) / (preview_scale*preview_scale)/* + (softness * 1000.f)*//* * preview_scale * preview_scale*/;
//  w = MIN(w, .499f);
//  const float w = .49f;
//  const float w = (2.0 - 1.0 / (0.1575 * sqrt ((width*height*ch1)*preview_scale)/* *preview_scale*preview_scale*/ + 0.8)) * .25f;
//  const float err_exit = (epsilon * epsilon * w * w) + (softness * 1000.f);
//  const float err_exit = MAX( ((softness) / (preview_scale*preview_scale)), 0.0f );
//  const float err_exit = MAX( ((0.) / (preview_scale*preview_scale)), 0.0f );
  int iter = 0;
  float err = 0;
  const float err_exit = .01f / (float)nmask/* * preview_scale * preview_scale*/;
  
  
//  for (int i = 0; i < nmask; i++)
//    Adiag[i] *= w;

//  max_iter -= (int)(softness * 10000.f);
  /* Gauss-Seidel with successive over-relaxation */
  for (iter = 0; iter < max_iter; iter++)
  {
    err = dt_heal_laplace_iteration(pixels, Adiag, Aidx, w, nmask, ch, preview_scale, use_sse) / (float)nmask;
    if (err <= err_exit)
      break;
  }

  printf("dt_heal iter=%i, err_exit=%f, w=%f, err=%f\n", iter, err_exit*1000.f, w, err);

cleanup:
  if (Adiag) dt_free_align(Adiag);
  if (Aidx) dt_free_align(Aidx);
}


/* Original Algorithm Design:
 *
 * T. Georgiev, "Photoshop Healing Brush: a Tool for Seamless Cloning
 * http://www.tgeorgiev.net/Photoshop_Healing.pdf
 */

void dt_heal(const float *const src_buffer, float *dest_buffer, const float *const mask_buffer, const int width, const int height, const int ch, 
    const float softness, const float preview_scale, const int use_sse)
{
//  float *diff_alloc;
//  float *diff_buffer;

//  diff_alloc = dt_alloc_align(64, sizeof(float) * (4 + (width * height + 1) * ch));
//  diff_buffer = (float*)(((uintptr_t)diff_alloc + 15) & ~15);
  float *diff_buffer = dt_alloc_align(64, width * (height + 1) * ch * sizeof(float));

  /* subtract pattern from image and store the result in diff */
  dt_heal_sub(dest_buffer, src_buffer, diff_buffer, width, height, ch);

  dt_heal_laplace_loop(diff_buffer, width, height, ch, mask_buffer, softness, preview_scale, use_sse);

  /* add solution to original image and store in dest */
  dt_heal_add(diff_buffer, src_buffer, dest_buffer, width, height, ch);

  dt_free_align(diff_buffer);
}


// test a new algorithm
    
static int get_steps(const float *const mask, const int width, const int height)
{
  int steps = 0;
  for (int i = 0; i < width * height; i++) if (mask[i]) steps++;
  return (steps);
}

static float laplacian(const float *const image_src, float *image_dest, const int width, const int height, const int ch)
{
  float diff = 0.f;
  
  memcpy(image_dest, image_src, width * ch * sizeof(float));
  memcpy(image_dest + (height-1) * width * ch, image_src + (height-1) * width * ch, width * ch * sizeof(float));
  
  for (int y = 1; y < height-1; y++)
  {
    for (int c = 0; c < ch-1; c++)
    {
      image_dest[(y * width * ch) + c] = image_src[(y * width * ch) + c];
      image_dest[(y * width * ch) + ((width-1) * ch) + c] = image_src[(y * width * ch) + ((width-1) * ch) + c];
    }
    
    for (int x = 1; x < width-1; x++)
    {
      for (int c = 0; c < ch-1; c++)
      {
        float k = ( image_src[(y * width * ch) + (x * ch) + c] * 4.f - 
            image_src[(y * width * ch) + ((x-1) * ch) + c] - 
            image_src[(y * width * ch) + ((x+1) * ch) + c] - 
            image_src[((y-1) * width * ch) + (x * ch) + c] -
            image_src[((y+1) * width * ch) + (x * ch) + c] );

        diff += (image_dest[(y * width * ch) + (x * ch) + c] - k) * (image_dest[(y * width * ch) + (x * ch) + c] - k);
        
        image_dest[(y * width * ch) + (x * ch) + c] = k;
      }
    }
  }
  
  return diff;
}

static float laplacian_iter(const float *const image, float *image1, const float *const image2, const int width, const int height, const int ch)
{
  float diff = 0.f;
  
  memcpy(image1, image2, width * ch * sizeof(float));
  memcpy(image1 + (height-1) * width * ch, image2 + (height-1) * width * ch, width * ch * sizeof(float));
  
  for (int y = 1; y < height-1; y++)
  {
    for (int c = 0; c < ch-1; c++)
    {
      image1[(y * width * ch) + c] = image2[(y * width * ch) + c];
      image1[(y * width * ch) + ((width-1) * ch) + c] = image2[(y * width * ch) + ((width-1) * ch) + c];
    }
    
    for (int x = 1; x < width-1; x++)
    {
      for (int c = 0; c < ch-1; c++)
      {
        float k = ( image[(y * width * ch) + (x * ch) + c] + 
            image2[(y * width * ch) + ((x-1) * ch) + c] + 
            image2[(y * width * ch) + ((x+1) * ch) + c] + 
            image2[((y-1) * width * ch) + (x * ch) + c] + 
            image2[((y+1) * width * ch) + (x * ch) + c] ) / 4.0f;

        diff += (image1[(y * width * ch) + (x * ch) + c] - k) * (image1[(y * width * ch) + (x * ch) + c] - k);
        
        image1[(y * width * ch) + (x * ch) + c] = k;
      }
    }
  }
  return diff;
}

void dt_heal1(const float *const src_buffer, float *dest_buffer, const float *const mask_buffer, const int width, const int height, const int ch, 
    const float softness, const float preview_scale, const int use_sse)
{
  float *dest1 = dt_alloc_align(64, width * height * ch * sizeof(float));
  float *dest2 = dt_alloc_align(64, width * height * ch * sizeof(float));
  
  memcpy(dest1, dest_buffer, width * height * ch * sizeof(float));
  memset(dest_buffer, 0, width * height * ch * sizeof(float));
  memset(dest2, 0, width * height * ch * sizeof(float));
  
  laplacian(src_buffer, dest_buffer, width, height, ch);
  
  unsigned int steps = get_steps(mask_buffer, width, height) / preview_scale;
//  steps *= steps;
  steps = MIN(steps, 2000);
  int steps2 = 0;
  float diff = 0.f;
//  const float diff_exit = MAX( ((0.001f + softness) / (preview_scale*preview_scale)), 0.00001f );
  const float diff_exit = MAX( ((softness) / (preview_scale*preview_scale)), 0.0f );
  
  printf("dt_heal2 diff_exit=%f\n", diff_exit);

  if (1){
  for (int i = 0; i < steps; i++)
  {
    laplacian_iter(dest_buffer, dest2, dest1, width, height, ch);
    diff = laplacian_iter(dest_buffer, dest1, dest2, width, height, ch);
    steps2++;
    if (diff <= diff_exit) break;
  }
  }
  if(0) {
  for (int i = 0; i < steps; i++)
  {
    laplacian(dest_buffer, dest1, width, height, ch);
    diff = laplacian(dest1, dest_buffer, width, height, ch);
    steps2++;
    if (diff <= diff_exit) break;
  }
  }
  printf("dt_heal2 steps=%i, steps2=%i, diff=%f\n", steps, steps2, diff);
  
  memcpy(dest_buffer, dest1, width * height * ch * sizeof(float));
  
  if (dest1) dt_free_align(dest1);
  if (dest2) dt_free_align(dest2);
}

