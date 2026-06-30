#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#define SS_RESTRICT __restrict
#else
#define SS_RESTRICT __restrict__
#endif

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// structstats -- single-pass gradient structure-tensor statistics.
//
// For uint8 luma the 3x3 Sobel responses Gx, Gy are exact integers, so the
// products Gx^2, Gy^2, Gx*Gy and their per-cell sums are exact integers: the
// hot loop is pure integer, no floating point. The image is read exactly ONCE
// whatever the pyramid depth -- every pixel scatters only into the FINEST grid,
// and coarser dyadic levels (exact sums of finer cells) are built by aggregating
// that small grid, not by re-scanning the image. Sobel is applied separably via
// per-row caches so neighbouring pixels reuse the vertical pass. Input may be a
// strided 2-D view (e.g. one interleaved HSV channel) -- read in place, no copy.
//
// raw():      one (cells_y, cells_x, 4) int64 array per level, [Sxx, Syy, Sxy, n].
// features(): the same in one call, then derived float32 maps (NF channels) per
//             cell, computed in C++ from the 3 sums (a closed form, no extra pass,
//             no atan2 -- orientation is the continuous double-angle vector).
// Replicate border; cell boundaries at full resolution (stride subsamples which
// pixels accumulate). Pyramid must be nested finest->coarsest; finest divides the
// image. eigenvalues do not pool across scales -- the sums do -- so coarse levels
// are summed from finer ones and derived per level.
// ---------------------------------------------------------------------------

namespace
{

  constexpr int SXX = 0, SYY = 1, SXY = 2, CNT = 3, NCOMP = 4;
  // Derived feature channels (float32), per cell:
  //   0 energy     = trace / n              (mean squared-gradient magnitude)
  //   1 coherence  = |lambda1-lambda2| / trace in [0,1] (0 isotropic, 1 single edge)
  //   2 ori_cos    = (Sxx-Syy) / trace      = coherence * cos(2*theta)
  //   3 ori_sin    = 2 Sxy / trace          = coherence * sin(2*theta)
  //   4 cornerness = lambda_min / n         (Shi-Tomasi, mean)
  constexpr int NF = 5;

  inline int clampi(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

  inline void derive_cell(const int64_t *SS_RESTRICT s, float *SS_RESTRICT f)
  {
    const double sxx = (double)s[SXX], syy = (double)s[SYY], sxy = (double)s[SXY];
    const double n = (double)s[CNT];
    const double tr = sxx + syy, d = sxx - syy;
    const double R = std::sqrt(d * d + 4.0 * sxy * sxy); // lambda1-lambda2
    const double invn = n > 0.0 ? 1.0 / n : 0.0;
    const double invt = tr > 0.0 ? 1.0 / tr : 0.0;
    f[0] = (float)(tr * invn);
    f[1] = (float)(R * invt);
    f[2] = (float)(d * invt);
    f[3] = (float)(2.0 * sxy * invt);
    f[4] = (float)((tr - R) * 0.5 * invn);
  }

  struct Level
  {
    int ny = 0, nx = 0;         // cells per axis (2^exp)
    int fy = 0, fx = 0;         // block factor down from the previous (finer) level
    std::vector<int64_t> buf;   // ny*nx*NCOMP raw sums, retained
    std::vector<float> feat;    // ny*nx*NF derived, retained
    std::vector<size_t> rshape; // {ny, nx, NCOMP}
    std::vector<size_t> fshape; // {ny, nx, NF}
  };

  class StructComputer
  {
    int h_ = 0, w_ = 0, sy_ = 1, sx_ = 1;
    bool want_global_ = false;
    std::vector<Level> levels_;    // finest first
    std::vector<int> row_cell_;    // finest row binning (full-res boundaries)
    std::vector<int16_t> vs_, vd_; // per-row separable Sobel caches
    int64_t graw_[NCOMP] = {0};    // global raw sums
    float gfeat_[NF] = {0};        // global derived
    std::vector<size_t> graw_shape_{(size_t)NCOMP}, gfeat_shape_{(size_t)NF};

    // Single pass over the image -> finest grid; then aggregate coarser levels and
    // (if requested) the global tensor. rs, cs are the input strides in elements.
    void accumulate(const uint8_t *SS_RESTRICT img, int64_t rs, int64_t cs)
    {
      const int hi_r = h_ - 1, hi_c = w_ - 1;
      Level &fine = levels_[0];
      const int nfx = fine.nx, cw = w_ / nfx;
      int64_t *SS_RESTRICT fb = fine.buf.data();
      int16_t *SS_RESTRICT vs = vs_.data();
      int16_t *SS_RESTRICT vd = vd_.data();
      std::fill(fine.buf.begin(), fine.buf.end(), (int64_t)0);

      for (int r = 0; r < h_; r += sy_)
      {
        const uint8_t *SS_RESTRICT r0 = img + (int64_t)clampi(r - 1, hi_r) * rs;
        const uint8_t *SS_RESTRICT r1 = img + (int64_t)r * rs;
        const uint8_t *SS_RESTRICT r2 = img + (int64_t)clampi(r + 1, hi_r) * rs;
        for (int c = 0; c < w_; ++c)
        { // vertical pass, reused by neighbours
          const int64_t k = (int64_t)c * cs;
          vs[c] = (int16_t)(r0[k] + 2 * r1[k] + r2[k]);
          vd[c] = (int16_t)(r2[k] - r0[k]);
        }
        int64_t *SS_RESTRICT rb = fb + (size_t)(row_cell_[r] * nfx) * NCOMP;
        for (int e = 0; e < nfx; ++e)
        { // one finest cell = one column run
          int64_t axx = 0, ayy = 0, axy = 0, acnt = 0;
          const int c1 = e * cw + cw;
          for (int c = e * cw; c < c1; c += sx_)
          {
            const int cm = clampi(c - 1, hi_c), cp = clampi(c + 1, hi_c);
            const int gx = vs[cp] - vs[cm];
            const int gy = vd[cm] + 2 * vd[c] + vd[cp];
            axx += (int64_t)gx * gx;
            ayy += (int64_t)gy * gy;
            axy += (int64_t)gx * gy;
            ++acnt;
          }
          int64_t *SS_RESTRICT acc = rb + (size_t)e * NCOMP;
          acc[SXX] += axx;
          acc[SYY] += ayy;
          acc[SXY] += axy;
          acc[CNT] += acnt;
        }
      }

      for (size_t k = 1; k < levels_.size(); ++k)
      { // cascade-aggregate coarser
        const Level &p = levels_[k - 1];
        Level &L = levels_[k];
        const int64_t *SS_RESTRICT pb = p.buf.data();
        int64_t *SS_RESTRICT lb = L.buf.data();
        const int pnx = p.nx;
        std::fill(L.buf.begin(), L.buf.end(), (int64_t)0);
        for (int i = 0; i < p.ny; ++i)
        {
          const int ci = (i / L.fy) * L.nx;
          for (int j = 0; j < pnx; ++j)
          {
            const int64_t *SS_RESTRICT s = pb + ((size_t)i * pnx + j) * NCOMP;
            int64_t *SS_RESTRICT d = lb + ((size_t)(ci + j / L.fx)) * NCOMP;
            d[SXX] += s[SXX];
            d[SYY] += s[SYY];
            d[SXY] += s[SXY];
            d[CNT] += s[CNT];
          }
        }
      }

      if (want_global_)
      {
        const Level &last = levels_.back();
        const int64_t *SS_RESTRICT lb = last.buf.data();
        const size_t nc = (size_t)last.ny * last.nx;
        int64_t g0 = 0, g1 = 0, g2 = 0, g3 = 0;
        for (size_t c = 0; c < nc; ++c)
        {
          const int64_t *SS_RESTRICT s = lb + c * NCOMP;
          g0 += s[0];
          g1 += s[1];
          g2 += s[2];
          g3 += s[3];
        }
        graw_[0] = g0;
        graw_[1] = g1;
        graw_[2] = g2;
        graw_[3] = g3;
      }
    }

    void derive()
    {
      for (Level &L : levels_)
      {
        const int64_t *SS_RESTRICT b = L.buf.data();
        float *SS_RESTRICT f = L.feat.data();
        const size_t nc = (size_t)L.ny * L.nx;
        for (size_t c = 0; c < nc; ++c)
          derive_cell(b + c * NCOMP, f + c * NF);
      }
      if (want_global_)
        derive_cell(graw_, gfeat_);
    }

  public:
    StructComputer() = default;

    void set_config(const std::vector<int64_t> &shape,
                    const std::vector<std::vector<int>> &grids,
                    const std::vector<int64_t> &stride, bool want_global)
    {
      h_ = (int)shape[0];
      w_ = (int)shape[1];
      sy_ = (int)stride[0];
      sx_ = (int)stride[1];
      want_global_ = want_global;

      levels_.clear();
      for (size_t k = 0; k < grids.size(); ++k)
      {
        Level L;
        L.ny = 1 << grids[k][0];
        L.nx = 1 << grids[k][1];
        if (k == 0)
        {
          if (h_ % L.ny != 0 || w_ % L.nx != 0)
            throw std::invalid_argument("finest grid must divide the image shape");
        }
        else
        {
          const Level &p = levels_[k - 1];
          if (p.ny % L.ny != 0 || p.nx % L.nx != 0)
            throw std::invalid_argument(
                "pyramid levels must be nested (each coarser grid divides the finer)");
          L.fy = p.ny / L.ny;
          L.fx = p.nx / L.nx;
        }
        L.buf.assign((size_t)L.ny * L.nx * NCOMP, 0);
        L.feat.assign((size_t)L.ny * L.nx * NF, 0.0f);
        L.rshape = {(size_t)L.ny, (size_t)L.nx, (size_t)NCOMP};
        L.fshape = {(size_t)L.ny, (size_t)L.nx, (size_t)NF};
        levels_.push_back(std::move(L));
      }

      const Level &f = levels_[0];
      row_cell_.resize(h_);
      for (int r = 0; r < h_; ++r)
        row_cell_[r] = (int)((int64_t)r * f.ny / h_);
      vs_.resize(w_);
      vd_.resize(w_);
    }

    nb::list raw(const uint8_t *SS_RESTRICT img, int64_t rs, int64_t cs)
    {
      accumulate(img, rs, cs);
      nb::list out;
      for (Level &L : levels_)
        out.append(nb::ndarray<nb::numpy, int64_t>(L.buf.data(), L.rshape.size(),
                                                   L.rshape.data(), nb::handle()));
      if (want_global_)
        out.append(nb::ndarray<nb::numpy, int64_t>(graw_, graw_shape_.size(),
                                                   graw_shape_.data(), nb::handle()));
      return out;
    }

    nb::list features(const uint8_t *SS_RESTRICT img, int64_t rs, int64_t cs)
    {
      accumulate(img, rs, cs);
      derive();
      nb::list out;
      for (Level &L : levels_)
        out.append(nb::ndarray<nb::numpy, float>(L.feat.data(), L.fshape.size(),
                                                 L.fshape.data(), nb::handle()));
      if (want_global_)
        out.append(nb::ndarray<nb::numpy, float>(gfeat_, gfeat_shape_.size(),
                                                 gfeat_shape_.data(), nb::handle()));
      return out;
    }
  };

  using Arr = nb::ndarray<nb::numpy, const uint8_t, nb::ndim<2>, nb::device::cpu>;

} // namespace

NB_MODULE(structstats_core, m)
{
  m.doc() = "structstats internal C++ module. Public API: structstats.StructComputer.";
  nb::class_<StructComputer>(m, "_StructComputerImpl")
      .def(nb::init<>())
      .def("set_config", &StructComputer::set_config, nb::arg("shape"),
           nb::arg("grids"), nb::arg("stride"), nb::arg("want_global"))
      .def(
          "raw",
          [](StructComputer &self, Arr a)
          {
            return self.raw(a.data(), a.stride(0), a.stride(1));
          },
          nb::arg("arr"))
      .def(
          "features",
          [](StructComputer &self, Arr a)
          {
            return self.features(a.data(), a.stride(0), a.stride(1));
          },
          nb::arg("arr"));
}