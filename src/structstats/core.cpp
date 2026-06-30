#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
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
// structstats -- single-pass gradient structure-tensor accumulation.
//
// For uint8 luma the 3x3 Sobel responses Gx, Gy are exact integers, so the
// products Gx^2, Gy^2, Gx*Gy and their per-cell sums are exact integers: the
// whole hot loop is integer, no floating point. The image is read exactly ONCE
// regardless of how many pyramid levels are requested -- every pixel scatters
// only into the FINEST grid, and coarser dyadic levels are exact sums of finer
// cells, so they are built by a cheap cascade aggregation of the small finest
// grid (not by re-scanning the image). Sobel is applied separably through
// per-row [1,2,1] / [-1,0,1] caches, so neighbouring pixels reuse the vertical
// pass instead of each redoing the full 3x3 (bit-identical to the full kernel).
//
// Output: one (cells_y, cells_x, 4) int64 array per level, last axis
// [Sxx, Syy, Sxy, count]; optionally a global (4,). Replicate border; cell
// boundaries at full resolution (stride subsamples which pixels accumulate).
// The caller derives eigenvalues / orientation / coherence from the raw sums --
// those are nonlinear and do not pool across scales, but the sums do.
// ---------------------------------------------------------------------------

namespace
{

  constexpr int SXX = 0, SYY = 1, SXY = 2, CNT = 3, NCOMP = 4;

  inline int clampi(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

  struct Level
  {
    int ny = 0, nx = 0;        // cells per axis (2^exp)
    int fy = 0, fx = 0;        // block factor down from the previous (finer) level
    std::vector<int64_t> buf;  // ny*nx*NCOMP, retained across calls
    std::vector<size_t> shape; // {ny, nx, NCOMP}
  };

  class StructComputer
  {
    int h_ = 0, w_ = 0, sy_ = 1, sx_ = 1;
    bool want_global_ = false;
    std::vector<Level> levels_;    // finest first
    std::vector<int> row_cell_;    // finest row binning (full-res boundaries)
    std::vector<int16_t> vs_, vd_; // per-row separable Sobel caches
    std::vector<int64_t> global_;  // NCOMP, retained
    std::vector<size_t> global_shape_;

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
        L.shape = {(size_t)L.ny, (size_t)L.nx, (size_t)NCOMP};
        levels_.push_back(std::move(L));
      }

      const Level &f = levels_[0];
      row_cell_.resize(h_);
      for (int r = 0; r < h_; ++r)
        row_cell_[r] = (int)((int64_t)r * f.ny / h_);
      vs_.resize(w_);
      vd_.resize(w_);
      global_.assign(NCOMP, 0);
      global_shape_ = {(size_t)NCOMP};
    }

    nb::list compute(const uint8_t *SS_RESTRICT img)
    {
      const int hi_r = h_ - 1, hi_c = w_ - 1;
      Level &fine = levels_[0];
      const int nfx = fine.nx, cw = w_ / nfx; // finest cells are cw-wide col runs
      int64_t *SS_RESTRICT fb = fine.buf.data();
      int16_t *SS_RESTRICT vs = vs_.data();
      int16_t *SS_RESTRICT vd = vd_.data();
      std::fill(fine.buf.begin(), fine.buf.end(), (int64_t)0);

      for (int r = 0; r < h_; r += sy_)
      {
        const uint8_t *SS_RESTRICT r0 = img + (size_t)clampi(r - 1, hi_r) * w_;
        const uint8_t *SS_RESTRICT r1 = img + (size_t)r * w_;
        const uint8_t *SS_RESTRICT r2 = img + (size_t)clampi(r + 1, hi_r) * w_;
        for (int c = 0; c < w_; ++c)
        { // vertical pass, reused by neighbours
          vs[c] = (int16_t)(r0[c] + 2 * r1[c] + r2[c]);
          vd[c] = (int16_t)(r2[c] - r0[c]);
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

      nb::list out;
      for (Level &L : levels_)
        out.append(nb::ndarray<nb::numpy, int64_t>(L.buf.data(), L.shape.size(),
                                                   L.shape.data(), nb::handle()));
      if (want_global_)
      {
        const Level &last = levels_.back();
        const int64_t *SS_RESTRICT lb = last.buf.data();
        const size_t n = (size_t)last.ny * last.nx;
        int64_t g0 = 0, g1 = 0, g2 = 0, g3 = 0;
        for (size_t c = 0; c < n; ++c)
        {
          const int64_t *SS_RESTRICT s = lb + c * NCOMP;
          g0 += s[0];
          g1 += s[1];
          g2 += s[2];
          g3 += s[3];
        }
        global_[0] = g0;
        global_[1] = g1;
        global_[2] = g2;
        global_[3] = g3;
        out.append(nb::ndarray<nb::numpy, int64_t>(global_.data(), global_shape_.size(),
                                                   global_shape_.data(), nb::handle()));
      }
      return out;
    }
  };

} // namespace

NB_MODULE(structstats_core, m)
{
  m.doc() = "structstats internal C++ module. Public API: structstats.StructComputer.";
  nb::class_<StructComputer>(m, "_StructComputerImpl")
      .def(nb::init<>())
      .def("set_config", &StructComputer::set_config, nb::arg("shape"),
           nb::arg("grids"), nb::arg("stride"), nb::arg("want_global"))
      .def(
          "compute",
          [](StructComputer &self,
             nb::ndarray<nb::numpy, uint8_t, nb::c_contig, nb::device::cpu> arr)
          {
            return self.compute(arr.data());
          },
          nb::arg("arr"));
}