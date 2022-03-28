// Adapted from DScribe: celllist.cpp
// https://github.com/SINGROUP/dscribe/blob/master/dscribe/ext/celllist.cpp
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;
using namespace std;

struct GridSearchResult {
  vector<int> indices;
  vector<double> distances;
  vector<double> distancesSquared;
};

#pragma GCC visibility push(hidden)
class GridSearch {

public:
  GridSearch(py::array_t<double> positions, double cutoff, double cutoffLower); // will add lower cutoff here
  GridSearchResult getNeighboursForPosition(const double x, const double y, const double z) const; 
  //  GridSearchResult getNeighboursForIndex(const int i) const; //probably will erase this

private:
  void init();
  const py::detail::unchecked_reference<double, 2> positions; //do not understand what is going on here
  vector<vector<vector<vector<int>>>> bins;
  const double cutoff;
  const double cutoffSquared;
  const double cutoffLower;
  const double cutoffLowerSquared;
  double xmin;
  double xmax;
  double ymin;
  double ymax;
  double zmin;
  double zmax;
  int nx;
  int ny;
  int nz;
  double dx;
  double dy;
  double dz;
};
