#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;
using namespace std;


int edge_search(
        const py::array_t<int> types,
        const py::array_t<double> positions,
        py::array_t<int> edge_index,
        py::array_t<double> edge_attr,
        int n_types,
        int n_bins,
        double cut_min,
        double cut_max,
        int buffersize
);


int mark_region(
    const py::array_t<int> centers,
    const py::array_t<double> positions,
    const py::array_t<int> mask_out,
    double cut_min,
    double cut_max
);


py::tuple hyperedge_search(
        const py::array_t<int> centers,
        const py::array_t<int> flags,
        const py::array_t<double> positions,
        py::array_t<int> center_reindex,
        py::array_t<int> node_index,
        py::array_t<int> hyperedge_index,
        py::array_t<double> hyperedge_attr,
        double cut_min,
        double cut_max_intra,
        double cut_max_inter,
        double cut_max_edge,
        int dim_hyperedge,
        int buffersize_node,
        int buffersize_edge,
        int flag_bound_lower,
        int flag_bound_upper
);


