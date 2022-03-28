#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  
#include <pybind11/stl.h>    

#include "gridsearch.hpp"
#include "edgesearch.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cxx, m) {
    m.def(
        "mark_region", 
        &mark_region, 
        "mark region around centers"
    );

    m.def(
        "edge_search", 
        &edge_search, 
        "generate edge_index and edge_attr"
    );

    m.def(
        "hyperedge_search", 
        &hyperedge_search, 
        "generate hyperedge_index and hyperedge_attr"
    );

    py::class_<GridSearch>(m, "GridSearch", py::module_local())
      .def(py::init<py::array_t<double>, double, double>())
      .def("getNeighboursForPosition", &GridSearch::getNeighboursForPosition);                                                  
}

