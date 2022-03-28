#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <map>
#include <utility>
#include <math.h>

#include "edgesearch.hpp"
#include "gridsearch.hpp"

namespace py = pybind11;


int mark_region(
        const py::array_t<int> centers,
        const py::array_t<double> positions,
        const py::array_t<int> mask_out,
        double cut_min,
        double cut_max
) {
    int *p_centers = (int*) centers.request().ptr;
    double *p_positions = (double*) positions.request().ptr;
    int *p_mask = (int*) mask_out.request().ptr;
    int n_centers = centers.shape(0);

    GridSearch cell_list(positions, cut_max, cut_min);

    for (int ii=0; ii<n_centers; ++ii) {
        int i = p_centers[ii];
        double ix = p_positions[i*3];
        double iy = p_positions[i*3 + 1];
        double iz = p_positions[i*3 + 2];

        GridSearchResult nbs = cell_list.getNeighboursForPosition(ix, iy, iz);

        for (int jj = 0; jj < (int) nbs.indices.size(); ++jj) {
            int j = nbs.indices[jj];
            p_mask[j] = 1;
        }
    }
    return 0;
}


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
) {

    int* p_types = (int*) types.request().ptr;
    double *p_positions = (double*) positions.request().ptr;
    int *p_edge_index = (int*) edge_index.request().ptr;
    double *p_edge_attr = (double*) edge_attr.request().ptr;
    double cut_bin_width = (cut_max - cut_min) / n_bins;

    int n_parts = positions.shape(0);
    int dim_edge = 2*n_types + n_bins;
    int edge = 0;

    GridSearch cell_list(positions, cut_max, cut_min);

    for (int i = 0; i < n_parts; ++i) {

        int it = p_types[i];
        double ix = p_positions[i*3];
        double iy = p_positions[i*3 + 1];
        double iz = p_positions[i*3 + 2];

        cout << i << " " << it << " @ " << ix << " " << iy << " " << iz << endl;
  
        GridSearchResult result = cell_list.getNeighboursForPosition(ix, iy, iz);
  
        for (int jj = 0; jj < (int) result.indices.size(); ++jj) {
            int j = result.indices[jj];
            if (j > i) {
                int jt = p_types[j];
                int off_1 = dim_edge*edge;
                int off_2 = off_1 + dim_edge;
                int bin_idx = int((result.distances[jj] - cut_min) / cut_bin_width + 0.5);

                p_edge_index[edge] = i;
                p_edge_index[edge+buffersize] = j;
                  
                p_edge_index[edge+1] = j;
                p_edge_index[edge+1+buffersize] = i;

                p_edge_attr[off_1 + bin_idx] = 1; 
                p_edge_attr[off_1 + n_bins + it] = 1;
                p_edge_attr[off_1 + n_bins + n_types + jt] = 1;

                p_edge_attr[off_2 + bin_idx] = 1; 
                p_edge_attr[off_2 + n_bins + jt] = 1;
                p_edge_attr[off_2 + n_bins + n_types + it] = 1;
 
                edge += 2;
            }
        }
  
    }
    return edge;
}


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
) {
    int *p_centers = (int*) centers.request().ptr;
    int *p_flags = (int*) flags.request().ptr;
    double *p_positions = (double*) positions.request().ptr;
    int *p_center_reindex = (int*) center_reindex.request().ptr;
    int *p_node_index = (int*) node_index.request().ptr;
    int *p_edge_index = (int*) hyperedge_index.request().ptr;
    double *p_edge_attr = (double*) hyperedge_attr.request().ptr;
    int n_centers = centers.shape(0);

    GridSearch cell_list(positions, cut_max_inter, cut_min);

    int node = 0;
    int edge = 0;
    for (int ii=0; ii<n_centers; ++ii) {

        int i = p_centers[ii];
        int fi = p_flags[i];
        double ix = p_positions[i*3];
        double iy = p_positions[i*3 + 1];
        double iz = p_positions[i*3 + 2];

        GridSearchResult nbs = cell_list.getNeighboursForPosition(ix, iy, iz);

        // Collect nodes
        int node_start_i = node;
        p_center_reindex[ii] = node_start_i;
        p_node_index[node++] = i;
        for (int jj = 0; jj < (int) nbs.indices.size(); ++jj) {
            int j = nbs.indices[jj];
            int fj = p_flags[j];
            // Flag-based exclusions
            if ((fj < flag_bound_lower || fj > flag_bound_upper) 
                    && fj != fi 
                    && fj != (fi+1)) 
                continue;
            // Apply intra-group cutoff
            if (fj == fi || fj == (fi+1)) {
                double rij = nbs.distances[jj];
                if (rij > cut_max_intra)
                    continue;
            }
            if (node > buffersize_node - 1) return py::make_tuple(-1, edge);
            p_node_index[node++] = j;
        }

        // Hyperedges on nodes
        for (int jj = 0, jjj = 0; jj < (int) nbs.indices.size(); ++jj) {

            int j = nbs.indices[jj];
            int fj = p_flags[j];
            double rij = nbs.distances[jj];
            double fij = 0;

            if ((fj < flag_bound_lower || fj > flag_bound_upper) 
                    && fj != fi 
                    && fj != (fi+1)) 
                continue;
            if (fj == fi || fj == (fi+1)) {
                fij = 1;
                if (rij > cut_max_intra)
                    continue;
            }
            ++jjj;

            double jx = p_positions[j*3];
            double jy = p_positions[j*3 + 1];
            double jz = p_positions[j*3 + 2];

            for (int kk = jj, kkk = jjj-1; kk < (int) nbs.indices.size(); ++kk) {

                int k = nbs.indices[kk];
                int fk = p_flags[k];
                double rik = nbs.distances[kk];
                double fik = 0;
                
                if ((fk < flag_bound_lower || fk > flag_bound_upper) 
                        && fk != fi 
                        && fk != (fi+1)) 
                    continue;
                if (fk == fi || fk == (fi+1)) {
                    fik = 1;
                    if (rik > cut_max_intra)
                        continue;
                }
                ++kkk; // Bloody delicate, be careful!

                double kx = p_positions[k*3];
                double ky = p_positions[k*3 + 1];
                double kz = p_positions[k*3 + 2];
                
                double dx = kx-jx;
                double dy = ky-jy;
                double dz = kz-jz;
                double rjk = sqrt(dx*dx + dy*dy + dz*dz);


                if (rjk > cut_max_edge) continue;
                if (edge > buffersize_edge - 2) return py::make_tuple(node, -1);

                // cout << " " << p_node_index[node_start_i + jjj] << " == " << j << " && " << p_node_index[node_start_i + kkk] << " == " << k << endl;

                // Edge index and attributes
                p_edge_index[edge] = node_start_i;
                p_edge_index[edge+buffersize_edge] = node_start_i+jjj;
                p_edge_index[edge+2*buffersize_edge] = node_start_i+kkk;

                int off_1 = dim_hyperedge*edge;
                p_edge_attr[off_1] = rij;
                p_edge_attr[off_1 + 1] = rik;
                p_edge_attr[off_1 + 2] = rjk;
                p_edge_attr[off_1 + 3] = fij;
                p_edge_attr[off_1 + 4] = fik;

                if (j != k) {
                    p_edge_index[edge+1] = node_start_i;
                    p_edge_index[edge+1+buffersize_edge] = node_start_i+kkk;
                    p_edge_index[edge+1+2*buffersize_edge] = node_start_i+jjj;

                    int off_2 = off_1 + dim_hyperedge;
                    p_edge_attr[off_2] = rik;
                    p_edge_attr[off_2 + 1] = rij;
                    p_edge_attr[off_2 + 2] = rjk;
                    p_edge_attr[off_2 + 3] = fik;
                    p_edge_attr[off_2 + 4] = fij;
                    edge += 2;
                } else {
                    edge += 1;
                }
            }

        }
    }
    return py::make_tuple(node, edge);
}


