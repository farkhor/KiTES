#ifndef GRAPH_OUTPUT_CUH_
#define GRAPH_OUTPUT_CUH_

#include "file_interactor.hpp"
#include "../graph/graph_csr.h"

namespace kites{

namespace io{

template < class vT, class eT, class vIdxT >
void out_vertices( kites::graph_csr<vT, eT, vIdxT>& inHostGraph, std::ofstream& outputFile ) {
  outputFile << inHostGraph.V;
}

template < class vT, class eT, class vIdxT >
void out_vertices( kites::graph_csr<vT, eT, vIdxT>& inHostGraph, std::string const inputFileStr ) {
  std::ofstream outputFile;
  kites::io::openFileToAccess( outputFile, inputFileStr );
  out_vertices( inHostGraph, outputFile );
}

template < class vT, class eT, class vIdxT >
void save_to_file( kites::graph_csr<vT, eT, vIdxT>& inHostGraph, std::ofstream& outputFile ) {
  outputFile << inHostGraph.get_num_vertices() << "\n";
  outputFile << inHostGraph.get_num_edges() << "\n";
  outputFile << ( ( inHostGraph.gProp == kites::graph_property::directed ) ? 1 : 0 ) << "\n";
  outputFile << inHostGraph.V << inHostGraph.R << inHostGraph.C << inHostGraph.E << inHostGraph.vBitmap;
  if( inHostGraph.gProp == kites::graph_property::directed )
    outputFile << inHostGraph.dir_R << inHostGraph.dir_C;
}

template < class vT, class eT, class vIdxT >
void save_to_file( kites::graph_csr<vT, eT, vIdxT>& inHostGraph, std::string const inputFileStr ) {
  std::ofstream outputFile;
  kites::io::openFileToAccess( outputFile, inputFileStr );
  save_to_file( inHostGraph, outputFile );
}

}	// end namespace io

}	// end namespace kites


#endif /* GRAPH_OUTPUT_CUH_ */
