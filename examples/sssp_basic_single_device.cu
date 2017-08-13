#include <iostream>
#include <stdlib.h>
#include <string>
#include <cstdlib>

// Includes all the header files of the library.
#include "../kites/kites.cuh"

template <typename vT, class eT>
bool pre_processing_function_sssp(
		const unsigned int argcount,	// The number of additional items in the line
		char** argvector,	// char* pointer for which dereferencing its elements provides us with the additional items in the line in form of char*.
		const unsigned int src_vertex_index,	// Source vertex index.
		const unsigned int dst_vertex_index,	// Destination vertex index.
		vT& src_vertex,
		vT& dst_vertex,
		eT& edge
		) {
  unsigned int const algorithm_source_index = 30;   // Source in the SSSP.
  src_vertex = ( src_vertex_index != algorithm_source_index ) ? 1073741823 : 0;
  dst_vertex = ( dst_vertex_index != algorithm_source_index ) ? 1073741823 : 0;
  edge = ( argcount > 0 ) ?
      atoi( argvector[0] ) :    // If edge weights are specified, read.
      // Otherwise, have deterministic random weights.
      ( ( src_vertex_index + 1 ) * ( dst_vertex_index + 2 ) % 127 );
  return ( src_vertex_index == algorithm_source_index );
}

int main( int argc, char **argv )
{

	try{

	  // Vertex type.
	  using vT = unsigned int;
	  // Edge type.
	  using eT = unsigned int;

	  // Functions to perform the vertex-centric sssp graph computation.
      auto initF		= [] __host__ __device__
              ( volatile vT& locV, vT& globV ) { locV = globV; };
      auto nbrCompF	= [] __host__ __device__
              ( volatile vT& partialV, vT srcV, eT* connE ) { partialV = srcV + (*connE); };
      auto redF		= [] __host__ __device__
              ( volatile vT& lhsV, volatile vT& rhsV ) { if( lhsV > rhsV ) lhsV = rhsV; };
      auto updF		= [] __host__ __device__
              ( volatile vT& newV, vT oldV ) { return newV < oldV; };

      // Create a graph at the host side.
      kites::graph_csr<vT, eT> hostGrph(
          "Sample-Graph-Wiki-Vote.txt",   // Graph edgelist. This graph is downloaded from SNAP dataset.
          pre_processing_function_sssp<vT, eT>,         // Pre-processing function.
          1,                                            // Vertex per group ratio for the CSC rep.
          kites::input_graph_form::edge_list_s_d        // Input graph form (here source to destination).
          );
      // Determine which GPU, or a subset of available GPUs process the graph.
      kites::nv_gpu mydev( 0 );
      std::cout << "Selected device: " << mydev.getDevIdx() << '\n';
      std::cout << "Processing the graph with one GPU ...\n";
      // Move created CSR graph to the device side.
      auto devGrph = kites::make_graph_csr_for_device( hostGrph, mydev );
      // Process the graph.
      kites::process< kites::launch::sync >(
          devGrph,
          mydev,
          initF, nbrCompF, redF, updF );
      // Make sure things went well.
      // This option is not available for specific configurations.
      kites::sanity_check( hostGrph, devGrph, initF, nbrCompF, redF, updF );

      std::cout << "Done." << std::endl;
      return( EXIT_SUCCESS );
	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n" << "Exiting." << std::endl;
		return( EXIT_FAILURE );
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}
}
