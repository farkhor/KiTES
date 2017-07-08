#ifndef SANITY_CHECK_CUH_
#define SANITY_CHECK_CUH_

#include "graph_csr.h"
#include "graph_csr_device.cuh"
#include "graph_csr_multi_device.cuh"
#include "graph_csr_mipmapped.hpp"
#include "graph_csr_mipmapped_device.cuh"

namespace kites
{

template < class vT, class eT, class vIdxT, class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
void sanity_check( std::size_t const numVertices,
		host_pinned_buffer<vT>& hGraphV, host_pinned_buffer<vIdxT>& hGraphR, host_pinned_buffer<vIdxT>& hGraphC, host_pinned_buffer<eT>& hGraphE,
		uva_buffer<vT>& dGraphV,
		funcInitT funcInit, funcCompNbrT funcCompNbr, funcRedT funcRed, funcUpdT funcUpd ) {

	std::cout << "### SANITY CHECK ...\n";
	unsigned int iterationCounter = 0;
	bool anyUpdate;
	do{
		anyUpdate = false;

		for( auto vIdx = 0; vIdx < numVertices; ++vIdx ) {

			vT preV = hGraphV[ vIdx ];
			vT VertexInHand;
			funcInit( VertexInHand, preV );
			for( auto eIdx = ( hGraphR[ vIdx ] & 0x7FFFFFFF ); eIdx < ( hGraphR[ vIdx + 1 ] & 0x7FFFFFFF ); ++eIdx ) {
				vT tmp;
				auto const el = hGraphC.at( eIdx );
				funcCompNbr(
					tmp,
					hGraphV.at( el ),
					hGraphE.get_ptr() + eIdx );
				funcRed( VertexInHand, tmp );
			}
			if( funcUpd( VertexInHand, preV ) ) {
				hGraphV[ vIdx ] = VertexInHand;
				anyUpdate = true;
			}

		}

		++iterationCounter;
	} while( anyUpdate == true );
	std::cout << "### SANITY CHECK: Single CPU thread performed " << iterationCounter << " iterations in total.\n";

	kites::host_pinned_buffer<vT> devV( dGraphV );
	std::size_t idx = 0;
	for( ; idx < devV.size(); ++idx )
		if( devV[ idx ] != hGraphV[ idx ] ) {
			std::cout << "### SANITY CHECK: MISMATCH at index " << idx << "! Expected " << hGraphV[ idx ] << "\tbut got " << devV[ idx ] << "\n";
			return;
		}
	std::cout << "### SANITY CHECK: PASSED!\n";

}

template < class vT, class eT, class vIdxT, class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
void sanity_check( kites::graph_csr<vT, eT, vIdxT>& hGraph, kites::graph_csr_device<vT, eT, vIdxT>& dGraph,
		funcInitT funcInit, funcCompNbrT funcCompNbr, funcRedT funcRed, funcUpdT funcUpd ) {
	sanity_check( hGraph.get_num_vertices(),
			hGraph.V, hGraph.R, hGraph.C, hGraph.E,
			dGraph.V,
			funcInit, funcCompNbr, funcRed, funcUpd );
}

template < class vT, class eT, class vIdxT, class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
void sanity_check( kites::graph_csr<vT, eT, vIdxT>& hGraph, kites::graph_csr_multi_device<vT, eT, vIdxT>& dGraph,
		funcInitT funcInit, funcCompNbrT funcCompNbr, funcRedT funcRed, funcUpdT funcUpd ) {
	host_pinned_buffer<vT> resV( hGraph.get_num_vertices() );
	dGraph.getVertexValues( resV );
	//std::cout << "resV\n" << resV;
	sanity_check( hGraph.get_num_vertices(),
			hGraph.V, hGraph.R, hGraph.C, hGraph.E,
			resV,
			funcInit, funcCompNbr, funcRed, funcUpd );
}

template < class vT, class eT, class vIdxT, class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
void sanity_check( kites::graph_csr_mipmapped<vT, eT, vIdxT>& hGraph, kites::graph_csr_mipmapped_device<vT, eT, vIdxT>& dGraph,
		funcInitT funcInit, funcCompNbrT funcCompNbr, funcRedT funcRed, funcUpdT funcUpd ) {
	sanity_check( hGraph.get_num_vertices(),
			hGraph.V, hGraph.R, hGraph.C, hGraph.E,
			dGraph.V,
			funcInit, funcCompNbr, funcRed, funcUpd );
}


}	// end namespace kites


#endif /* SANITY_CHECK_CUH_ */
