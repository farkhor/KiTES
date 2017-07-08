#ifndef GRAPH_CSR_MIPMAPPED_HPP_
#define GRAPH_CSR_MIPMAPPED_HPP_

#include <cmath>

#include "graph_base.h"
#include "graph_raw.h"

#include "../common/globals.cuh"
#include "../io/file_interactor.hpp"
#include "../memory/host_pinned_buffer.cuh"

namespace kites{


template <class vT, class eT, class vIdxT = unsigned int>
class graph_csr_mipmapped : public graph_base
{

	using eIdxT = vIdxT;

public:

	host_pinned_buffer<vT> V;
	host_pinned_buffer<eIdxT> R;
	host_pinned_buffer<vIdxT> C;
	host_pinned_buffer<eT> E;
	host_pinned_buffer<unsigned int> updateFlag;

	unsigned int compressionRatio;
	unsigned int mipmapDim;
	host_bitmap vBitmap;
	host_bitmap mipmap;

private:

	template <class funcT>
	void construct_me(
			std::ifstream& inputFile,
			funcT f,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d
			) {
		kites::graph_raw<vT, eT> gRaw( inputFile, f, gForm, kites::graph_property::directed, kites::edge_list_expression::complete );

		auto const nVertices = gRaw.get_num_vertices();
		auto const nEdges = gRaw.get_num_edges();
		this->set_num_edges( nEdges );
		this->set_num_vertices( nVertices );

		V.alloc( nVertices );
		R.alloc( nVertices + 1 );
		R.at(0) = 0;
		C.alloc( nEdges );
		E.alloc( nEdges );
		updateFlag.alloc( 1 );
		updateFlag.at( 0 ) = 0;

		mipmapDim = static_cast<unsigned int>( std::ceil( static_cast<double>( nVertices ) / compressionRatio ) );
		while( ( mipmapDim % 32 ) != 0 )
			++mipmapDim;
		mipmap.alloc( mipmapDim * mipmapDim );
		mipmap.reset();
		vBitmap.alloc( mipmapDim );
		vBitmap.reset();

		std::memcpy( V.get_ptr(), gRaw.rawVVec.data(), V.sizeInBytes() );
		for( auto vIdx = 0; vIdx < nVertices; ++vIdx ) {
			auto const nNbrs = gRaw.rawNbrsVec.at( vIdx ).size();
			auto const edgeIdxOffset = R[ vIdx ];
			for( auto nbrIdx = 0; nbrIdx < nNbrs; ++nbrIdx ) {
				C[ edgeIdxOffset + nbrIdx ] = gRaw.rawNbrsVec.at( vIdx ).at( nbrIdx ).nbrIdx;
				E[ edgeIdxOffset + nbrIdx ] = gRaw.rawNbrsVec.at( vIdx ).at( nbrIdx ).EdgeVal;
			}
			R[ vIdx + 1 ] = edgeIdxOffset + nNbrs;

			if( gRaw.vBitmap.at( vIdx ) )
				vBitmap.setAt( vIdx / compressionRatio );

			for( auto& el: gRaw.rawIVecOutgoing.at( vIdx ) )
				mipmap.setAt( ( vIdx / compressionRatio ) * mipmapDim + ( el / compressionRatio ) );

		}


	}

public:

	template <class funcT>
	graph_csr_mipmapped(
			std::ifstream& inputFile,
			funcT f,
			unsigned int const compressionRatioIn,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d
	): compressionRatio{ compressionRatioIn } {
		construct_me( inputFile, f, gForm );
	}

	template <class funcT>
	graph_csr_mipmapped(
			std::string const inputFileStr,
			funcT f,
			unsigned int const compressionRatioIn,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d
	): compressionRatio{ compressionRatioIn } {
		std::ifstream inputFile;
		kites::io::openFileToAccess( inputFile, inputFileStr );
		construct_me( inputFile, f, gForm );
	}

};

}	// end namespace kites



#endif /* GRAPH_CSR_MIPMAPPED_HPP_ */
