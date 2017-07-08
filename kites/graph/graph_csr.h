#ifndef GRAPH_CSR_H_
#define GRAPH_CSR_H_

#include <string>
#include <cstring>
#include <fstream>
#include <cmath>

#include "graph_base.h"
#include "graph_raw.h"

#include "../common/globals.cuh"
#include "../io/file_interactor.hpp"
#include "../memory/host_pinned_buffer.cuh"

namespace kites{

/**
 * \brief The class template represents a graph kept inside the host memory while pinned, with CSR format.
 */
template <class vT, class eT, class vIdxT = unsigned int>
class graph_csr : public graph_base
{
	using eIdxT = vIdxT;

public:

	host_pinned_buffer<vT> V;
	host_pinned_buffer<eIdxT> R;
	host_pinned_buffer<vIdxT> C;
	host_pinned_buffer<eT> E;
	host_bitmap vBitmap;
	host_pinned_buffer<uint> updateFlag;
	kites::graph_property gProp;

	// For directed graphs.
	uint vertexPerGroup;
	host_pinned_buffer<eIdxT> dir_R;
	host_pinned_buffer<vIdxT> dir_C;


private:

	template <class funcT>
	void construct_me(
			std::ifstream& inputFile,
			funcT f,
			uint const vpg,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d,
			kites::graph_property const gProp = kites::graph_property::directed,
			kites::edge_list_expression const elExp = kites::edge_list_expression::complete
			) {

		kites::graph_raw<vT, eT> gRaw( inputFile, f, vpg, gForm, gProp, elExp );

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
		vBitmap.alloc( gRaw.vBitmap.size() );
		vBitmap.reset();

		if( gProp == kites::graph_property::directed ) {
			dir_R.alloc( gRaw.rawIVecOutgoing.size() + 1 );
			dir_R.at( 0 ) = 0;
			dir_C.alloc( gRaw.getTotalNumOutEdges() );
		}

		std::memcpy( V.get_ptr(), gRaw.rawVVec.data(), V.sizeInBytes() );
		std::size_t dir_nbrCnt = 0;
		for( auto vIdx = 0; vIdx < nVertices; ++vIdx ) {
			auto const nNbrs = gRaw.rawNbrsVec.at( vIdx ).size();
			auto const edgeIdxOffset = R[ vIdx ];
			for( auto nbrIdx = 0; nbrIdx < nNbrs; ++nbrIdx ) {
				C[ edgeIdxOffset + nbrIdx ] = gRaw.rawNbrsVec.at( vIdx ).at( nbrIdx ).nbrIdx;
				E[ edgeIdxOffset + nbrIdx ] = gRaw.rawNbrsVec.at( vIdx ).at( nbrIdx ).EdgeVal;
			}
			R[ vIdx + 1 ] = edgeIdxOffset + nNbrs;
			/*
			if( gRaw.vBitmap.at( vIdx ) )
				vBitmap.setAt( vIdx );

			if( gProp == kites::graph_property::directed ) {
				for( auto& el: gRaw.rawIVecOutgoing.at( vIdx ) )
					dir_C[ dir_nbrCnt++ ] = el;
				dir_R[ vIdx + 1 ] = dir_nbrCnt;
			}
			*/
		}


		for( auto vgIdx = 0; vgIdx < gRaw.rawIVecOutgoing.size(); ++vgIdx ) {
			if( gRaw.vBitmap.at( vgIdx ) )
				vBitmap.setAt( vgIdx );
			if( gProp == kites::graph_property::directed ) {
				for( auto& el: gRaw.rawIVecOutgoing.at( vgIdx ) )
					dir_C[ dir_nbrCnt++ ] = el;
				dir_R[ vgIdx + 1 ] = dir_nbrCnt;
			}
		}

	}

	void construct_prepared( std::ifstream& inputFile ) {
		std::string line;
		std::getline( inputFile, line );
		std::size_t const nVertices = std::stoll( line );
		std::getline( inputFile, line );
		std::size_t const nEdges = std::stoll( line );
		this->set_num_edges( nEdges );
		this->set_num_vertices( nVertices );
		std::getline( inputFile, line );
		gProp = ( ( std::stoi( line ) == 1 ) ? kites::graph_property::directed : kites::graph_property::undirected );

		V.alloc( nVertices );
		R.alloc( nVertices + 1 );
		C.alloc( nEdges );
		E.alloc( nEdges );
		vBitmap.alloc( nVertices );
		updateFlag.alloc( 1 );
		updateFlag.at( 0 ) = 0;
		if( gProp == kites::graph_property::directed ) {
			dir_R.alloc( nVertices + 1 );
			dir_C.alloc( nEdges );
		}
		for( auto vIdx = 0; vIdx < nVertices; ++vIdx ) {
			std::getline( inputFile, line );
			V[ vIdx ] = static_cast<vT>( std::stoull( line ) );
		}
		for( auto vIdx = 0; vIdx < ( nVertices + 1 ); ++vIdx ) {
			std::getline( inputFile, line );
			R[ vIdx ] = static_cast<eIdxT>( std::stoull( line ) );
		}
		for( auto eIdx = 0; eIdx < nEdges; ++eIdx ) {
			std::getline( inputFile, line );
			C[ eIdx ] = static_cast<vIdxT>( std::stoull( line ) );
		}
		for( auto eIdx = 0; eIdx < nEdges; ++eIdx ) {
			std::getline( inputFile, line );
			E[ eIdx ] = static_cast<eT>( std::stoull( line ) );
		}
		for( auto vIdx = 0; vIdx < vBitmap.size(); ++vIdx ) {
			std::getline( inputFile, line );
			vBitmap[ vIdx ] = static_cast<unsigned int>( std::stoull( line ) );
		}
		if( gProp == kites::graph_property::directed ) {
			for( auto vIdx = 0; vIdx < ( nVertices + 1 ); ++vIdx ) {
				std::getline( inputFile, line );
				dir_R[ vIdx ] = static_cast<eIdxT>( std::stoull( line ) );
			}
			for( auto eIdx = 0; eIdx < nEdges; ++eIdx ) {
				std::getline( inputFile, line );
				dir_C[ eIdx ] = static_cast<vIdxT>( std::stoull( line ) );
			}
		}


	}

public:

	uint getVertexPerGroup() const {
		return vertexPerGroup;
	}

	template <class funcT>
	graph_csr(
			std::ifstream& inputFile,
			funcT f,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d,
			kites::graph_property const gPropIn = kites::graph_property::directed,
			kites::edge_list_expression const elExp = kites::edge_list_expression::complete
		): gProp{ gPropIn }, vertexPerGroup{ 1 } {
		construct_me( inputFile, f, 1, gForm, gProp, elExp );
	}

	template <class funcT>
	graph_csr(
			std::string const inputFileStr,
			funcT f,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d,
			kites::graph_property const gPropIn = kites::graph_property::directed,
			kites::edge_list_expression const elExp = kites::edge_list_expression::complete
		): gProp{ gPropIn }, vertexPerGroup{ 1 } {
		std::ifstream inputFile;
		kites::io::openFileToAccess( inputFile, inputFileStr );
		construct_me( inputFile, f, 1, gForm, gProp, elExp );
	}

	graph_csr( std::ifstream& inputFile	 ):
		vertexPerGroup{ 1 } {
		construct_prepared( inputFile );
	}

	graph_csr( std::string const inputFileStr ):
		vertexPerGroup{ 1 } {
		std::ifstream inputFile;
		kites::io::openFileToAccess( inputFile, inputFileStr );
		construct_prepared( inputFile );
	}

	template <class funcT>
	graph_csr(
			std::ifstream& inputFile,
			funcT f,
			uint const vpg,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d
		): gProp{ kites::graph_property::directed }, vertexPerGroup{ vpg } {
		construct_me( inputFile, f, vpg, gForm );
	}

	template <class funcT>
	graph_csr(
			std::string const inputFileStr,
			funcT f,
			uint const vpg,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d
		): gProp{ kites::graph_property::directed }, vertexPerGroup{ vpg } {
		std::ifstream inputFile;
		kites::io::openFileToAccess( inputFile, inputFileStr );
		construct_me( inputFile, f, vpg, gForm );
	}

};


}	// end namespace kites

#endif /* GRAPH_CSR_H_ */
