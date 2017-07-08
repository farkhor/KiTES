#ifndef GRAPH_RAW_H_
#define GRAPH_RAW_H_


#include <string>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <utility>
#include <stdlib.h>

#include "graph_base.h"

#include "../common/globals.cuh"
#include "../io/file_interactor.hpp"


namespace kites{


template <typename vT, class eT, class vIdxT = unsigned int>
class graph_raw : public graph_base
{

public:

	class neighbor{
	public:
		eT EdgeVal;
		vIdxT nbrIdx;
		neighbor( vIdxT const inNbrIdx ):
			nbrIdx{ inNbrIdx }
		{}
		neighbor( vIdxT const inNbrIdx, eT const inEVal ):
			nbrIdx{ inNbrIdx }, EdgeVal{ inEVal }
		{}
		eT& getEdgeVal() { return EdgeVal; }
		bool operator< ( neighbor const& rhs ) const { return ( nbrIdx <  rhs.nbrIdx ); }
		bool operator==( neighbor const& rhs ) const { return ( nbrIdx == rhs.nbrIdx ); }
	};

	// Main CSR buffers.
	std::vector< vT > rawVVec;
	std::vector< std::vector< neighbor > > rawNbrsVec;

	// Auxiliary buffers.
	std::vector< std::vector< vIdxT > > rawIVecOutgoing;
	std::vector<bool> vBitmap;

	uint vertexPerGroup;
	uint totalNumOutEdges;

private:

	template <class funcT>
	void construct_me(
			std::ifstream& inputFile,
			funcT f,
			kites::input_graph_form const gForm,
			kites::graph_property const gProp,
			kites::edge_list_expression const elExp = kites::edge_list_expression::complete
			) {

		std::string line;
		char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
		char* pch;
		unsigned int Additionalargc=0;
		char* Additionalargv[ 61 ];

		// Read the input graph line-by-line.
		while( std::getline( inputFile, line ) ) {
			if( line[0] < '0' || line[0] > '9' )	// Skipping any line blank or starting with a character rather than a number.
				continue;
			char cstrLine[256];
			std::strcpy( cstrLine, line.c_str() );
			vIdxT firstIndex, secondIndex;

			pch = strtok( cstrLine, delim );
			if( pch != NULL ) firstIndex = static_cast<vIdxT>( atoll( pch ) );
			else continue;
			pch = strtok( NULL, delim );
			if( pch != NULL ) secondIndex = static_cast<vIdxT>( atoll( pch ) );
			else continue;

			auto const theMax = ( firstIndex >= secondIndex ) ? firstIndex : secondIndex;
			auto const srcVertexIndex = ( gForm == kites::input_graph_form::edge_list_s_d ) ? firstIndex : secondIndex;
			auto const dstVertexIndex = ( gForm == kites::input_graph_form::edge_list_s_d ) ? secondIndex : firstIndex;
			if( rawVVec.size() <= theMax ) {
				rawNbrsVec.resize( theMax + 1 );
				rawVVec.resize( theMax + 1 );
				vBitmap.resize( std::ceil( static_cast<double>( theMax + 1 ) / vertexPerGroup ), false );
			}

			Additionalargc=0;
			Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			while( Additionalargv[ Additionalargc ] != NULL ){
				++Additionalargc;
				Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			}
			rawNbrsVec.at( dstVertexIndex ).push_back( neighbor( srcVertexIndex ) );
			vBitmap.at( dstVertexIndex / vertexPerGroup ) = vBitmap.at( dstVertexIndex / vertexPerGroup ) | f(
				Additionalargc,
				Additionalargv,
				srcVertexIndex,
				dstVertexIndex,
				rawVVec.at( srcVertexIndex ),
				rawVVec.at( dstVertexIndex ),
				rawNbrsVec.at( dstVertexIndex ).back().getEdgeVal() );
			if( gProp == kites::graph_property::undirected && elExp == kites::edge_list_expression::incomplete ) {
				rawNbrsVec.at( srcVertexIndex ).push_back( neighbor( dstVertexIndex ) );
				vBitmap.at( srcVertexIndex / vertexPerGroup ) = vBitmap.at( srcVertexIndex / vertexPerGroup ) | f(
					Additionalargc,
					Additionalargv,
					dstVertexIndex,
					srcVertexIndex,
					rawVVec.at( dstVertexIndex ),
					rawVVec.at( srcVertexIndex ),
					rawNbrsVec.at( srcVertexIndex ).back().getEdgeVal() );
			}

		}


		set_num_edges( 0 );
		if( gProp == kites::graph_property::directed )
			rawIVecOutgoing.resize( std::ceil( static_cast<double>( rawVVec.size() ) / vertexPerGroup ) );
		vIdxT dstVIdx = 0;
		for( auto& vNbrs: rawNbrsVec ) {
			std::sort( vNbrs.begin(), vNbrs.end() );
			//if( gProp == kites::graph_property::undirected )
			//	vNbrs.erase( std::unique( vNbrs.begin(), vNbrs.end() ), vNbrs.end() );
			set_num_edges( get_num_edges() + vNbrs.size() );
			if( gProp == kites::graph_property::directed ) {
				for( auto& el: vNbrs )
					rawIVecOutgoing.at( el.nbrIdx / vertexPerGroup ).push_back( dstVIdx / vertexPerGroup );
				++dstVIdx;
			}
		}

		if( gProp == kites::graph_property::directed && vertexPerGroup != 1 ) {
			//totalNumOutEdges = 0;	// no need for it because initializer already does it.
			for( auto& el: rawIVecOutgoing ) {
				std::sort( el.begin(), el.end() );
				el.erase( std::unique( el.begin(), el.end() ), el.end() );
				totalNumOutEdges += el.size();
			}
		} else {
			totalNumOutEdges = get_num_edges();
		}

		// Below commented section is for sanity check.
		/*
		for( auto el: vBitmap )
			if( el )
				std::cout << "true\n";
			else
				std::cout << "false\n";
		std::cout << rawVVec.size() << "\t" << get_num_edges() << "\n";
		for( auto el: rawNbrsVec ) {
			for( auto ele: el )
				std::cout << ele.nbrIdx << "\t";
			std::cout << "\n";
		}
		for( auto el: rawIVecOutgoing ) {
			for( auto ele: el )
				std::cout << ele << "\t";
			std::cout << "\n";
		}
*/

		auto const initialNumVertices = rawVVec.size();
		if( ( initialNumVertices % 128 ) != 0 ) {
			rawNbrsVec.resize( ( ( initialNumVertices / 128 ) + 1 ) * 128 );
			rawVVec.resize( ( ( initialNumVertices / 128 ) + 1 ) * 128 );
			if( gProp == kites::graph_property::directed && vertexPerGroup == 1 ) {
				rawIVecOutgoing.resize( ( ( initialNumVertices / 128 ) + 1 ) * 128 );
				vBitmap.resize( ( ( initialNumVertices / 128 ) + 1 ) * 128 );
			}
		}
		set_num_vertices( rawVVec.size() );

		//if( gProp == kites::graph_property::directed && ( rawIVecOutgoing.size() % 32 ) != 0 ) {
		//	auto const initSize = rawIVecOutgoing.size();
		//	rawIVecOutgoing.resize( ( ( initSize / 32 ) + 1 ) * 32 );
		//	vBitmap.resize( ( ( initSize / 32 ) + 1 ) * 32 );
		//}

	}



public:
	template <class funcT>
	graph_raw(
			std::string const inputFileStr,
			funcT f,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d,
			kites::graph_property const gProp = kites::graph_property::directed,
			kites::edge_list_expression const elExp = kites::edge_list_expression::complete
			) : graph_base{}, rawNbrsVec{ 0 }, rawVVec{ 0 }, rawIVecOutgoing{ 0 }, vBitmap{ 0 }, vertexPerGroup{ 1 }, totalNumOutEdges{ 0 }
	{
		std::ifstream inputFile;
		kites::io::openFileToAccess< std::ifstream > ( inputFile, inputFileStr );
		construct_me( inputFile, f, gForm, gProp, elExp );
	}

	template <class funcT>
	graph_raw(
			std::ifstream& inputFile,
			funcT f,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d,
			kites::graph_property const gProp = kites::graph_property::directed,
			kites::edge_list_expression const elExp = kites::edge_list_expression::complete
			) : graph_base{}, rawNbrsVec{ 0 }, rawVVec{ 0 }, rawIVecOutgoing{ 0 }, vBitmap{ 0 }, vertexPerGroup{ 1 }, totalNumOutEdges{ 0 }
	{
		construct_me( inputFile, f, gForm, gProp, elExp );
	}

	template <class funcT>
	graph_raw(
			std::string const inputFileStr,
			funcT f,
			uint const vpg,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d,
			kites::graph_property const gProp = kites::graph_property::directed,
			kites::edge_list_expression const elExp = kites::edge_list_expression::complete
			) : graph_base{}, rawNbrsVec{ 0 }, rawVVec{ 0 }, rawIVecOutgoing{ 0 }, vBitmap{ 0 }, vertexPerGroup{ vpg }, totalNumOutEdges{ 0 }
	{
		std::ifstream inputFile;
		kites::io::openFileToAccess< std::ifstream > ( inputFile, inputFileStr );
		construct_me( inputFile, f, gForm, gProp, elExp );
	}

	template <class funcT>
	graph_raw(
			std::ifstream& inputFile,
			funcT f,
			uint const vpg,
			kites::input_graph_form const gForm = kites::input_graph_form::edge_list_s_d,
			kites::graph_property const gProp = kites::graph_property::directed,
			kites::edge_list_expression const elExp = kites::edge_list_expression::complete
			) : graph_base{}, rawNbrsVec{ 0 }, rawVVec{ 0 }, rawIVecOutgoing{ 0 }, vBitmap{ 0 }, vertexPerGroup{ vpg }, totalNumOutEdges{ 0 }
	{
		construct_me( inputFile, f, gForm, gProp, elExp );
	}

	uint getVertexPerGroup() const {
		return vertexPerGroup;
	}

	uint getTotalNumOutEdges() const {
		return totalNumOutEdges;
	}



};

}	// end namespace kites

#endif /* GRAPH_RAW_H_ */
