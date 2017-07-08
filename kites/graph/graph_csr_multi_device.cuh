#ifndef GRAPH_CSR_MULTI_DEVICE_CUH_
#define GRAPH_CSR_MULTI_DEVICE_CUH_

#include "graph_csr_device.cuh"

namespace kites{

template <class vT, class eT, class vIdxT = unsigned int>
class graph_csr_multi_device
{
	using eIdxT = vIdxT;

public:

	// A vector holds the essential parts of the graphs over GPUs.
	std::vector< graph_csr_device<vT, eT, vIdxT> > graphParts;

	// Host as hub variables.
	bool host_as_hub;
	host_pinned_buffer<vT> HAHvertexValue_odd;
	host_pinned_buffer<vT> HAHvertexValue_even;
	host_pinned_buffer<vIdxT> HAHvertexIndices_odd;
	host_pinned_buffer<vIdxT> HAHvertexIndices_even;

	// Vectors specifying the range of vertices and edges
	//  stored on each device. These vectors are set upon
	//  construction and do not change during the computation.
	std::vector<std::pair<uint, uint>> vertexRangeStorage;
	std::vector<std::pair<uint, uint>> edgeRangeStorage;
	std::vector<std::pair<uint, uint>> dir_vertexRangeStorage;
	std::vector<std::pair<uint, uint>> dir_edgeRangeStorage;

	// Vectors specifying the range of vertices and edges
	//  assigned to each device. These vectors may get changed
	// during graph computation.
	std::vector<uint> vertexRangeAssignment;
	std::vector<uint> edgeRangeAssignment;
	std::vector<uint> dir_vertexRangeAssignment;
	std::vector<uint> dir_edgeRangeAssignment;

	std::vector<uint> initialBoundaryMarking;
	std::vector<uint> maxNumVerticesSend;

	host_pinned_buffer<uint> activeVerticesNumCollector;
	host_pinned_buffer<uint> PSNumEdgesPerSection;
	host_pinned_buffer<uint> dir_PSNumEdgesPerSection;


	/**
	 * \brief A member function providing a reference
	 *        to the specific graph piece requested.
	 *
	 * @param idx The index of the graph part.
	 *
	 * @return The requested CSR graph piece.
	 */
	graph_csr_device<vT, eT, vIdxT>& at( std::size_t const idx ) {
		return graphParts.at( idx );
	}

	/**
	 * \brief A member function telling if any of the devices
	 *        has seen an update.
	 *
	 * @return True if any update has been seen otherwise false.
	 */
	bool any_update() {
		for( auto& grph: graphParts )
			if( grph.updateFlagHost[0] != 0 )
				return true;
		return false;
	}

	uint getVertexPerGroup() const {
		return graphParts.at( 0 ).getVertexPerGroup();
	}

	bool if_host_as_hub() const {
		return host_as_hub;
	}

	/**
	 * \brief Collect the set of vertex values belonging
	 *        to the set of devices inside the given buffer.
	 *
	 *  @param dstBuffer The buffer to collect the concatenated
	 *         vertex value pieces into.
	 */
	void getVertexValues( uva_buffer<vT>& dstBuffer ) {
		for( uint devID = 0; devID < graphParts.size(); ++devID )
			kites::copyMem( dstBuffer, graphParts[ devID ].V,
				vertexRangeAssignment[ devID + 1 ] - vertexRangeAssignment[ devID ],
				vertexRangeAssignment[ devID ], vertexRangeAssignment[ devID ] );
	}

	graph_csr_multi_device(
			kites::graph_csr<vT, eT, vIdxT>& inHostGraph,
			kites::nv_gpus& devs )
	{
		std::cout << "Vertex Grouping Ratio is " << inHostGraph.getVertexPerGroup() << ".\n";

		/*********************************
		 * DETERMING PARTITION SIZES.
		 *********************************/
		// Default assumption about the storage and the assignment of the vertices.
		vertexRangeAssignment.resize( devs.num_devices() + 1 );
		edgeRangeAssignment.resize( devs.num_devices() + 1 );
		vertexRangeAssignment.front() = 0;
		vertexRangeAssignment.back() = inHostGraph.get_num_vertices();
		edgeRangeAssignment.front() = 0;
		edgeRangeAssignment.back() = inHostGraph.get_num_edges();
		dir_edgeRangeAssignment.resize( devs.num_devices() + 1, 0 );
		dir_vertexRangeAssignment.resize( devs.num_devices() + 1, 0 );
		if( inHostGraph.gProp == kites::graph_property::directed ) {
			dir_edgeRangeAssignment.back() = inHostGraph.dir_C.size();
			dir_vertexRangeAssignment.back() = inHostGraph.dir_R.size() - 1;
		}
		uint approxmiateNumEdgesPerDevice = inHostGraph.get_num_edges() / devs.num_devices();
		for( unsigned int dev = 1; dev < devs.num_devices(); ++dev ) {
			unsigned int accumulatedEdges = 0;
			uint movingVertexIndex = vertexRangeAssignment[ dev - 1 ];
			// TODO: Replace it with a binary search.
			while( accumulatedEdges < approxmiateNumEdgesPerDevice ) {
				//std::cout << "accum now: " << movingVertexIndex << "\t" << accumulatedEdges << "\n";
				accumulatedEdges += ( inHostGraph.R[ movingVertexIndex + 1 ] - inHostGraph.R[ movingVertexIndex ] );
				++movingVertexIndex;
			}
			movingVertexIndex &= ~( COMPILE_TIME_DETERMINED_BLOCK_SIZE - 1 );
			vertexRangeAssignment[ dev ] = movingVertexIndex;
			edgeRangeAssignment[ dev ] = inHostGraph.R[ movingVertexIndex ];
			if( inHostGraph.gProp == kites::graph_property::directed ) {
				uint const dir_idx = movingVertexIndex / inHostGraph.getVertexPerGroup();
				dir_vertexRangeAssignment[ dev ] = dir_idx;
				dir_edgeRangeAssignment[ dev ] = inHostGraph.dir_R[ dir_idx ];
			}
		}
		std::cout << "Vertex partitions: "; for( auto& el: vertexRangeAssignment ) std::cout << el << "\t"; std::cout << "\n";
		std::cout << "Edge partitions: "; for( auto& el: edgeRangeAssignment ) std::cout << el << "\t"; std::cout << "\n";
		std::cout << "dir_Vertex partitions: "; for( auto& el: dir_vertexRangeAssignment ) std::cout << el << "\t"; std::cout << "\n";
		std::cout << "dir_Edge partitions: "; for( auto& el: dir_edgeRangeAssignment ) std::cout << el << "\t"; std::cout << "\n";

		/*********************************
		 * ALLOW PEER-DEVICE MEMORY ACCESS.
		 *********************************/
		for( uint devID = 0; devID < devs.num_devices(); ++devID ) {
			devs.at( devID ).setAsActive();
			for( uint trgtDevID = 0; trgtDevID < devs.num_devices(); ++trgtDevID )
				if( trgtDevID != devID )
					CUDAErrorCheck( cudaDeviceEnablePeerAccess( devs.at( trgtDevID ).getDevIdx() , 0 ) );
		}

		/*********************************
		 * MARK BOUNDARIES.
		 *********************************/
		initialBoundaryMarking = vertexRangeAssignment;
		// A vector that will hold true for a vertex's corresponding index if it is referred to by other devices.
		std::vector< bool > vertexMarker( inHostGraph.get_num_vertices(), false );
		// Mark the vertices that are referred to by other devices.
		for( uint devID = 0; devID < devs.num_devices(); ++devID ) {
			for( uint edgeIdx = inHostGraph.R[ initialBoundaryMarking[ devID ] ];
					edgeIdx < inHostGraph.R[ initialBoundaryMarking[ devID + 1 ] ];
					++edgeIdx ) {
				auto const nbrIdx = inHostGraph.C[ edgeIdx ];
				auto const bound = std::equal_range( initialBoundaryMarking.begin(), initialBoundaryMarking.end(), nbrIdx );
				uint const nbrDevID = ( bound.first - initialBoundaryMarking.begin() - 1);
				if( nbrDevID != devID )
					vertexMarker[ nbrIdx ] = true;
			}
		}

		for( uint iii = 0; iii < vertexRangeAssignment.size(); ++iii )
			vertexRangeStorage.push_back( std::make_pair( vertexRangeAssignment[ iii ], vertexRangeAssignment[ iii + 1 ] ) );
		for( uint iii = 0; iii < edgeRangeAssignment.size(); ++iii )
			edgeRangeStorage.push_back( std::make_pair( edgeRangeAssignment[ iii ], edgeRangeAssignment[ iii + 1 ] ) );
		for( uint iii = 0; iii < dir_vertexRangeAssignment.size(); ++iii )
			dir_vertexRangeStorage.push_back( std::make_pair( dir_vertexRangeAssignment[ iii ], dir_vertexRangeAssignment[ iii + 1 ] ) );
		for( uint iii = 0; iii < dir_edgeRangeAssignment.size(); ++iii )
			dir_edgeRangeStorage.push_back( std::make_pair( dir_edgeRangeAssignment[ iii ], dir_edgeRangeAssignment[ iii + 1 ] ) );

		// TODO: COMPLETE.
		host_as_hub = ( devs.num_devices() > 2 );


		/*********************************
		 * EXTRACT INFORMATION OF BOUNDARIES PER DEVICE.
		 *********************************/
		maxNumVerticesSend.resize( devs.num_devices(), 0 );
		uint totalMaxVerticesCommunicated = 0;
		for( uint devID = 0; devID < devs.num_devices(); ++devID ) {
			auto const markedVertices = std::count( vertexMarker.begin() + initialBoundaryMarking[ devID ],
					vertexMarker.begin() + initialBoundaryMarking[ devID + 1 ], true );
			maxNumVerticesSend[ devID ] = markedVertices +
					( initialBoundaryMarking[ devID ] - vertexRangeStorage[ devID ].first ) +
					( vertexRangeStorage[ devID ].second - initialBoundaryMarking[ devID + 1 ] );
			totalMaxVerticesCommunicated += maxNumVerticesSend[ devID ];
		}
		std::cout << "Total num of marked vertices: " << totalMaxVerticesCommunicated << "\n";
		std::cout << "num vertices to send:\n"; for( auto& el: maxNumVerticesSend ) std::cout << el << "\t"; std::cout << "\n";

		/*********************************
		 * ALLOCATE HOST AS HUB.
		 *********************************/
		if( host_as_hub ) {
			HAHvertexValue_odd.alloc( totalMaxVerticesCommunicated );
			HAHvertexIndices_odd.alloc( totalMaxVerticesCommunicated );
			HAHvertexValue_even.alloc( totalMaxVerticesCommunicated );
			HAHvertexIndices_even.alloc( totalMaxVerticesCommunicated );
		}

		/*********************************
		 * ALLOCATE DEVICE BUFFERS.
		 *********************************/
		graphParts.resize( devs.num_devices() );
		for( uint devID = 0; devID < devs.num_devices(); ++devID ) {
			kites::nv_gpu& dev = devs.at( devID );
			dev.setAsActive();
			graph_csr_device<vT, eT, vIdxT>& graphPart = graphParts.at( devID );
			auto const residing_device_id = dev.getDevIdx();
			auto const nDedicatedVertices = vertexRangeStorage[ devID ].second - vertexRangeStorage[ devID ].first;
			auto const nDedicatedEdges = edgeRangeStorage[ devID ].second - edgeRangeStorage[ devID ].first;
			//graphPart.set_num_edges( vertexRangeStorage[ devID ] );
			//graphPart.set_num_vertices( edgeRangeStorage[ devID ] );

			graphPart.residing_device_id = residing_device_id;
			graphPart.vertexPerGroup = inHostGraph.getVertexPerGroup();
			graphPart.V.alloc( inHostGraph.V.size(), residing_device_id );

			graphPart.R.alloc( nDedicatedVertices + 1, residing_device_id );
			graphPart.C.alloc( nDedicatedEdges, residing_device_id );
			graphPart.E.alloc( nDedicatedEdges, residing_device_id );

			graphPart.vBitmapEven.alloc( inHostGraph.vBitmap.getnBits(), residing_device_id );
			graphPart.vBitmapOdd.alloc( inHostGraph.vBitmap.getnBits(), residing_device_id );
			graphPart.updateFlag.alloc( inHostGraph.updateFlag.size(), residing_device_id );
			graphPart.updateFlagHost.alloc( inHostGraph.updateFlag.size() );
			graphPart.gProp = inHostGraph.gProp;

			if( inHostGraph.gProp == kites::graph_property::directed ) {
				auto const dir_nDedicatedVertices = dir_vertexRangeStorage[ devID ].second - dir_vertexRangeStorage[ devID ].first;
				auto const dir_nDedicatedEdges = dir_edgeRangeStorage[ devID ].second - dir_edgeRangeStorage[ devID ].first;
				graphPart.dir_R.alloc( dir_nDedicatedVertices + 1, residing_device_id );
				graphPart.dir_C.alloc( dir_nDedicatedEdges, residing_device_id );
			}


			// Allocate box buffers.
			graphPart.inboxTop_odd.alloc( 1 );
			graphPart.inboxTop_odd[ 0 ] = 0;
			graphPart.inboxTop_even.alloc( 1 );
			graphPart.inboxTop_even[ 0 ] = 0;
			graphPart.outboxTop.alloc( 1, residing_device_id );
			graphPart.outboxIndices.alloc( maxNumVerticesSend[ devID ], residing_device_id );
			graphPart.outboxVertices.alloc( maxNumVerticesSend[ devID ], residing_device_id );
			if( !host_as_hub ) { // If we don't have host as the hub, we need inbox buffers in each device.
				graphPart.inboxIndices_odd.alloc( maxNumVerticesSend[ devID ^ 0x1 ], residing_device_id );
				graphPart.inboxVertices_odd.alloc( maxNumVerticesSend[ devID ^ 0x1 ], residing_device_id );
				graphPart.inboxIndices_even.alloc( maxNumVerticesSend[ devID ^ 0x1 ], residing_device_id );
				graphPart.inboxVertices_even.alloc( maxNumVerticesSend[ devID ^ 0x1 ], residing_device_id );
			}

			auto const vpg = inHostGraph.getVertexPerGroup();

		}

		// Mark boundaries.
		for( uint iii = 0; iii < vertexMarker.size(); ++iii )
			if( vertexMarker[ iii ] )
				inHostGraph.R[ iii ] |= 0x80000000;

		/*********************************
		 * COPY FROM HOST TO DEVICE BUFFERS.
		 *********************************/
		using timer = std::chrono::high_resolution_clock;
		timer::time_point const t1 = timer::now();
		inHostGraph.updateFlag[ 0 ] = 0;
		for( uint devID = 0; devID < devs.num_devices(); ++devID ) {
			kites::nv_gpu& dev = devs.at( devID );
			dev.setAsActive();
			graph_csr_device<vT, eT, vIdxT>& graphPart = graphParts.at( devID );
			auto const residing_device_id = dev.getDevIdx();
			auto const nDedicatedVertices = vertexRangeStorage[ devID ].second - vertexRangeStorage[ devID ].first;
			auto const nDedicatedEdges = edgeRangeStorage[ devID ].second - edgeRangeStorage[ devID ].first;

			kites::copyMemOnStream( graphPart.updateFlag, inHostGraph.updateFlag, dev.getSsEven() );
			kites::copyMemOnStream( graphPart.V, inHostGraph.V, dev.getSsEven() );
			kites::copyMemOnStream( graphPart.vBitmapEven, inHostGraph.vBitmap, dev.getSsEven() );
			graphPart.vBitmapOdd.reset( dev.getSsEven() );

			kites::copyMemOnStream( graphPart.R, inHostGraph.R, nDedicatedVertices + 1, dev.getSsEven(), 0, vertexRangeStorage[ devID ].first );
			kites::copyMemOnStream( graphPart.C, inHostGraph.C, nDedicatedEdges, dev.getSsEven(), 0, edgeRangeStorage[ devID ].first );
			kites::copyMemOnStream( graphPart.E, inHostGraph.E, nDedicatedEdges, dev.getSsEven(), 0, edgeRangeStorage[ devID ].first );

			if( inHostGraph.gProp == kites::graph_property::directed ) {
				kites::copyMemOnStream( graphPart.dir_R, inHostGraph.dir_R, graphPart.dir_R.size(), dev.getSsEven(), 0, dir_vertexRangeStorage[ devID ].first );
				kites::copyMemOnStream( graphPart.dir_C, inHostGraph.dir_C, graphPart.dir_C.size(), dev.getSsEven(), 0, dir_edgeRangeStorage[ devID ].first );
			}

		}
		devs.sync();
		timer::time_point const t2 = timer::now();
		double const accumulatedH2DCopyTime = ( static_cast<double>( std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() ) / 1000.0 );
		std::cout << "H2D copies took " << accumulatedH2DCopyTime << " (ms) in total.\n";

	}

};

/**
 * \brief A function to make CSR graphs for a device group.
 *
 *  The main purpose of this function is to create a
 *  multi-GPU CSR graph object without having to specify
 *  the template parameters.
 *
 *  @param inHostGraph The input host CSR graph.
 *  @param devs The device group collaboratively holding the graph.
 *  @param lb Specifying the inter-device load balancer existence.
 *
 *  @return Created multi-GPU CSR graph.
 */
template < class vT, class eT, class vIdxT >
kites::graph_csr_multi_device<vT, eT, vIdxT> make_graph_csr_for_devices(
		kites::graph_csr<vT, eT, vIdxT>& inHostGraph,
		kites::nv_gpus& devs
		) {
	return kites::graph_csr_multi_device<vT, eT, vIdxT>( inHostGraph, devs );
}


}	// end namespace kites


#endif /* GRAPH_CSR_MULTI_DEVICE_CUH_ */
