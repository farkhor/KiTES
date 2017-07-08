#ifndef KERNELS_SINGLE_DEVICE_CUH_
#define KERNELS_SINGLE_DEVICE_CUH_

#include "../common/globals.cuh"
#include "utility_device_functions.cuh"

namespace kites
{


/*
 * \brief The main graph processing CUDA kernel for a single device.
 *
 * It assumes the number of vertices is a multiple of 32,
 *   the assumption that is fulfilled in the pre-processing step
 *   by adding disconnected virtual vertices.
 */
template < graph_property gProp, typename vertexT, typename edgeT,
	class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
__global__ void iteration_kernel_single_device(
		uint const nVerticesToProcess,
		const uint* C,
		const uint*  R,
		vertexT* V,
		edgeT* E,
		uint* devUpdateFlag,
		const uint* bitmapPtrRead,
		uint* bitmapPtrWrite,
		funcInitT funcInit, funcCompNbrT funcCompNbr, funcRedT funcRed, funcUpdT funcUpd,
		const uint* dir_C = nullptr,
		const uint* dir_R = nullptr
		) {

	volatile __shared__ vertexT fetchedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ uint nbrIdxBegin[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ uint nbrSizeExclusivePS[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ vertexT locallyComputedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];

	uint const tidWithinCTA = threadIdx.x;
	uint const vertexID = tidWithinCTA + blockIdx.x * blockDim.x;
	// Short-circuit the warp in case they are assigned out-of-range.
	// Note that due to having the number of vertices being multiple of 32,
	//   all the threads inside the warp stay or retire together.
	if( vertexID >= nVerticesToProcess )
		return;

	// Get the assigned 32-bit register for the warp from the bitmap.
	uint const globalWarpID = vertexID >> WARP_SIZE_SHIFT;
	uint const BMSec = dev_utils::getBitmapForWarpID( bitmapPtrRead, globalWarpID );
	// If none of the assigned vertices are active, retire the warp.
	if( BMSec == 0 )
		return;

	// Get the lane ID and the lane mask for the thread.
	uint const laneID = dev_utils::getLaneID();
	uint const laneMask = dev_utils::getLaneMask();

	uint const warpOffsetWithinCTA = tidWithinCTA & ( ~ ( WARP_SIZE - 1 ) );

	bool const isMyAssignedActive = ( dev_utils::BFExtract( BMSec, laneID, 1 ) != 0 );
	uint const intrawarpBinaryPS = __popc( BMSec & laneMask );
	uint const activePos = warpOffsetWithinCTA + intrawarpBinaryPS;

	// Initialize vertices.
	nbrIdxBegin[ tidWithinCTA ] = UINT_MAX;	//0xffffffff
	nbrSizeExclusivePS[ tidWithinCTA ] = UINT_MAX;	//0xffffffff
	uint tBeginIdx, tSum;
	vertexT assignedVertex;

	if( isMyAssignedActive ) {
		assignedVertex = V[ vertexID ];
		funcInit( fetchedVertexValues[ activePos ], assignedVertex );
		tBeginIdx = R[ vertexID ];
		tSum = R[ vertexID + 1 ] - tBeginIdx;
		nbrIdxBegin[ activePos ] = tBeginIdx;
	}
	uint const intrawarpScanInclusive = dev_utils::scanWarpShuffle( isMyAssignedActive ? tSum : 0 );
	uint const nTotalEdgesToProcess = __shfl( intrawarpScanInclusive, 31 );//nActives - 1 );
	if( isMyAssignedActive )
		nbrSizeExclusivePS[ activePos ] = intrawarpScanInclusive - tSum;
	uint const nActives = __popc( BMSec );
	if( laneID == 0 & nActives != 32 )
		nbrSizeExclusivePS[ warpOffsetWithinCTA + nActives ] = nTotalEdgesToProcess;

	// Iterate over the virtually expanded neighbors of active vertices.
	for(	uint virtualEdgeIdx = laneID;
			virtualEdgeIdx < nTotalEdgesToProcess;
			virtualEdgeIdx += WARP_SIZE ) {

		// Operations to find the neighbor for the assigned vertex,
	    //   which results in grabbing its value from V array.
		uint const belongingVertexIdx = dev_utils::find_belonging_vertex_index_inside_warp(
				nbrSizeExclusivePS + warpOffsetWithinCTA,
				virtualEdgeIdx );
		uint const addrInShared = warpOffsetWithinCTA + belongingVertexIdx;
		uint const edgeOffset = nbrIdxBegin[ addrInShared ];
		uint const location = virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ];
		uint const targetVertexIndex = C[ edgeOffset + location ];
		vertexT const srcValue = V[ targetVertexIndex ];

		funcCompNbr(
				locallyComputedVertexValues[ tidWithinCTA ],
				srcValue,
				E + location + edgeOffset );

		uint const intraSegIdx = min( laneID, virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ] );
		uint const intraSegIdxRev = min(
				( ( belongingVertexIdx != 31 ) ? nbrSizeExclusivePS[ addrInShared + 1 ] : nTotalEdgesToProcess ) - virtualEdgeIdx,
				WARP_SIZE - laneID );
		uint const segmentSize = intraSegIdxRev + intraSegIdx + 1;
		volatile vertexT* threadPosPtr = ( intraSegIdx != 0 ) ?
				( locallyComputedVertexValues + tidWithinCTA - 1 ) :
				( fetchedVertexValues + addrInShared );
		#pragma unroll 6
		for( uint iii = WARP_SIZE; iii > 0; iii /= 2 )
			if( segmentSize > iii && ( intraSegIdx + iii ) < segmentSize )
				funcRed( *threadPosPtr, locallyComputedVertexValues[ tidWithinCTA + iii - 1 ] );

	}

	// Update vertices.
	bool const updatedVertex = isMyAssignedActive ?
			funcUpd( fetchedVertexValues[ activePos ], assignedVertex  ) : false;
	if( updatedVertex )
		V[ vertexID ] = fetchedVertexValues[ activePos ];

	// Update outgoing vertices.
	// First get the distribution of updated vertices.
	// Retire the warp if no update has happened.
	uint const UPwarpBallot = __ballot( updatedVertex );
	if( UPwarpBallot == 0 ) return;
	else if( laneID == 0 ) (*devUpdateFlag) = 1;

	uint const UPintrawarpBinaryPS = __popc( UPwarpBallot & laneMask );
	uint const UPactivePos = warpOffsetWithinCTA + UPintrawarpBinaryPS;

	nbrIdxBegin[ tidWithinCTA ] = UINT_MAX;	//0xffffffff
	nbrSizeExclusivePS[ tidWithinCTA ] = UINT_MAX;	//0xffffffff
	if( gProp == kites::graph_property::directed && updatedVertex ) {
		tBeginIdx = dir_R[ vertexID ];
		tSum = dir_R[ vertexID + 1 ] - tBeginIdx;
	}
	uint const UPintrawarpScanInclusive = dev_utils::scanWarpShuffle( updatedVertex ? tSum : 0 );
	uint const UPnTotalEdgesToProcess = __shfl( UPintrawarpScanInclusive, 31 );
	if( updatedVertex ) {
		nbrSizeExclusivePS[ UPactivePos ] = UPintrawarpScanInclusive - tSum;
		nbrIdxBegin[ UPactivePos ] = tBeginIdx;
	}

	for(	uint virtualEdgeIdx = laneID;
			virtualEdgeIdx < UPnTotalEdgesToProcess;
			virtualEdgeIdx += WARP_SIZE ) {

		uint const belongingVertexIdx = dev_utils::find_belonging_vertex_index_inside_warp(
				nbrSizeExclusivePS + warpOffsetWithinCTA,
				virtualEdgeIdx );

		uint const addrInShared = warpOffsetWithinCTA + belongingVertexIdx;
		uint const edgeOffset = nbrIdxBegin[ addrInShared ];
		uint const location = virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ];
		uint const targetVertexIndex = ( gProp == kites::graph_property::undirected ) ?
				C[ edgeOffset + location ] :
				dir_C[ edgeOffset + location ];

		dev_utils::setBitmapAt( bitmapPtrWrite, targetVertexIndex );

	}

}




template < typename vertexT, typename edgeT,
  class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
__global__ void iteration_kernel_single_device_incompletevpg(
		uint const nVerticesToProcess,
		const uint* C,
		const uint*  R,
		vertexT* V,
		edgeT* E,
		uint* devUpdateFlag,
		const uint* bitmapPtrRead,
		uint* bitmapPtrWrite,
		funcInitT funcInit, funcCompNbrT funcCompNbr, funcRedT funcRed, funcUpdT funcUpd,
		const uint* dir_C = nullptr,
		const uint* dir_R = nullptr,
		const uint vpg_shift = 0
		) {

	volatile __shared__ vertexT fetchedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ uint nbrIdxBegin[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ uint nbrSizeExclusivePS[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ vertexT locallyComputedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];

	uint const tidWithinCTA = threadIdx.x;
	uint const vertexID = tidWithinCTA + blockIdx.x * blockDim.x;
	// Short-circuit the warp in case they are assigned out-of-range.
	// Note that due to having the number of vertices being multiple of 32,
	//   all the threads inside the warp stay or retire together.
	if( vertexID >= nVerticesToProcess )
		return;

	// Get the lane ID and the lane mask for the thread.
	uint const laneID = dev_utils::getLaneID();
	uint const laneMask = dev_utils::getLaneMask();

	uint const warpOffsetWithinCTA = tidWithinCTA & ( ~ ( WARP_SIZE - 1 ) );

	uint const bitIdx = vertexID >> vpg_shift;
	bool const isMyAssignedActive = dev_utils::getBitmapAt( bitmapPtrRead, bitIdx );
	uint const BMSec = __ballot( isMyAssignedActive );

	uint const intrawarpBinaryPS = __popc( BMSec & laneMask );
	uint const activePos = warpOffsetWithinCTA + intrawarpBinaryPS;

	// Initialize vertices.
	nbrIdxBegin[ tidWithinCTA ] = UINT_MAX;	//0xffffffff
	nbrSizeExclusivePS[ tidWithinCTA ] = UINT_MAX;	//0xffffffff
	uint tBeginIdx, tSum;
	vertexT assignedVertex;

	if( isMyAssignedActive ) {
		assignedVertex = V[ vertexID ];
		funcInit( fetchedVertexValues[ activePos ], assignedVertex );
		tBeginIdx = R[ vertexID ];
		tSum = R[ vertexID + 1 ] - tBeginIdx;
		nbrIdxBegin[ activePos ] = tBeginIdx;
	}
	uint const intrawarpScanInclusive = dev_utils::scanWarpShuffle( isMyAssignedActive ? tSum : 0 );
	uint const nTotalEdgesToProcess = __shfl( intrawarpScanInclusive, 31 );//nActives - 1 );
	if( isMyAssignedActive )
		nbrSizeExclusivePS[ activePos ] = intrawarpScanInclusive - tSum;
	uint const nActives = __popc( BMSec );
	if( laneID == 0 & nActives != 32 )
		nbrSizeExclusivePS[ warpOffsetWithinCTA + nActives ] = nTotalEdgesToProcess;

	// Iterate over the virtually expanded neighbors of active vertices.
	for(	uint virtualEdgeIdx = laneID;
			virtualEdgeIdx < nTotalEdgesToProcess;
			virtualEdgeIdx += WARP_SIZE ) {

		// Operations to find the neighbor for the assigned vertex,
	    //   which results in grabbing its value from V array.
		uint const belongingVertexIdx = dev_utils::find_belonging_vertex_index_inside_warp(
				nbrSizeExclusivePS + warpOffsetWithinCTA,
				virtualEdgeIdx );
		uint const addrInShared = warpOffsetWithinCTA + belongingVertexIdx;
		uint const edgeOffset = nbrIdxBegin[ addrInShared ];
		uint const location = virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ];
		uint const targetVertexIndex = C[ edgeOffset + location ];
		vertexT const srcValue = V[ targetVertexIndex ];

		funcCompNbr(
				locallyComputedVertexValues[ tidWithinCTA ],
				srcValue,
				E + location + edgeOffset );

		uint const intraSegIdx = min( laneID, virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ] );
		uint const intraSegIdxRev = min(
				( ( belongingVertexIdx != 31 ) ? nbrSizeExclusivePS[ addrInShared + 1 ] : nTotalEdgesToProcess ) - virtualEdgeIdx,
				WARP_SIZE - laneID );
		uint const segmentSize = intraSegIdxRev + intraSegIdx + 1;
		volatile vertexT* threadPosPtr = ( intraSegIdx != 0 ) ?
				( locallyComputedVertexValues + tidWithinCTA - 1 ) :
				( fetchedVertexValues + addrInShared );
		#pragma unroll 6
		for( uint iii = WARP_SIZE; iii > 0; iii /= 2 )
			if( segmentSize > iii && ( intraSegIdx + iii ) < segmentSize )
				funcRed( *threadPosPtr, locallyComputedVertexValues[ tidWithinCTA + iii - 1 ] );

	}

	// Update vertices.
	bool const updatedVertex = isMyAssignedActive ?
			funcUpd( fetchedVertexValues[ activePos ], assignedVertex  ) : false;
	if( updatedVertex )
		V[ vertexID ] = fetchedVertexValues[ activePos ];

	// Update outgoing vertices.
	// First get the distribution of updated vertices. Retire the warp if no update has happened.
	uint const UPwarpBallot = __ballot( updatedVertex );
	if( UPwarpBallot == 0 )
		return;
	else if( laneID == 0 )
		(*devUpdateFlag) = 1;

	bool const responsibleLane = ( dev_utils::BFExtract( laneID, 0, vpg_shift ) == 0 )
			& ( dev_utils::BFExtract( UPwarpBallot, laneID, 1 << vpg_shift ) != 0 );

	uint const UPintrawarpBinaryPS = __popc( __ballot( responsibleLane ) & laneMask );
	uint const UPactivePos = warpOffsetWithinCTA + UPintrawarpBinaryPS;

	nbrIdxBegin[ tidWithinCTA ] = UINT_MAX;	//0xffffffff
	nbrSizeExclusivePS[ tidWithinCTA ] = UINT_MAX;	//0xffffffff
	if( responsibleLane ) {
		tBeginIdx = dir_R[ bitIdx ];
		tSum = dir_R[ bitIdx + 1 ] - tBeginIdx;
	}
	uint const UPintrawarpScanInclusive = dev_utils::scanWarpShuffle( responsibleLane ? tSum : 0 );
	uint const UPnTotalEdgesToProcess = __shfl( UPintrawarpScanInclusive, 31 );
	if( responsibleLane ) {
		nbrSizeExclusivePS[ UPactivePos ] = UPintrawarpScanInclusive - tSum;
		nbrIdxBegin[ UPactivePos ] = tBeginIdx;
	}

	for(	uint virtualEdgeIdx = laneID;
			virtualEdgeIdx < UPnTotalEdgesToProcess;
			virtualEdgeIdx += WARP_SIZE ) {

		uint const belongingVertexIdx = dev_utils::find_belonging_vertex_index_inside_warp(
				nbrSizeExclusivePS + warpOffsetWithinCTA,
				virtualEdgeIdx );

		uint const addrInShared = warpOffsetWithinCTA + belongingVertexIdx;
		uint const edgeOffset = nbrIdxBegin[ addrInShared ];
		uint const location = virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ];
		uint const targetVertexIndex = dir_C[ edgeOffset + location ];

		dev_utils::setBitmapAt( bitmapPtrWrite, targetVertexIndex );
		//if( __clz( __brev( laneID ) ) == vpg_shift )
		//	atomicOr( bitmapPtrWrite + ( targetVertexIndex >> 5 ), 1 << ( targetVertexIndex & 31 ) );

	}

}


// The main graph processing CUDA kernel for a single device.
// It assumes the number of vertices is a multiple of 32.
template < typename vertexT, typename edgeT,
  class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
__global__ void iteration_kernel_single_device_vpg(
		uint const nVerticesToProcess,
		const uint* C,
		const uint*  R,
		vertexT* V,
		edgeT* E,
		uint* devUpdateFlag,
		const uint* bitmapPtrRead,
		uint* bitmapPtrWrite,
		funcInitT funcInit, funcCompNbrT funcCompNbr, funcRedT funcRed, funcUpdT funcUpd,
		const uint* dir_C,
		const uint* dir_R,
		uint const vpg_shift
		) {

	volatile __shared__ vertexT fetchedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ uint fetchedEdgesIndices[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ vertexT locallyComputedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];

	uint const tidWithinCTA = threadIdx.x;
	uint const globalThreadID = tidWithinCTA + blockIdx.x * blockDim.x;

	// Short-circuit the warp in case they are assigned out-of-range or they are not active.
	// Note that due to having the number of vertices being multiple of 32,
	//   all the threads inside the warp stay or retire together.
	if( globalThreadID >= nVerticesToProcess )
		return;

	uint const bitIdx = globalThreadID >> vpg_shift;
	if( !dev_utils::getBitmapAt( bitmapPtrRead, bitIdx ) )
		return;

	uint const vertexID = globalThreadID;
	uint const laneID = dev_utils::getLaneID();
	uint const warpOffsetWithinCTA = tidWithinCTA & ( ~ ( WARP_SIZE - 1 ) );
	uint const warpGlobalVertexOffset = vertexID & ( ~ ( WARP_SIZE - 1 ) );

	vertexT assignedVertex = V[ vertexID ];
	funcInit( fetchedVertexValues[ tidWithinCTA ], assignedVertex );
	fetchedEdgesIndices[ tidWithinCTA ] = R[ vertexID ];
	uint const endEdgeIdx = R[ warpGlobalVertexOffset + WARP_SIZE ];
	uint const startEdgeIdx = fetchedEdgesIndices[ warpOffsetWithinCTA ];

	// Iterate over the virtually expanded neighbors of active vertices.
	for(	uint edgeIdx = startEdgeIdx + laneID;
			edgeIdx < endEdgeIdx;
			edgeIdx += WARP_SIZE ) {

		uint const targetVertexIndex = C[ edgeIdx ];
		vertexT const srcValue = V[ targetVertexIndex ];

		// Operations to find the neighbor for the assigned vertex, which results in grabbing its value from V array.
		uint const belongingVertexIdx = dev_utils::find_belonging_vertex_index_inside_warp(
				fetchedEdgesIndices + warpOffsetWithinCTA,
				edgeIdx );
		uint const addrInShared = warpOffsetWithinCTA + belongingVertexIdx;
		uint const intraSegIdx = min( laneID, edgeIdx - fetchedEdgesIndices[ addrInShared ] );
		uint const intraSegIdxRev = min(
				( ( belongingVertexIdx != 31 ) ? fetchedEdgesIndices[ addrInShared + 1 ] : endEdgeIdx ) - edgeIdx,
				WARP_SIZE - laneID );
		uint const segmentSize = intraSegIdxRev + intraSegIdx + 1;

		funcCompNbr(
				locallyComputedVertexValues[ tidWithinCTA ],
				srcValue,
				E + edgeIdx );

		volatile vertexT* threadPosPtr = ( intraSegIdx != 0 ) ?
				( locallyComputedVertexValues + tidWithinCTA - 1 ) :
				( fetchedVertexValues + addrInShared );
		#pragma unroll 6
		for( uint iii = WARP_SIZE; iii > 0; iii /= 2 )
			if( segmentSize > iii && ( intraSegIdx + iii ) < segmentSize )
				funcRed( *threadPosPtr, locallyComputedVertexValues[ tidWithinCTA + iii - 1 ] );

	}

	// Update vertices.
	bool const updatedVertex = funcUpd( fetchedVertexValues[ tidWithinCTA ], assignedVertex  );
	if( updatedVertex )
		V[ vertexID ] = fetchedVertexValues[ tidWithinCTA ];
	bool const anyUpdate = __any( updatedVertex );
	if( !anyUpdate ) return;
	else if ( laneID == 0 ) (*devUpdateFlag) = 1;

	uint const beginNbrIdx = dir_R[ bitIdx ];
	uint const endNbrIdx = dir_R[ bitIdx + 1 ];
	for(	uint itemIdx = beginNbrIdx + laneID;
			itemIdx < endNbrIdx;
			itemIdx += WARP_SIZE ) {
		dev_utils::setBitmapAt( bitmapPtrWrite, dir_C[ itemIdx ] );
	}

}


}	// end namespace kites


#endif /* KERNELS_SINGLE_DEVICE_CUH_ */
