#ifndef KERNELS_MULTI_DEVICE_CUH_
#define KERNELS_MULTI_DEVICE_CUH_

#include "../common/globals.cuh"
#include "utility_device_functions.cuh"

namespace kites
{

/**
 * \brief Bitmap update kernel.
 */
__global__ void kernel_update_bitmap(
		uint* dstBitmap,
		const uint* srcBitmap,
		uint const nRegs
		) {
	uint const globalTID = threadIdx.x + blockIdx.x * blockDim.x;
	if( globalTID < nRegs )
		dstBitmap[ globalTID ] |= srcBitmap[ globalTID ];
		//atomicOr( dstBitmap + globalTID, srcBitmap[ globalTID ] );
}

void update_bitmap(
		uint* dstBitmap,
		uint* srcBitmap,
		uint const nBits,
		cudaStream_t sss,
		uint const dstOffsetBits = 0,
		uint const srcOffsetBits = 0
		) {
	uint const nRegs = std::ceil( static_cast<float>( nBits ) / 32 );
	uint const gridDim = std::ceil( static_cast<float>( nRegs ) /
			COMPILE_TIME_DETERMINED_BLOCK_SIZE );
	if( gridDim != 0 )
		kernel_update_bitmap<<< gridDim, COMPILE_TIME_DETERMINED_BLOCK_SIZE, 0, sss >>>
		( dstBitmap + ( dstOffsetBits / 32 ), srcBitmap + ( srcOffsetBits / 32 ),
				nRegs );
}

/**
 * \brief Inbox unloading kernel.
 */
template < typename idxT, typename vT >
__global__ void kernel_unload_inbox(
		vT* VertexValue,
		uint const loadSize,
		const idxT* inboxIndices,
		const vT* inboxVertices
		) {
	uint const globalTID = threadIdx.x + blockIdx.x * blockDim.x;
	if( globalTID < loadSize )
		VertexValue[ inboxIndices[ globalTID ] ] = inboxVertices[ globalTID ];
}

template < typename idxT, typename vT >
void unload_inbox(
		vT* VertexValue,
		uint const loadSize,
		const idxT* inboxIndices,
		const vT* inboxVertices,
		cudaStream_t sss,
		uint const offset
		) {
	uint const gridDim =
			std::ceil( static_cast<float>( loadSize ) /
			COMPILE_TIME_DETERMINED_BLOCK_SIZE );
	if( gridDim != 0 )
		kernel_unload_inbox
		<<< gridDim, COMPILE_TIME_DETERMINED_BLOCK_SIZE, 0, sss >>>
		( VertexValue, loadSize, inboxIndices + offset, inboxVertices + offset );
}

/**
 * \brief GPU kernel distributing (copying) the content of the outbox.
 *
 * Here GPU threads perform the copy operation instead of an explicit memory copy
 * command issued by the host since the load size is known by the device only,
 * and making it known to the host and then performing the copy is costly.
 *
 * @param loadSizePtr The pointer its pointee has the load size to transfer.
 * @param srcIndices The pointer to indices for the vertices to transfer.
 * @param srcVertices The pointer to the vertices to transfer.
 * @param dstIndices The pointer to the destination buffer for indices.
 * @param dstVertices The pointer to the destination buffer for vertices.
 */
template < typename vertexT >
__global__ void dev_distribute_outbox(
		const uint* loadSizePtr,
		const uint* srcIndices,
		const vertexT* srcVertices,
		uint* dstIndices,
		vertexT* dstVertices
		) {
	uint const globalTID = threadIdx.x + blockIdx.x * blockDim.x;
	if( globalTID < (*loadSizePtr) ) {
		dstIndices[ globalTID ] = srcIndices[ globalTID ];
		dstVertices[ globalTID ] = srcVertices[ globalTID ];
	}
}

/**
 * \brief The method to distribute (copy) the content of the outbox.
 */
template < typename vertexT >
void distribute_outbox(
		uint const maxLoadSize,
		const uint* loadSizePtr,
		const uint* srcIndices,
		const vertexT* srcVertices,
		uint* dstIndices,
		vertexT* dstVertices,
		cudaStream_t sss,
		uint const dstOffset
		) {
	uint const gridDim =
			std::ceil( static_cast<double>( maxLoadSize ) /
			COMPILE_TIME_DETERMINED_BLOCK_SIZE );
	if( gridDim != 0 )
		dev_distribute_outbox
		<<< gridDim, COMPILE_TIME_DETERMINED_BLOCK_SIZE, 0, sss >>>
		( loadSizePtr, srcIndices, srcVertices,
		dstIndices + dstOffset, dstVertices + dstOffset );
}



/**
 * \brief The main graph processing CUDA kernel for multiple devices.
 *
 * It assumes the number of vertices to be processed is a multiple of 32.
 */
template < graph_property gProp, typename vertexT, typename edgeT,
	class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
__global__ void iteration_kernel_multi_device(
		uint const nVerticesToProcess,
		const uint* C,
		const uint*  R,
		vertexT* V,
		edgeT* E,
		uint* devUpdateFlag,
		const uint* bitmapPtrRead,
		uint* bitmapPtrWrite,
		funcInitT funcInit,
		funcCompNbrT funcCompNbr,
		funcRedT funcRed,
		funcUpdT funcUpd,
		uint const globalVertexIndexOffsetForDevice,
		uint const globalEdgeIndexOffsetForDevice,
		uint* outboxTop,
		vertexT* outboxVertices,
		uint* outboxIndices,
		const uint* dir_C = nullptr,
		const uint* dir_R = nullptr,
		uint const dir_globalVertexIndexOffsetForDevice = 0,
		uint const dir_globalEdgeIndexOffsetForDevice = 0
		) {

	// Static shared memory declaration.
	volatile __shared__ uint
		nbrIdxBegin[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ uint
		nbrSizeExclusivePS[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ vertexT
		fetchedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ vertexT
		locallyComputedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];

	// The ID of the thread within the kernel being executed in the device.
	uint const inDeviceThreadID = threadIdx.x + blockIdx.x * blockDim.x;

	// Short-circuit the warp in case they are assigned out-of-range.
	// Note that due to having the number of vertices being multiple of 32,
	// all the threads inside the warp stay or retire together.
	if( inDeviceThreadID >= nVerticesToProcess )
		return;

	// Get the assigned vertex ID within the device and also globally.
	uint const inDeviceVertexID = inDeviceThreadID;
	uint const globalVertexID = globalVertexIndexOffsetForDevice + inDeviceVertexID;

	// Get the assigned 32-bit register for the warp from the bitmap.
	uint const BMSec =
		dev_utils::getBitmapForWarpWithPos( bitmapPtrRead, globalVertexID );

	// If none of the assigned vertices are active, retire the whole warp.
	if( BMSec == 0 )
		return;

	// Get the lane ID and the lane mask for the thread.
	uint const laneID = dev_utils::getLaneID();
	uint const laneMask = dev_utils::getLaneMask();

	// Get the warp offset within CTA consistent for all the warp threads.
	uint const warpOffsetWithinCTA = threadIdx.x & ( ~ ( WARP_SIZE - 1 ) );

	// Figure out if the assigned vertex is active.
	bool const isMyAssignedActive = ( dev_utils::BFExtract( BMSec, laneID, 1 ) != 0 );

	// Then using the activeness status, calculate the intra-warp binary prefix sum.
	// This will tell the index of the thread with active vertex among other
	//  threads with active vertices inside the warp.
	uint const intrawarpBinaryPS = __popc( BMSec & laneMask );

	// Now tell which position inside the warp this active vertex is assigned to.
	uint const activePos = warpOffsetWithinCTA + intrawarpBinaryPS;

	// Initialize shared memory buffers standing for nbr indices and their sizes.
	nbrIdxBegin[ threadIdx.x ] = UINT_MAX;	//0xffffffff
	nbrSizeExclusivePS[ threadIdx.x ] = UINT_MAX;	//0xffffffff

	// Helper variables.
	uint tBeginIdx, tSum;
	vertexT assignedVertex;
	bool isBoundary = isMyAssignedActive;

	// Those threads assigned to an active vertex.
	if( isMyAssignedActive ) {

		// Get the assigned vertex content and save it into a thread-private register.
		assignedVertex = V[ globalVertexID ];

		// Perform initialization function.
		funcInit( fetchedVertexValues[ activePos ], assignedVertex );

		// Read the corresponding element from R array.
		uint const rawRElement = R[ inDeviceVertexID ];

		// Purify the read R element from possible boundary marking bit.
		tBeginIdx = rawRElement & 0x7FFFFFFF;

		// A vertex is a boundary if its MSB is set,
		//  or conservatively, it is outside the valid marking boundary.
		isBoundary = ( rawRElement >= 0x80000000 );

		// Get the total number of neighbors for the active vertex.
		tSum = ( R[ inDeviceVertexID + 1 ] & 0x7FFFFFFF ) - tBeginIdx;

		// Calculate the accurate location in which the adjacency list starts,
		//  and save it inside the designated shared memory location.
		tBeginIdx =  tBeginIdx - globalEdgeIndexOffsetForDevice;
		nbrIdxBegin[ activePos ] = tBeginIdx;

	}	// End of the code block for threads with active vertices.

	// Using an intra-warp scan, threads with active vertices
	//  calculate the prefix sum of the number of neighbors
	//  have to be processed.
	uint const intrawarpScanInclusive =
			dev_utils::scanWarpShuffle( isMyAssignedActive ? tSum : 0 );

	// Calculate the total number of edges to be processed by the warp threads.
	uint const nTotalEdgesToProcess = __shfl( intrawarpScanInclusive, 31 );

	// Convert the calculated prefix sums from inclusive to exclusive
	//  and save them inside the shared memory.
	if( isMyAssignedActive )
		nbrSizeExclusivePS[ activePos ] = intrawarpScanInclusive - tSum;

	// Calculate the number of threads assigned to active vertices.
	uint const nActives = __popc( BMSec );

	// Modify the one last element in the shared memory location for the prefix sum.
	if( laneID == 0 & nActives != 32 )
		nbrSizeExclusivePS[ warpOffsetWithinCTA + nActives ] = nTotalEdgesToProcess;

	// Iterate over the virtually expanded neighbors of active vertices.
	for(	uint virtualEdgeIdx = laneID;
			virtualEdgeIdx < nTotalEdgesToProcess;
			virtualEdgeIdx += WARP_SIZE ) {

		// Operations to find the neighbor for the assigned vertex,
		//  which results in grabbing its value from V array.
		uint const belongingVertexIdx =
				dev_utils::find_belonging_vertex_index_inside_warp(
						nbrSizeExclusivePS + warpOffsetWithinCTA,
						virtualEdgeIdx );
		uint const addrInShared = warpOffsetWithinCTA + belongingVertexIdx;
		uint const edgeOffset = nbrIdxBegin[ addrInShared ];
		uint const location = virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ];
		uint const targetVertexIndex = C[ edgeOffset + location ];
		vertexT const srcValue = V[ targetVertexIndex ];

		// Perform the neighbor computation operation.
		funcCompNbr(
				locallyComputedVertexValues[ threadIdx.x ],
				srcValue,
				E + location + edgeOffset );

		// Calculate the intra-segment index from both directions
		//  and also the segment size.
		uint const intraSegIdx = min( laneID,
				virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ] );
		uint const intraSegIdxRev = min(
				( ( belongingVertexIdx != 31 ) ?
						nbrSizeExclusivePS[ addrInShared + 1 ] :
						nTotalEdgesToProcess ) - virtualEdgeIdx,
				WARP_SIZE - laneID );
		uint const segmentSize = intraSegIdxRev + intraSegIdx + 1;

		// Using a helper pointer, perform the reduction function iteratively
		//  in minimum number of steps.
		volatile vertexT* threadPosPtr = ( intraSegIdx != 0 ) ?
				( locallyComputedVertexValues + threadIdx.x - 1 ) :
				( fetchedVertexValues + addrInShared );
		#pragma unroll 6
		for( uint iii = WARP_SIZE; iii > 0; iii /= 2 )
			if( segmentSize > iii && ( intraSegIdx + iii ) < segmentSize )
				funcRed( *threadPosPtr,
						locallyComputedVertexValues[ threadIdx.x + iii - 1 ] );

	}	// End of iteration over the neighbors of active vertices.

	// Active vertices are checked using the update function.
	// The global vertex value is updated if it is true.
	bool const updatedVertex = isMyAssignedActive ?
			funcUpd( fetchedVertexValues[ activePos ], assignedVertex  ) : false;
	if( updatedVertex )
		V[ globalVertexID ] = fetchedVertexValues[ activePos ];

	///////////////////////////
	// ACTIVENESS PROPAGATION.
	///////////////////////////

	// Now mark outgoing vertices that need to be activated.
	// First get the distribution of updated vertices.
	uint const UPwarpBallot = __ballot( updatedVertex );

	// Retire the warp if no update has happened.
	// Else update the global flag signaling the computation is not converged.
	if( UPwarpBallot == 0 )
		return;
	else if( laneID == 0 )
		(*devUpdateFlag) = 1;

	// Then using the update status, calculate the intra-warp binary prefix sum.
	// Similar to what we did before,
	//  this will tell the index of the thread with updated vertex among other
	//  threads with active vertices inside the warp.
	uint const UPintrawarpBinaryPS = __popc( UPwarpBallot & laneMask );

	// Now tell which position inside the warp this updated vertex is assigned to.
	uint const UPactivePos = warpOffsetWithinCTA + UPintrawarpBinaryPS;

	// Reinitialize the shared memory buffers.
	nbrIdxBegin[ threadIdx.x ] = UINT_MAX;	//0xffffffff
	nbrSizeExclusivePS[ threadIdx.x ] = UINT_MAX;	//0xffffffff

	// If the graph is directed and the vertex is updated.
	if( gProp == kites::graph_property::directed && updatedVertex ) {

		// Calculate the set of variables holding the adjacency list
		//  for the CSC representation.
		uint const dir_inDeviceVertexID =
				inDeviceThreadID;
		tBeginIdx = dir_R[ dir_inDeviceVertexID ];
		tSum = dir_R[ dir_inDeviceVertexID + 1 ] - tBeginIdx;
		tBeginIdx =  tBeginIdx - dir_globalEdgeIndexOffsetForDevice;

	}

	// Calculate the intra-warp scan of the adjacency list of updated vertices.
	uint const UPintrawarpScanInclusive =
			dev_utils::scanWarpShuffle( updatedVertex ? tSum : 0 );

	// Get the total number of edges to visit.
	uint const UPnTotalEdgesToProcess = __shfl( UPintrawarpScanInclusive, 31 );

	// Update the shared memory buffers using calculated data.
	if( updatedVertex ) {
		nbrSizeExclusivePS[ UPactivePos ] = UPintrawarpScanInclusive - tSum;
		nbrIdxBegin[ UPactivePos ] = tBeginIdx;
	}

	// Iterate over the outgoing edges belonging to updated vertices.
	for(	uint virtualEdgeIdx = laneID;
			virtualEdgeIdx < UPnTotalEdgesToProcess;
			virtualEdgeIdx += WARP_SIZE ) {

		// Figure out which vertices they point to.
		uint const belongingVertexIdx =
				dev_utils::find_belonging_vertex_index_inside_warp(
						nbrSizeExclusivePS + warpOffsetWithinCTA,
						virtualEdgeIdx );
		uint const addrInShared = warpOffsetWithinCTA + belongingVertexIdx;
		uint const edgeOffset = nbrIdxBegin[ addrInShared ];
		uint const location = virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ];
		uint const targetVertexIndex =
				( gProp == kites::graph_property::undirected ) ?
				C[ edgeOffset + location ] :
				dir_C[ edgeOffset + location ];

		// Accordingly update the bitmask data structure.
		dev_utils::setBitmapAt( bitmapPtrWrite, targetVertexIndex );

	}

	///////////////////////////
	// UPDATE DISTRIBUTION.
	///////////////////////////

	// Distribute boundary vertices that are updated to other devices.
	// Retire the warp if no update has happened.
	bool const updatedAndBoundary = updatedVertex && isBoundary;
	uint const distWarpBallot = __ballot( updatedAndBoundary );
	if( distWarpBallot == 0 )
		return;

	// Warp-aggregated atomics to reduce the contention over the atomic variable.
	uint const distNum = __popc( distWarpBallot );
	uint outboxReservedPosition;
	if( laneID == 0 )
		outboxReservedPosition = atomicAdd( outboxTop, distNum );
	outboxReservedPosition = __shfl( outboxReservedPosition, 0 );

	// Intra-warp binary prefix-sum to realize the exact position
	//  for lanes to write in the buffer.
	uint const positionToWrite =
			outboxReservedPosition + __popc( distWarpBallot & laneMask );

	// If the vertex is updated and is boundary, write it to the outbox buffer.
	if( updatedAndBoundary ) {
		outboxVertices[ positionToWrite ] = fetchedVertexValues[ activePos ];
		outboxIndices[ positionToWrite ] = globalVertexID;
	}

}	// End of the kernel.

/**
 * \brief The graph processing CUDA kernel for multiple devices when
 *        vertex grouping with ratios 2, 4, or 8, or 16 is on.
 *
 * It assumes the number of vertices to be processed is a multiple of 32.
 */
template < typename vertexT, typename edgeT,
	class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
__global__ void iteration_kernel_multi_device_incompletevpg(
		uint const nVerticesToProcess,
		const uint* C,
		const uint*  R,
		vertexT* V,
		edgeT* E,
		uint* devUpdateFlag,
		const uint* bitmapPtrRead,
		uint* bitmapPtrWrite,
		funcInitT funcInit,
		funcCompNbrT funcCompNbr,
		funcRedT funcRed,
		funcUpdT funcUpd,
		uint const vpg_shift,
		uint const globalVertexIndexOffsetForDevice,
		uint const globalEdgeIndexOffsetForDevice,
		uint* outboxTop,
		vertexT* outboxVertices,
		uint* outboxIndices,
		const uint* dir_C,
		const uint* dir_R,
		uint const dir_globalVertexIndexOffsetForDevice,
		uint const dir_globalEdgeIndexOffsetForDevice
		) {

	// Static shared memory declaration.
	volatile __shared__ uint
		nbrIdxBegin[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ uint
		nbrSizeExclusivePS[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ vertexT
		fetchedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ vertexT
		locallyComputedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];

	// The ID of the thread within the kernel being executed in the device.
	uint const inDeviceThreadID = threadIdx.x + blockIdx.x * blockDim.x;

	// Short-circuit the warp in case they are assigned out-of-range.
	// Note that due to having the number of vertices being multiple of 32,
	// all the threads inside the warp stay or retire together.
	if( inDeviceThreadID >= nVerticesToProcess )
		return;

	// Get the assigned vertex ID within the device and also globally.
	uint const inDeviceVertexID = inDeviceThreadID;
	uint const globalVertexID = globalVertexIndexOffsetForDevice + inDeviceVertexID;

	// Get the lane ID and the lane mask for the thread.
	uint const laneID = dev_utils::getLaneID();
	uint const laneMask = dev_utils::getLaneMask();

	// Get the warp offset within CTA consistent for all the warp threads.
	uint const warpOffsetWithinCTA = threadIdx.x & ( ~ ( WARP_SIZE - 1 ) );

	uint const bitIdx = globalVertexID >> vpg_shift;
	bool const isMyAssignedActive = dev_utils::getBitmapAt( bitmapPtrRead, bitIdx );
	uint const BMSec = __ballot( isMyAssignedActive );

	// If none of the assigned vertices are active, retire the whole warp.
	if( BMSec == 0 )
		return;

	// Then using the activeness status, calculate the intra-warp binary prefix sum.
	// This will tell the index of the thread with active vertex among other
	//  threads with active vertices inside the warp.
	uint const intrawarpBinaryPS = __popc( BMSec & laneMask );

	// Now tell which position inside the warp this active vertex is assigned to.
	uint const activePos = warpOffsetWithinCTA + intrawarpBinaryPS;

	// Initialize shared memory buffers standing for nbr indices and their sizes.
	nbrIdxBegin[ threadIdx.x ] = UINT_MAX;	//0xffffffff
	nbrSizeExclusivePS[ threadIdx.x ] = UINT_MAX;	//0xffffffff

	// Helper variables.
	uint tBeginIdx, tSum;
	vertexT assignedVertex;
	bool isBoundary = isMyAssignedActive;

	// Those threads assigned to an active vertex.
	if( isMyAssignedActive ) {

		// Get the assigned vertex content and save it into a thread-private register.
		assignedVertex = V[ globalVertexID ];

		// Perform initialization function.
		funcInit( fetchedVertexValues[ activePos ], assignedVertex );

		// Read the corresponding element from R array.
		uint const rawRElement = R[ inDeviceVertexID ];

		// Purify the read R element from possible boundary marking bit.
		tBeginIdx = rawRElement & 0x7FFFFFFF;

		// A vertex is a boundary if its MSB is set.
		isBoundary = ( rawRElement >= 0x80000000 );

		// Get the total number of neighbors for the active vertex.
		tSum = ( R[ inDeviceVertexID + 1 ] & 0x7FFFFFFF ) - tBeginIdx;

		// Calculate the accurate location in which the adjacency list starts,
		//  and save it inside the designated shared memory location.
		tBeginIdx =  tBeginIdx - globalEdgeIndexOffsetForDevice;
		nbrIdxBegin[ activePos ] = tBeginIdx;

	}	// End of the code block for threads with active vertices.

	// Using an intra-warp scan, threads with active vertices
	//  calculate the prefix sum of the number of neighbors
	//  have to be processed.
	uint const intrawarpScanInclusive =
			dev_utils::scanWarpShuffle( isMyAssignedActive ? tSum : 0 );

	// Calculate the total number of edges to be processed by the warp threads.
	uint const nTotalEdgesToProcess = __shfl( intrawarpScanInclusive, 31 );

	// Convert the calculated prefix sums from inclusive to exclusive
	//  and save them inside the shared memory.
	if( isMyAssignedActive )
		nbrSizeExclusivePS[ activePos ] = intrawarpScanInclusive - tSum;

	// Calculate the number of threads assigned to active vertices.
	uint const nActives = __popc( BMSec );

	// Modify the one last element in the shared memory location for the prefix sum.
	if( laneID == 0 & nActives != 32 )
		nbrSizeExclusivePS[ warpOffsetWithinCTA + nActives ] = nTotalEdgesToProcess;

	// Iterate over the virtually expanded neighbors of active vertices.
	for(	uint virtualEdgeIdx = laneID;
			virtualEdgeIdx < nTotalEdgesToProcess;
			virtualEdgeIdx += WARP_SIZE ) {

		// Operations to find the neighbor for the assigned vertex,
		//  which results in grabbing its value from V array.
		uint const belongingVertexIdx =
				dev_utils::find_belonging_vertex_index_inside_warp(
						nbrSizeExclusivePS + warpOffsetWithinCTA,
						virtualEdgeIdx );
		uint const addrInShared = warpOffsetWithinCTA + belongingVertexIdx;
		uint const edgeOffset = nbrIdxBegin[ addrInShared ];
		uint const location = virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ];
		uint const targetVertexIndex = C[ edgeOffset + location ];
		vertexT const srcValue = V[ targetVertexIndex ];

		// Perform the neighbor computation operation.
		funcCompNbr(
				locallyComputedVertexValues[ threadIdx.x ],
				srcValue,
				E + location + edgeOffset );

		// Calculate the intra-segment index from both directions
		//  and also the segment size.
		uint const intraSegIdx = min( laneID,
				virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ] );
		uint const intraSegIdxRev = min(
				( ( belongingVertexIdx != 31 ) ?
						nbrSizeExclusivePS[ addrInShared + 1 ] :
						nTotalEdgesToProcess ) - virtualEdgeIdx,
				WARP_SIZE - laneID );
		uint const segmentSize = intraSegIdxRev + intraSegIdx + 1;

		// Using a helper pointer, perform the reduction function iteratively
		//  in minimum number of steps.
		volatile vertexT* threadPosPtr = ( intraSegIdx != 0 ) ?
				( locallyComputedVertexValues + threadIdx.x - 1 ) :
				( fetchedVertexValues + addrInShared );
		#pragma unroll 6
		for( uint iii = WARP_SIZE; iii > 0; iii /= 2 )
			if( segmentSize > iii && ( intraSegIdx + iii ) < segmentSize )
				funcRed( *threadPosPtr,
						locallyComputedVertexValues[ threadIdx.x + iii - 1 ] );

	}	// End of iteration over the neighbors of active vertices.

	// Active vertices are checked using the update function.
	// The global vertex value is updated if it is true.
	bool const updatedVertex = isMyAssignedActive ?
			funcUpd( fetchedVertexValues[ activePos ], assignedVertex  ) : false;
	if( updatedVertex )
		V[ globalVertexID ] = fetchedVertexValues[ activePos ];

	///////////////////////////
	// ACTIVENESS PROPAGATION.
	///////////////////////////

	// Now mark outgoing vertices that need to be activated.
	// First get the distribution of updated vertices.
	uint const UPwarpBallot = __ballot( updatedVertex );

	// Retire the warp if no update has happened.
	// Else update the global flag signaling the computation is not converged.
	if( UPwarpBallot == 0 )
		return;
	else if( laneID == 0 )
		(*devUpdateFlag) = 1;

	// One thread per vertex group becomes responsible to gather
	//  the adjacency list region.
	bool const responsibleLane = ( dev_utils::BFExtract( laneID, 0, vpg_shift ) == 0 )
			& ( dev_utils::BFExtract( UPwarpBallot, laneID, 1 << vpg_shift ) != 0 );

	// Then using the responsibility status, calculate the intra-warp binary prefix sum.
	// Similar to what we did before,
	//  this will tell the index of the thread with activated vertex groups among other
	//  threads with active vertices inside the warp.
	uint const UPintrawarpBinaryPS = __popc( __ballot( responsibleLane ) & laneMask );

	// Now tell which position inside the warp this updated vertex is assigned to.
	uint const UPactivePos = warpOffsetWithinCTA + UPintrawarpBinaryPS;

	// Reinitialize the shared memory buffers.
	nbrIdxBegin[ threadIdx.x ] = UINT_MAX;	//0xffffffff
	nbrSizeExclusivePS[ threadIdx.x ] = UINT_MAX;	//0xffffffff

	// If the vertex is responsible.
	if( responsibleLane ) {

		// Calculate the set of variables holding the adjacency list
		//  for the CSC representation.
		uint const dir_inDeviceVertexID =
				inDeviceThreadID >> vpg_shift;
		tBeginIdx = dir_R[ dir_inDeviceVertexID ];
		tSum = dir_R[ dir_inDeviceVertexID + 1 ] - tBeginIdx;
		tBeginIdx =  tBeginIdx - dir_globalEdgeIndexOffsetForDevice;

	}

	// Calculate the intra-warp scan of the adjacency list of updated vertices.
	uint const UPintrawarpScanInclusive =
			dev_utils::scanWarpShuffle( responsibleLane ? tSum : 0 );

	// Get the total number of edges to visit.
	uint const UPnTotalEdgesToProcess = __shfl( UPintrawarpScanInclusive, 31 );

	// Update the shared memory buffers using calculated data.
	if( responsibleLane ) {
		nbrSizeExclusivePS[ UPactivePos ] = UPintrawarpScanInclusive - tSum;
		nbrIdxBegin[ UPactivePos ] = tBeginIdx;
	}

	// Iterate over the outgoing edges belonging to updated vertices.
	for(	uint virtualEdgeIdx = laneID;
			virtualEdgeIdx < UPnTotalEdgesToProcess;
			virtualEdgeIdx += WARP_SIZE ) {

		// Figure out which vertices they point to.
		uint const belongingVertexIdx =
				dev_utils::find_belonging_vertex_index_inside_warp(
						nbrSizeExclusivePS + warpOffsetWithinCTA,
						virtualEdgeIdx );
		uint const addrInShared = warpOffsetWithinCTA + belongingVertexIdx;
		uint const edgeOffset = nbrIdxBegin[ addrInShared ];
		uint const location = virtualEdgeIdx - nbrSizeExclusivePS[ addrInShared ];
		uint const targetVertexIndex = dir_C[ edgeOffset + location ];

		// Accordingly update the bitmask data structure.
		dev_utils::setBitmapAt( bitmapPtrWrite, targetVertexIndex );

	}

	///////////////////////////
	// UPDATE DISTRIBUTION.
	///////////////////////////

	// Distribute boundary vertices that are updated to other devices.
	// Retire the warp if no update has happened.
	bool const updatedAndBoundary = updatedVertex && isBoundary;
	uint const distWarpBallot = __ballot( updatedAndBoundary );
	if( distWarpBallot == 0 )
		return;

	// Warp-aggregated atomics to reduce the contention over the atomic variable.
	uint const distNum = __popc( distWarpBallot );
	uint outboxReservedPosition;
	if( laneID == 0 )
		outboxReservedPosition = atomicAdd( outboxTop, distNum );
	outboxReservedPosition = __shfl( outboxReservedPosition, 0 );

	// Intra-warp binary prefix-sum to realize the exact position
	//  for lanes to write in the buffer.
	uint const positionToWrite =
			outboxReservedPosition + __popc( distWarpBallot & laneMask );

	// If the vertex is updated and is boundary, write it to the outbox buffer.
	if( updatedAndBoundary ) {
		outboxVertices[ positionToWrite ] = fetchedVertexValues[ activePos ];
		outboxIndices[ positionToWrite ] = globalVertexID;
	}

}	// End of the kernel.


/**
 * \brief The graph processing CUDA kernel for multiple devices when
 *        vertex grouping with ratios 32, 64, or 128 is on.
 *
 * It assumes the number of vertices to be processed is a multiple of 32.
 */
template < typename vertexT, typename edgeT,
	class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
__global__ void iteration_kernel_multi_device_vpg(
		uint const nVerticesToProcess,
		const uint* C,
		const uint*  R,
		vertexT* V,
		edgeT* E,
		uint* devUpdateFlag,
		const uint* bitmapPtrRead,
		uint* bitmapPtrWrite,
		funcInitT funcInit,
		funcCompNbrT funcCompNbr,
		funcRedT funcRed,
		funcUpdT funcUpd,
		uint const vpg_shift,
		uint const globalVertexIndexOffsetForDevice,
		uint const globalEdgeIndexOffsetForDevice,
		uint* outboxTop,
		vertexT* outboxVertices,
		uint* outboxIndices,
		const uint* dir_C,
		const uint* dir_R,
		uint const dir_globalVertexIndexOffsetForDevice,
		uint const dir_globalEdgeIndexOffsetForDevice
		) {

	// Static shared memory declaration.
	volatile __shared__ uint
		fetchedEdgesIndices[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ vertexT
		fetchedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ vertexT
		locallyComputedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];

	// The ID of the thread within the kernel being executed in the device.
	uint const inDeviceThreadID = threadIdx.x + blockIdx.x * blockDim.x;

	// Short-circuit the warp in case they are assigned out-of-range.
	// Note that due to having the number of vertices being multiple of 32,
	//  all the threads inside the warp stay or retire together.
	if( inDeviceThreadID >= nVerticesToProcess )
		return;

	// Get the assigned vertex ID within the device and also globally.
	uint const inDeviceVertexID = inDeviceThreadID;
	uint const globalVertexID = globalVertexIndexOffsetForDevice + inDeviceVertexID;

	// If the bit for the threads of the warp is not set, i.e.
	//  if the group of vertices is not active, retire the warp.
	uint const bitIdx = globalVertexID >> vpg_shift;
	if( !dev_utils::getBitmapAt( bitmapPtrRead, bitIdx ) )
		return;

	// Get the lane ID and the lane mask for the thread.
	uint const laneID = dev_utils::getLaneID();

	// Get the warp offset within CTA consistent for all the warp threads.
	uint const warpOffsetWithinCTA = threadIdx.x & ( ~ ( WARP_SIZE - 1 ) );

	// Get the warp offset within the device consistent for all the warp threads.
	const uint warpInDeviceVertexOffset = inDeviceVertexID & ( ~ ( WARP_SIZE - 1 ) );

	// Get the assigned vertex content and save it into a thread-private register.
	vertexT assignedVertex = V[ globalVertexID ];

	// Perform initialization function.
	funcInit( fetchedVertexValues[ threadIdx.x ], assignedVertex );

	// Construct the prefix sum of the number of neighbors for the warp threads.
	uint const rawRElement = R[ inDeviceVertexID ];
	fetchedEdgesIndices[ threadIdx.x ] = ( rawRElement & 0x7FFFFFFF )
		- globalEdgeIndexOffsetForDevice;
	bool const isBoundary = ( rawRElement >= 0x80000000 );
	uint const endEdgeIdx = ( R[ warpInDeviceVertexOffset + WARP_SIZE ] & 0x7FFFFFFF )
		- globalEdgeIndexOffsetForDevice;
	uint const startEdgeIdx = fetchedEdgesIndices[ warpOffsetWithinCTA ];

	// Iterate over the virtually expanded neighbors of active vertices.
	for(	uint edgeIdx = startEdgeIdx + laneID;
			edgeIdx < endEdgeIdx;
			edgeIdx += WARP_SIZE ) {

		// Get the neighbor index and the content of the neighbor vertex.
		uint const targetVertexIndex = C[ edgeIdx ];
		vertexT const srcValue = V[ targetVertexIndex ];

		// Operations to find the vertex the neighbor belongs to,
		//  as well as figuring out the intra-segment index and the segment size.
		uint const belongingVertexIdx =
				dev_utils::find_belonging_vertex_index_inside_warp(
				fetchedEdgesIndices + warpOffsetWithinCTA,
				edgeIdx );
		uint const addrInShared = warpOffsetWithinCTA + belongingVertexIdx;
		uint const intraSegIdx =
				min( laneID, edgeIdx - fetchedEdgesIndices[ addrInShared ] );
		uint const intraSegIdxRev = min(
				( ( belongingVertexIdx != 31 ) ?
				fetchedEdgesIndices[ addrInShared + 1 ] : endEdgeIdx ) - edgeIdx,
				WARP_SIZE - laneID );
		uint const segmentSize = intraSegIdxRev + intraSegIdx + 1;

		// Perform the neighbor computation function.
		funcCompNbr(
				locallyComputedVertexValues[ threadIdx.x ],
				srcValue,
				E + edgeIdx );

		// Using a helper pointer, perform the reduction function in parallel
		//  and in minimum number of steps.
		volatile vertexT* threadPosPtr = ( intraSegIdx != 0 ) ?
				( locallyComputedVertexValues + threadIdx.x - 1 ) :
				( fetchedVertexValues + addrInShared );
		#pragma unroll 6
		for( uint iii = WARP_SIZE; iii > 0; iii /= 2 )
			if( segmentSize > iii && ( intraSegIdx + iii ) < segmentSize )
				funcRed( *threadPosPtr,
						locallyComputedVertexValues[ threadIdx.x + iii - 1 ] );

	}	// End of iteration over the neighbors of warp's assigned vertices.

	// Perform the update check function and update the global content
	//  of the vertex if needed.
	bool const updatedVertex =
			funcUpd( fetchedVertexValues[ threadIdx.x ], assignedVertex  );
	if( updatedVertex )
		V[ globalVertexID ] = fetchedVertexValues[ threadIdx.x ];

	///////////////////////////
	// ACTIVENESS PROPAGATION.
	///////////////////////////

	// If no vertex has been updated, retire the warp otherwise signal the host.
	bool const anyUpdate = __any( updatedVertex );
	if( !anyUpdate )
		return;
	else if ( laneID == 0 )
		(*devUpdateFlag) = 1;

	// Figure out the outgoing neighbor range of the current group.
	uint const beginNbrIdx = dir_R[ bitIdx - dir_globalVertexIndexOffsetForDevice ]
	                                - dir_globalEdgeIndexOffsetForDevice;
	uint const endNbrIdx = dir_R[ bitIdx + 1 - dir_globalVertexIndexOffsetForDevice ]
	                                - dir_globalEdgeIndexOffsetForDevice;

	// Warp threads collaboratively mark the bitmask.
	for(	uint itemIdx = beginNbrIdx + laneID;
			itemIdx < endNbrIdx;
			itemIdx += WARP_SIZE )
		dev_utils::setBitmapAt( bitmapPtrWrite, dir_C[ itemIdx ] );

	///////////////////////////
	// UPDATE DISTRIBUTION.
	///////////////////////////

	// Distribute boundary vertices that are updated to other devices.
	// Retire the warp if no update has happened.
	bool const updatedAndBoundary = updatedVertex && isBoundary;
	uint const distWarpBallot = __ballot( updatedAndBoundary );
	if( distWarpBallot == 0 )
		return;

	// Warp-aggregated atomics to reduce the contention over the atomic variable.
	uint const distNum = __popc( distWarpBallot );
	uint outboxReservedPosition;
	if( laneID == 0 )
		outboxReservedPosition = atomicAdd( outboxTop, distNum );
	outboxReservedPosition = __shfl( outboxReservedPosition, 0 );

	// Intra-warp binary prefix-sum to realize the exact position for lanes
	//  to write in the buffer.
	uint const laneMask = dev_utils::getLaneMask();
	uint const positionToWrite =
			outboxReservedPosition + __popc( distWarpBallot & laneMask );
	if( updatedAndBoundary ) {
		outboxVertices[ positionToWrite ] = fetchedVertexValues[ threadIdx.x ];
		outboxIndices[ positionToWrite ] = globalVertexID;///multiDeviceVertexID;
	}

}

}	// end namespace kites

#endif /* KERNELS_MULTI_DEVICE_CUH_ */
