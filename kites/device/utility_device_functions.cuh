/*
 *		utility_device_functions.cuh
 *
 *		Created on: Jul 12, 2016
 *		Author: farzad
 */

#ifndef UTILITY_DEVICE_FUNCTIONS_CUH_
#define UTILITY_DEVICE_FUNCTIONS_CUH_


#include "../common/globals.cuh"

namespace kites
{

/**
 * \brief This namespace represents the set of utility device functions.
 *
 * The functions in this namespace are used by the main processing kernels.
 * Thanks goes to Nicholas Wilt (Archaea Software) for introducing some of these
 *   functions in his book: The CUDA Handbook, 2013.
 */
namespace dev_utils
{

/**
 * \brief PTX routine for the shuffle-based intra-warp prefix-sum.
 */
__device__ __forceinline__ int
scanWarpShuffle_step( int partial, int offset )
{
  int result;
  asm(
      "{.reg .u32 r0;"
       ".reg .pred p;"
       "shfl.up.b32 r0|p, %1, %2, 0;"
       "@p add.u32 r0, r0, %3;"
       "mov.u32 %0, r0;}"
      : "=r"(result) : "r"(partial), "r"(offset), "r"(partial));
  return result;
}

/**
 * \brief Inclusive intra-warp prefix-sum using shuffle.
 */
template <int levels = 5>
__device__ __forceinline__ int
scanWarpShuffle( int mysum )
{
  #pragma unroll 5
  for( int i = 0; i < levels; ++i )
      mysum = scanWarpShuffle_step( mysum, 1 << i );
  return mysum;
}

/**
 * \brief Bit-field extraction wrapper.
 */
__device__ __forceinline__ uint BFExtract(
      uint const src,
      uint const start,
      uint const len
      ) {
  uint ret;
  asm ("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(src), "r"(start), "r"(len) );
  return ret;
}

/**
 * \brief Set bitmap at an specific bit.
 */
__device__ __inline__ void
setBitmapAt( uint* ptr, const uint pos ) {
  atomicOr( ptr + ( pos >> 5 ), 1 << ( pos & 31 ) );
}

/**
 * \brief Get the content of bitmap at an specific bit.
 */
__device__ __inline__ bool
getBitmapAt( const uint* ptr, uint const pos ) {
  return ( BFExtract( ptr[ pos >> 5 ], pos & 31, 1 ) != 0 ) ? true : false;
}

/**
 * \brief Get the content of bitmap at an specific 32-bit register.
 */
__device__ __inline__ uint
getBitmapForWarpID( const uint* ptr, uint const warpID ) {
  return ptr[ warpID ];
}

/**
 * \brief Get the content of bitmap at an specific 32-bit register
 *        specified by the bit index at its beginning.
 */
__device__ __inline__ uint
getBitmapForWarpWithPos( const uint* ptr, uint const pos ) {
  return ptr[ pos >> 5 ];
}

/**
 * \brief Get the lane ID for the thread inside the warp.
 *
 * @return Thread's lane ID.
 */
__device__ __inline__ uint
getLaneID() {
  //return ( threadIdx.x & ( WARP_SIZE - 1 ) );
  uint laneID;
  asm( "mov.u32 %0, %laneid;" : "=r"(laneID) );
  return laneID;
}

/**
 * \brief Get the lane mask for the thread inside the warp.
 *
 * @return Thread's less-than-or-equal lane mask.
 */
__device__ __inline__ uint
getLaneMask() {
  //return ( 0xFFFFFFFF >> ( WARP_SIZE - laneID ) );
  uint laneMask;
  asm( "mov.u32 %0, %lanemask_lt;" : "=r"(laneMask) );
  return laneMask;
}

/**
 * \brief Device function to find the belonging vertex index
 * 		  for the thread's assigned edge via a binary search.
 */
template< typename idxT >
__device__ __inline__ idxT
find_belonging_vertex_index_inside_warp(
      volatile const idxT* edgesIndicesShared,
      idxT const currentEdgeIdx
      ) {
  idxT startIdx = 0, endIdx = WARP_SIZE;
  #pragma unroll 5
  for( int iii = 0; iii < WARP_SIZE_SHIFT; ++iii ) {
    idxT middle = ( startIdx + endIdx ) >> 1;
    if( currentEdgeIdx < edgesIndicesShared[ middle ] )
      endIdx = middle;
    else
      startIdx = middle;
  }
  return startIdx;
}


/**
 * \brief A stream compaction kernel that extracts the ID of set bits
 *        inside a bit vector and puts them inside a compaction buffer.
 */
__global__ void bitmapStreamExtraction(
      const uint* bitmapPtrRead,
      uint const nItems,
      uint* compactionBuffer
      ) {
  uint const bitID = threadIdx.x + blockIdx.x * blockDim.x;
  if( bitID >= nItems )
    return;
  uint const warpReg = dev_utils::getBitmapForWarpWithPos( bitmapPtrRead, bitID );
  if( warpReg == 0 )
    return;

  // Get the lane ID and the lane mask for the thread.
  uint const laneID = getLaneID();
  uint const laneMask = getLaneMask();

  uint const totalNumSet = __popc( warpReg );
  uint reservedLoc;
  if( laneID == 0 )
    reservedLoc = atomicAdd( compactionBuffer, totalNumSet );
  reservedLoc = __shfl( reservedLoc, 0, WARP_SIZE );

  uint const intraWarpBinaryPS = __popc( warpReg & laneMask );
  bool const isMyAssignedSet = ( dev_utils::BFExtract( warpReg, laneID, 1 ) != 0 );
  if( isMyAssignedSet )
    compactionBuffer[ 1 + reservedLoc + intraWarpBinaryPS ] = bitID;

}

}	// end namespace dev_utils

}	// end namespace kites

#endif /* UTILITY_DEVICE_FUNCTIONS_CUH_ */
