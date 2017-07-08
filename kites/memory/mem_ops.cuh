#ifndef MEM_OPS_CUH_
#define MEM_OPS_CUH_

#include "device_buffer.cuh"
#include "host_pinned_buffer.cuh"
#include "../device/stream.cuh"

#include <chrono>

namespace kites
{

  /**
   * \brief Copy with the specified length.
   */
  template< class T, kites::launch lMode = kites::launch::sync >
  double copyMem(
          uva_buffer<T>& dst,
          uva_buffer<T>&src,
          std::size_t const length,
          std::size_t const dstOffst = 0,
          std::size_t const srcOffst = 0 ) {
    if( lMode == kites::launch::sync ) {
        using timer = std::chrono::high_resolution_clock;
        timer::time_point const t1 = timer::now();
        CUDAErrorCheck( cudaMemcpy(
            dst.get_ptr() + dstOffst,
            src.get_ptr() + srcOffst,
            length*sizeof(T),
            cudaMemcpyDefault ) );
        timer::time_point const t2 = timer::now();
        return ( static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() ) / 1000.0 );
    } else {
      CUDAErrorCheck( cudaMemcpyAsync(
          dst.get_ptr() + dstOffst,
          src.get_ptr() + srcOffst,
          length*sizeof(T),
          cudaMemcpyDefault ) );
      return 0;
    }
  }

  /**
   * \brief Copy one buffer to another.
   */
  template< class T, kites::launch lMode = kites::launch::sync >
  double copyMem(
          uva_buffer<T>& dst,
          uva_buffer<T>&src,
          std::size_t const dstOffst = 0,
          std::size_t const srcOffst = 0 ) {
    std::size_t const length = std::min( src.size(), dst.size() );
    return kites::copyMem<T, lMode>( dst, src, length, dstOffst, srcOffst );
  }

  /**
   * \brief Copy with the specified length asynchronously on the given stream.
   */
  template< class T >
  void copyMemOnStream(
          uva_buffer<T>& dst,
          uva_buffer<T>&src,
          std::size_t const length,
          kites::stream& inStream,
          std::size_t const dstOffst = 0,
          std::size_t const srcOffst = 0 ) {
    CUDAErrorCheck( cudaMemcpyAsync(
        dst.get_ptr() + dstOffst,
        src.get_ptr() + srcOffst,
        length*sizeof(T),
        cudaMemcpyDefault,
        inStream.get() ) );
  }

  /**
   * \brief Copy asynchronously on the given stream.
   */
  template< class T >
  void copyMemOnStream(
          uva_buffer<T>& dst,
          uva_buffer<T>&src,
          kites::stream& inStream,
          std::size_t const dstOffst = 0,
          std::size_t const srcOffst = 0 ) {
    std::size_t const length = std::min( src.size(), dst.size() );
    kites::copyMemOnStream( dst, src, length, inStream, dstOffst, srcOffst );
  }

}	// end namespace kites

#endif /* MEM_OPS_CUH_ */
