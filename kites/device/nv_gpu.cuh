#ifndef NV_GPU_CUH_
#define NV_GPU_CUH_


#include "../common/cuda_error_check.cuh"
#include "stream.cuh"

#include <vector>
#include <algorithm>
#include <initializer_list>
#include <utility>
#include <cuda.h>
#include <cuda_runtime.h>

namespace kites
{

/**
 * \brief Represents a GPU.
 */
class nv_gpu
{

private:

  int devIdx;
  cudaDeviceProp devProp;
  kites::stream ssEven, ssOdd;

public:

  nv_gpu( int const inIdx )
    : devIdx{ inIdx }, ssEven{ inIdx }, ssOdd{ inIdx }
  {
    CUDAErrorCheck( cudaGetDeviceProperties( &devProp, inIdx ) );
    CUDAErrorCheck( cudaSetDevice( inIdx ) );
  }
  nv_gpu()
    : nv_gpu{ 0 }
  {}

  nv_gpu( nv_gpu const& srcDev ) = delete;
  nv_gpu& operator=( nv_gpu const& srcDev ) = delete;
  nv_gpu& operator=( nv_gpu&& in ) = delete;
  nv_gpu( nv_gpu&& in ):
    devIdx{ in.getDevIdx() },
    devProp( in.getDevProp() ),
    ssEven{ std::move( in.getSsEven() ) },
    ssOdd{ std::move( in.getSsOdd() ) }
  {}
  ~nv_gpu() = default;

  int getDevIdx() const {
    return devIdx;
  }

  cudaDeviceProp& getDevProp() {
    return devProp;
  }

  void setAsActive() const {
    CUDAErrorCheck( cudaSetDevice( this->devIdx ) );
  }

  void sync() {
    this->setAsActive();
    CUDAErrorCheck( cudaDeviceSynchronize() );
  }

  kites::stream& getSsEven() {
    return ssEven;
  }

  kites::stream& getSsOdd() {
    return ssOdd;
  }

  std::pair<double, double> queryMemUsage() const {
    this->setAsActive();
    std::pair<double, double> ret;
    std::size_t free, total;
    CUDAErrorCheck( cudaMemGetInfo( &free, &total ) );
    ret.first = ( total - free ) / 1000000.0;
    ret.second = total / 1000000.0;
    return ret;
  }

};



/**
 * \brief Represents a set of available GPUs.
 */
class nv_gpus
{

  std::vector< nv_gpu > devs;

public:

  nv_gpus( std::initializer_list<int> args ) {
    for( auto el: args )
      devs.emplace_back( el );
  }

  auto num_devices() -> decltype( devs.size() )
  {
    return devs.size();
  }

  void sync() {
    for( auto& d: devs )
      d.sync();
  }

  nv_gpu& at( std::size_t const idx ) {
    return devs.at( idx );
  }

};

}	// end namespace kites

#endif /* NV_GPU_CUH_ */
