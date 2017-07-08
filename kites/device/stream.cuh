#ifndef STREAM_CUH_
#define STREAM_CUH_

#include "../common/cuda_error_check.cuh"

namespace kites
{

/**
 * \brief This class encapsulates the stream objects.
 */
class stream {

private:
  int residing_device_id;
  cudaStream_t strm;

public:

  stream() = delete;
  stream( stream const& instrm ) = delete;
  stream& operator=( stream const& srcDev ) = delete;
  stream& operator=( stream&& in ) = delete;
  stream( stream&& instrm ):
    residing_device_id{ instrm.getResidingDeviceId() }
  {
    strm = instrm.strm;
    instrm.strm = nullptr;
  }

  stream( int const devID ):
    residing_device_id( devID )
  {
    CUDAErrorCheck( cudaSetDevice( residing_device_id ) );
    CUDAErrorCheck( cudaStreamCreate( &strm ) );
  }

  ~stream() {
    if( strm != nullptr ) {
      CUDAErrorCheck( cudaSetDevice( residing_device_id ) );
      CUDAErrorCheck( cudaStreamDestroy( strm ) );
    }
  }

  void sync() {
    CUDAErrorCheck( cudaSetDevice( residing_device_id ) );
    CUDAErrorCheck( cudaStreamSynchronize( strm ) );
  }

  cudaStream_t& get() {
    return strm;
  }
  cudaStream_t* get_ptr() {
    return &strm;
  }

  int getResidingDeviceId() const {
    return residing_device_id;
  }

};

}	// end namespace kites

#endif /* STREAM_CUH_ */
