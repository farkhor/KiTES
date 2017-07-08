#ifndef DEVICE_BUFFER_CUH_
#define DEVICE_BUFFER_CUH_


#include "uva_buffer.cuh"
#include "../common/cuda_error_check.cuh"
#include "../device/stream.cuh"

#include <cuda.h>
#include <cuda_runtime.h>


namespace kites
{

/**
 * \brief This class represents a global memory buffer on a particular device.
 */
template <typename T>
class device_buffer : public uva_buffer<T>{

private:

  int residing_device_id;

  void construct( std::size_t const n ) {
    CUDAErrorCheck( cudaSetDevice( this->residing_device_id ) );
    CUDAErrorCheck( cudaMalloc( (void**)&this->ptr, n*sizeof(T) ) );
    this->nElems = n;
  }

public:

  void free() override {
    if( this->nElems!=0 ) {
      this->nElems = 0;
      CUDAErrorCheck( cudaFree( this->ptr ) );
    }
  }

  device_buffer():
    residing_device_id{ 0 }
  {}

  device_buffer( std::size_t const n, int const devID  ):
    residing_device_id{ devID }
  {
    if( n > 0 )
      construct( n );
  }

  /**
   * \brief Copy construction on \p device_buffer.
   */
  device_buffer( uva_buffer<T> const& inDB, int const devID = 0 ):
    device_buffer{ inDB.size(), devID }
  {
    if( inDB.size() != 0 )
      CUDAErrorCheck( cudaMemcpy( this->ptr, inDB.get_ptr(), inDB.sizeInBytes(), cudaMemcpyDefault ) );
  }

  /**
   * \brief Disallowing copy assignment operation on \p device_buffer since it may cause ambiguity.
   */
  device_buffer& operator=( device_buffer const& inDB ) = delete;

  /**
   * \brief constructs the \p device_buffer by moving the content of an already existing
   * \p device_buffer.
   *
   * This form of construction is expected to have no allocation overhead
   * by moving the content of a temporary/ (a.k.a. rvalue) \p device_buffer.
   *
   * @param inPB The \p device_buffer that its content needs to be moved
   *             to the new \p device_buffer being constructed.
   */
  device_buffer( device_buffer&& inDB ):
    uva_buffer<T>{ inDB.size(), inDB.get_ptr() },
    residing_device_id{ inDB.residing_device_id }
  {
    inDB.nElems = 0;
    inDB.ptr = nullptr;
  }

  /**
   * \brief Move assignment operator.
   *
   * Similar to the move constructor, move assignment operator creates
   * the \p device_buffer by assuming the ownership of an already existing temporary buffer.
   *
   * @param inPB The \p device_buffer that its content needs to be moved
   *             to the new \p device_buffer being constructed.
   */
  device_buffer& operator=( device_buffer&& inDB ) {
    this->nElems = inDB.nElems;
    this->ptr = inDB.ptr;
    inDB.nElems = 0;
    inDB.ptr = nullptr;
    return *this;
  }

  /**
   * Upon the destruction, the allocated memory is freed.
   */
  virtual ~device_buffer() override {
    this->free();
  }

  virtual void alloc( std::size_t const n, int const devID = 0 ) {
    residing_device_id = devID;
    if( this->nElems == 0 && n > 0 )
      construct( n );
  }

  int get_residing_device_id() const {
    return this->residing_device_id;
  }

  void reset() {
    CUDAErrorCheck( cudaSetDevice( this->get_residing_device_id() ) );
    CUDAErrorCheck( cudaMemset( (void*)this->ptr, 0, this->sizeInBytes() ) );
  }

  void reset( kites::stream& ss ) {
    CUDAErrorCheck( cudaSetDevice( this->get_residing_device_id() ) );
    CUDAErrorCheck( cudaMemsetAsync( (void*)this->ptr, 0, this->sizeInBytes(), ss.get() ) );
  }

};


class device_bitmap : public device_buffer<unsigned int> {

  using dev_idx_T = int;

private:

  std::size_t nBits;

public:

  device_bitmap():
    nBits{ 0 }
  {}

  device_bitmap( std::size_t const n, dev_idx_T const devID = 0 ):
    device_buffer{ static_cast<std::size_t>( std::ceil( static_cast<double>( n ) / 32.0 ) ), devID }, nBits{ n }
  {}

  void alloc( std::size_t const n, dev_idx_T const devID = 0 ) override {
    if( n > 0 )
      this->nBits = n;
      device_buffer<unsigned int>::alloc( std::ceil( static_cast<double>( n ) / 32.0 ), devID );
  }

  std::size_t getnBits() const {
    return nBits;
  }

};

}	// end namespace kites

#endif /* DEVICE_BUFFER_CUH_ */
