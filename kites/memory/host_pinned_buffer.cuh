#ifndef HOST_PINNED_BUFFER_CUH_
#define HOST_PINNED_BUFFER_CUH_


#include "uva_buffer.cuh"
#include "../common/cuda_error_check.cuh"

#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


namespace kites
{

/**
 * \brief This class represents a host pinned memory buffer.
 */
template <typename T>
class host_pinned_buffer : public uva_buffer<T>{

private:

  void construct( std::size_t const n ){
    CUDAErrorCheck( cudaHostAlloc( (void**)&this->ptr, n*sizeof(T), cudaHostAllocPortable ) );
    this->nElems = n;
  }

public:

  void free() override {
    if( this->nElems != 0 ) {
      this->nElems = 0;
      CUDAErrorCheck( cudaFreeHost( this->ptr ) );
    }
  }

  host_pinned_buffer():
    uva_buffer<T>()
  {}

  host_pinned_buffer( std::size_t const n ){
    if( n > 0 )
      construct( n );
  }

  /**
   * \brief Copy construction on \p host_pinned_buffer.
   */
  host_pinned_buffer( uva_buffer<T> const& inHPB ):
    host_pinned_buffer{ inHPB.size() }
  {
    if( inHPB.size() != 0 )
      CUDAErrorCheck( cudaMemcpy( this->ptr, inHPB.get_ptr(), inHPB.sizeInBytes(), cudaMemcpyDefault ) );
  }

  /**
   * \brief Disallowing copy assignment operation on \p host_pinned_buffer since it may cause ambiguity.
   */
  host_pinned_buffer& operator=( host_pinned_buffer const& inHPB ) = delete;

  /**
   * \brief constructs the \p host_pinned_buffer by moving the content of an already existing
   * \p host_pinned_buffer.
   *
   * This form of construction is expected to have no allocation overhead
   * by moving the content of a temporary/ (a.k.a. rvalue) \p host_pinned_buffer.
   *
   * @param inPB The \p host_pinned_buffer that its content needs to be moved
   *             to the new \p host_pinned_buffer being constructed.
   */
  host_pinned_buffer( host_pinned_buffer&& inHPB ):
    uva_buffer<T>{ inHPB.size(), inHPB.get_ptr() }
  {
    inHPB.nElems = 0;
    inHPB.ptr = nullptr;
  }

  /**
   * \brief Move assignment operator.
   *
   * Similar to the move constructor, move assignment operator creates
   * the \p host_pinned_buffer by assuming the ownership of an already existing temporary buffer.
   *
   * @param inPB The \p host_pinned_buffer that its content needs to be moved
   *             to the new \p host_pinned_buffer being constructed.
   */
  host_pinned_buffer& operator=( host_pinned_buffer&& inHPB ) {
    this->nElems = inHPB.nElems;
    this->ptr = inHPB.ptr;
    inHPB.nElems = 0;
    inHPB.ptr = nullptr;
    return *this;
  }

  /**
   * Upon the destruction, the allocated memory is freed.
   */
  ~host_pinned_buffer() override {
    this->free();
  }

  virtual void alloc( std::size_t const n ){
    if( this->nElems == 0 && n > 0 )
      construct(n);
  }

  T& at( std::size_t const index ){
    if( index >= this->nElems )
      throw std::runtime_error( "The referred element does not exist in the buffer." );
    return this->ptr[ index ];
  }

  T& operator[]( std::size_t const index ) {
    return this->ptr[ index ];
  }

  friend std::ostream& operator<<( std::ostream& output, host_pinned_buffer& D ) {
    for( std::size_t iii = 0; iii < D.size(); ++iii )
      output << D.at( iii ) << "\n";
    return output;
  }

};


class host_bitmap : public host_pinned_buffer<unsigned int> {

private:
  std::size_t nBits;

public:

  host_bitmap():
    nBits{ 0 }
  {}

  host_bitmap( std::size_t const n ):
    nBits{ n },
    host_pinned_buffer{ static_cast<std::size_t>(std::ceil( static_cast<double>( n ) / 32.0 )) }
  {}

  void alloc( std::size_t const n ) override {
    if( n > 0 )
      this->nBits = n;
    host_pinned_buffer<unsigned int>::alloc( std::ceil( static_cast<double>( n ) / 32.0 ) );
  }

  void setAt( std::size_t const pos ) {
    this->ptr[ pos >> 5 ] |= ( 1 << ( pos & 31 ) );
  }

  void reset() {
    std::memset( this->ptr, 0, this->sizeInBytes() );
  }

  std::size_t getnBits() const {
    return nBits;
  }

  unsigned int count() const {
    unsigned int const nnn = this->size();
    unsigned int ret = 0;
    for( unsigned int iii = 0; iii < nnn; ++iii ) {
      ret += __builtin_popcount( this->get_ptr()[ iii ] );
    }
    return ret;
  }

};

}	// end namespace kites

#endif /* HOST_PINNED_BUFFER_CUH_ */
