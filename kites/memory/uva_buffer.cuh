#ifndef UVA_BUFFER_CUH_
#define UVA_BUFFER_CUH_


namespace kites
{

/**
 * \brief Abstract buffer class template that other buffers are inherited from.
 *
 *	This acts as an interface for all type of buffers that obey Unified Virtual Addressing.
 *
 */
template < typename T >
class uva_buffer{

protected:

  std::size_t nElems;
  T* ptr;

public:

  uva_buffer():
    nElems( 0 ), ptr( nullptr )
  {
      //std::cout << "The default constructor for uva_buffer is called.\n";
  }

  uva_buffer( std::size_t const nElemsIn, T* const ptrIn ):
    nElems{ nElemsIn },
    ptr{ ptrIn }
  {}

  T* get_ptr() const{
    return ptr;
  }

  std::size_t size() const {
    return nElems;
  }

  std::size_t sizeInBytes() const {
    return nElems*sizeof(T);
  }

  // pure virtual member function hence abstract (interface) class.
  virtual void free() = 0;

  virtual ~uva_buffer(){};

};

}	// end namespace kites

#endif /* UVA_BUFFER_CUH_ */
