#ifndef FILE_INTERACTOR_HPP_
#define FILE_INTERACTOR_HPP_

#include <fstream>
#include <string>
#include <stdexcept>

namespace kites{

namespace io{

/**
 * \brief A function to open files safely for the read or the write.
 */
template <typename T_file>
void openFileToAccess( T_file& input_file, const std::string& file_name ) {
  input_file.open( file_name.c_str() );
  if( !input_file )
    throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}

}	// end namespace io

}	// end namespace kites


#endif /* FILE_INTERACTOR_HPP_ */
