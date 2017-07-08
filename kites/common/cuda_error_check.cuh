#ifndef	CUDA_ERROR_CHECK_CUH_
#define	CUDA_ERROR_CHECK_CUH_

#include <string>
#include <sstream>
#include <stdexcept>

//Error checking mechanism
#define CUDAErrorCheck(err) { CUDAAssert((err), __FILE__, __LINE__); }
inline void CUDAAssert( cudaError_t err, const char *file, int line )
{
 if ( err != cudaSuccess )
 {
    std::ostringstream errStream;
    errStream << "CUDAAssert: " << cudaGetErrorString(err) << " " << file << " " << line << "\n";
    throw std::runtime_error(errStream.str());
 }
}

#endif /* CUDA_ERROR_CHECK_CUH_ */
