#ifndef GRAPH_CSR_MIPMAPPED_DEVICE_CUH_
#define GRAPH_CSR_MIPMAPPED_DEVICE_CUH_


#include "graph_csr_mipmapped.hpp"
#include "../memory/mem_ops.cuh"
#include "../device/nv_gpu.cuh"


namespace kites{

template <class vT, class eT, class vIdxT = unsigned int>
class graph_csr_mipmapped_device : public graph_base
{
	using eIdxT = vIdxT;

public:

	int residing_device_id;	// Should be first.

	device_buffer<vT> V;
	device_buffer<eIdxT> R;
	device_buffer<vIdxT> C;
	device_buffer<eT> E;
	host_pinned_buffer<unsigned int> updateFlagHost;

	// For directed graphs.
	unsigned int compressionRatio;
	unsigned int mipmapDim;
	device_bitmap mipmap;
	device_bitmap vBitmapEven;
	device_bitmap vBitmapOdd;
	device_buffer<uint> compactionBuffer;

	int getResidingDeviceID() const {
		return residing_device_id;
	}


	/// Class members after this point are for profiling/experiment purposes.
	double accumulatedH2DCopyTime;
	graph_csr_mipmapped_device( graph_csr_mipmapped<vT, eT, vIdxT>& inHostGraph, kites::nv_gpu& dev ):
		residing_device_id{ [&]{ dev.setAsActive(); return dev.getDevIdx(); }() },
		compressionRatio{ inHostGraph.compressionRatio },
		mipmapDim{ inHostGraph.mipmapDim },
		V{ inHostGraph.V.size(), residing_device_id },
		R{ inHostGraph.R.size(), residing_device_id },
		C{ inHostGraph.C.size(), residing_device_id },
		E{ inHostGraph.E.size(), residing_device_id },
		vBitmapEven{ inHostGraph.vBitmap.getnBits(), residing_device_id },
		vBitmapOdd{ inHostGraph.vBitmap.getnBits(), residing_device_id },
		updateFlagHost{ inHostGraph.updateFlag.size() },
		mipmap{ inHostGraph.mipmap.getnBits(), residing_device_id },
		compactionBuffer{ inHostGraph.mipmapDim + 1, residing_device_id },
		accumulatedH2DCopyTime{ 0 }
		{
			using timer = std::chrono::high_resolution_clock;
			timer::time_point const t1 = timer::now();

			kites::copyMemOnStream( V, inHostGraph.V, dev.getSsEven() );
			kites::copyMemOnStream( R, inHostGraph.R, dev.getSsEven() );
			kites::copyMemOnStream( C, inHostGraph.C, dev.getSsEven() );
			kites::copyMemOnStream( E, inHostGraph.E, dev.getSsEven() );
			kites::copyMemOnStream( vBitmapEven, inHostGraph.vBitmap, dev.getSsEven() );
			kites::copyMemOnStream( mipmap, inHostGraph.mipmap, dev.getSsEven() );

			dev.getSsEven().sync();
			timer::time_point const t2 = timer::now();
			accumulatedH2DCopyTime = ( static_cast<double>( std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() ) / 1000.0 );

			vBitmapOdd.reset( dev.getSsEven() );
			this->set_num_edges( inHostGraph.get_num_edges() );
			this->set_num_vertices( inHostGraph.get_num_vertices() );
			std::cout << "H2D copies took " << accumulatedH2DCopyTime << " (ms) in total.\n";
		}

};

template < class vT, class eT, class vIdxT >
kites::graph_csr_mipmapped_device<vT, eT, vIdxT> make_graph_csr_for_device( kites::graph_csr_mipmapped<vT, eT, vIdxT>& inHostGraph, kites::nv_gpu& dev ) {
	return kites::graph_csr_mipmapped_device<vT, eT, vIdxT>( inHostGraph, dev );
}


}	// end namespace kites

#endif /* GRAPH_CSR_MIPMAPPED_DEVICE_CUH_ */
