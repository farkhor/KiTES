#ifndef GRAPH_CSR_DEVICE_CUH_
#define GRAPH_CSR_DEVICE_CUH_

#include "graph_csr.h"
#include "../memory/mem_ops.cuh"
#include "../device/nv_gpu.cuh"

namespace kites{

/**
 * \brief The class template represents a CSR graph kept inside
 *        the device memory.
 */
template <class vT, class eT, class vIdxT = unsigned int>
class graph_csr_device : public graph_base
{
	using eIdxT = vIdxT;

public:

	// The ID for the device holding the graph.
	// Should come first in the list of member variables.
	int residing_device_id;

	// Essential objects to represent the graph.
	device_buffer<vT> V;
	device_buffer<eIdxT> R;
	device_buffer<vIdxT> C;
	device_buffer<eT> E;
	device_bitmap vBitmapEven;
	device_bitmap vBitmapOdd;
	device_buffer<uint> updateFlag;
	host_pinned_buffer<uint> updateFlagHost;
	kites::graph_property gProp;

	// Variables for directed graphs.
	device_buffer<eIdxT> dir_R;
	device_buffer<vIdxT> dir_C;
	uint vertexPerGroup;

	// Outbox buffers, used exclusively in multi-GPU scenarios.
	device_buffer<uint> outboxTop;
	device_buffer< vIdxT > outboxIndices;
	device_buffer< vT > outboxVertices;

	// Inbox buffers, used exclusively in multi-GPU scenarios.
	device_buffer< vIdxT > inboxIndices_odd;
	device_buffer< vT > inboxVertices_odd;
	host_pinned_buffer<uint> inboxTop_odd;
	device_buffer< vIdxT > inboxIndices_even;
	device_buffer< vT > inboxVertices_even;
	host_pinned_buffer<uint> inboxTop_even;


	/**
	 * \brief Returns the device ID on which the graph is residing.
	 *
	 * @return The ID of the associated device.s
	 */
	int getResidingDeviceID() const {
		return residing_device_id;
	}

	/**
	 * \brief Returns the vertex grouping ratio for the graph.
	 *
	 * Note that it has a meaning only when the graph is directed.
	 *
	 * @return Vertex grouping ratio.
	 */
	uint getVertexPerGroup() const {
		return vertexPerGroup;
	}

	graph_csr_device() = default;

	/**
	 * \brief The main constructor for the object using a host CSR graph.
	 *
	 *  @param inHostGraph The input host CSR graph.
	 *  @param dev The device holding the graph.
	 */
	graph_csr_device(
		kites::graph_csr<vT, eT, vIdxT>& inHostGraph,
		kites::nv_gpu& dev ):
			residing_device_id{
				[&]{ dev.setAsActive(); return dev.getDevIdx(); }()
			},
			vertexPerGroup{ inHostGraph.getVertexPerGroup() },
			V{ inHostGraph.V.size(), residing_device_id },
			R{ inHostGraph.R.size(), residing_device_id },
			C{ inHostGraph.C.size(), residing_device_id },
			E{ inHostGraph.E.size(), residing_device_id },
			vBitmapEven{ inHostGraph.vBitmap.getnBits(), residing_device_id },
			vBitmapOdd{ inHostGraph.vBitmap.getnBits(), residing_device_id },
			updateFlag{ inHostGraph.updateFlag, residing_device_id },
			updateFlagHost{ inHostGraph.updateFlag.size() },
			gProp{ inHostGraph.gProp },
			dir_R{ inHostGraph.dir_R.size(), residing_device_id },
			dir_C{ inHostGraph.dir_C.size(), residing_device_id }
			{
				using timer = std::chrono::high_resolution_clock;
				timer::time_point const t1 = timer::now();

				kites::copyMemOnStream( V, inHostGraph.V, dev.getSsEven() );
				kites::copyMemOnStream( R, inHostGraph.R, dev.getSsEven() );
				kites::copyMemOnStream( C, inHostGraph.C, dev.getSsEven() );
				kites::copyMemOnStream( E, inHostGraph.E, dev.getSsEven() );
				kites::copyMemOnStream
					( vBitmapEven, inHostGraph.vBitmap, dev.getSsEven() );
				if( gProp == kites::graph_property::directed ) {
					kites::copyMemOnStream
						( dir_R, inHostGraph.dir_R, dev.getSsEven() );
					kites::copyMemOnStream
						( dir_C, inHostGraph.dir_C, dev.getSsEven() );
				}
				vBitmapOdd.reset( dev.getSsEven() );
				this->set_num_edges( inHostGraph.get_num_edges() );
				this->set_num_vertices( inHostGraph.get_num_vertices() );

				dev.getSsEven().sync();
				timer::time_point const t2 = timer::now();
				double const accumulatedH2DCopyTime =
					( static_cast<double>(
						std::chrono::duration_cast<std::chrono::microseconds>
							( t2 - t1 ).count() ) / 1000.0 );
				std::cout << "H2D copies took " <<
					accumulatedH2DCopyTime << " (ms) in total.\n";
			}


};

/**
 * \brief A function to make CSR graphs for a device.
 *
 *  The main purpose of this function is to create a
 *  device-specific CSR graph object without having to specify
 *  the template parameters.
 *
 *  @param inHostGraph The input host CSR graph.
 *  @param dev The device holding the graph.
 *
 *  @return Created device-specific CSR graph.
 */
template < class vT, class eT, class vIdxT >
kites::graph_csr_device<vT, eT, vIdxT> make_graph_csr_for_device(
		kites::graph_csr<vT, eT, vIdxT>& inHostGraph,
		kites::nv_gpu& dev
		) {
	return kites::graph_csr_device<vT, eT, vIdxT>( inHostGraph, dev );
}


}	// end namespace kites


#endif /* GRAPH_CSR_DEVICE_CUH_ */
