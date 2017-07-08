#ifndef COMPUTE_CUH_
#define COMPUTE_CUH_

#include "../common/globals.cuh"
#include "nv_gpu.cuh"
#include "kernels_single_device.cuh"
#include "kernels_multi_device.cuh"
#include "../graph/graph_csr_device.cuh"
#include "../graph/graph_csr_multi_device.cuh"
#include "../graph/graph_csr_mipmapped_device.cuh"

#include <chrono>
#include <cmath>

namespace kites
{

/**
 * \brief The main processing routine for processing a graph
 *        collected in one GPU.
 *
 * @param grph CSR graph collected inside the device memory.
 * @param dev Device that will process the graph.
 * @param funcInit Passed initialization function.
 * @param funcCompNbr Passed neighbor computation function.
 * @param funcRed Passed reduction function.
 * @param funcUpd Passed update function.
 */
template< launch launchMode, class vT, class eT, class vIdxT,
	class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
void process(
		graph_csr_device<vT, eT, vIdxT>& grph,
		nv_gpu& dev,
		funcInitT funcInit,
		funcCompNbrT funcCompNbr,
		funcRedT funcRed,
		funcUpdT funcUpd
		) {

	unsigned int const bDim = COMPILE_TIME_DETERMINED_BLOCK_SIZE;
	auto const gDim = static_cast<unsigned int>
		( std::ceil( static_cast<double>( grph.get_num_vertices() ) / bDim ) );
	unsigned int iterationCounter( 0 );
	dev.setAsActive();

	// Getting information about consumed memory.
	auto const consumedAndTotal = dev.queryMemUsage();
	std::cout << consumedAndTotal.first << " MB is consumed out of the total " <<
			consumedAndTotal.second << " MB global memory.\n";
	uint const vpg_shift = static_cast<uint>
		( std::log( grph.getVertexPerGroup() ) / std::log( 2 ) );

	std::cout << "Launching iterative procedure ...\n";
	using timer = std::chrono::high_resolution_clock;
	timer::time_point const t1 = timer::now();

	do {
		grph.updateFlagHost[0] = 0;
		kites::copyMemOnStream
			( grph.updateFlag, grph.updateFlagHost, dev.getSsEven() );
		bool const evenIter = ( iterationCounter & 1 ) == 0 ;
		device_bitmap& bitmapToRead = evenIter ? grph.vBitmapEven : grph.vBitmapOdd;
		device_bitmap& bitmapToWrite = evenIter ? grph.vBitmapOdd : grph.vBitmapEven;

		//host_bitmap hb( bitmapToRead.getnBits() );
		//cudaMemcpy( hb.get_ptr(), bitmapToRead.get_ptr(), bitmapToRead.getnBits() / 8, cudaMemcpyDeviceToHost );
		//std::cout << hb.count() << "\t";

		if( grph.getVertexPerGroup() == 1 ) {
			if( grph.gProp == kites::graph_property::undirected )
				iteration_kernel_single_device	< kites::graph_property::undirected >
					<<<gDim, bDim, 0, dev.getSsEven().get()>>>
					( static_cast<uint>( grph.get_num_vertices() ),
						grph.C.get_ptr(),
						grph.R.get_ptr(),
						grph.V.get_ptr(),
						grph.E.get_ptr(),
						grph.updateFlag.get_ptr(),
						bitmapToRead.get_ptr(), bitmapToWrite.get_ptr(),
						funcInit, funcCompNbr, funcRed, funcUpd );
			else
				iteration_kernel_single_device	< kites::graph_property::directed >
					<<<gDim, bDim, 0, dev.getSsEven().get()>>>
					( static_cast<uint>( grph.get_num_vertices() ),
						grph.C.get_ptr(),
						grph.R.get_ptr(),
						grph.V.get_ptr(),
						grph.E.get_ptr(),
						grph.updateFlag.get_ptr(),
						bitmapToRead.get_ptr(), bitmapToWrite.get_ptr(),
						funcInit, funcCompNbr, funcRed, funcUpd,
						grph.dir_C.get_ptr(), grph.dir_R.get_ptr() );
		} else if( grph.getVertexPerGroup() >= 32 ){
			iteration_kernel_single_device_vpg
				<<<gDim, bDim, 0, dev.getSsEven().get()>>>
				( static_cast<uint>( grph.get_num_vertices() ),
					grph.C.get_ptr(),
					grph.R.get_ptr(),
					grph.V.get_ptr(),
					grph.E.get_ptr(),
					grph.updateFlag.get_ptr(),
					bitmapToRead.get_ptr(), bitmapToWrite.get_ptr(),
					funcInit, funcCompNbr, funcRed, funcUpd,
					grph.dir_C.get_ptr(), grph.dir_R.get_ptr(), vpg_shift );
		} else {
			iteration_kernel_single_device_incompletevpg
				<<<gDim, bDim, 0, dev.getSsEven().get()>>>
				( static_cast<uint>( grph.get_num_vertices() ),
					grph.C.get_ptr(),
					grph.R.get_ptr(),
					grph.V.get_ptr(),
					grph.E.get_ptr(),
					grph.updateFlag.get_ptr(),
					bitmapToRead.get_ptr(), bitmapToWrite.get_ptr(),
					funcInit, funcCompNbr, funcRed, funcUpd,
					grph.dir_C.get_ptr(), grph.dir_R.get_ptr(), vpg_shift );

		}
		CUDAErrorCheck( cudaPeekAtLastError() );

		bitmapToRead.reset( dev.getSsEven() );
		kites::copyMemOnStream
			( grph.updateFlagHost, grph.updateFlag, dev.getSsEven() );

		++iterationCounter;
		dev.getSsEven().sync();

	} while( grph.updateFlagHost[0] != 0 );

	timer::time_point const t2 = timer::now();
	auto const processing_time = std::chrono::duration_cast
			<std::chrono::microseconds>( t2 - t1 ).count();
	std::cout << "Processing finished in " <<
			static_cast<double>( processing_time ) / 1000.0 << " (ms).\n";
	std::cout << "Performed " << iterationCounter << " iterations in total.\n";

}

/**
 * \brief The main processing routine for processing a graph
 *        collected in multiple GPUS.
 *
 * @param grph CSR graph collected inside the device memory.
 * @param devs Device group that will process the graph.
 * @param funcInit Passed initialization function.
 * @param funcCompNbr Passed neighbor computation function.
 * @param funcRed Passed reduction function.
 * @param funcUpd Passed update function.
 */
template< kites::launch launchMode, class vT, class eT, class vIdxT,
	class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
void process(
		kites::graph_csr_multi_device<vT, eT, vIdxT>& grph,
		kites::nv_gpus& devs,
		funcInitT funcInit,
		funcCompNbrT funcCompNbr,
		funcRedT funcRed,
		funcUpdT funcUpd
		) {

	unsigned int const bDim = COMPILE_TIME_DETERMINED_BLOCK_SIZE;
	unsigned int iterationCounter( 0 );
	uint const vpg_shift = static_cast<uint>
		( std::log( grph.getVertexPerGroup() ) / std::log( 2 ) );
	bool const hostAsHub = grph.if_host_as_hub();

	std::vector<cudaEvent_t> events( devs.num_devices() );


	std::cout << "Launching iterative procedure ...\n";
	using timer = std::chrono::high_resolution_clock;
	timer::time_point const t1 = timer::now();

	do {
		bool const evenIter = ( iterationCounter & 1 ) == 0;


		uint devInHandHostInboxOffset = 0;

		// A loop on all devices involved.
		for( uint devID = 0; devID < devs.num_devices(); ++devID ) {

			// Define some variables for the device in hand.
			kites::nv_gpu& devInHand = devs.at( devID );
			graph_csr_device<vT, eT, vIdxT>& grphPart = grph.at( devID );
			devInHand.setAsActive();
			device_bitmap& bitmapToRead = evenIter ?
					grphPart.vBitmapEven : grphPart.vBitmapOdd;
			device_bitmap& bitmapToWrite = evenIter ?
					grphPart.vBitmapOdd : grphPart.vBitmapEven;

			// Initialize the update flag.
			grphPart.updateFlagHost[ 0 ] = 0;
			kites::copyMemOnStream
				( grphPart.updateFlag, grphPart.updateFlagHost, devInHand.getSsEven() );

			// Initialize the outbox top.
			host_pinned_buffer<uint>& inboxTop
				= evenIter ? grphPart.inboxTop_odd : grphPart.inboxTop_even;
			inboxTop[ 0 ] = 0;
			kites::copyMemOnStream
				( grphPart.outboxTop, inboxTop, devInHand.getSsEven() );

			////////////////////////////////////
			// UNLOAD INBOX AND UPDATE BITMAP.
			////////////////////////////////////

			if( iterationCounter > 0 ) {

				uint hostInboxOffset = 0;

				// For all other devices.
				for( uint targetDevID = 0;
						targetDevID < devs.num_devices();
						++targetDevID ) {

					if( targetDevID != devID ) {

						unload_inbox(
							grphPart.V.get_ptr(),
							evenIter ? grph.at( targetDevID ).inboxTop_even[ 0 ] : grph.at( targetDevID ).inboxTop_odd[ 0 ],
							evenIter ?
									( hostAsHub ? grph.HAHvertexIndices_even.get_ptr() : grphPart.inboxIndices_even.get_ptr() ) :
									( hostAsHub ? grph.HAHvertexIndices_odd.get_ptr() : grphPart.inboxIndices_odd.get_ptr() ),
							evenIter ?
									( hostAsHub ? grph.HAHvertexValue_even.get_ptr() : grphPart.inboxVertices_even.get_ptr() ) :
									( hostAsHub ? grph.HAHvertexValue_odd.get_ptr() : grphPart.inboxVertices_odd.get_ptr() ),
							devInHand.getSsEven().get(),
							hostAsHub ? hostInboxOffset : 0
						);
						update_bitmap(
							bitmapToRead.get_ptr(),
							evenIter ? grph.at( targetDevID ).vBitmapEven.get_ptr() : grph.at( targetDevID ).vBitmapOdd.get_ptr(),
							( grph.vertexRangeAssignment[ devID + 1 ] - grph.vertexRangeAssignment[ devID ] ) / grph.getVertexPerGroup(),
							devInHand.getSsEven().get(),
							grph.vertexRangeAssignment[ devID ] / grph.getVertexPerGroup(),
							grph.vertexRangeAssignment[ devID ] / grph.getVertexPerGroup()
						);

					}	// End of condition for avoiding self-update for device.

					if( hostAsHub )
						hostInboxOffset += grph.maxNumVerticesSend[ targetDevID ];
				}

			}	// End of condition for the first iteration.

			////////////////////////////////////
			// LAUNCHING THE KERNEL.
			////////////////////////////////////

			// Reset the write bitmask for the new iteration.
			bitmapToWrite.reset( devInHand.getSsEven() );

			// Caclculate the number of vertices to process and required grid dim.
			uint const nVerticesToProcess =
					grph.vertexRangeAssignment[ devID + 1 ] -
					grph.vertexRangeAssignment[ devID ];
			uint const gDim = std::ceil( static_cast<float>
				( nVerticesToProcess ) / COMPILE_TIME_DETERMINED_BLOCK_SIZE );

			// KERNEL.
			if( gDim != 0 ) {
				if( grph.getVertexPerGroup() == 1 ) {
					if( grph.at( 0 ).gProp == kites::graph_property::undirected )
						iteration_kernel_multi_device < kites::graph_property::undirected >
						<<< gDim, bDim, 0, devInHand.getSsEven().get() >>>
						( nVerticesToProcess, grphPart.C.get_ptr(), grphPart.R.get_ptr(), grphPart.V.get_ptr(), grphPart.E.get_ptr(),
						grphPart.updateFlag.get_ptr(), bitmapToRead.get_ptr(), bitmapToWrite.get_ptr(),
						funcInit, funcCompNbr, funcRed, funcUpd,

						grph.vertexRangeStorage[ devID ].first,
						grph.edgeRangeStorage[ devID ].first,

						grphPart.outboxTop.get_ptr(), grphPart.outboxVertices.get_ptr(), grphPart.outboxIndices.get_ptr()
						);
					else {
						iteration_kernel_multi_device < kites::graph_property::directed >
						<<< gDim, bDim, 0, devInHand.getSsEven().get() >>>
						( nVerticesToProcess, grphPart.C.get_ptr(), grphPart.R.get_ptr(), grphPart.V.get_ptr(), grphPart.E.get_ptr(),
						grphPart.updateFlag.get_ptr(), bitmapToRead.get_ptr(), bitmapToWrite.get_ptr(),
						funcInit, funcCompNbr, funcRed, funcUpd,

						grph.vertexRangeStorage[ devID ].first,
						grph.edgeRangeStorage[ devID ].first,

						grphPart.outboxTop.get_ptr(), grphPart.outboxVertices.get_ptr(), grphPart.outboxIndices.get_ptr(),
						grphPart.dir_C.get_ptr(), grphPart.dir_R.get_ptr(),

						grph.dir_vertexRangeStorage[ devID ].first,
						grph.dir_edgeRangeStorage[ devID ].first

						);
					}
				} else if( grph.getVertexPerGroup() >= 32 ) {
					iteration_kernel_multi_device_vpg
						<<< gDim, bDim, 0, devInHand.getSsEven().get() >>>
						( nVerticesToProcess, grphPart.C.get_ptr(), grphPart.R.get_ptr(), grphPart.V.get_ptr(), grphPart.E.get_ptr(),
						grphPart.updateFlag.get_ptr(), bitmapToRead.get_ptr(), bitmapToWrite.get_ptr(),
						funcInit, funcCompNbr, funcRed, funcUpd, vpg_shift,

						grph.vertexRangeStorage[ devID ].first,
						grph.edgeRangeStorage[ devID ].first,

						grphPart.outboxTop.get_ptr(), grphPart.outboxVertices.get_ptr(), grphPart.outboxIndices.get_ptr(),
						grphPart.dir_C.get_ptr(), grphPart.dir_R.get_ptr(),

						grph.dir_vertexRangeStorage[ devID ].first,
						grph.dir_edgeRangeStorage[ devID ].first
						);
				} else {
					iteration_kernel_multi_device_incompletevpg
						<<< gDim, bDim, 0, devInHand.getSsEven().get() >>>
						( nVerticesToProcess, grphPart.C.get_ptr(), grphPart.R.get_ptr(), grphPart.V.get_ptr(), grphPart.E.get_ptr(),
						grphPart.updateFlag.get_ptr(), bitmapToRead.get_ptr(), bitmapToWrite.get_ptr(),
						funcInit, funcCompNbr, funcRed, funcUpd, vpg_shift,

						grph.vertexRangeStorage[ devID ].first,
						grph.edgeRangeStorage[ devID ].first,

						grphPart.outboxTop.get_ptr(), grphPart.outboxVertices.get_ptr(), grphPart.outboxIndices.get_ptr(),
						grphPart.dir_C.get_ptr(), grphPart.dir_R.get_ptr(),

						grph.dir_vertexRangeStorage[ devID ].first,
						grph.dir_edgeRangeStorage[ devID ].first
						);
				}
			}
			CUDAErrorCheck( cudaPeekAtLastError() );

			kites::copyMemOnStream( grphPart.updateFlagHost, grphPart.updateFlag, devInHand.getSsEven() );
			kites::copyMemOnStream( evenIter ? grphPart.inboxTop_odd : grphPart.inboxTop_even, grphPart.outboxTop, devInHand.getSsEven() );

			////////////////////////////////////
			// DISTRIBUTING UPDATED DATA.
			////////////////////////////////////

			// Distribute/copy the outbox to host/other device's inboxes.
			if( hostAsHub ) {
				distribute_outbox(
					grph.maxNumVerticesSend[ devID ],
					grphPart.outboxTop.get_ptr(),
					grphPart.outboxIndices.get_ptr(),
					grphPart.outboxVertices.get_ptr(),
					evenIter ? grph.HAHvertexIndices_odd.get_ptr() : grph.HAHvertexIndices_even.get_ptr(),
					evenIter ? grph.HAHvertexValue_odd.get_ptr() : grph.HAHvertexValue_even.get_ptr(),
					devInHand.getSsEven().get(),
					devInHandHostInboxOffset );
				devInHandHostInboxOffset += grph.maxNumVerticesSend[ devID ];
			} else {	// Two device holding inboxes.
				for( uint targetDevID = 0; targetDevID < devs.num_devices(); ++targetDevID )
					if( targetDevID != devID )	// Made possible by direct peer memory access.
						distribute_outbox(
							grph.maxNumVerticesSend[ devID ],
							grphPart.outboxTop.get_ptr(),
							grphPart.outboxIndices.get_ptr(),
							grphPart.outboxVertices.get_ptr(),
							evenIter ? grph.at( targetDevID ).inboxIndices_odd.get_ptr() : grph.at( targetDevID ).inboxIndices_even.get_ptr(),
							evenIter ? grph.at( targetDevID ).inboxVertices_odd.get_ptr() : grph.at( targetDevID ).inboxVertices_even.get_ptr(),
							devInHand.getSsEven().get(),
							0 );
			}

		} // End of the for loop on the device.

		// Sync with the host at the end of an iteration.
		++iterationCounter;
		devs.sync();
	} while( grph.any_update() );

	// Get processing time and other information.
	timer::time_point const t2 = timer::now();
	auto const processing_time = std::chrono::duration_cast
		<std::chrono::microseconds>( t2 - t1 ).count();
	std::cout << "Processing finished in " <<
		static_cast<double>( processing_time ) / 1000.0 << " (ms).\n";
	std::cout << "Performed " << iterationCounter << " iterations in total.\n";

}


/**
 * \brief The main processing routine for processing a graph
 *        collected in the host pinned memory by a device.
 *
 * This routine first puts the graph pieces into appropriate
 *  data structures inside the specified device and then runs the
 *  core single-GPU graph processing routine.
 *
 * @param grph CSR graph collected inside the host pinned memory.
 * @param dev Device that will process the graph.
 * @param funcInit Passed initialization function.
 * @param funcCompNbr Passed neighbor computation function.
 * @param funcRed Passed reduction function.
 * @param funcUpd Passed update function.
 */
template< kites::launch launchMode, class vT, class eT, class vIdxT,
	class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
void process(
		kites::graph_csr<vT, eT, vIdxT>& grph,
		kites::nv_gpu& dev,
		funcInitT funcInit,
		funcCompNbrT funcCompNbr,
		funcRedT funcRed,
		funcUpdT funcUpd
		) {
	auto grphDev = kites::make_graph_csr_for_device( grph, dev );
	kites::process< kites::launch::sync >
		( grphDev, dev, funcInit, funcCompNbr, funcRed, funcUpd );
	kites::copyMem( grph.V, grphDev.V );	// sync with the host thread.
}

/**
 * \brief The main processing routine for processing a graph
 *        collected in the host pinned memory by a device group.
 *
 * This routine first puts the graph pieces into appropriate
 * data structures inside specified devices and then runs the
 * core multi-GPU graph processing routine.
 *
 * @param grph CSR graph collected inside the host pinned memory.
 * @param devs Device group that will process the graph.
 * @param funcInit Passed initialization function.
 * @param funcCompNbr Passed neighbor computation function.
 * @param funcRed Passed reduction function.
 * @param funcUpd Passed update function.
 */
template< kites::launch launchMode,
	class vT, class eT, class vIdxT,
	class funcInitT, class funcCompNbrT, class funcRedT, class funcUpdT >
void process(
		kites::graph_csr<vT, eT, vIdxT>& grph,
		kites::nv_gpus& devs,
		funcInitT funcInit,
		funcCompNbrT funcCompNbr,
		funcRedT funcRed,
		funcUpdT funcUpd
		) {
	auto grphDevs = kites::make_graph_csr_for_devices( grph, devs );
	kites::process< kites::launch::sync >
		( grphDevs, devs, funcInit, funcCompNbr, funcRed, funcUpd );
	grphDevs.getVertexValues( grph.V );
}


/**
 * \brief Synchronizes the given GPU object with the host.
 *
 * @param trgtDev The device the host will be synchronized with.
 */
void sync_with( kites::nv_gpu& trgtDev )
{
	trgtDev.sync();
}

/**
 * \brief Synchronizes the given GPU object group with the host.
 *
 * @param trgtDev The device group the host will be synchronized with.
 */
void sync_with( kites::nv_gpus& trgtDev )
{
	trgtDev.sync();
}


}	// end namespace kites

#endif /* COMPUTE_CUH_ */
