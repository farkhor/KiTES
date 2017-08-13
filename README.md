# KiTES
KiTES is a CUDA C++11 template library for high-performance single/multi GPU vertex-centric graph processing.

Requirements
------------------
* A GPU from Kepler family or newer (Compute Capability 3.0 or higher).
* CUDA 7.5 compiler or higher (for C++11 support).
* `--expt-extended-lambda` compilation flag to compile with GPU (or heterogeneous) lambdas.

Usage
------------------
KiTES is designed as a template library meaning it does not need separate compilation but rather gets compiled by NVCC alongside the program that includes its header(s). This design maximizes the solution's portability and ease-of-use. Examples in the [example folder](https://github.com/farkhor/KiTES/tree/master/examples) show some samples of KiTES usage. Here we move forward with SSSP example.

KiTES requires four device functions for vertex initialization, neighbor visitation, partial value reduction, and update predicate creation. Below device functions (described by lambda expressions) describe the procedure to perform on a vertex during an iteration so as to carry out SSSP iteratively in our example.
```
auto initF    = [] __host__ __device__
  ( volatile vT& locV, vT& globV ) { locV = globV; };
auto nbrCompF = [] __host__ __device__
  ( volatile vT& partialV, vT srcV, eT* connE ) { partialV = srcV + (*connE); };
auto redF     = [] __host__ __device__
  ( volatile vT& lhsV, volatile vT& rhsV ) { if( lhsV > rhsV ) lhsV = rhsV; };
auto updF     = [] __host__ __device__
  ( volatile vT& newV, vT oldV ) { return newV < oldV; };
```
   
All the graphs processed by the GPU have to be first constructed at the host side, implicitly or explicitly. Below piece of code, creates a host graph with specified pre-processing function, specified vertex-grouping ratio, and the arrangement of vertex indices in the input edgelist.
```
kites::graph_csr<vT, eT> hostGrph(
  "Sample-Graph-Wiki-Vote.txt",          // Graph edgelist. This graph is downloaded from SNAP dataset.
  pre_processing_function_sssp<vT, eT>,  // Pre-processing function.
  1,                                     // Vertex per group for the CSC rep.
  kites::input_graph_form::edge_list_s_d // Input graph form (here source to destination).
  );
```

We also create an `nv_gpu` object which specifies a particular GPU with an specific index in the system. This GPU object will be used by the main computation routine to process the graph.
```
kites::nv_gpu mydev( 0 );
```
Here we specified the GPU with index `0`. We can also specify a combination of the available GPUs using `kites::nv_gpu` object like below:
```
kites::nv_gpus mydevs{ 0, 1, 2 };
```

Finally, the processing function is called by specifying the graph object, device object, and the GPU device functions. This essentially means that the specified graph will be processed by the set of GPUs specified as the compute device. Behind the scene, the library transfers the graph data into GPUs' DRAM and organizes the graph processing procedure while utilizing specified device functions iteratively.
```
kites::process< kites::launch::sync >(
          devGrph,
          mydev,
          initF, nbrCompF, redF, updF );
```
