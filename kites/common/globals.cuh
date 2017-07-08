#ifndef GLOBALS_CUH_
#define GLOBALS_CUH_

namespace kites
{

enum{
  COMPILE_TIME_DETERMINED_BLOCK_SIZE = 128,
  WARP_SIZE = 32,
  WARP_SIZE_SHIFT = 5
};

enum class launch
{
  sync = true,
  async = false
};

enum class operation_mode
{
  iterative = true,
  frontier_based = false
};

enum class input_graph_form
{
  // Edge list with the source index first and then destination index.
  edge_list_s_d = true,
  // Edge list with the destination index first and then source index.
  edge_list_d_s = false
};

enum class graph_property
{
  // It is a directed graph. In other words, every edge has been expressed in the input.
  directed = true,
  // It is a non-directed graph.
  undirected = false
};

enum class edge_list_expression
{
  // It is a completely-expressed undirected graph:
  //   every edge has been expressed in the input.
  complete = true,
  // It is a non-directed graph and two incoming and outgoing edges
  //   have been expressed by one entry in the input file.
  incomplete = false
};

enum class semi_dynamic_load_balancer
{
  ON = true,
  OFF = false
};

}	// end namespace kites

#endif /* GLOBALS_CUH_ */
