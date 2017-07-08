#ifndef GRAPH_BASE_H_
#define GRAPH_BASE_H_

#include <cstddef>

namespace kites
{

class graph_base
{
  using size_T = std::size_t;

protected:
  size_T numVertices;
  size_T numEdges;


public:
  graph_base( size_T nV = 0, size_T nE = 0 ):
    numVertices( nV ), numEdges( nE )
  {}

  void set_num_vertices( size_T numVertices_ ) {
    numVertices = numVertices_;
  }
  void set_num_edges( size_T numEdges_ ) {
    numEdges = numEdges_;
  }

  auto get_num_vertices() -> decltype( numVertices )
  {
    return numVertices;
  }

  auto get_num_edges() -> decltype( numEdges )
  {
    return numEdges;
  }

};

}	// end namespace kites

#endif /* GRAPH_BASE_H_ */
