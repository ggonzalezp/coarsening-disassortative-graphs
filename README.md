# coarsening-disassortative-graphs

### Pooling operators available in torch-geometric

#### Sparse
* TopKPooling - from the “Graph U-Nets”, “Towards Sparse Hierarchical Graph Classifiers” and “Understanding Attention and Generalization in Graph Neural Networks” papers.

* SAGPooling - The self-attention pooling operator from the “Self-Attention Graph Pooling” and “Understanding Attention and Generalization in Graph Neural Networks” papers.

* ASAPooling - The Adaptive Structure Aware Pooling operator from the “ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations” paper.

* PANPooling - The path integral based pooling operator from the “Path Integral Based Convolution and Pooling for Graph Neural Networks” paper.

* MemPooling - Memory based pooling layer from “Memory-Based Graph Networks” paper, which learns a coarsened graph representation based on soft cluster assignments.



#### Dense
* dense_diff_pool - Differentiable pooling operator from the “Hierarchical Graph Representation Learning with Differentiable Pooling” paper

* dense_mincut_pool - MinCUt pooling operator from the “Mincut Pooling in Graph Neural Networks” paper