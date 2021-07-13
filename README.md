# coarsening-disassortative-graphs

### Pooling operators available in torch-geometric

#### Sparse
* **TopKPooling** - from the “Graph U-Nets”, “Towards Sparse Hierarchical Graph Classifiers” and “Understanding Attention and Generalization in Graph Neural Networks” papers. Needs supervised loss. Use if setting up task with supervised loss.

* **SAGPooling** - The self-attention pooling operator from the “Self-Attention Graph Pooling” and “Understanding Attention and Generalization in Graph Neural Networks” papers. Needs supervised loss. Use if setting up task with supervised loss


* PANPooling - The path integral based pooling operator from the “Path Integral Based Convolution and Pooling for Graph Neural Networks” paper. Needs supervised loss

* MemPooling - Memory based pooling layer from “Memory-Based Graph Networks” paper, which learns a coarsened graph representation based on soft cluster assignments. Needs supervised loss

* ASAPooling - The Adaptive Structure Aware Pooling operator from the “ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations” paper.

* **Graclus** - A greedy clustering algorithm from the “Weighted Graph Cuts without Eigenvectors: A Multilevel Approach” paper of picking an unmarked vertex and matching it with one of its unmarked neighbors (that maximizes its edge weight). Trainable as a stand-alone.



#### Dense
* **dense_diff_pool** - Differentiable pooling operator from the “Hierarchical Graph Representation Learning with Differentiable Pooling” paper. Trainable as a stand-alone.

* **dense_mincut_pool** - MinCUt pooling operator from the “Mincut Pooling in Graph Neural Networks” paper. Trainable as a stand-alone.