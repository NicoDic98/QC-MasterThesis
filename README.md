The code herein has been organized in a modular fashion. 
Different physical models can be implemented by inheriting from $\texttt{BaseHamiltonian}$. And different algorithms can be implemented by inheriting from $\texttt{BaseSolver}$.

Here already implemented are:
- A Hamiltonian for 2+1 dimensional free Wilson fermions in the class  $\texttt{FreeWilson2D}$.
- The ED algorithm in the class $\texttt{ED}$. This implementation uses the sparse linear algebra submodule of [scipy](https://arxiv.org/abs/1907.10121).
- The VQE algorithm in the class $\texttt{VQE}$. For the gradient-free optimizers we use the implementations provided by [scipy](https://arxiv.org/abs/1907.10121).
- The ADAPT-VQE algorithm in the class $\texttt{AdaptVQE}$.

The implementation further utilizes the following python packages: 
- [Qiskit](https://arxiv.org/abs/2405.08810)
- [scipy](https://arxiv.org/abs/1907.10121)
- [numpy](https://arxiv.org/abs/2006.10256)
- [sympy](https://doi.org/10.7717/peerj-cs.103)
- [h5py](https://www.h5py.org/)
- [matplotlib](https://doi.org/10.1109/MCSE.2007.55)