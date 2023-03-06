# Some Physics-Informed Machine Learning Tests

### Equation solving with DeepXDE

 - Effective one-body equations ([Google Colab](https://colab.research.google.com/drive/1z7ljMLqoWYTORzLUbecFoK1gMNvNQ8Z8?usp=sharing)) -- An orbital mechanics integrator. Includes transfer learning solutions to increase integration time, hard boundary constraints through output transformations, and solutions with parametrized initial conditions as inputs.

 - DGP Gravity ([Google colab](https://colab.research.google.com/drive/1fDCsrSRxDRzX4EJf--pBkWkRP_tIOpaA?usp=sharing) -- Testing a solver for waves in DGP modified gravity.
 
 - Inverse diffusion ([Google colab](https://colab.research.google.com/drive/1WHNbn3X3lxdOWQnlBm4OAwxi8tedjQAu) -- Testing a solver for the inverse diffusion problem (an ill-posed system of equations, so difficult to solve with traditional methods). Includes a basic grid search over some network hyperameters.

### Modulus Tests

Files in this repository mostly contain scratch work for PDE solving using NVIDIA's [Modulus](https://developer.nvidia.com/modulus). These include some assessments of how relevant physics-informed nets might be for studying cosmological systems, including self-gravitating matter and modifications to general relativity.
