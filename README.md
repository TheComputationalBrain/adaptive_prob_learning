# adaptive_prob_learning

This repository provide all code used to generate plots and results of the paper "Sophistication of Human Adaptive Probability Learning‬
‭ in Dynamic Environments".

The data folder contains the three datasets we used in the paper.
- Foucault, C., & Meyniel, F. (2024). Two Determinants of Dynamic Adaptive Learning for Magnitudes and Probabilities. OPEN MIND, 8, 615–638. https://doi.org/10.1162/opmi_a_00139
- Gallistel, C. R., Krishan, M., Liu, Y., Miller, R., & Latham, P. E. (2014). The Perception of Probability. Psychological Review, 121, 99–123.
- Khaw, M. W., Stevens, L., & Woodford, M. (2017). Discrete adjustment to a changing environment: Experimental evidence. Journal of Monetary Economics, 91, 88–103. https://doi.org/10.1016/j.jmoneco.2017.09.001

The models folder contains code to implement models described in the manuscript.
- HMM (Meyniel et al, 2015;2020)
- Change-point model (Gallistel et al.,2014)
- Delta rule (Rescorla & Wagner 1972)
- Pearce-Hall model (Pearce & Hall, 1980)
- Reduced Bayesian model (Nasser et al.,2010)
- Mixture of delta rules (Wilson et al.,2013)
- Volatile Kalman filter (Piray et al., 2020,2021)
- Hierarchical Gaussian Filters (Mathys et al., 2011, 2014)
- Proportional–integral–derivative controller (Ritz et al., 2018)

The figures folder contains figures in the paper and all the figures can be genenerate using the *plots.py* script. 

The *analysis.py* script contains code we used to generate the results, including optimisation of parameters, cross-validation, models/parameters recovery. Running this script will take times, therefore the results is already included in the results folder. 
