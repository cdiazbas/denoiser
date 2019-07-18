# Solar image denoising with convolutional neural networks

We have designed a neural network that is capable of recovering weak signals under a complex noise corruption using deep-learning techniques. It is described in the paper, that you can find in [https://arxiv.org/abs/XXXX](https://arxiv.org/abs/XXXXX).

![example](docs/prediction_sst.png?raw=true "")
**Figure 1** — Example of the neural applied to real data.


## Abstract

The topology and dynamics of the solar chromosphere are dominated by magnetic fields. The latter can be inferred by analyzing polarimetric observations of spectral lines. However, polarimetric signals induced by chromospheric magnetic fields are particularly weak, and in most cases very close to the detection limit of current instrumentation. Therefore there are few observational studies that have successfully reconstructed the three components of the magnetic field vector in the chromosphere. Traditionally, the signal-to-noise ratio of observations has been improved by performing time-averages or spatial averages, but in both cases some information is lost. More advanced techniques, like Principal-component-analysis, have also been employed to take advantage of the sparsity of the observations in the spectral direction. In the present study, we propose to use the spatial coherence of the observations to reduce the noise using deep-learning techniques. We have designed a neural network that is capable of recovering weak signals under a complex noise corruption (including instrumental artifacts and non-linear post-processing). The training of the network is carried out without a priori knowledge of the clean signals, or an explicit statistical characterization of the noise or other corruption, only using the same observations as our generative model. Synthetic experiments are used to demonstrate the performance of this method. After that, we show examples of the improvement in typical signals obtained in current telescopes such as the Swedish 1-meter Solar Telescope.
