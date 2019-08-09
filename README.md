# Solar image denoising with convolutional neural networks

We have designed a neural network that is capable of recovering weak signals under a complex noise corruption using deep-learning techniques. It is described in the paper, that you can find in [http://arxiv.org/abs/1908.02815](http://arxiv.org/abs/1908.02815).

![example](docs/prediction_sst.png?raw=true "")
**Figure 1** — Example of the neural network applied to real data.


## Motivation

The topology and dynamics of the solar chromosphere are dominated by magnetic fields. The latter can be inferred by analyzing polarimetric observations of spectral lines. However, polarimetric signals induced by chromospheric magnetic fields are particularly weak, and in most cases very close to the detection limit of current instrumentation. Therefore there are few observational studies that have successfully reconstructed the three components of the magnetic field vector in the chromosphere. Traditionally, the signal-to-noise ratio of observations has been improved by performing time-averages or spatial averages, but in both cases some information is lost. More advanced techniques, like Principal-component-analysis, have also been employed to take advantage of the sparsity of the observations in the spectral direction. In the present study, we propose to use the spatial coherence of the observations to reduce the noise using deep-learning techniques. We have designed a neural network that is capable of recovering weak signals under a complex noise corruption (including instrumental artifacts and non-linear post-processing). The training of the network is carried out without a priori knowledge of the clean signals, or an explicit statistical characterization of the noise or other corruption, only using the same observations as our generative model. Synthetic experiments are used to demonstrate the performance of this method. After that, we show examples of the improvement in typical signals obtained in current telescopes such as the Swedish 1-meter Solar Telescope.

## Using the neural network for prediction

A list of example command lines you can use with the pre-trained models provided in the GitHub releases. The script we provide reads the observations from a numpy file, but you can modify the script to provide the observations from any other file. Then you just type the following to infer the clean image of the example: `python prediction_sst.py -i example_sst.npy -o output/prediction_sst.npy`.

In case you want to check the network trained with synthetic observations you can use the same script but changing the input and the name of the network typing `python prediction_sst.py -i example_synthetic.npy -m weights/simulation_noisy -p _synthetic -o output/prediction_synthetic.npy`.

If you want to use the network for scientific purposes with CRISP data, you can use this Python module to work with CRISP data ([https://github.com/AlexPietrow/CRISpy](https://github.com/AlexPietrow/CRISpy)). You can load the data, predict the output with the network and then create another ".fcube" with the result. Important note: if the cleaning process is not well done, first check the units/range of your data and/or increase the number of cycles that the network is executed in case you have removed important information and the residuals are not flat. We provide an example with the script `profilePredicction.py`.

As this software was developed with some specific libraries you have to install: "keras" and "tensorflow". If `conda` is installed in your machine you can type:  `conda install tensorflow`, `conda install keras`. We only support keras v2.

In case one wants to also predict the uncertainties, you have to install the `astroNN`library. In [their webpage](https://astronn.readthedocs.io) you will 
find how to install it. Also many examples are provided in the [github repository](https://github.com/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis) to implement your own bayesian network. Then, you just type the following to infer the clean image of the example with its uncertainty: `python bayesPrediction.py -i example_synthetic.npy -o output/prediction_bayes_simulation.npy`.


