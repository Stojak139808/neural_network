# neural_network

This repository has three implemenations of a simple neural network framwork, for teaching them using stochastic gradient descent.

First version uses vectors for most operation, which affected the speed. 
Second implementation (neural_network_v2) uses simple arrays for its operations and also the code is much clearer, becuase 
I first implemented needed matrix operations and then wrote SGD using these operations.
Last version utilizes cuda, for making the learning process faster, this is mostly done by rewriting the Matrix.h header to run on GPU.
