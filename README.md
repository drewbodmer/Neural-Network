# Neural-Network

A first attempt at building a simple neural network to recognize handwritten digits. This project is based off [this tutorial](http://neuralnetworksanddeeplearning.com/) by Michael 
Nielson and [this Youtube series](https://youtu.be/aircAruvnKk) on neural networks.
This project was built as a learning project, so I did my best to not look at any previously-written code for neural networks. 

The trained network currently has an accuracy of around 85%. 

#### Known Issues:
- Currently no mini-batch implementation, so learning is quite slow.
- Learning speed is not proportional to the magnitude of the cost function, so there is likely some "bouncing"
