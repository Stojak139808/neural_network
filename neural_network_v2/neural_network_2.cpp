#include <iostream>
#include "Matrix.h"
#include "MNIST_loader.h"
#include "Network.h"

using namespace std;

int main() {
	
	//loading in traind and test batch
	MNIST_database* data = new MNIST_database;
	data->ReadMNIST();

	//initilizing the network
	int number_of_layers = 3;
	int* structure = new int[number_of_layers];
	structure[0] = 784;
	structure[1] = 30;
	structure[2] = 10;
	Network* net = new Network(structure, number_of_layers);
	delete[] structure;
	
	//loading data from MNIST object to network
	net->load_training_sets(	data->training_image_data,			//training inputs
								data->training_image_labels,		//training outputs
								data->number_of_training_sets,		//number of training sets
								data->testing_image_data,			//testing inputs
								data->testing_image_labels,			//testing outputs
								data->number_of_testing_sets		//number of testing sets
							);

	delete data;

	net->stochastic_gradient_descent(1, 10, 0.3);

	delete net;

}
