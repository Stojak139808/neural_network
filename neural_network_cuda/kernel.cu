#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "Matrixgpu.h"
#include "MNIST_loader.h"

#include <random>
#include <chrono>
#include <iostream> //replace with printf kekw

__global__ void activate_layer_Kernel(float* x, float* y) {
	//get nx1 matrix, activate for each row
	const int j = blockIdx.y * blockDim.y + threadIdx.y; //row

	float ex = expf(x[j]);

	y[j] = ex / (ex + 1);

}

__global__ void activate_layer_prime_Kernel(float* x, float* y) {
	//get nx1 matrix, activate for each row
	const int j = blockIdx.y * blockDim.y + threadIdx.y; //row

	float ex = expf(x[j]);
	ex = ex / (ex + 1);

	y[j] = ex * (1 - ex);
}

class Networkgpu {

public:
	//network values
	int* sizes;				//list with sizes of layers
	int number_of_layers;	//number of neuron layers in the network
	Matrixgpu** weights;		//list with weight Matrix for each layer
	Matrixgpu** biases;		//list of bias vector for each layer
	Matrixgpu** activations;	//list of vectors with neuron outputs for each layer
	Matrixgpu** z_values;		//list of vectors with neuron inputs for each layer

	//training values
	int number_of_training_sets;
	float** training_image_data;		//list of vectors with image pixel data, but normalized to the (0, 1) range
	float** training_image_labels;		//list of vecotrs with expected net outputs, based on labels

	//testing data
	int number_of_testing_sets;
	float** testing_image_data;		//list of vectors with image pixel data, but normalized to the (0, 1) range
	float** testing_image_labels;		//list of vecotrs with expected net outputs, based on labels

	Networkgpu(int* structure, int n_layers) {

		//loading values and reserving space for each list

		//cloning *structure (couldn't find a better way)
		sizes = new int[n_layers];
		for (int i = 0; i < n_layers; ++i) {
			*(sizes + i) = *(structure + i);
		}

		number_of_layers = n_layers;
		weights = new Matrixgpu * [number_of_layers - 1];
		biases = new Matrixgpu * [number_of_layers - 1];
		activations = new Matrixgpu * [number_of_layers];
		z_values = new Matrixgpu * [number_of_layers];

		for (int i = 0; i < number_of_layers - 1; ++i) {
			weights[i] = new Matrixgpu(sizes[i + 1], sizes[i]);
			biases[i] = new Matrixgpu(sizes[i + 1], 1);

			//initializing random weight and biases
			weights[i]->randomize();
			biases[i]->randomize();
		}

		for (int i = 0; i < number_of_layers; ++i) {
			activations[i] = new Matrixgpu(sizes[i], 1);
			z_values[i] = new Matrixgpu(sizes[i], 1);
		}

	}
private:

	//functions for feeding forward

	Matrixgpu activate_layer(Matrixgpu x) {
		//x has to be a vector (n x 1), where (n x 1) is (rows x columns)
		//return a new (n x 1) matrix, where the activation functions has beed applied element-wise


		//////////////////////////////////////////////////////////
		////////MAKE THEM ACTIVATE AT ONCE WITH ONE KERNEL///////////////////////DONE AND TESTED
		//////////////////////////////////////////////////////////
		Matrixgpu y(x.number_of_rows(), x.number_of_columns());

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid;
		dimGrid.x = (x.number_of_columns() + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (x.number_of_rows() + dimBlock.y - 1) / dimBlock.y;
		activate_layer_Kernel << <dimGrid, dimBlock >> > (x.get_pointer(), y.get_pointer());

		return y;

	}

public:
	void feed_forard(Matrixgpu x) {
		//calculating input and output for each neuron, based on porvided "x" input layer

		*activations[0] = x;

		for (int i = 0; i < number_of_layers - 1; ++i) {
			*z_values[i + 1] = *weights[i] * *activations[i] + *biases[i];
			*activations[i + 1] = activate_layer(*z_values[i + 1]);
			//printf("layer: %d\n", i + 1);
			//activations[i + 1]->get_values();
			//activations[i + 1]->print_matrix();//////////////////////////////////////////////////////////////////////////////////////////////////////

		}

	}

	void load_training_sets(int** n_training_image_data, int** n_training_image_labels, int training_set_size, int** n_testing_image_data, int** n_testing_image_labels, int testing_set_size) {
		//cloning data sets, so they are independent from the MNIST object
		number_of_training_sets = training_set_size;
		number_of_testing_sets = testing_set_size;

		training_image_data = new float* [training_set_size];
		training_image_labels = new float* [training_set_size];
		testing_image_data = new float* [testing_set_size];
		testing_image_labels = new float* [testing_set_size];

		for (int i = 0; i < training_set_size; ++i) {

			training_image_data[i] = new float[sizes[0]];
			for (int n = 0; n < sizes[0]; ++n) {
				training_image_data[i][n] = (float)(n_training_image_data[i][n]) / 255;
			}

			training_image_labels[i] = new float[sizes[number_of_layers - 1]];
			for (int n = 0; n < sizes[number_of_layers - 1]; ++n) {
				training_image_labels[i][n] = (float)(n_training_image_labels[i][n]);
			}

		}

		for (int i = 0; i < testing_set_size; ++i) {

			testing_image_data[i] = new float[sizes[0]];
			for (int n = 0; n < sizes[0]; ++n) {
				testing_image_data[i][n] = (float)(n_testing_image_data[i][n]) / 255;
			}

			testing_image_labels[i] = new float[sizes[number_of_layers - 1]];
			for (int n = 0; n < sizes[number_of_layers - 1]; ++n) {
				testing_image_labels[i][n] = (float)(n_testing_image_labels[i][n]);
			}

		}

	}

private:
	void shuffle_data() {
		//making a list with indexes from 0 to number_of_training_sets
		int* indx = new int[number_of_training_sets];
		for (int i = 0; i < number_of_training_sets; ++i) {
			indx[i] = i;
		}
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		shuffle(indx, indx + number_of_training_sets, std::default_random_engine(seed));

		//new lists that will replace the old one with cloned data
		float** tmp_image_data = new float* [number_of_training_sets];
		float** tmp_image_labels = new float* [number_of_training_sets];

		for (int i = 0; i < number_of_training_sets; ++i) {

			tmp_image_data[i] = new float[sizes[0]];
			for (int j = 0; j < sizes[0]; ++j) {
				tmp_image_data[i][j] = training_image_data[indx[i]][j];
			}
			tmp_image_labels[i] = new float[sizes[number_of_layers - 1]];
			for (int j = 0; j < sizes[number_of_layers - 1]; ++j) {
				tmp_image_labels[i][j] = training_image_labels[indx[i]][j];
			}

		}

		//deleting old lists
		for (int i = 0; i < number_of_training_sets; ++i) {
			delete[] training_image_data[i];
			delete[] training_image_labels[i];
		}
		delete[] training_image_data;
		delete[] training_image_labels;
		delete[] indx;

		//giving pointers to the new shuffeled lists
		training_image_data = tmp_image_data;
		training_image_labels = tmp_image_labels;

	}

	//functions and values for SGD

	Matrixgpu** nabla_weights;
	Matrixgpu** nabla_biases;

	Matrixgpu nabla_cost_function(Matrixgpu y, Matrixgpu desired_y) {
		//return a matrix with derivative of the cost function
		return y - desired_y;
	}

	Matrixgpu activate_layer_prime(Matrixgpu x) {
		//x has to be a vector (n x 1), where (n x 1) is (rows x columns)
		//return a new (n x 1) matrix, where the derivative of activation functions has beed applied element-wise


		//////////////////////////////////////////////////////////
		////////MAKE THEM ACTIVATE AT ONCE WITH ONE KERNEL////////////////DONE AND TESTED
		//////////////////////////////////////////////////////////
		Matrixgpu y(x.number_of_rows(), x.number_of_columns());

		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid;
		dimGrid.x = (x.number_of_columns() + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (x.number_of_rows() + dimBlock.y - 1) / dimBlock.y;
		activate_layer_prime_Kernel << <dimGrid, dimBlock >> > (x.get_pointer(), y.get_pointer());
		return y;
	}

	void backpropagation(Matrixgpu x, Matrixgpu y) {
		//this function will backporpagate and update nabla_weights and nabla_biases 

		feed_forard(x);

		//output error
		Matrixgpu delta = nabla_cost_function(*activations[number_of_layers - 1], y) % activate_layer_prime(*z_values[number_of_layers - 1]);

		*nabla_weights[number_of_layers - 2] = *nabla_weights[number_of_layers - 2] + delta * activations[number_of_layers - 2]->transpose();
		*nabla_biases[number_of_layers - 2] = *nabla_biases[number_of_layers - 2] + delta;

		//backpropagating the error
		for (int l = 2; l < number_of_layers; ++l) {
			delta = (weights[number_of_layers - l]->transpose() * delta) % activate_layer_prime(*z_values[number_of_layers - l]);
			*nabla_weights[number_of_layers - 1 - l] = *nabla_weights[number_of_layers - 1 - l] + delta * activations[number_of_layers - 1 - l]->transpose();
			*nabla_biases[number_of_layers - 1 - l] = *nabla_biases[number_of_layers - 1 - l] + delta;
		}

	}


	void update_mini_batch(int batch_size, int batch_number, float eta) {

		//data sets are stored in a bunch of float** arrays to save space
		//here these arrays are changed into matrix based inpiuts and outputs
		Matrixgpu x(sizes[0], 1);
		Matrixgpu y(sizes[number_of_layers - 1], 1);

		int sizein = sizes[0] * sizeof(float);
		int sizeout = sizes[number_of_layers - 1] * sizeof(float);

		for (int n = 0; n < batch_size; ++n) {

			//data sets are stored in a bunch of float** arrays to save space
			//here these arrays are changed into matrix based inpiuts and outputs
			cudaMemcpy(x.get_pointer(),training_image_data[batch_size * batch_number + n],sizein, cudaMemcpyHostToDevice);
			cudaMemcpy(y.get_pointer(), training_image_labels[batch_size * batch_number + n], sizeout, cudaMemcpyHostToDevice);

			backpropagation(x, y);

		}

		//updating weights and biases
		for (int i = 0; i < number_of_layers - 1; ++i) {
			*weights[i] = *weights[i] - (*nabla_weights[i] * (eta / batch_size));
			*biases[i] = *biases[i] - (*nabla_biases[i] * (eta / batch_size));
		}

	}

	float evaluation() {

		Matrixgpu x(sizes[0], 1);
		int sizein = sizes[0] * sizeof(float);

		int y;
		float error = 0;

		float max = 0;
		int indx_of_max = 0;

		for (int i = 0; i < number_of_testing_sets; ++i) {

			max = 0;

			//////////////////////////////////////////
			////////ONE CUDAMEMCPY SHOULD WORK////////////////////DONE AND TESTED I GUESS
			//////////////////////////////////////////
			/*
			for (int n = 0; n < sizes[0]; ++n) {
				x.matrix[n][0] = testing_image_data[i][n];
			}
			*/
			cudaMemcpy(x.get_pointer(), testing_image_data[i], sizein, cudaMemcpyHostToDevice);
			for (int n = 0; n < sizes[number_of_layers - 1]; ++n) {
				if (testing_image_labels[i][n] != 0) {
					y = n;
				}
			}

			feed_forard(x);

			//////////////////////////////////////////
			////////get values for last layer?////////
			//////////////////////////////////////////
			activations[number_of_layers - 1]->get_values();
			for (int n = 0; n < sizes[number_of_layers - 1]; ++n) {
				if (activations[number_of_layers - 1]->matrix[n][0] > max) {
					max = activations[number_of_layers - 1]->matrix[n][0];
					indx_of_max = n;
				}
			}

			if (indx_of_max == y) {
				error += 1;
			}

		}

		return 100 * (error / number_of_testing_sets);
	}

public:
	void stochastic_gradient_descent(int epochs, int mini_batch_size, float eta) {

		//new temporary matrixes
		nabla_weights = new Matrixgpu * [number_of_layers - 1];
		nabla_biases = new Matrixgpu * [number_of_layers - 1];

		for (int i = 0; i < number_of_layers - 1; ++i) {
			nabla_weights[i] = new Matrixgpu(sizes[i + 1], sizes[i]);
			nabla_biases[i] = new Matrixgpu(sizes[i + 1], 1);
		}

		cout << "Before learning, the accuracy is: " << evaluation() << "%\n";

		for (int epoch = 0; epoch < epochs; ++epoch) {

			cout << "\nStarting epoch " << epoch + 1 << "\n";

			//shuffling data before each epoch
			shuffle_data();


			for (int i = 0; i < number_of_training_sets / mini_batch_size; ++i) {

				//emptying temporaty matrixes before another batch update
				for (int j = 0; j < number_of_layers - 1; ++j) {
					nabla_weights[j]->fill(0);
					nabla_biases[j]->fill(0);
				}

				update_mini_batch(mini_batch_size, i, eta);

			}

			cout << "Epoch " << epoch + 1 << " is done. ";
			cout << "Accuracy is: " << evaluation() << "%\n";

		}

		//deleting temporary matrixes
		for (int i = 0; i < number_of_layers - 1; ++i) {
			delete[] nabla_weights[i];
			delete[] nabla_biases[i];
		}
		delete[] nabla_weights;
		delete[] nabla_biases;

	}

	void print_image(int image_number, bool which_set) {
		//prints out the chosen image in CMD
		//where which_set == 0 is training set, testing set otherwise

		char full = 178;
		char medium = 177;
		char empty = 176;

		int n = (int)sqrt(sizes[0]);
		if (image_number < number_of_training_sets && which_set == false) {

			cout << "The number is: ";
			for (int i = 0; i < 10; ++i) {
				if (training_image_labels[image_number][i] == 1) {
					cout << i << "\n";
				}
			}

			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < n; ++j) {

					if (training_image_data[image_number][i * n + j] >= 0.5) {
						cout << full;
					}
					else if (training_image_data[image_number][i * n + j] != 0) {
						cout << medium;
					}
					else {
						cout << empty;
					}

				}
				cout << "\n";
			}
		}
		else if (image_number < number_of_testing_sets) {
			if (image_number < number_of_testing_sets && which_set == true) {

				cout << "The number is: ";
				for (int i = 0; i < 10; ++i) {
					if (testing_image_labels[image_number][i] == 1) {
						cout << i << "\n";
					}
				}

				for (int i = 0; i < n; ++i) {
					for (int j = 0; j < n; ++j) {

						if (testing_image_data[image_number][i * n + j] >= 0.5) {
							cout << full;
						}
						else if (testing_image_data[image_number][i * n + j] != 0) {
							cout << medium;
						}
						else {
							cout << empty;
						}

					}
					cout << "\n";
				}
			}
		}
	}

	//destructor
	~Networkgpu() {

		for (int i = 0; i < number_of_layers - 1; ++i) {
			delete weights[i];
			delete biases[i];
		}
		for (int i = 0; i < number_of_layers; ++i) {
			delete activations[i];
			delete z_values[i];
		}
		delete[] weights;
		delete[] biases;
		delete[] activations;
		delete[] z_values;

		for (int i = 0; i < number_of_training_sets; ++i) {
			delete[] training_image_data[i];
			delete[] training_image_labels[i];
		}
		delete[] training_image_data;
		delete[] training_image_labels;

		for (int i = 0; i < number_of_testing_sets; ++i) {
			delete[] testing_image_data[i];
			delete[] testing_image_labels[i];
		}
		delete[] testing_image_data;
		delete[] testing_image_labels;

	}

};

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
	Networkgpu* net = new Networkgpu(structure, number_of_layers);
	delete[] structure;

	//loading data from MNIST object to network
	net->load_training_sets(data->training_image_data,			//training inputs
		data->training_image_labels,		//training outputs
		data->number_of_training_sets,		//number of training sets
		data->testing_image_data,			//testing inputs
		data->testing_image_labels,			//testing outputs
		data->number_of_testing_sets		//number of testing sets
	);

	delete data;

	net->stochastic_gradient_descent(30, 10, 0.3);

	delete net;

	return 0;
}