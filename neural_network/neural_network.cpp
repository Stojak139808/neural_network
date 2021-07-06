#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stack>
#include <algorithm>
#include <random>

using namespace std;

struct image {

    vector<float> image_data;
    int label;

};

int ReverseInt(int i) {

    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMNIST(vector<image>& training_data, vector<image>& testing_data) {

    ifstream test_images_read("t10k-images.idx3-ubyte", ios::binary);

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    if (test_images_read.is_open()) {

        test_images_read.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        test_images_read.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        test_images_read.read((char*)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        test_images_read.read((char*)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        testing_data.resize(number_of_images);

        cout << magic_number << " " << number_of_images << " " << n_rows << " " << n_cols << "\n";

        for (int i = 0; i < number_of_images; ++i) {

            testing_data[i].image_data.resize(n_cols * n_cols);

            for (int r = 0; r < n_rows; ++r) {

                for (int c = 0; c < n_cols; ++c) {

                    unsigned char temp = 0;
                    test_images_read.read((char*)&temp, sizeof(temp));
                    testing_data[i].image_data[(n_rows * r) + c] = (int)temp;

                }
            }
        }

        test_images_read.close();
    }
    else {
        cout << "cannot open test image file\n";
    }

    ifstream train_images_read("train-images.idx3-ubyte", ios::binary);

    if (train_images_read.is_open()) {

        train_images_read.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        train_images_read.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        train_images_read.read((char*)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        train_images_read.read((char*)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        training_data.resize(number_of_images);

        cout << magic_number << " " << number_of_images << " " << n_rows << " " << n_cols << "\n";

        for (int i = 0; i < number_of_images; ++i) {

            training_data[i].image_data.resize(n_cols * n_cols);

            for (int r = 0; r < n_rows; ++r) {

                for (int c = 0; c < n_cols; ++c) {

                    unsigned char temp = 0;
                    train_images_read.read((char*)&temp, sizeof(temp));
                    training_data[i].image_data[(n_rows * r) + c] = (int)temp;

                }
            }
        }

        test_images_read.close();

    }
    else {
        cout << "cannot open train image file\n";
    }

    ifstream train_images_labels_read("train-labels.idx1-ubyte", ios::binary);

    if (train_images_labels_read.is_open()) {

        train_images_labels_read.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        train_images_labels_read.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        cout << magic_number << " " << number_of_images << "\n";

        for (int i = 0; i < number_of_images; ++i) {

            unsigned char temp = 0;
            train_images_labels_read.read((char*)&temp, sizeof(temp));
            training_data[i].label = (int)temp;

        }

        train_images_labels_read.close();

    }
    else {
        cout << "cannot open train image labels file\n";
    }

    ifstream test_images_labels_read("t10k-labels.idx1-ubyte", ios::binary);

    if (test_images_labels_read.is_open()) {

        test_images_labels_read.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        test_images_labels_read.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        cout << magic_number << " " << number_of_images << "\n";

        for (int i = 0; i < number_of_images; ++i) {

            unsigned char temp = 0;
            test_images_labels_read.read((char*)&temp, sizeof(temp));
            testing_data[i].label = (int)temp;

        }

        test_images_labels_read.close();

    }
    else {
        cout << "cannot open test image labels file\n";
    }

}

void print_image(vector<image>& images, int n) {

    for (int i = 0; i < 784; ++i) {
        if (i % 28 == 0) {
            cout << "\n";
        }
        cout.width(4);
        cout.fill(' ');
        cout << images[n].image_data[i];
    }

    cout << "\nthe number is: " << images[n].label << "\n";

}

class Network {

public:
    int number_of_layers;
    vector<int> sizes;
    vector<vector<float>> biases; //vector of vectors with weights for biases
    vector<vector<vector<float>>> weights; //vector of matixes with weights
    vector<vector<float>> net_values; //values in every neuron for the network
    vector<vector<float>> net_values_z; //values of inputs for each neuron in the network

    Network(vector<int>& x) {

        srand(time(NULL));
        number_of_layers = x.size();
        sizes = x;

        //expanding net_values vector to the appropriate size
        net_values.resize(number_of_layers);
        net_values_z.resize(number_of_layers);
        for (int i = 0; i < number_of_layers; ++i) {
            net_values[i].resize(sizes[i]);
            net_values_z[i].resize(sizes[i]);
        }

        //expanding vecotr with weight matixes to the appropriate size
        weights.resize(sizes.size() - 1);
        for (int i = 0; i < number_of_layers - 1; ++i) {
            weights[i].resize(sizes[i]);
            for (int j = 0; j < sizes[i]; ++j) {
                weights[i][j].resize(sizes[i + 1]);
            }
        }

        //loading random values for weight matrixes
        for (int i = 0; i < number_of_layers - 1; ++i) {
            for (int j = 0; j < sizes[i]; ++j) {
                for (int k = 0; k < sizes[i + 1]; ++k) {
                    weights[i][j][k] = (float)(rand() % 2000 - 1000) / 10000000;
                }
            }
        }

        //expanding bias vectors to the appropriate size
        biases.resize(number_of_layers - 1);
        for (int i = 0; i < number_of_layers - 1; ++i) {
            biases[i].resize(sizes[i + 1]);
        }

        //loading bias vector with random values
        for (int i = 0; i < number_of_layers - 1; ++i) {
            for (int j = 0; j < sizes[i + 1]; ++j) {
                biases[i][j] = (float)(rand() % 2000 - 1000) / 10000000;
            }
        }

    }

private:

    float sigmoid(float x) {
        return (1 / (1 + exp(-x)));
    }

    float sigmoid_prime(float x) {
        x = sigmoid(x);
        return x * (1 - x);
    }

    void output_form_label(vector<float>& y, int n) {
        y.resize(10);
        y[n] = 1;
    }

    float cost_derivative(float output_activations, float y) {
        // returns the derivative of cost function
        return output_activations - y;
    }

public:
    void feed_foward(vector<float>& input, bool calculate_z = false) { //calculating values for each neuron based on the provided input

        float tmp_z = 0;

        if (input.size() == sizes[0]) {

            if (calculate_z) {

                net_values[0] = input; //loading input vector to input neurons

                for (int i = 0; i < number_of_layers - 1; ++i) { //calculating values for each layer: w' = sigmoid(w*input + bias)
                    for (int k = 0; k < sizes[i + 1]; ++k) {
                        for (int j = 0; j < sizes[i]; ++j) {
                            tmp_z += (net_values[i][j] * weights[i][j][k]);

                        }

                        tmp_z += biases[i][k];
                        net_values_z[i + 1][k] = tmp_z;
                        net_values[i + 1][k] = sigmoid(tmp_z);
                        tmp_z = 0;

                    }
                }
            }
            else {

                net_values[0] = input; //loading input vector to input neurons

                for (int i = 0; i < number_of_layers - 1; ++i) { //calculating values for each layer: a' = sigmoid(w*a + bias)
                    for (int k = 0; k < sizes[i + 1]; ++k) {
                        for (int j = 0; j < sizes[i]; ++j) {
                            tmp_z = tmp_z + (net_values[i][j] * weights[i][j][k]);
                        }

                        tmp_z += biases[i][k];
                        net_values[i + 1][k] = sigmoid(tmp_z);
                        tmp_z = 0;

                    }
                }
            }
        }
        else {
            cout << "\nwrong input layer!\n";
        }

    }

private:

    void back_propagation(vector<float>& x, vector<float>& y, vector<vector<float>>& b, vector<vector<vector<float>>>& w) {

        //1. input and 2. feedfoward
        feed_foward(x, true);

        //3. output error
        vector<float> delta;

        for (int i = 0; i < sizes.back(); ++i) {
            delta.push_back(cost_derivative(net_values[number_of_layers - 1][i], y[i]) * (sigmoid_prime(net_values_z[number_of_layers - 1][i])));
            b[number_of_layers - 2][i] += delta[i];
        }

        //b[number_of_layers - 2] += delta;

        for (int j = 0; j < sizes[number_of_layers - 1]; ++j) {
            for (int k = 0; k < sizes[number_of_layers - 2]; ++k) {
                w[w.size() - 1][k][j] += delta[j] * net_values[number_of_layers - 2][k];

            }
        }

        //4. backpropagate the error
        for (int l = 2; l < number_of_layers; ++l) {

            vector<float> tmp;
            tmp.resize(sizes[number_of_layers - l]);

            //w(l+1)^T * delta(l+1)
            for (int j = 0; j < sizes[number_of_layers - l]; ++j) {
                for (int k = 0; k < sizes[number_of_layers - l + 1]; ++k) {

                    tmp[j] += weights[number_of_layers - l][j][k] * delta[k];
                }
            }
            //(w(l+1)^T * delta(l+1)) dot deltasigma(z(l))
            delta.resize(sizes[number_of_layers - l]);
            for (int j = 0; j < sizes[number_of_layers - l]; ++j) {
                delta[j] = tmp[j] * sigmoid_prime(net_values_z[number_of_layers - l][j]);
                b[number_of_layers - l - 1][j] += delta[j];

            }


            //b[number_of_layers - l -1] = delta;

            for (int j = 0; j < sizes[number_of_layers - l]; ++j) {
                for (int k = 0; k < sizes[number_of_layers - l - 1]; ++k) {
                    w[w.size() - l][k][j] += delta[j] * net_values[number_of_layers - l - 1][k];
                }
            }

        }

        //b = delta_nabla_biases;
        //w = delta_nabla_weights;

    }

    void update_mini_batch(vector<image>& batch, float eta, vector<vector<float>>& nabla_biases, vector<vector<vector<float>>>& nabla_weights) {



        //vector<vector<float>> delta_nabla_biases;
        //vector<vector<vector<float>>> delta_nabla_weights;




        for (int i = 0; i < batch.size(); ++i) {

            vector<float> y;
            output_form_label(y, batch[i].label);


            back_propagation(batch[i].image_data, y, nabla_biases, nabla_weights);


        }

        float tmp_eta = eta / (float)batch.size();
        //loading newly calculated weights and biases from the batch
        for (int n = 0; n < nabla_weights.size(); ++n) {
            for (int j = 0; j < nabla_weights[n].size(); ++j) {
                for (int k = 0; k < nabla_weights[n][j].size(); ++k) {
                    weights[n][j][k] -= (tmp_eta) * (nabla_weights[n][j][k]);
                    nabla_weights[n][j][k] = 0;
                }
                for (int k = 0; k < nabla_biases[n].size(); ++k) {
                    biases[n][k] -= (tmp_eta) * (nabla_biases[n][k]);
                    nabla_biases[n][k] = 0;
                }
            }
        }

    }

    float evaluate(vector<image>& testing_data) {

        float accuracy = 0;
        float tmp_max;
        int tmp_max_id = 0;

        for (int i = 0; i < testing_data.size(); ++i) {

            //feeding data
            feed_foward(testing_data[i].image_data);
            //getting the output
            tmp_max = 0;
            for (int k = 0; k < net_values[number_of_layers - 1].size(); ++k) {
                if (net_values[number_of_layers - 1][k] > tmp_max) {
                    tmp_max = net_values[number_of_layers - 1][k];
                    tmp_max_id = k;
                }

            }

            if (tmp_max_id == testing_data[i].label) {
                accuracy += 1;
            }

        }

        return 100 * accuracy / testing_data.size();

    }

public:

    void SGD(vector<image>& training_data, int epochs, int train_batch_size, float eta, vector<image>& testing_data) {

        cout << evaluate(testing_data) << "\n";
        cout << "\nStart\n";

        vector<vector<float>> nabla_biases; //vectors for backprop
        vector<vector<vector<float>>> nabla_weights;

        //expanding new vecotrs to the appropriate size
        nabla_weights.resize(sizes.size() - 1);
        nabla_biases.resize(number_of_layers - 1);
        for (int i = 0; i < number_of_layers - 1; ++i) {
            nabla_weights[i].resize(sizes[i]);
            for (int j = 0; j < sizes[i]; ++j) {
                nabla_weights[i][j].resize(sizes[i + 1]);
            }
            nabla_biases[i].resize(sizes[i + 1]);
        }


        //start of each epoch
        for (int j = 0; j < epochs; ++j) {

            //shuffling data
            auto rng = default_random_engine{};
            shuffle(training_data.begin(), training_data.end(), rng);

            //creating small batches of data
            vector<vector<image>> mini_batches;
            vector<image> tmp;

            for (int i = 0; i < (training_data.size() / train_batch_size); ++i) {

                tmp.resize(train_batch_size);

                for (int k = 0; k < train_batch_size; ++k) {
                    tmp[k] = training_data[i * train_batch_size + k];
                }

                mini_batches.push_back(tmp);
                tmp.clear();

            }

            for (int i = 0; i < (training_data.size() / train_batch_size); ++i) {

                update_mini_batch(mini_batches[i], eta, nabla_biases, nabla_weights);
                
                /*
                //progress bar
                if (i % train_batch_size == 0) {

                    cout << (float)i / (training_data.size() / train_batch_size) << "\n";


                }
                */

            }

            cout << "epoch " << j + 1 << " is done\n";

            cout << "the accuracy is: " << evaluate(testing_data) << "%\n";


        }

    }

};

int main() {

    vector<image> training_data;
    vector<image> testing_data;

    ReadMNIST(training_data, testing_data);

    for (int i = 0; i < training_data.size(); ++i) {
        for (int j = 0; j < training_data[i].image_data.size(); ++j) {
            training_data[i].image_data[j] = training_data[i].image_data[j] / 255;
        }
    }

    for (int i = 0; i < testing_data.size(); ++i) {
        for (int j = 0; j < testing_data[i].image_data.size(); ++j) {
            testing_data[i].image_data[j] = testing_data[i].image_data[j] / 255;
        }
    }

    //print_image(training_data, 69);

    vector<int> z;
    z.push_back(784);
    z.push_back(30);
    z.push_back(10);

    Network ass(z);

    ass.SGD(training_data, 50, 10, 0.1, testing_data);

}