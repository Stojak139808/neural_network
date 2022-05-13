#ifndef MNIST
#define MNIST


#include <iostream>
#include <fstream>

using namespace std;

class MNIST_database {

public:

    int** training_image_data;
    int** training_image_labels;
    int** testing_image_data;
    int** testing_image_labels;

    int number_of_training_sets = 0;
    int number_of_testing_sets = 0;
    int image_size = 0;

private:
    int ReverseInt(int i) {

        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
    }
public:
    void ReadMNIST() {

        ifstream test_images_read("t10k-images.idx3-ubyte", ios::binary);

        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        cout << "Loading test images.\n";

        if (test_images_read.is_open()) {

            test_images_read.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ReverseInt(magic_number);
            test_images_read.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = ReverseInt(number_of_images);
            test_images_read.read((char*)&n_rows, sizeof(n_rows));
            n_rows = ReverseInt(n_rows);
            test_images_read.read((char*)&n_cols, sizeof(n_cols));
            n_cols = ReverseInt(n_cols);

            number_of_testing_sets = number_of_images;
            image_size = n_cols * n_cols;

            cout << "\nReading: t10k-images.idx3-ubyte\n";
            cout << "Magic number: " << magic_number << " Number of images: " << number_of_images << " Rows: " << n_rows << " Columns: " << n_cols << "\n";


            testing_image_data = new int* [number_of_images];

            for (int i = 0; i < number_of_images; ++i) {

                testing_image_data[i] = new int[image_size];

                for (int r = 0; r < n_rows; ++r) {

                    for (int c = 0; c < n_cols; ++c) {

                        unsigned char temp = 0;
                        test_images_read.read((char*)&temp, sizeof(temp));
                        testing_image_data[i][(n_rows * r) + c] = (int)temp;

                    }
                }
            }

            test_images_read.close();
        }
        else {
            cout << "cannot open test image file\n";
        }

        ifstream test_images_labels_read("t10k-labels.idx1-ubyte", ios::binary);

        if (test_images_labels_read.is_open()) {

            test_images_labels_read.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ReverseInt(magic_number);
            test_images_labels_read.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = ReverseInt(number_of_images);

            cout << "\nReading: t10k-labels.idx1-ubyte\n";
            cout << "Magic number: " << magic_number << " Number of images: " << number_of_images << "\n";

            testing_image_labels = new int* [number_of_images];

            for (int i = 0; i < number_of_images; ++i) {

                unsigned char temp = 0;
                test_images_labels_read.read((char*)&temp, sizeof(temp));

                //chenging labels to expected net output here
                testing_image_labels[i] = new int[10];
                for (int j = 0; j < 10; ++j) {
                    testing_image_labels[i][j] = 0;
                }
                testing_image_labels[i][(int)temp] = 1;

            }

            test_images_labels_read.close();

        }
        else {
            cout << "cannot open test image labels file\n";
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

            number_of_training_sets = number_of_images;

            cout << "\nReading: train-images.idx3-ubyte\n";
            cout << "Magic number: " << magic_number << " Number of images: " << number_of_images << " Rows: " << n_rows << " Columns: " << n_cols << "\n";

            training_image_data = new int* [number_of_images];

            for (int i = 0; i < number_of_images; ++i) {

                training_image_data[i] = new int[image_size];

                for (int r = 0; r < n_rows; ++r) {

                    for (int c = 0; c < n_cols; ++c) {

                        unsigned char temp = 0;
                        train_images_read.read((char*)&temp, sizeof(temp));
                        training_image_data[i][(n_rows * r) + c] = (int)temp;

                    }
                }
            }

            train_images_read.close();

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

            cout << "\nReading: train-labels.idx1-ubyte\n";
            cout << "Magic number: " << magic_number << " Number of images: " << number_of_images << "\n";

            training_image_labels = new int* [number_of_images];

            for (int i = 0; i < number_of_images; ++i) {

                unsigned char temp = 0;
                train_images_labels_read.read((char*)&temp, sizeof(temp));

                //changing labels to expected net output here
                training_image_labels[i] = new int[10];
                for (int j = 0; j < 10; ++j) {
                    training_image_labels[i][j] = 0;
                }
                training_image_labels[i][(int)temp] = 1;

            }

            train_images_labels_read.close();

        }
        else {
            cout << "cannot open train image labels file\n";
        }

    }

    ~MNIST_database(){

        for (int i = 0; i < number_of_training_sets; ++i) {
            delete[] training_image_data[i];
            delete[] training_image_labels[i];
        }
        for (int i = 0; i < number_of_testing_sets; ++i) {
            delete[] testing_image_data[i];
            delete[] testing_image_labels[i];
        }

        delete[] training_image_data;
        delete[] training_image_labels;
        delete[] testing_image_data;
        delete[] testing_image_labels;

    }

};

#endif
