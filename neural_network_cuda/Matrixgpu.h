#ifndef MATRIXGPU
#define MATRIXGPU

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#define BLOCK_SIZE 16

__global__ void fillKernel(float* matrix, float x, const int rows, const int columns) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; //column
    const int j = blockIdx.y * blockDim.y + threadIdx.y; //row
    if ((j < rows) && (i < columns)) {
        matrix[j * columns + i] = x;
    }
}

__global__ void transposeKernel(float* A, float* y, int width, int height) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1]; //padding for avoiding bank conflicts?
    int i_column = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int i_row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Load matrix into tile
    for (int i = 0; i < BLOCK_SIZE; i += BLOCK_SIZE) {
        if (i_column < width && (i_row + i) < height) {
            tile[threadIdx.y + i][threadIdx.x] = A[(i_row + i) * width + i_column];
        }
    }
    __syncthreads();

    //nedded replacement for non-square matrices
    i_column = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    i_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    //load tiles into output matrix
    for (int i = 0; i < BLOCK_SIZE; i += BLOCK_SIZE) {
        if (i_column < height && (i_row + i) < width) {
            y[(i_row + i) * height + i_column] = tile[threadIdx.x][threadIdx.y + i];

        }
    }
}

__global__ void addKernel(const float* A, const float* B, float* Y, const int rows, const int columns) {

    //Y = A + B
    const int i = blockIdx.x * blockDim.x + threadIdx.x; //column
    const int j = blockIdx.y * blockDim.y + threadIdx.y; //row
    if ((j < rows) && (i < columns)) {
        Y[j * columns + i] = A[j * columns + i] + B[j * columns + i];
    }
}

__global__ void substractKernel(const float* A, const float* B, float* Y, const int rows, const int columns) {

    //Y = A - B
    const int i = blockIdx.x * blockDim.x + threadIdx.x; //column
    const int j = blockIdx.y * blockDim.y + threadIdx.y; //row
    if ((j < rows) && (i < columns)) {
        Y[j * columns + i] = A[j * columns + i] - B[j * columns + i];
    }
}

__global__ void multipyscalarKernel(const float* A, const float b, float* Y, const int rows, const int columns) {

    //Y = A * b (b is a scalar)
    const int i = blockIdx.x * blockDim.x + threadIdx.x; //column
    const int j = blockIdx.y * blockDim.y + threadIdx.y; //row
    if ((j < rows) && (i < columns)) {
        Y[j * columns + i] = A[j * columns + i] * b;
    }
}

__global__ void multiplymatrixKernel(const float* A, const float* B, float* Y, const int rows, const int columns, const int columns_A) {

    //Y = A * B
    const int column = blockIdx.x * blockDim.x + threadIdx.x; //column
    const int row = blockIdx.y * blockDim.y + threadIdx.y; //row
    float Yval = 0;

    if (row >= rows || column >= columns) return;
    for (int e = 0; e < columns_A; ++e) {
        Yval += A[row * columns_A + e] * B[e * columns + column];
    }
    Y[row * columns + column] = Yval;
}

__global__ void hadamardKernel(const float* A, const float* B, float* Y, const int rows, const int columns) {

    //Y = hadamard product of A and B
    const int i = blockIdx.x * blockDim.x + threadIdx.x; //column
    const int j = blockIdx.y * blockDim.y + threadIdx.y; //row
    if ((j < rows) && (i < columns)) {
        Y[j * columns + i] = A[j * columns + i] * B[j * columns + i];
    }
}


class Matrixgpu {

private:
    int rows; //height
    int columns; //width
    int size;   //size of the dev_matrix vector ( rows*columns*sizeof(float) )
    float* dev_matrix; //pointer to allocated memory in gpu
public:

    float** matrix; //values stored on host memory, needs to be loaded from gpu before reading

    Matrixgpu(int number_of_rows, int number_of_columns) {
        //standard constructor, allocates memory on device and host
        rows = number_of_rows;
        columns = number_of_columns;
        size = rows * columns * sizeof(float);
        cudaMalloc(&dev_matrix, size);

        matrix = (float**)malloc(rows * sizeof(float*));
        for (int i = 0; i < rows; ++i) {
            matrix[i] = (float*)calloc(columns, sizeof(float));
        }
    }

    Matrixgpu(const Matrixgpu& x) {
        //copy constructor for deep copy
        rows = x.rows;
        columns = x.columns;
        size = x.size;
        cudaMalloc(&dev_matrix, size);
        cudaMemcpy(dev_matrix, x.dev_matrix, size, cudaMemcpyDeviceToDevice);
        matrix = (float**)malloc(rows * sizeof(float*));
        for (int i = 0; i < rows; ++i) {
            matrix[i] = (float*)malloc(columns * sizeof(float));
            for (int j = 0; j < columns; ++j) {
                matrix[i][j] = x.matrix[i][j];
            }
        }

    }

    void fill(float x) {

        //fills every value in a matrice with x

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 dimGrid;
        dimGrid.x = (this->columns + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (this->rows + dimBlock.y - 1) / dimBlock.y;
        dimGrid.z = 1;

        fillKernel << <dimGrid, dimBlock >> > (dev_matrix, x, rows, columns);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "ERROR fill: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
    }

    void randomize() {
        //fills matrix with random values
        srand(time(NULL));
        float* tmp = (float*)malloc(size);
        for (int i = 0; i < columns * rows; ++i) {
            *(tmp + i) = (float)(rand() % 2000 - 1000) / 100000;
        }
        cudaMemcpy(this->dev_matrix, tmp, size, cudaMemcpyHostToDevice);
        free(tmp);
    }

    int number_of_rows() {
        return rows;
    }

    int number_of_columns() {
        return columns;
    }

    float* get_pointer() {
        return dev_matrix;
    }

    void print_matrix() {
        //print loaded data into the console
        printf("rows: %d   columns: %d\n", rows, columns);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                printf("%f ", this->matrix[i][j]);
            }
            printf("\n");
        }
    }

    Matrixgpu& operator = (const Matrixgpu& x) {
        // this = x

        //deleting the old matrix
        cudaFree(dev_matrix);
        for (int i = 0; i < this->rows; ++i) {
            free(*(this->matrix + i));
        }
        free(matrix);

        this->rows = x.rows;
        this->columns = x.columns;
        size = rows * columns * sizeof(float);

        cudaMalloc(&dev_matrix, size);
        cudaMemcpy(dev_matrix, x.dev_matrix, size, cudaMemcpyDeviceToDevice);

        matrix = (float**)malloc(rows * sizeof(float*));
        for (int i = 0; i < rows; ++i) {
            matrix[i] = (float*)malloc(columns * sizeof(float));
            for (int j = 0; j < columns; ++j) {
                matrix[i][j] = x.matrix[i][j];
            }
        }

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "ERROR assigment: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        return *this;
    }

    Matrixgpu operator + (const Matrixgpu& x) {
        // y = this + x
        if (this->columns == x.columns && this->rows == x.rows) {

            Matrixgpu y(this->rows, this->columns);
            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
            dim3 dimGrid;
            dimGrid.x = (this->columns + dimBlock.x - 1) / dimBlock.x;
            dimGrid.y = (this->rows + dimBlock.y - 1) / dimBlock.y;
            dimGrid.z = 1;
            addKernel << <dimGrid, dimBlock >> > (this->dev_matrix, x.dev_matrix, y.dev_matrix, this->rows, this->columns);

            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                fprintf(stderr, "ERROR addition: %s\n", cudaGetErrorString(error));
                exit(-1);

            }

            return y;
        }
        else {
            printf("dimensions don't match, cannot multiply\n");
            exit(-1);
        }
    }

    Matrixgpu operator - (const Matrixgpu& x) {
        // y = this - x
        if (this->columns == x.columns && this->rows == x.rows) {

            Matrixgpu y(this->rows, this->columns);
            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
            dim3 dimGrid;
            dimGrid.x = (this->columns + dimBlock.x - 1) / dimBlock.x;
            dimGrid.y = (this->rows + dimBlock.y - 1) / dimBlock.y;
            dimGrid.z = 1;
            substractKernel << <dimGrid, dimBlock >> > (this->dev_matrix, x.dev_matrix, y.dev_matrix, this->rows, this->columns);

            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                fprintf(stderr, "ERROR substraction: %s\n", cudaGetErrorString(error));
                exit(-1);

            }

            return y;
        }
        else {
            printf("dimensions don't match, cannot multiply\n");
            exit(-1);
        }
    }

    Matrixgpu operator * (const float& x) {
        // y = this * x (x is a scalar)

        Matrixgpu y(this->rows, this->columns);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid;
        dimGrid.x = (this->columns + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (this->rows + dimBlock.y - 1) / dimBlock.y;
        multipyscalarKernel << <dimGrid, dimBlock >> > (this->dev_matrix, x, y.dev_matrix, this->rows, this->columns);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "ERROR multiply: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        return y;

    }

    Matrixgpu operator * (const Matrixgpu& X) {
        // y = this * X
        if (this->columns == X.rows) {
            Matrixgpu y(this->rows, X.columns);

            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 dimGrid;
            dimGrid.x = (X.columns + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dimGrid.y = (this->rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
            multiplymatrixKernel << <dimGrid, dimBlock >> > (this->dev_matrix, X.dev_matrix, y.dev_matrix, y.rows, y.columns, this->columns);

            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                fprintf(stderr, "ERROR multiply: %s\n", cudaGetErrorString(error));
                exit(-1);
            }

            return y;
        }
        else {
            printf("dimensions don't match, cannot multiply\n");
            exit(-1);
        }
    }

    Matrixgpu operator % (const Matrixgpu& X) {
        // y = Hadamard product of this and X
        if (this->columns == X.columns && this->rows == X.rows) {

            Matrixgpu y(this->rows, this->columns);
            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
            dim3 dimGrid;
            dimGrid.x = (this->columns + dimBlock.x - 1) / dimBlock.x;
            dimGrid.y = (this->rows + dimBlock.y - 1) / dimBlock.y;
            hadamardKernel << <dimGrid, dimBlock >> > (this->dev_matrix, X.dev_matrix, y.dev_matrix, this->rows, this->columns);

            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                fprintf(stderr, "ERROR Hadamard: %s\n", cudaGetErrorString(error));
                exit(-1);
            }

            return y;
        }
        else {
            printf("dimensions don't match, cannot multiply\n");
            exit(-1);
        }
    }

    Matrixgpu transpose() {
        //y = this^T
        Matrixgpu y = Matrixgpu(this->columns, this->rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid;
        dimGrid.x = (this->columns + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (this->rows + dimBlock.y - 1) / dimBlock.y;

        transposeKernel << <dimGrid, dimBlock >> > (this->dev_matrix, y.dev_matrix, this->columns, this->rows);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "ERROR transpose: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        return y;

    }

    void get_values() {
        //load data from device to host

        float* tmp = (float*)malloc(size);
        cudaMemcpy(tmp, dev_matrix, size, cudaMemcpyDeviceToHost);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "ERROR getvalues: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                matrix[i][j] = tmp[i * columns + j];
            }
        }

    }

    ~Matrixgpu() {
        //destructor
        //frees memeory from both host and device
        cudaFree(dev_matrix);
        for (int i = 0; i < rows; ++i) {
            free(matrix[i]);
        }
        free(matrix);
    }

};

#endif