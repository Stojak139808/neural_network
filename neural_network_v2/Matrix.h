#ifndef MATRIX
#define MATRIX

#include <iostream>

#include <cstdlib>
#include <ctime>

using namespace std;

class Matrix {

private:

	int rows;
	int columns;

public:
	float** matrix;
	
	//standard constructor
	Matrix(int number_of_rows, int number_of_columns) {
		rows = number_of_rows;
		columns = number_of_columns;
		matrix = new float* [rows];
		for (int i = 0; i < rows; ++i) {
			*(matrix + i) = new float[columns];
			for (int j = 0; j < columns; ++j) {
				*(*(matrix + i) + j) = 0;
			}
		}
	}

	//copy constructor, standard bit-wise copy wont work correctly when there are pointers in an object
	Matrix(const Matrix& x) {

		rows = x.rows;
		columns = x.columns;

		matrix = new float* [rows];
		for (int i = 0; i < rows; ++i) {
			*(matrix + i) = new float[columns];
			for (int j = 0; j < columns; ++j) {
				*(*(matrix + i) + j) = x.matrix[i][j];
			}
		}

	}

	void print_matrix() {
		cout << "rows: " << rows << " columns: " << columns << "\n";
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < columns; ++j) {
				cout << *(*(matrix + i) + j) << " ";
			}
			cout << "\n";
		}
	}

	void fill(float x) {
		for (int j = 0; j < columns; ++j) {
			for (int i = 0; i < rows; ++i) {
				*(*(matrix + i) + j) = x;
			}
		}
	}

	void randomize() {

		srand(time(NULL));
		for (int j = 0; j < columns; ++j) {
			for (int i = 0; i < rows; ++i) {
				*(*(matrix + i) + j) = (float)(rand() % 2000 - 1000) / 100000;
			}
		}
		
	}

	int number_of_rows() {
		return rows;
	}

	int number_of_columns() {
		return columns;
	}


	//operator overloads

	Matrix& operator = (const Matrix& x) {
		// this = x

		//deleting the old matrix
		for (int i = 0; i < this->rows; ++i) {
			delete[] *(this->matrix + i);
		}
		delete[] matrix;
		
		this->rows = x.rows;
		this->columns = x.columns;

		//creating a new matrix, which is a clone of x
		matrix = new float* [this->rows];
		for (int i = 0; i < this->rows; ++i) {
			*(matrix + i) = new float[this->columns];
			for (int j = 0; j < this->columns; ++j) {
				*(*(this->matrix + i) + j) = x.matrix[i][j];
			}
		}
		return *this;
	}

	Matrix operator + (const Matrix& x) {
		// y = this + x
		if (x.rows == (this->rows) and x.columns == this->columns) {
			Matrix y(this->rows, this->columns);
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < columns; ++j) {
					y.matrix[i][j] = x.matrix[i][j] + this->matrix[i][j];
				}
			}
			return y;
		}
	}

	Matrix operator - (const Matrix& x) {
		// y = this - x
		if (x.rows == this->rows and x.columns == this->columns) {
			Matrix y(this->rows, this->columns);
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < columns; ++j) {
					y.matrix[i][j] = this->matrix[i][j] - x.matrix[i][j];
				}
			}

			return y;
		}
	}

	Matrix operator * (const Matrix& x) {
		// y = this * x
		if (this->columns == x.rows) {
			Matrix y(this->rows, x.columns);
			for (int i = 0; i < y.rows; ++i) {
				for (int j = 0; j < y.columns; ++j) {
					for (int n = 0; n < this->columns; ++n) {
						y.matrix[i][j] += this->matrix[i][n] * x.matrix[n][j];
					}
				}
			}
			return y;
		}
	}

	Matrix operator * (const float& x) {
		// y = this * x (where x is a scalar)
		Matrix y(this->rows, this->columns);
		for (int i = 0; i < y.rows; ++i) {
			for (int j = 0; j < y.columns; ++j) {
				y.matrix[i][j] = this->matrix[i][j] * x;
			}
		}
		return y;
	}

	Matrix transpose() {
		//y = this^T
		Matrix y(this->columns, this->rows);
		for (int i = 0; i < this->rows; ++i) {
			for (int j = 0; j < this->columns; ++j) {
				y.matrix[j][i] = this->matrix[i][j];
			}
		}
		return y;
	}

	Matrix operator % (const Matrix& x) {
		// y = hadamard product of this and x
		// can be called with the "%" symbol
		if(x.rows == (this->rows) and x.columns == this->columns) {
			Matrix y(this->rows, this->columns);
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < columns; ++j) {
					y.matrix[i][j] = x.matrix[i][j] * this->matrix[i][j];
				}
			}
			return y;
		}
	}

	//destructor
	~Matrix() {
		for (int i = 0; i < this->rows; ++i) {
			delete[] * (this->matrix + i);
		}
		delete[] this->matrix;
	}

};

#endif