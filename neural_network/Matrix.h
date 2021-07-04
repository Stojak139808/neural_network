#include <iostream>

using namespace std;

class Matrix {

public:
	int** matrix;
	int* rows;
	int* columns;

	Matrix(int number_of_rows, int number_of_columns) {
		rows = new int;
		columns = new int;
		*rows = number_of_rows;
		*columns = number_of_columns;
		matrix = new int* [*rows];
		for (int i = 0; i < *rows; ++i) {
			*(matrix + i) = new int[*columns];
			for (int j = 0; j < *columns; ++j) {
				*(*(matrix + i) + j) = 0;
			}
		}

	}

	void print_matrix() {

		cout << "rows: " << *rows << " columns: " << *columns << "\n";
		for (int i = 0; i < *rows; ++i) {
			for (int j = 0; j < *columns; ++j) {
				cout << *(*(matrix + i) + j) << " ";
			}
			cout << "\n";
		}
	}

	void fill(int x) {
		for (int j = 0; j < *columns; ++j) {
			for (int i = 0; i < *rows; ++i) {
				*(*(matrix + i) + j) = x;
			}
		}
	}

	Matrix operator + (const Matrix& x) {
		// y = this + x
		if (*(x.rows) == *(this->rows) and *(x.columns) == *(this->columns)) {
			Matrix y(*(this->rows), *(this->columns));
			for (int i = 0; i < *rows; ++i) {
				for (int j = 0; j < *columns; ++j) {
					y.matrix[i][j] = x.matrix[i][j] + this->matrix[i][j];
				}
			}

			return y;
		}
		else {
			cout << "dimensions don't match, cannot add\n";
		}
	}

	Matrix operator - (const Matrix& x) {
		// y = this - x
		if (*(x.rows) == *(this->rows) and *(x.columns) == *(this->columns)) {
			Matrix y(*(this->rows), *(this->columns));
			for (int i = 0; i < *rows; ++i) {
				for (int j = 0; j < *columns; ++j) {
					y.matrix[i][j] = this->matrix[i][j] - x.matrix[i][j];
				}
			}

			return y;
		}
		else {
			cout << "dimensions don't match, cannot subtract\n";
		}
	}

	Matrix operator * (const Matrix& x) {
		// y = this * x
		if (*(this->columns) == *(x.rows)) {
			Matrix y(*(this->rows), *(x.columns));
			for (int i = 0; i < *(y.rows); ++i) {
				for (int j = 0; j < *(y.columns); ++j) {
					for (int n = 0; n < *(this->columns); ++n) {
						y.matrix[i][j] += this->matrix[i][n] * x.matrix[n][j];
					}
				}
			}
			return y;
		}
		else {
			cout << "dimensions don't match, cannot multiply\n";
		}
	}

	Matrix transpose() {
		//y = this^T

		Matrix y(*(this->columns), *(this->rows));
		for (int i = 0; i < *(this->rows); ++i) {
			for (int j = 0; j < *(this->columns); ++j) {
				y.matrix[j][i] = this->matrix[i][j];
			}
		}

		return y;

	}

};