#include <iostream>
#include "Matrix.h"

using namespace std;

int main(){

	int* rozmiar;
	rozmiar = new int;

	cin >> *rozmiar;

	Matrix* A = new Matrix(*rozmiar, 1);
	Matrix* B = new Matrix(*rozmiar, 1);
	A->fill(1);
	A->print_matrix();
	cout << "\n";
	B->fill(2);
	B->print_matrix();
	cout << "\n";

	A = new Matrix(*A * (B->transpose()));
	A->print_matrix();

}