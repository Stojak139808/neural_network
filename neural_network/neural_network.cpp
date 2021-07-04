#include <iostream>
#include "Matrix.h"

using namespace std;

int main()
{
	int* rozmiar;
	rozmiar = new int;

	cin >> *rozmiar;

	Matrix* A = new Matrix(*rozmiar, *rozmiar);
	Matrix* B = new Matrix(*rozmiar, 1);
	A->fill(1);
	A->print_matrix();
	cout << "\n";
	B->fill(2);
	B->print_matrix();
	cout << "\n";

	A = new Matrix(*A * *B);
	A->print_matrix();

}