/*
Xiaohang Su
sxhdragon@gmail.com
Layer.cpp
Layer implementation
*/

#include "Layer.h"
#include <Eigen\Dense>
#include <iostream>
using namespace Eigen;



/*
	r == row | c == column | b == batch | range == error output array size
*/
Layer::Layer(int r, int c, int b, int range) {
	if (range == 0) {
		output = new MatrixXd(r, c);
		error = new MatrixXd(r, c);
		batch = b;
		output->setZero();
		error->setZero();
	}
	else {
		output = new MatrixXd[range];
		error = new MatrixXd[range];
		for (int i = 0; i < range; i++) {
			*(output + i) = MatrixXd(r, c).setZero();
			*(error + i) = MatrixXd(r, c).setZero();

		}
		batch = b;
	}

}




void Layer::forwardPropagation(MatrixXd *x, int n) {
	std::cout << "Layer FowardPropagation involve~" << std::endl << std::endl;
}

void Layer::backwardPropagation(MatrixXd *x, MatrixXd *y, int n) {
	std::cout << "Layer BackPropagation involve~" << std::endl << std::endl;
}

void Layer::descentGradient(MatrixXd* x) {
	std::cout << "Layer descentGradient involve~" << std::endl << std::endl;

}


/*
	the derived class should initialize output
	OR ERROR occur!
*/
MatrixXd* Layer::getOutput() {
	return output;
}

/*
	the derived class should initialize error
	OR ERROR occur!
*/
MatrixXd* Layer::getError() {
	return error;
}

void Layer::setBatch(int b) {
	batch = b;
}

/*
	the derived class should initialize batch
	OR ERROR occur!
*/
int Layer::getBatch() {
	return batch;
}