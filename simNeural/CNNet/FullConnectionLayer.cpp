/*
Xiaohang Su
sxhdragon@gmail.com
FullConnectionLayer.cpp
FullConnectionLayer implementation
*/

#include "FullConnectionLayer.h"
#include <iostream>
#include <Windows.h>

using namespace Eigen;

FullConnectionLayer::FullConnectionLayer(int in, int out,
	double lr, int b) : Layer(out, b, b) {

	LearnRate = lr;

	column = in + 1; // + 1 for bias
	row = out;

	theta = MatrixXd::Random(out, in + 1) / 10;
	

}


void FullConnectionLayer::forwardPropagation(MatrixXd* in, int null) {
	MatrixXd *output = getOutput();
	MatrixXd input = MatrixXd(column, getBatch());
	input << *in, input.row(column - 1).setOnes();

	*output = theta * input;
	//std::cout << *output << std::endl << std::endl;
	//std::cout << theta.cols() << " " << theta.rows() << std::endl << std::endl;
	//system("pause");
	sigmoid(*output);

}

void FullConnectionLayer::backwardPropagation(MatrixXd* preError,
	MatrixXd* lastTheta, int) {
	MatrixXd *error = getError();
	MatrixXd *output = getOutput();

	MatrixXd sigmoidReverseValue = sigmoidReverse(output);

	*error = (lastTheta->leftCols(lastTheta->cols() - 1).transpose()) * (*preError);
	*error = ((error->array()) *(sigmoidReverseValue.array())).matrix();


}

void FullConnectionLayer::descentGradient(MatrixXd* in){
	MatrixXd *error = getError();

	MatrixXd input = MatrixXd(column, getBatch());
	input << *in, input.row(column - 1).setOnes();
	theta -= (LearnRate * (*error * input.transpose()) / getBatch());
	error->setZero();
	
}

void FullConnectionLayer::backwardPropagationForOutputLayer(MatrixXd& standOutput) {
	MatrixXd *error = getError();
	MatrixXd *output = getOutput();
	*error = *output - standOutput;

}


MatrixXd& FullConnectionLayer::getTheta() {
	return theta;
}


int FullConnectionLayer::getRow() {
	return row;
}

int FullConnectionLayer::getColumn() {
	return column;
}

void FullConnectionLayer::sigmoid(MatrixXd &in) {
	in = (1 / (1 + (-1 * in.array()).exp())).matrix();
}

MatrixXd FullConnectionLayer::sigmoidReverse(MatrixXd* in) {
	return (in->array() * (1 - in->array())).matrix();
}