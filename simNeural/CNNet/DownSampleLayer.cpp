/*
Xiaohang Su
sxhdragon@gmail.com
DownSampleLayer.cpp
DownSampleLayer implementation
*/

#include "DownSampleLayer.h"
#include <iostream>
#include <Windows.h>
#include <Eigen\Dense>
using namespace Eigen;


/************************************************
	pooling using means pooling
*************************************************/


DownSampleLayer::DownSampleLayer(int kr, int kc, int inputRow,
	int inputColumn, int inNumber, int b) : Layer(inputRow / kr,
	inputColumn / kc, b, inNumber) {

	if (inputColumn % kc != 0 || inputRow % kr != 0) {
		std::cout <<
		"ERROR: input And kernel DIDNT match£¡~(DownSampleLayer)"
			<< std::endl;
		return;
	}
	
	theta = MatrixXd::Constant(kr, kc, 1.0 / (kr*kc));
	row = kr;
	column = kc;
	MatrixXd *output = getOutput(),
		*error = getError();
	inputNumber = inNumber;
	output = new MatrixXd[inputNumber];
	error = new MatrixXd[inputNumber];
	for (int i = 0; i < inputNumber; i++) {
		output[i] = MatrixXd(inputRow / kr, inputColumn / kc);
		error[i] = MatrixXd(inputRow / kr, inputColumn / kc);

	}
}

void DownSampleLayer::forwardPropagation(MatrixXd* in, int inputNumber) {
	MatrixXd* output = getOutput();

	// exception catch
	if (in->cols() % column != 0 || in->rows() % row != 0) {
		std::cout <<
			"Input cols or rows error! ERROR:NOT FIT POOLING"
			<< std::endl;
		return;
	}
	
	for (int i = 0; i < inputNumber; i++) {
		for (int r = 0; r < output->rows(); r++) {
			for (int c = 0; c < output->cols(); c++) {
				//std::cout << (in[i]).block(r, c, row, column) << std::endl << std::endl;
				//std::cout << theta << std::endl << std::endl;
				 
				(*(output + i))(r,c) = ((in[i]).block(r * row, c * column, row, column) * theta).mean();
				
				//std::cout << *(output + i) << std::endl << std::endl;
				//system("pause");

			}
		}

		sigmoid(output + i);
	}




}

/*
	* preError is error from last layer
	* lastKernel is last convolutionLayer kernel or NULL
	* lastRow is last Layer row measures the preError numbers
	
	(if connect to fullConnectionLayer)
		lastKernel = NULL && lastRow = MatrixXd ' s Number

*/
void DownSampleLayer::backwardPropagation(MatrixXd* preError, MatrixXd *lastKernel, int lastRow)  {
	MatrixXd *error = getError();
	if (lastKernel == NULL) {
		for (int i = 0; i < lastRow; i++) {
			*(error+i) = *(preError+i);
		}
	}
	else {
		for (int r = 0; r < inputNumber; r++) {
			for (int l = 0; l < lastRow; l++) {


				convolutionForBackPropagation(lastKernel[l * row + r], preError[l], error[r]);
				

			}

			error[r] = sigmoidReverse(&(error[r]));
		}
	}
}

void DownSampleLayer::descentGradient(MatrixXd *x) {
	MatrixXd *error = getError();
	for (int i = 0; i < inputNumber; i++) {
		(error[i]).setZero();
	}
	//std::cout << "DownSampleLayer::descentGradient(MatrixXd *x, int n)" << std::endl;
}


MatrixXd& DownSampleLayer::getTheta() {
	return theta;
}

void DownSampleLayer::sigmoid(MatrixXd *in) {
	*in = (1 / (1 + (-1 * in->array()).exp())).matrix();
}

MatrixXd DownSampleLayer::sigmoidReverse(MatrixXd* in) {
	return (in->array() * (1 - in->array())).matrix();
}


/*
	in1 size < in2 size
*/
void DownSampleLayer::convolutionForBackPropagation(MatrixXd& in1, MatrixXd & in2, MatrixXd &result) {
	int resultRow = result.rows(),
		resultCol = result.cols();

	// Matrix in2 push inside to the result Matrix (Zero padding)
	
	MatrixXd paddingMatrix = MatrixXd(
		in2.rows() + 2 * (in1.rows() - 1),
		in2.cols() + 2 * (in1.cols() - 1)).setZero();
	paddingMatrix.block(in1.rows() - 1, in1.cols() - 1,
		in2.rows(), in2.cols()) = in2;

	for (int r = 0; r < resultRow; r++) {
		for (int c = 0; c < resultCol; c++) {
			result(r, c) += (in1.reverse().array() * paddingMatrix.block(r, c,
				in1.rows(), in1.cols()).array()).sum();
		}
	}

}


int DownSampleLayer::getRow() {
	return row;
}

int DownSampleLayer::getColumn() {
	return column;
}