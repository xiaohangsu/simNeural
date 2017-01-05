/*
Xiaohang Su
sxhdragon@gmail.com
ConvolutionLayer.cpp
ConvolutionLayer implementation
*/

#include "ConvolutionLayer.h"
#include <iostream>
#include <Windows.h>
#include <Eigen\Dense>
using namespace Eigen;

ConvolutionLayer::ConvolutionLayer(int r, int c, int in, int kn,
	int kr,int kc, double lr, int b) 
	: Layer(r - kr + 1, c - kc + 1, b, kn) {
	row = r;
	column = c;
	kernel = new MatrixXd[in * kn];
	bias = new double[kn];
	inputNumber = in;
	kernelNumber = kn;
	kernelRow = kr;
	kernelColumn = kc;
	learningRate = lr;

	for (int k = 0; k < kn; k++) {
		for (int i = 0; i < in; i++){
			*(kernel + k * in + i) = MatrixXd::Random(kr, kc) / 10;
		}
		
		*(bias + k) = (MatrixXd::Random(1,1) / 10).sum();
		std::cout << *(bias + k) << std::endl;
	}

	MatrixXd *output = getOutput(),
		*error = getError();
	output = new MatrixXd[kernelNumber];
	error = new MatrixXd[kernelNumber];

	for (int k = 0; k < kernelNumber; k++) {
		*(output + k) = MatrixXd(r - kr + 1, c - kc + 1).setZero();
		*(error + k) = MatrixXd(kr, kc).setZero();
	}
}

void ConvolutionLayer::forwardPropagation(MatrixXd* input, int in) {


	MatrixXd *output = getOutput();
	for (int k = 0; k < kernelNumber; k++) {
		for (int i = 0; i < in; i++) {
			convolution(kernel[k * in + i],
				input[i], *(output + k));

		}
		*(output + k) = ((output + k)->array() + *(bias + k)).matrix();
		//std::cout << *(output + k) << std::endl << std::endl;
		//system("pause");
	}

	
	
}

/*
	* preError is error from last Layer
	* Theta is last downSampleLayer kernel (normally)
	* preErrorNumber measures the preError Numbers
*/
void ConvolutionLayer::backwardPropagation(MatrixXd *preError, MatrixXd * theta, int preErrorNumber) {
	MatrixXd *error = getError();
	int preErrorRow = preError->rows(),
		preErrorCol = preError->cols();
	int thetaRow = theta->rows(),
		thetaCol = theta->cols();
	for (int e = 0; e < preErrorNumber; e++) {
		for (int r = 0; r < preErrorRow; r++) {
			for (int c = 0; c < preErrorCol; c++) {
				(error[e]).block(r * thetaRow, c * thetaCol,
					thetaRow, thetaCol) =
					MatrixXd::Constant(thetaRow, thetaCol, (preError[e])(r, c))
					/ (preErrorCol * preErrorRow);
			}
		}

	}
}

/*
	*
	*	lastOutput means the output From last Layer (Input or pooling)
	*
*/
void ConvolutionLayer::descentGradient(MatrixXd *lastOutput) {
	MatrixXd *error = getError();
	MatrixXd *output = getOutput();
	for (int e = 0; e < kernelNumber; e++) {
		for (int l = 0; l < inputNumber; l++) {
			//std::cout << error[e] << std::endl << std::endl;
			//std::cout << lastOutput[l] << std::endl << std::endl;
			//std::cout << e << "" << l << " " <<  e * column + l << std::endl << std::endl;
			convolutionFordescentGradient(error[e], lastOutput[l], kernel[e * inputNumber + l]);
		}
		bias[e] -= learningRate * error[e].sum();
		//std::cout << "e: " << e << " " << bias[e] << std::endl << std::endl;
		(error[e]).setZero();
		(output[e]).setZero();
	}
}

/*
	* Convolution
	* in1 and in2 (in1 size < in2 size)
	*
*/
void ConvolutionLayer::convolution(MatrixXd& in1, MatrixXd & in2, MatrixXd &result) {
	int resultRow = result.rows(),
		resultCol = result.cols();

	for (int r = 0; r < resultRow; r++) {
		for (int c = 0; c < resultCol; c++) {

			result(r, c) += (in1.array() * in2.block(r, c,
				in1.rows(), in1.cols()).array()).sum();
		}
	}

}

/*
* Convolution For descent Gradient
* in1 and in2 (in1 size < in2 size)
*
*/
void ConvolutionLayer::convolutionFordescentGradient(MatrixXd& in1,
	MatrixXd & in2, MatrixXd &result) {
	int resultRow = result.rows(),
		resultCol = result.cols();
	//std::cout << "in1:" << in1 << std::endl << std::endl;
	//std::cout << "in2:" << in2 << std::endl << std::endl;
	//std::cout << "result:" << result << std::endl << std::endl;
	for (int r = 0; r < resultRow; r++) {
		for (int c = 0; c < resultCol; c++) {
			result(r, c) -= learningRate * (in1.array() * in2.block(r, c,
				in1.rows(), in1.cols()).array()).sum() / kernel->size();
		}
	}
}

/*
	getKernel return convolution layer Kernel
*/
MatrixXd* ConvolutionLayer::getKernel() {
	return kernel;
}

int ConvolutionLayer::getRow() {
	return row;
}

int ConvolutionLayer::getColumn() {
	return column;
}