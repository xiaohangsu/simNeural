/*
Xiaohang Su
sxhdragon@gmail.com
ConvolutionLayer.h
ConvolutionLayer classification
*/

#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H

#include "Layer.h"
#include <Eigen\Dense>

class ConvolutionLayer : public Layer {
private:
	int row;
	int column;

	Eigen::MatrixXd *kernel;
	double *bias;
	int kernelNumber;
	int kernelRow;
	int kernelColumn;
	int inputNumber;
	double learningRate;
	// input Matrix | input Matrix | result
	void convolution(Eigen::MatrixXd&, Eigen::MatrixXd&,
		Eigen::MatrixXd&);
	void convolutionFordescentGradient(Eigen::MatrixXd&,
		Eigen::MatrixXd&, Eigen::MatrixXd&);

public:
	ConvolutionLayer(int, int, int, int,
		int, int, double, int);

	Eigen::MatrixXd* getKernel();
	virtual void forwardPropagation(Eigen::MatrixXd*, int);
	virtual void backwardPropagation(Eigen::MatrixXd*, Eigen::MatrixXd*, int);
	virtual void descentGradient(Eigen::MatrixXd *);
	int getRow();
	int getColumn();
};


#endif