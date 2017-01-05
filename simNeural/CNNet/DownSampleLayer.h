/*
Xiaohang Su
sxhdragon@gmail.com
DownSampleLayer.h
DownSampleLayer classification
*/

#ifndef DOWNSAMPLELAYER_H
#define DOWNSAMPLELAYER_H

#include "Layer.h"
#include <Eigen\Dense>

class DownSampleLayer : public Layer {
private:
	Eigen::MatrixXd theta;
	int row;
	int column;
	int inputNumber;
	void convolutionForBackPropagation(Eigen::MatrixXd&,
		Eigen::MatrixXd&, Eigen::MatrixXd&);
public:
	DownSampleLayer(int r, int c, int inputRow,
		int inputColumn, int inputNumber, int batch);

	Eigen::MatrixXd& getTheta();
	int getRow();
	int getColumn();

	virtual void forwardPropagation(Eigen::MatrixXd*, int);
	virtual void backwardPropagation(Eigen::MatrixXd*,
		Eigen::MatrixXd*, int);

	void sigmoid(Eigen::MatrixXd *input);
	virtual void descentGradient(Eigen::MatrixXd*);
	Eigen::MatrixXd sigmoidReverse(Eigen::MatrixXd* input);
};

#endif