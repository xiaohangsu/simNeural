/*
Xiaohang Su
sxhdragon@gmail.com
Layer.h
Layer classification
*/
#ifndef LAYER_H
#define LAYER_H
#include <Eigen/Dense>

class Layer {
private:
	int batch;
	Eigen::MatrixXd *output;
	Eigen::MatrixXd *error;

public:

	Layer(int r, int c, int b, int range = 0);
	Eigen::MatrixXd * getOutput();
	Eigen::MatrixXd * getError();

	virtual void forwardPropagation(Eigen::MatrixXd*, int);
	virtual void backwardPropagation(Eigen::MatrixXd*,
		Eigen::MatrixXd*, int);
	virtual void descentGradient(Eigen::MatrixXd*);
	void setBatch(int);
	int getBatch();
};

#endif
