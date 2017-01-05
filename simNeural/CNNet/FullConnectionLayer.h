/*
Xiaohang Su
sxhdragon@gmail.com
FullConnectionLayer.h
FullConnectionLayer classification
*/

#ifndef FULLCONNECTIONLAYER_H
#define FULLCONNECTIONLAYER_H

#include "Layer.h"
#include <Eigen\Dense>

class FullConnectionLayer : public Layer{
private:
	Eigen::MatrixXd theta;
	int row;
	int column;
public:
	double LearnRate;
	FullConnectionLayer(int inNumber, int outNumber,double learningRate,int batch);
	Eigen::MatrixXd& getTheta();

	virtual void forwardPropagation(Eigen::MatrixXd*, int);
	virtual void backwardPropagation(Eigen::MatrixXd*, Eigen::MatrixXd*, int);
	virtual void descentGradient(Eigen::MatrixXd* input);

	void backwardPropagationForOutputLayer(Eigen::MatrixXd&);
	
	int getRow();
	int getColumn();

	void sigmoid(Eigen::MatrixXd& input);
	Eigen::MatrixXd sigmoidReverse(Eigen::MatrixXd* input);
};



#endif
