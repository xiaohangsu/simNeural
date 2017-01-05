/*
	Xiaohang Su
	sxhdragon@gmail.com
	CNNet.h
	CNNet Classification
*/

#ifndef CNNET_H
#define CNNET_H

#include "ConvolutionLayer.h"
#include "DownSampleLayer.h"
#include "FullConnectionLayer.h"

class CNNet {
public:
	CNNet(double, int);
	void forwardPropagation();
	void backwardPropagation();
	void descentGradient();
	void setBatch(int);
	void setLearningRate(int);

private:
	int batch;
	double learningRate;
	Layer *layer;
	int layerNumber;
};

#endif