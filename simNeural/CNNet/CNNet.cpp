/*
	CNNet.cpp
	sxhdragon@gmail.com
	CNNet.cpp
	CNNet Implementation
*/

#include <iostream>
#include <Windows.h>
#include <Eigen/Dense>
#include "FullConnectionLayer.h"
#include "DownSampleLayer.h"
#include "ConvolutionLayer.h"

#include "CNNet.h"
using namespace std;
using namespace Eigen;

CNNet::CNNet(double lr, int b) {
	learningRate = lr;
	batch = b;

	ConvolutionLayer con1(28, 28, 1, 20, 5, 5, 0.1, 1);
	ConvolutionLayer con2(12, 12, 20, 50, 5, 5, 0.1, 1);

	DownSampleLayer down1(2, 2, 24, 24, 20, 1);

	DownSampleLayer down2(2, 2, 8, 8, 50, 1);

	FullConnectionLayer full1(800, 300, 0.1, 1);
	FullConnectionLayer full2(300, 10, 0.1, 1);

}