/*
Xiaohang Su
sxhdragon@gmail.com
CNNetDataHandler.cpp
CNNetDataHandler implementation
*/

#include <iostream>
#include <fstream>
#include <string>

#include "CNNet.h"
#include "Layer.h"
#include "DownSampleLayer.h"
#include "ConvolutionLayer.h"
#include "FullConnectionLayer.h"
#include "CNNetDataHandler.h"
#include <Eigen\Dense>
using namespace Eigen;
using namespace std;

CNNetDataHandler::CNNetDataHandler(string f, bool i) {
	fileName = f;
	isBinary = i;
}

void CNNetDataHandler::readData(CNNet *cNNet) {
	
}

void CNNetDataHandler::writeData(CNNet *cNNet) {

}

void CNNetDataHandler::readFullConnectionLayer(FullConnectionLayer * fullConnectionLayer) {
	
}

void CNNetDataHandler::writeFullConnectionLayer(FullConnectionLayer *fullConnectionLayer) {

}

void CNNetDataHandler::readDownSampleLayer(DownSampleLayer *downSampleLayer){

}

void CNNetDataHandler::writeDownSampleLayer(DownSampleLayer* downSampleLayer) {

}

void CNNetDataHandler::readConvolutionLayer(ConvolutionLayer* convolutionLayer) {

}

void CNNetDataHandler::writeConvolutionLayer(ConvolutionLayer* convolutionLayer) {

}