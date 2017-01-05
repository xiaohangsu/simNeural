/*
Xiaohang Su
sxhdragon@gmail.com
CNNetDataHandler.h
CNNetDataHandler classification
*/

#include "Layer.h"
#include "CNNet.h"
#include "ConvolutionLayer.h"
#include "DownSampleLayer.h"
#include "FullConnectionLayer.h"
#include <string>

#ifndef CNNETDATAHANDLER_H
#define CNNETDATAHANDLER_H

class CNNetDataHandler {
private:
	std::string fileName;
	bool isBinary;
	void readFullConnectionLayer(FullConnectionLayer*);
	void writeFullConnectionLayer(FullConnectionLayer*);
	void readDownSampleLayer(DownSampleLayer*);
	void writeDownSampleLayer(DownSampleLayer*);
	void readConvolutionLayer(ConvolutionLayer*);
	void writeConvolutionLayer(ConvolutionLayer*);
public:
	CNNetDataHandler(std::string, bool);
	void readData(CNNet*);
	void writeData(CNNet*);
	void setFileName(std::string);
};


#endif