//
//  main.cpp
//  simNeural
//
//  Created by Xiaohang Su on 12/21/16.
//  Copyright © 2016 Xiaohang Su. All rights reserved.
//

#include <iostream>
#include "Net.hpp"
#include "FullConnectionLayer.hpp"
#include "ConvolutionalLayer.hpp"
#include "MeanPoolingLayer.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include "DebugTool.h"
using namespace Eigen;
using namespace std;

#define TRAINNING_DATA_NUMEBR 60000
#define TEST_BATCH 10000


void smallSampleTest();
void readMNIST(vector<MatrixXd> &imageData, MatrixXd &imageLabel, vector<MatrixXd> &testImageData, vector<int> &imageTestLabel);

int main(int argc, const char * argv[]) {

//    vector<MatrixXd> imageData(TRAINNING_DATA_NUMEBR, MatrixXd(28, 28));
//    MatrixXd imageLabel(10, TRAINNING_DATA_NUMEBR);
//    vector<MatrixXd> imageTestData(TEST_BATCH, MatrixXd(28, 28));
//    vector<int> imageTestLabel(TEST_BATCH, 0);
//    readMNIST(imageData, imageLabel, imageTestData, imageTestLabel);
    smallSampleTest();
    return 0;
}



void smallSampleTest() {

    
    ConvolutionalLayer conv1(28, 28, 1, 20, 5, 5, 0.1, 1);
    ConvolutionalLayer conv2(12, 12, 20, 50, 5, 5, 0.1, 1);
    
    MeanPoolingLayer pool1(2, 2, 24, 24, 20, 1);
    
    MeanPoolingLayer pool2(2, 2, 8, 8, 50, 1);
    
    FullConnectionLayer full1(800, 300, 0.1, 1);
    FullConnectionLayer full2(300, 10, 0.1, 1);
    
    //    MatrixXd input(10, 1);
//    input << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
//    MatrixXd standardOutput(10, 1);
//    standardOutput <<
//        0, 0, 0, 0, 0,
//        0, 0, 0, 0, 1;
    
    vector<MatrixXd> imageData;
    MatrixXd imageLabel;
    vector<MatrixXd> testImageData;
    vector<int> imageTestLabel;
    readMNIST(imageData, imageLabel, testImageData, imageTestLabel);
    
    for (int i = 0; i < 20; i++) {
        conv1.forward(imageData, 1);
        pool1.forward(conv1.getOutputVec());
        conv2.forward(pool1.getOutputVec(), 10);
        pool2.forward(conv2.getOutputVec());
        vector<MatrixXd> pool2Out = pool2.getOutputVec();
        MatrixXd full1Input(pool2.getInputNum() * pool2.getOutputNum() * pool2.getCount(), 1);
        for (int i = 0; i < pool2Out.size(); i++) {
            full1Input.block(i * 16, 0, 16, 1) << Map<VectorXd>(pool2Out[i].data(), pool2Out[i].size());
        }
        
        full1.forward(full1Input);
        full2.forward(full1.getOutput());

        // backward Output
        MatrixXd correctLabel = imageLabel.col(0);
        full2.backwardForOutputLayer(correctLabel);
        full1.backward(full2.getError(), full2.getTheta());
        debug::printOutput(full2);
        MatrixXd error = MatrixXd();
        MatrixXd preError = full1.getError();
        MatrixXd lastTheta = full1.getTheta();
        MatrixXd sigmoidReverseValue = neu_alg::sigmoidReverse(full1Input);
        
        error = (lastTheta.leftCols(lastTheta.cols() - FCL_BIAS_NUM).transpose()) * (preError);
        
        error = ((error.array()) * (sigmoidReverseValue.array())).matrix();
        
        std::vector<MatrixXd> fullToPoolError = vector<MatrixXd>(50, MatrixXd());
        
        for (int i = 0; i < 50; i++) {
            fullToPoolError[i] = MatrixXd::Map(error.block(i * 16, 0, 16, 1).data(), 4, 4);

        }
        
        pool2.backward(fullToPoolError, vector<vector<MatrixXd>>());
        conv2.backward(pool2.getErrorVec(), pool2.getTheta());
        pool1.backward(conv2.getErrorVec(), conv2.getKernels());
        conv1.backward(pool1.getErrorVec(), pool1.getTheta());
        
        full2.descentGradient(full1.getOutput());
        full1.descentGradient(full1Input);
        pool2.descentGradient();
        conv2.descentGradient(pool1.getOutputVec());
        pool1.descentGradient();
        vector<MatrixXd> temp(1, imageData[0]);
        conv1.descentGradient(temp);
    }

}

void readMNIST(vector<MatrixXd> &imageData, MatrixXd &imageLabel, vector<MatrixXd> &testImageData, vector<int> &imageTestLabel) {
    
    // init params
    imageData = vector<MatrixXd>(TRAINNING_DATA_NUMEBR, MatrixXd(28, 28));
    imageLabel = MatrixXd(10, TRAINNING_DATA_NUMEBR);
    testImageData = vector<MatrixXd>(TEST_BATCH, MatrixXd(28, 28));
    
    imageTestLabel = vector<int>(TEST_BATCH, 0);
    
    /*
     read file stream
     */
    ifstream *imaget10kStream = new ifstream("data/t10k-images.idx3-ubyte", ifstream::binary);
    ifstream *labelt10kStream = new ifstream("data/t10k-labels.idx1-ubyte", ifstream::binary);
    ifstream *trainImageStream = new ifstream("data/train-images.idx3-ubyte", ifstream::binary);
    ifstream *trainLabelStream = new ifstream("data/train-labels.idx1-ubyte", ifstream::binary);
    
    /*
     Skip migic number & number of items & row and column
     *	char * 4 + number of items(60000) + row and column
     */
    int magicNumber[4], row, column, itemNumber;
    
    imaget10kStream->read((char*)&magicNumber[0], 4);
    labelt10kStream->read((char*)&magicNumber[1], 4);
    trainImageStream->read((char*)&magicNumber[2], 4);
    trainLabelStream->read((char*)&magicNumber[3], 4);
    imaget10kStream->read((char*)&itemNumber, 4);
    labelt10kStream->read((char*)&itemNumber, 4);
    trainImageStream->read((char*)&itemNumber, 4);
    trainLabelStream->read((char*)&itemNumber, 4);
    trainImageStream->read((char*)&row, 4);
    trainImageStream->read((char*)&column, 4);
    imaget10kStream->read((char*)&row, 4);
    imaget10kStream->read((char*)&column, 4);
    
    cout << "readImageAndLabel Function~" << endl;
    cout << "read Trainning Image And Label" << endl;
    /*
     * ROW * COLUMN LOOP read an Image
     */
    unsigned int buffer;
    for (int t = 0; t < TRAINNING_DATA_NUMEBR; t++){
        
        trainLabelStream->read((char*)&buffer, 1);
        int label = (unsigned int)buffer & 0x0000000f;
        for (int i = 0; i < 10; i++) {
            if (label == i)
                imageLabel(i, t) = 1;
            else imageLabel(i, t) = 0;
        }
        //cout << "readImageAndLabel";
        
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                trainImageStream->read((char*)&buffer, 1);
                (imageData[t])(r, c) = ((unsigned int)buffer & 0x000000ff) / 256.0;
            }
        }
    }
    
    cout << "Finished read Trainning Image And Label" << endl;
    
    cout << "read Testing Image And Label" << endl;
    
    // TEST -----------------------------------------
    // READ TEST DATA
    for (int t = 0; t < TEST_BATCH; t++) {
        // read the test Image & Label
        unsigned int buffer;
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                imaget10kStream->read((char*)&buffer, 1);
                (testImageData[t])(r, c) = ((unsigned int)buffer & 0x000000ff) / 256.0;
            }
        }
        labelt10kStream->read((char*)&buffer, 1);
        imageTestLabel[t] = (unsigned int)buffer & 0x0000000f;
    }
    
    cout << "Finished read Testing Image And Label" << endl;
    
}
