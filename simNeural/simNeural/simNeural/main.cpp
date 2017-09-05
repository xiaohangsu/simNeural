//
//  main.cpp
//  simNeural
//
//  Created by Xiaohang Su on 8/28/17.
//  Copyright © 2017 XiaohangSu. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include "Net.hpp"
#include "preDefine.h"
using namespace std;

#define ROW 28
#define COLUMN 28
#define BATCH 60000
#define BPNET_BATCH 100
#define TRAINNING_DATA_NUMEBR 60000
#define TRAINNING_TIME 20
#define TEST_BATCH 10000
#define LEARNING_RATE 0.0001
#define uint32 unsigned int
#define BigtoLittle32(A) ((((uint32)(A) & 0xff000000) >> 24) | (((uint32)(A) & 0x00ff0000) >> 8) | \
	(((uint32)(A)& 0x0000ff00) << 8) | (((uint32)(A)& 0x000000ff) << 24))


void readImageAndLabel(ifstream* imageStream, ifstream* labelStream, vec_Matrix_r imageData, Matrix_r labelData,
	ifstream* imaget10k, ifstream* labelt10k, vec_Matrix_r imageTestData, vector<int>& imageTestLabel) {

	cout << "readImageAndLabel Function~" << endl;
	cout << "read Trainning Image And Label" << endl;

	/*
	* ROW * COLUMN LOOP read an Image
	*/
	unsigned int buffer;
	for (int t = 0; t < TRAINNING_DATA_NUMEBR; t++){
		imageData[t] = Matrix(ROW, COLUMN);

		labelStream->read((char*)&buffer, 1);
		int label = (unsigned int)buffer & 0x0000000f;
		for (int i = 0; i < 10; i++) {
			if (label == i)
				labelData(i, t) = 1;
			else labelData(i, t) = 0;
		}
		//cout << "readImageAndLabel";

		for (int r = 0; r < ROW; r++) {
			for (int c = 0; c < COLUMN; c++) {
				imageStream->read((char*)&buffer, 1);
				(imageData[t])(r, c) = (unsigned int)buffer & 0x000000ff;
			}
		}
	}

	cout << "Finished read Trainning Image And Label" << endl;

	cout << "read Testing Image And Label" << endl;

	// TEST -----------------------------------------
	// READ TEST DATA
	for (int t = 0; t < TEST_BATCH; t++) {
		imageTestData[t] = Matrix(ROW, COLUMN);

		// read the test Image & Label
		unsigned int buffer;
		for (int r = 0; r < ROW; r++) {
			for (int c = 0; c < COLUMN; c++) {
				imaget10k->read((char*)&buffer, 1);
				(imageTestData[t])(r, c) = (unsigned int)buffer & 0x000000ff;
			}
		}
		labelt10k->read((char*)&buffer, 1);
		imageTestLabel.push_back((unsigned int)buffer & 0x0000000f);
	}

	cout << "Finished read Testing Image And Label" << endl;
}

bool correct(Matrix_cr standardOuput, Matrix_cr output) {
    int max = 0;
    for (int i = 0; i < output.rows(); i++) {
        if (output(i, 0) > output(max, 0)) {
            max = i;
        }
    }
    
    return standardOuput(max, 0) > 0;
}

void trainning(vec_Matrix_cr imageData, Matrix_cr imageLabel) {
    Net net = Net();
    
    int count = 101;
    double c = 0;
    for (int i = 0; i < TRAINNING_TIME * TRAINNING_DATA_NUMEBR; i++) {
        net.forward(imageData[i % TRAINNING_DATA_NUMEBR]);
        net.backward(imageLabel.col(i % TRAINNING_DATA_NUMEBR));
        net.descendGraident();
        
        if (correct(imageLabel.col(i % TRAINNING_DATA_NUMEBR), net.getOutput())) {
            c++;
        }
        
        if (i != 0 && i % count == 0) {
            cout << "Correct Rate is: " << c / count << endl;
            c = 0.0;
        }
    }
}

int main(int argc, const char * argv[]) {
	/*
		read file stream
	*/
	ifstream *imaget10k = new ifstream("data/t10k-images.idx3-ubyte", ifstream::binary);
	ifstream *labelt10k = new ifstream("data/t10k-labels.idx1-ubyte", ifstream::binary);
	ifstream *trainImage = new ifstream("data/train-images.idx3-ubyte", ifstream::binary);
	ifstream *trainLabel = new ifstream("data/train-labels.idx1-ubyte", ifstream::binary);

	/*
	Skip migic number & number of items & row and column
	*	char * 4 + number of items(60000) + row and column
	*/
	int magicNumber[4], row, column, itemNumber;

	imaget10k->read((char*)&magicNumber[0], 4);
	labelt10k->read((char*)&magicNumber[1], 4);
	trainImage->read((char*)&magicNumber[2], 4);
	trainLabel->read((char*)&magicNumber[3], 4);
	imaget10k->read((char*)&itemNumber, 4);
	labelt10k->read((char*)&itemNumber, 4);
	trainImage->read((char*)&itemNumber, 4);
	trainLabel->read((char*)&itemNumber, 4);
	trainImage->read((char*)&row, 4);
	trainImage->read((char*)&column, 4);
	imaget10k->read((char*)&row, 4);
	imaget10k->read((char*)&column, 4);


	vec_Matrix imageData = vec_Matrix(TRAINNING_DATA_NUMEBR, Matrix());
	Matrix imageLabel(10, TRAINNING_DATA_NUMEBR);
	vec_Matrix imageTestData = vec_Matrix(TEST_BATCH, Matrix());
    vector<int> imageTestLabel(TEST_BATCH, 0);
	readImageAndLabel(trainImage, trainLabel, imageData, imageLabel,
		imaget10k, labelt10k, imageTestData, imageTestLabel);


    trainning(imageData, imageLabel);
    return 0;
}
