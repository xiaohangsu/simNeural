/*
	main.cpp
	for CNNet project
	sxhdragon@gmail.com
	*****************************************************
*/

#include <iostream>
#include <Windows.h>
#include <Eigen/Dense>
#include <ctime>
#include <fstream>

#include "FullConnectionLayer.h"
#include "DownSampleLayer.h"
#include "ConvolutionLayer.h"

using namespace std;
using namespace Eigen;


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

void costFunction(double *cost, MatrixXd &standardOutput, MatrixXd &output) {
	int max = 0;
	//cout << output << endl << endl;
	for (int i = 0; i < output.rows(); i++) {
		if (output(i, 0) > output(max, 0)) {
			max = i;
		}
	}
	if (standardOutput(max, 0) > 0) {
		*cost += 1;
	}
	//cout << max << " " << standardOutput(max, 0) << output(max, 0) << endl;
}


void readImageAndLabel(ifstream* imageStream, ifstream* labelStream, MatrixXd imageData[], MatrixXd &labelData,
	ifstream* imaget10k, ifstream* labelt10k, MatrixXd imageTestData[], int imageTestLabel[]) {

	cout << "readImageAndLabel Function~" << endl;
	cout << "read Trainning Image And Label" << endl;

	/*
	* ROW * COLUMN LOOP read an Image
	*/
	unsigned int buffer;
	for (int t = 0; t < TRAINNING_DATA_NUMEBR; t++){
		*(imageData+t) = MatrixXd(ROW, COLUMN);

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
		*(imageTestData + t) = MatrixXd(ROW, COLUMN);

		// read the test Image & Label
		unsigned int buffer;
		for (int r = 0; r < ROW; r++) {
			for (int c = 0; c < COLUMN; c++) {
				imaget10k->read((char*)&buffer, 1);
				(imageTestData[t])(r, c) = (unsigned int)buffer & 0x000000ff;
			}
		}
		labelt10k->read((char*)&buffer, 1);
		imageTestLabel[t] = (unsigned int)buffer & 0x0000000f;
	}

	cout << "Finished read Testing Image And Label" << endl;


}

void trainningFunction(MatrixXd imageData[], MatrixXd &imageLabel) {
	double startTime, endTime;
	startTime = clock();

	MatrixXd standardOutput(10, 10);
	standardOutput <<
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 1;


	double *cost = new double[1];
	*cost = 0;
	int a[7] = { 28, 28, 1, 20, 5, 5, 1 };
	ConvolutionLayer con1(28, 28, 1, 20, 5, 5, 0.1, 1);
	ConvolutionLayer con2(12, 12, 20, 50, 5, 5, 0.1, 1);

	DownSampleLayer down1(2, 2, 24, 24, 20, 1);
	
	DownSampleLayer down2(2, 2, 8, 8, 50, 1);
	
	FullConnectionLayer full1(800, 300, 0.1, 1);
	FullConnectionLayer full2(300, 10, 0.1, 1);
	
	for (int t = 0; t < TRAINNING_TIME; t++) {
		for (int b = 0; b < BATCH; b++) {
			MatrixXd imageTempData = imageData[b];
			MatrixXd imageTempLabel = imageLabel.col(b);

			//cout << "forwardPropagation~~~~~~~~~~~~~~~~~~" << endl;
			con1.forwardPropagation(&imageTempData, 1);
			down1.forwardPropagation(con1.getOutput(), 20);

			con2.forwardPropagation(down1.getOutput(), 20);

			down2.forwardPropagation(con2.getOutput(), 50);

			MatrixXd *forwardTempOutput = down2.getOutput();
			VectorXd temp(800);
			for (int i = 0; i < 50; i++) {
				temp.segment(i * 16, (i+1)*16) << VectorXd::Map(forwardTempOutput[i].data(), forwardTempOutput[i].rows() * forwardTempOutput[i].cols());
			}
			
			full1.forwardPropagation(&MatrixXd(temp), 0);
	

			full2.forwardPropagation(full1.getOutput(), 0);
			
			full2.backwardPropagationForOutputLayer(imageTempLabel);
			full1.backwardPropagation(full2.getError(), &(full2.getTheta()), 0);


			/*
				turn full1 error to correct format for down2 backwardPropagation
			*/
			MatrixXd *down2PreError = new MatrixXd[50];
			for (int i = 0; i < 50; i++) {
				*(down2PreError + i) = MatrixXd::Map(temp.segment(i * 16, (i + 1) * 16).data(), 4, 4);

			}


			down2.backwardPropagation(down2PreError, NULL, 50);
			con2.backwardPropagation(down2.getError(), &(down2.getTheta()), 50);
			down1.backwardPropagation(con2.getError(), con2.getKernel(), 50);
 			con1.backwardPropagation(down1.getError(), &(down1.getTheta()), 20);
			////cout << imageTempLabel << endl;
			//system("pause");
			//cout << full2.getTheta().block(24, 24, 24, 24) << endl;
			//system("pause");
			//cout << full1.getTheta().block(24,24, 24,24) << endl;
			//system("pause");
			//cout << down2.getTheta() << endl;
			//system("pause");
			//cout << *(con2.getKernel()) << endl;
			//system("pause");
			//cout << down1.getTheta() << endl;
			//system("pause");
			//cout << *(con1.getKernel()) << endl;
			//system("pause");

			//cout << "descentGradient~~~~~~~~~~~~~~~~~"  << endl;
			full2.descentGradient(full1.getOutput());
			full1.descentGradient(&MatrixXd(temp));
			down2.descentGradient(con2.getOutput());
			con2.descentGradient(down1.getOutput());
			down1.descentGradient(con1.getOutput());


			con1.descentGradient(&imageTempData);
			costFunction(cost, imageTempLabel, *(full2.getOutput()));
			if ((b+1) % 100 == 0) {
				cout << "Trainning Time:\t" << t + 1 << "\tBatch:\t"
					<< b + 1 << "\tcost:\t" << *cost << endl;
				*cost = 0;

			}

			delete[] down2PreError;
		}
	}

	cout << "Finished Training!~" << endl;
	endTime = clock();
	std::cout << "Finished Time: " << (endTime - startTime) / TRAINNING_DATA_NUMEBR << " minutes." << endl;

}

int main() {

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


	MatrixXd *imageData = new MatrixXd[TRAINNING_DATA_NUMEBR];
	MatrixXd imageLabel(10, TRAINNING_DATA_NUMEBR);
	MatrixXd *imageTestData = new MatrixXd[TEST_BATCH];
	int imageTestLabel[TEST_BATCH];
	readImageAndLabel(trainImage, trainLabel, imageData, imageLabel,
		imaget10k, labelt10k, imageTestData, imageTestLabel);

	trainningFunction(imageData, imageLabel);



	system("pause");
	return 0;
}
