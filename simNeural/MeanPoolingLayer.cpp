//
//  MeanPoolingLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 1/25/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "MeanPoolingLayer.hpp"

MeanPoolingLayer::MeanPoolingLayer(int t_kernel_row, int t_kernel_col, int t_inputRow, int t_inputCol, int t_inputNum, int t_batch) : PoolingLayer(t_kernel_row, t_kernel_col, t_inputRow, t_inputCol, t_inputNum, t_batch){
    theta = Eigen::MatrixXd::Constant(t_kernel_row, t_kernel_col, 1.0); // mapping to kernel matrix
}

void MeanPoolingLayer::forwardCaculateForPoolingLayer(std::vector<Eigen::MatrixXd> &in) {
    std::vector<Eigen::MatrixXd> &output = getOutputVec();
    int inputNum = getInputNum();
    int outputRow = static_cast<int>(output[0].rows());
    int outputCol = static_cast<int>(output[0].cols());
    int row = getKernelRow();
    int col = getKernelCol();
    for (int i = 0; i < inputNum; i++) {
        for (int r = 0; r < outputRow; r++) {
            for (int c = 0; c < outputCol; c++) {
                output[i](r, c) = ((in[i].block(r * row, c * col, row, col)) * theta).mean();
            }
        }
    }
}

void MeanPoolingLayer::backwardCaculateForPoolingLayer(std::vector<Eigen::MatrixXd> &preError, std::vector<Eigen::MatrixXd> &lastTheta, int t_batch) {
    int inputNum = getInputNum();
    std::vector<Eigen::MatrixXd> &error = getErrorVec();
    for (int r = 0; r < inputNum; r++) {
        for (int b = 0; b < t_batch; b++) {
            
        }
    }
}