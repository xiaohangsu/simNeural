//
//  MeanPoolingLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 1/25/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "MeanPoolingLayer.hpp"
#include "SigmoidLayer.hpp"
#include <iostream>
#include "Neural_Algorithms.h"
MeanPoolingLayer::MeanPoolingLayer(int t_kernel_row, int t_kernel_col, int t_inputRow, int t_inputCol, int t_inputNum, int t_batch) : PoolingLayer(t_kernel_row, t_kernel_col, t_inputRow, t_inputCol, t_inputNum, t_batch){
    theta = Eigen::MatrixXd::Constant(t_kernel_row, t_kernel_col, 1.0 / (t_kernel_col * t_kernel_row)); // mapping to kernel matrix

}

void MeanPoolingLayer::forwardCaculateForPoolingLayer(const std::vector<Eigen::MatrixXd> &in) {
    std::vector<Eigen::MatrixXd> &output = getOutputVec();
    int inputNum = getCount();
    int outputRow = static_cast<int>(output[0].rows());
    int outputCol = static_cast<int>(output[0].cols());
    int row = getKernelRow();
    int col = getKernelCol();
    
    for (int i = 0; i < inputNum; i++) {
        for (int r = 0; r < outputRow; r++) {
            for (int c = 0; c < outputCol; c++) {
                output[i](r, c) = ((in[i].block(r * row, c * col, row, col)).array() * theta.array()).sum();
            }
        }
    }
}

void MeanPoolingLayer::backwardCaculateForPoolingLayer(const std::vector<Eigen::MatrixXd> &preError, const std::vector<std::vector<Eigen::MatrixXd>> &lastTheta) {
    std::vector<Eigen::MatrixXd> &error = getErrorVec();
    
    int inputNum = getCount();
    int preErrorRow = static_cast<int>(preError[0].rows());
    int preErrorCol = static_cast<int>(preError[0].cols());
    int lastThetaRow = static_cast<int>(lastTheta[0][0].rows());
    int lastThetaCol = static_cast<int>(lastTheta[0][0].cols());
    int outputNum = static_cast<int>(preError.size());

    for (int k = 0; k < inputNum; k++) {
        for (int b = 0; b < outputNum; b++) {
            Eigen::MatrixXd paddingMatrix = Eigen::MatrixXd(preErrorRow + 2 * (lastThetaRow) - 1, preErrorCol + 2 * (lastThetaCol) - 1).setZero();
            paddingMatrix.block(lastThetaRow - 1, lastThetaCol - 1, preErrorRow, preErrorCol) = preError[b];
            Eigen::MatrixXd lastThetaReverse = lastTheta[b][k].reverse();
            neu_alg::convolution(lastThetaReverse, paddingMatrix, error[k]);
        }
    }
}
