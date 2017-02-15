//
//  PoolingLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 1/19/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "PoolingLayer.hpp"
PoolingLayer::PoolingLayer(int t_kernelRow, int t_kernelCol, int t_inputRow, int t_inputCol, int t_inputNumber, int t_batch) : Layer (t_inputRow / t_kernelRow, t_inputCol / t_kernelCol, t_batch, t_inputNumber) {
    
    m_kernel_col = t_kernelCol;
    m_kernel_row = t_kernelRow;
}

void PoolingLayer::forward(const std::vector<Eigen::MatrixXd> &in) {
    assert(!in.empty());

    forwardCaculateForPoolingLayer(in);
    
}

// t_lastTheta if empty means top layer is FCLayer
void PoolingLayer::backward(const std::vector<Eigen::MatrixXd> & t_preError, const std::vector<std::vector<Eigen::MatrixXd>> &t_lastTheta) {
    std::vector<Eigen::MatrixXd> &error = getErrorVec();
    int errorNum = static_cast<int>(error.size());
    if (t_lastTheta.empty()) {
        for (int i = 0; i < errorNum; i++) {
            error[i] = t_preError[i];
        }
    } else {
        backwardCaculateForPoolingLayer(t_preError, t_lastTheta);
    }
}

void PoolingLayer::descentGradient() {
    std::vector<Eigen::MatrixXd> &error = getErrorVec();
    std::vector<Eigen::MatrixXd> &output = getOutputVec();
    int inputNum = getCount();
    for (int i = 0; i < inputNum; i++) {
        error[i].setZero();
        output[i].setZero();
    }
}

int PoolingLayer::getKernelCol() {
    return m_kernel_col;
}

int PoolingLayer::getKernelRow() {
    return m_kernel_row;
}
