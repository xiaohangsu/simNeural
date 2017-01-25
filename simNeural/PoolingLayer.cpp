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

void PoolingLayer::forward(std::vector<Eigen::MatrixXd> &in) {
    int inputNumber = static_cast<int>(in.size());
    
    std::vector<Eigen::MatrixXd> output = getOutputVec();
    
    assert(!in.empty());
    assert(in[0].cols() % m_kernel_col != 0 || in[0].rows() % m_kernel_row != 0);
    
    for (int i = 0; i < inputNumber; i++) {
        for (int r = 0; r < output[0].rows(); r++) {
            for (int c = 0; c < output[0].cols(); c++) {
                forwardCaculateForPoolingLayer();
            }
        }
    }
}

void PoolingLayer::backward(std::vector<Eigen::MatrixXd> & t_preError, std::vector<Eigen::MatrixXd> &t_lastTheta, int t_batch) {
    std::vector<Eigen::MatrixXd> error = getErrorVec();
    if (t_lastTheta.empty()) {
        for (int i = 0; i < t_batch; i++) {
            error[i] = t_preError[i];
        }
    } else {
        int inputNum = getInputNum();
        for (int r = 0; r < inputNum; r++) {
            for (int b = 0; b < t_batch; b++) {
                backwardCaculateForPoolingLayer();
            }
        }
    }
}

void PoolingLayer::descentGradient() {
    std::vector<Eigen::MatrixXd> error = getErrorVec();
    int inputNum = getInputNum();
    for (int i = 0; i < inputNum; i++) {
        error[i].setZero();
    }
}
