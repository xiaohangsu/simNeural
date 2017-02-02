//
//  ConvolutionalLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 1/5/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "ConvolutionalLayer.hpp"
#include <iostream>
ConvolutionalLayer::ConvolutionalLayer(const int t_row, const int t_col, const int t_inputNumber, const int t_kernel_number, const int t_kernel_row, const int t_kernel_col, double t_lr, int t_batch) : Layer(t_row - t_kernel_row + 1, t_col - t_kernel_col + 1, t_batch, t_kernel_number) {
    m_row = t_row;
    m_col = t_col;
    m_kernel_row = t_kernel_row;
    m_kernel_col = t_kernel_col;
    
    m_kernel = std::vector<std::vector<Eigen::MatrixXd>>(t_kernel_number, std::vector<Eigen::MatrixXd>(t_inputNumber, Eigen::MatrixXd::Random(t_kernel_row, t_kernel_col)));
    
    m_bias = std::vector<double>(t_kernel_number, neu_alg::randomDouble(CONV_BIAS_LOWERBOUND, CONV_BIAS_UPPERBOUND));
}

void ConvolutionalLayer::forward(std::vector<Eigen::MatrixXd>& t_input, int t_in) {
    std::vector<Eigen::MatrixXd> &output = getOutputVec();
    
    for (int k = 0; k < m_kernel.size(); k++) {
        for (int i = 0; i < t_in; i++) {
            neu_alg::convolution(m_kernel[k][i], t_input[i], output[k]);
        }
        output[k] = (output[k].array() + m_bias[k]).matrix();
    }
}

void ConvolutionalLayer::backward(std::vector<Eigen::MatrixXd>& preError, Eigen::MatrixXd& lastTheta) {
    
    std::vector<Eigen::MatrixXd> &error = getErrorVec();
    long preErrorRow = preError[0].rows(),
    preErrorCol = preError[0].cols();
    long thetaRow = lastTheta.rows(),
    thetaCol = lastTheta.cols();
    int preErrorNumber = static_cast<int>(preError.size());
    for (int e = 0; e < preErrorNumber; e++) {
        for (int r = 0; r < preErrorRow; r++) {
            for (int c = 0; c < preErrorCol; c++) {
                (error[e]).block(r * thetaRow, c * thetaCol, thetaRow, thetaCol) = Eigen::MatrixXd::Constant(thetaRow, thetaCol, (preError[e])(r, c) / (preErrorCol * preErrorRow));
            }
        }
    }
};

void ConvolutionalLayer::descentGradient(std::vector<Eigen::MatrixXd>& lastOutput) {
    std::vector<Eigen::MatrixXd> &error = getErrorVec();
    std::vector<Eigen::MatrixXd> &output = getOutputVec();
    
    int inputNumber = static_cast<int>(lastOutput.size());
    int kernelNumber = static_cast<int>(m_kernel.size());
    for (int e = 0; e < kernelNumber; e++) {
        for (int l = 0; l < inputNumber; l++) {
            neu_alg::conv_descent_gradient(m_learningRate, error[e], lastOutput[l], m_kernel[e][l], kernelNumber);
        }
        m_bias[e] -= m_learningRate * error[e].sum();
        
        (error[e]).setZero();
        (output[e]).setZero();
    }
};

const int ConvolutionalLayer::getKernelNum() {
    return static_cast<int>(m_kernel.size());
}

const int ConvolutionalLayer::getCol() {
    return m_col;
}

const int ConvolutionalLayer::getRow() {
    return m_row;
}

std::vector<std::vector<Eigen::MatrixXd>>& ConvolutionalLayer::getKernels() {
    return m_kernel;
}
