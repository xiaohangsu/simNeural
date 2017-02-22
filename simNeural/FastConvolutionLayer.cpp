//
//  FastConvolutionLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 2/20/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "FastConvolutionLayer.hpp"
#include "ReluLayer.hpp"
#include "SigmoidLayer.hpp"
FastConvolutionLayer::FastConvolutionLayer(){};

FastConvolutionLayer::FastConvolutionLayer(const int t_input_row, const int t_input_col, const int t_inputNum, const int t_kernel_number, const int t_kernel_row, const int t_kernel_col, const double t_lr, const int t_stride) {
    m_input_row = t_input_row;
    m_input_col = t_input_col;
    m_input_num = t_inputNum;
    m_learningRate = t_lr;
    m_kernel_num = t_kernel_number;
    m_kernel_row = t_kernel_row;
    m_kernel_col = t_kernel_col;
    m_kernel = Eigen::MatrixXd::Random(m_kernel_col * m_kernel_row * m_kernel_num, m_input_num);
    m_output = Eigen::MatrixXd(m_kernel_num, (m_input_row - m_kernel_row + 1) * (m_input_col - m_kernel_col + 1));
    
    for (int i = 0; i < m_kernel_num; i++) {
        Eigen::MatrixXd temp((m_input_row - m_kernel_row + 1), (m_input_col - m_kernel_col + 1));
        m_outputVec.push_back(temp);
        m_errorVec.push_back(temp);
    }
    m_input = Eigen::MatrixXd((m_input_row - m_kernel_row + 1) * (m_input_col - m_kernel_col + 1), m_kernel_row * m_kernel_col * m_input_num);
}

void FastConvolutionLayer::forward(std::vector<Eigen::MatrixXd> &t_input) {
    
    for(int i = 0; i < m_input.rows(); i++) {
        for (int j = 0; j < m_input.cols() / (m_kernel_row * m_kernel_col); j++) {
            m_input.block(i, j * (m_kernel_row * m_kernel_col), 1, (m_kernel_row * m_kernel_col));
        }
    }
    m_output = m_kernel * m_input;
    m_activateLayer->activate(m_output);
    for (int i = 0; i < m_kernel_num; i++) {
        m_outputVec[i] = Eigen::MatrixXd::Map(m_output.row(i).data(), (m_input_row - m_kernel_row + 1), (m_input_col - m_kernel_col + 1));
    }
}

void FastConvolutionLayer::backward(std::vector<Eigen::MatrixXd> &t_preError, Eigen::MatrixXd &t_lastTheta) {
    
}



void FastConvolutionLayer::setActivateLayer(ACTIVATE_TYPE t_TYPE) {
    switch (t_TYPE) {
        case SIGMOID:
            m_activateLayer = new SigmoidLayer();
            break;
        case RELU:
            m_activateLayer = new ReluLayer();
            break;
        default:
            break;
    }
}

void FastConvolutionLayer::setLearningRate(const double t_lr) {
    m_learningRate = t_lr;
}

ActivateLayer FastConvolutionLayer::getActivateLayer() {
    return *m_activateLayer;
}
