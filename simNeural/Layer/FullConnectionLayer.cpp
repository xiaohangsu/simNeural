//
//  FullConnectionLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 1/3/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "FullConnectionLayer.hpp"
#include "preDefine.h"
#include "Neural_Algorithms.h"
#include "SigmoidLayer.hpp"
#include "ReluLayer.hpp"
FullConnectionLayer::FullConnectionLayer() : Layer() {
    
};

FullConnectionLayer::FullConnectionLayer(const int t_inNumber, const int t_outNumber, const double t_learningRate, const int t_batch) : Layer(t_inNumber, t_outNumber, t_batch) {
    m_learningRate = t_learningRate;
    m_row = t_outNumber;
    m_col = t_inNumber + FCL_BIAS_NUM; // + 1 for bias
    m_theta = Eigen::MatrixXd::Random(m_row, m_col) / FCL_THETA_RANDOM_DIV;
}

Eigen::MatrixXd& FullConnectionLayer::getTheta() {
    return m_theta;
}

//
// input = t_input + bias
// forward computing: Theta * input -> output
//
void FullConnectionLayer::forward(Eigen::MatrixXd &t_input) {
    Eigen::MatrixXd& output = getOutput();
    Eigen::MatrixXd input = Eigen::MatrixXd(m_col, getBatch());
    input << t_input, input.row(m_col - FCL_BIAS_NUM).setOnes();

    output = m_theta * input;

    m_activateLayer->activate(output);
}
void FullConnectionLayer::backward(Eigen::MatrixXd &t_preError, Eigen::MatrixXd& t_lastTheta) {
    Eigen::MatrixXd& error = getError();
    Eigen::MatrixXd& output = getOutput();
    
    error = (t_lastTheta.leftCols(t_lastTheta.cols() - FCL_BIAS_NUM).transpose()) * (t_preError);
    m_activateLayer->deactivate(output, error);
}

void FullConnectionLayer::descentGradient(Eigen::MatrixXd & t_input) {
    Eigen::MatrixXd& error = getError();
    
    Eigen::MatrixXd input = Eigen::MatrixXd(m_col, getBatch());
    input << t_input, FCL_BIAS_VALUE;
    m_theta += (m_learningRate * (error * input.transpose()));
    error.setZero();
}

void FullConnectionLayer::backwardForOutputLayer(Eigen::MatrixXd &standOutput) {
    Eigen::MatrixXd& error = getError();
    Eigen::MatrixXd& output = getOutput();
    error = (standOutput - output);
}

const int FullConnectionLayer::getRow() {
    return m_row;
}

const int FullConnectionLayer::getCol() {
    return m_col;
}

void FullConnectionLayer::setActivateLayer(ACTIVATE_TYPE t_TYPE) {
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

ActivateLayer FullConnectionLayer::getActivateLayer() {
    return *m_activateLayer;
}

