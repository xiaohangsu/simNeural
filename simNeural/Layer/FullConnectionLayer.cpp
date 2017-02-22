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
#include <iostream>
using namespace std;
FullConnectionLayer::FullConnectionLayer(const int t_input_row, const int t_input_col, const int t_output_row, const int t_output_col, const double t_learningRate, const int t_batch) : Layer() {
    m_learningRate = t_learningRate;
    m_input_row = t_input_row;
    m_input_col = t_input_col;
    m_output_row = t_output_row;
    m_output_col = t_output_col;
    m_batch = t_batch;
    m_theta = Eigen::MatrixXd::Random(m_output_row, m_input_row + FCL_BIAS_NUM) / FCL_THETA_RANDOM_DIV;
    m_output = Eigen::MatrixXd(m_output_row, m_output_col);
    m_error = Eigen::MatrixXd(m_input_row, m_input_col);
    m_input = Eigen::MatrixXd(m_input_row + FCL_BIAS_NUM, m_input_col);
}

const Eigen::MatrixXd& FullConnectionLayer::getTheta() {
    return m_theta;
}

const Eigen::MatrixXd& FullConnectionLayer::getOutput() {
    return m_output;
}

const Eigen::MatrixXd& FullConnectionLayer::getError() {
    return m_error;
}

//
// input = t_input + bias
// forward computing: Theta * input -> output
//
void FullConnectionLayer::forward(const Eigen::MatrixXd &t_input) {
    
    m_input << t_input, Eigen::MatrixXd(FCL_BIAS_NUM, m_input_col).setOnes();
    m_output = m_theta * m_input;
}
void FullConnectionLayer::backward(const Eigen::MatrixXd &t_preError) {
//    Eigen::MatrixXd& error = getError();
//    Eigen::MatrixXd& output = getOutput();
//    
//    error = (t_lastTheta.leftCols(t_lastTheta.cols() - FCL_BIAS_NUM).transpose()) * (t_preError);
//    m_activateLayer->deactivate(output, error);
    m_error = (m_theta.rightCols(m_input_row).transpose()) * t_preError;
}

void FullConnectionLayer::descentGradient(const Eigen::MatrixXd& t_preError) {
//    Eigen::MatrixXd& error = getError();
//    
//    Eigen::MatrixXd input = Eigen::MatrixXd(m_col, getBatch());
//    input << t_input, FCL_BIAS_VALUE;
//    m_theta += (m_learningRate * (error * input.transpose()));
//    error.setZero();
    m_theta += (m_learningRate * (t_preError * m_input.transpose()));
}

//void FullConnectionLayer::backwardForOutputLayer(Eigen::MatrixXd &standOutput) {
////    Eigen::MatrixXd& error = getError();
////    Eigen::MatrixXd& output = getOutput();
////    error = (standOutput - output);
//}

const int FullConnectionLayer::getInputRow() {
    return m_input_row;
}


const int FullConnectionLayer::getInputCol() {
    return m_input_col;
}

const int FullConnectionLayer::getOutputRow() {
    return m_output_row;
}

const int FullConnectionLayer::getOutputCol() {
    return m_output_col;
}

const int FullConnectionLayer::getBatch() {
    return m_batch;
}

const double FullConnectionLayer::getLearningRate() {
    return m_learningRate;
}

void FullConnectionLayer::setLearningRate(const double t_lr) {
    m_learningRate = t_lr;
}
