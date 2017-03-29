//
//  FullConnectionLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 1/3/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "FullConnectionLayer.hpp"

FullConnectionLayer::FullConnectionLayer(
                                         const int t_input_row,
                                         const int t_input_col,
                                         const int t_output_row,
                                         const int t_output_col,
                                         const double t_learningRate,
                                         const int t_batch) : Layer() {
    m_learningRate = t_learningRate;
    m_input_row = t_input_row;
    m_input_col = t_input_col;
    m_output_row = t_output_row;
    m_output_col = t_output_col;
    m_batch = t_batch;
    m_theta = Matrix::Random(m_output_row, m_input_row + FCL_BIAS_NUM) / FCL_THETA_RANDOM_DIV;
    m_output = Matrix(m_output_row, m_output_col);
    m_error = Matrix(m_input_row, m_input_col);
    m_input = Matrix(m_input_row + FCL_BIAS_NUM, m_input_col);
}

//
// input = t_input + bias
// forward computing: Theta * input -> output
//
void FullConnectionLayer::forward(Matrix_cr t_input) {
    
    m_input << t_input, Matrix(FCL_BIAS_NUM, m_input_col).setOnes();
    m_output = m_theta * m_input;
}

void FullConnectionLayer::backward(Matrix_cr t_preError) {
    m_error = (m_theta.rightCols(m_input_row).transpose()) * t_preError;
}

void FullConnectionLayer::descentGradient(Matrix_cr t_preError) {
    m_theta += (m_learningRate * (t_preError * m_input.transpose()));
}


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
