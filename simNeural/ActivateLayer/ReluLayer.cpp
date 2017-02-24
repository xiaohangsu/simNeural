//
//  ReluLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 2/9/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "ReluLayer.hpp"

ReluLayer::ReluLayer(const int t_input_row, const int t_input_col, const int t_output_row, const int t_output_col) {
    m_input_row = t_input_row;
    m_input_col = t_input_col;
    m_output_row= t_output_row;
    m_output_col = t_output_col;
    m_output = Eigen::MatrixXd(m_output_row, m_output_col);
    m_error = Eigen::MatrixXd(m_input_row, m_input_col);
}

void ReluLayer::forward(const Eigen::MatrixXd& t_input) {
    m_output = t_input.array().max(0).matrix();
}

void ReluLayer::backward(const Eigen::MatrixXd &t_preError) {
    Eigen::MatrixXd reluReverseValue = m_output.array().max(0).ceil().min(1).matrix();
    m_error = (t_preError.array() * reluReverseValue.array()).matrix();
}

const Eigen::MatrixXd& ReluLayer::getOutput() {
    return m_output;
}

const Eigen::MatrixXd& ReluLayer::getError() {
    return m_error;
}

const int ReluLayer::getInputRow() {
    return m_input_row;
}

const int ReluLayer::getInputCol() {
    return m_input_col;
}

const int ReluLayer::getOutputRow() {
    return m_output_row;
}

const int ReluLayer::getOutputCol() {
    return m_output_col;
}
