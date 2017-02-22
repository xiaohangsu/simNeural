//
//  OutputLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 2/22/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "OutputLayer.hpp"

OutputLayer::OutputLayer(const int t_input_row, const int t_input_col, const int t_output_row, const int t_output_col) {
    m_input_row = t_input_row;
    m_input_col = t_input_col;
    m_output_row = t_output_row;
    m_output_col = t_output_col;
    m_output = Eigen::MatrixXd(m_output_row, m_output_col);
    m_error = Eigen::MatrixXd(m_input_row, m_input_col);
}

void OutputLayer::forward(const Eigen::MatrixXd &t_input) {
    m_output = t_input;
}

void OutputLayer::backward(const Eigen::MatrixXd &t_standard_output) {
    m_error = t_standard_output - m_output;
}

void OutputLayer::gradientDescent() {
    
}

const Eigen::MatrixXd& OutputLayer::getOutput() {
    return m_output;
}

const Eigen::MatrixXd& OutputLayer::getError() {
    return m_error;
}

const int OutputLayer::getInputRow() {
    return m_input_row;
}

const int OutputLayer::getInputCol() {
    return m_input_col;
}

const int OutputLayer::getOutputRow() {
    return m_output_row;
}

const int OutputLayer::getOutputCol() {
    return m_output_col;
}
