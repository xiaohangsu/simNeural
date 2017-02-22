//
//  SigmoidLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 2/3/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "SigmoidLayer.hpp"

SigmoidLayer::SigmoidLayer(const int t_input_row, const int t_input_col, const int t_output_row, const int t_output_col) {
    m_input_row = t_input_row;
    m_input_col = t_input_col;
    m_output_row= t_output_row;
    m_output_col = t_output_col;
    m_output = Eigen::MatrixXd(m_output_row, m_output_col);
    m_error = Eigen::MatrixXd(m_input_row, m_input_col);
}

void SigmoidLayer::forward(const Eigen::MatrixXd& t_input) {
    m_output = (1 / (1 + (-1 * t_input.array()).exp())).matrix();
}

void SigmoidLayer::backward(const Eigen::MatrixXd &t_preError) {
    m_output = (m_output.array() * (1 - m_output.array())).matrix();
    m_error = ((t_preError.array()) * (m_output.array())).matrix();
}

//void SigmoidLayer::activate(Eigen::MatrixXd &t_input) {
//    t_input = (1 / (1 + (-1 * t_input.array()).exp())).matrix();
//}
//
//void SigmoidLayer::activate(std::vector<Eigen::MatrixXd> & t_input) {
//    for (int i = 0; i < t_input.size(); i++) {
//        SigmoidLayer::activate(t_input[i]);
//    }
//}
//
//void SigmoidLayer::deactivate(Eigen::MatrixXd &t_output, Eigen::MatrixXd &t_error) {
//    Eigen::MatrixXd sigmoidReverseValue = (t_output.array() * (1 - t_output.array())).matrix();
//    t_error = ((t_error.array() * (sigmoidReverseValue.array()))).matrix();
//}
//
//void SigmoidLayer::deactivate(std::vector<Eigen::MatrixXd> & t_output, std::vector<Eigen::MatrixXd> & t_error) {
//    assert(t_error.size() == t_output.size());
//    for (int i = 0; i < t_error.size(); i++) {
//        SigmoidLayer::deactivate(t_output[i], t_error[i]);
//    }
//}
//
//void SigmoidLayer::deactivate(std::vector<Eigen::MatrixXd> &t_error) {
//    for (int i = 0; i < t_error.size(); i++)
//        t_error[i] = (t_error[i].array() * (1 - t_error[i].array())).matrix();
//}

const Eigen::MatrixXd& SigmoidLayer::getOutput() {
    return m_output;
}

const Eigen::MatrixXd& SigmoidLayer::getError() {
    return m_error;
}

const int SigmoidLayer::getInputRow() {
    return m_input_row;
}

const int SigmoidLayer::getInputCol() {
    return m_input_col;
}

const int SigmoidLayer::getOutputRow() {
    return m_output_row;
}

const int SigmoidLayer::getOutputCol() {
    return m_output_col;
}
