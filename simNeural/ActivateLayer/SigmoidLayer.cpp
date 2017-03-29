//
//  SigmoidLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 2/3/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "SigmoidLayer.hpp"

SigmoidLayer::SigmoidLayer(
                           const int t_input_row,
                           const int t_input_col,
                           const int t_output_row,
                           const int t_output_col) {
    m_input_row = t_input_row;
    m_input_col = t_input_col;
    m_output_row= t_output_row;
    m_output_col = t_output_col;
    m_output = Matrix(m_output_row, m_output_col);
    m_error = Matrix(m_input_row, m_input_col);
}

void SigmoidLayer::forward(Matrix_cr t_input) {
    m_output = (1 / (1 + (-1 * t_input.array()).exp())).matrix();
}

void SigmoidLayer::backward(Matrix_cr t_preError) {
    m_output = (m_output.array() * (1 - m_output.array())).matrix();
    m_error = ((t_preError.array()) * (m_output.array())).matrix();
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
