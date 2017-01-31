//
//  Layer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 12/21/16.
//

#include "Layer.hpp"

Layer::Layer(){
    m_inputNum = 0;
    m_outputNum = 0;
    m_batch = 0;
};

Layer::Layer(const int t_row, const int t_col, const int t_batch) {
    m_inputNum = t_row;
    m_outputNum = t_col;
    m_batch = t_batch;
    m_error.push_back(Eigen::MatrixXd(m_outputNum, t_batch));
    m_output.push_back(Eigen::MatrixXd(m_outputNum, t_batch));
}

/**
    For convolution layer
**/
Layer::Layer(const int t_row, const int t_col, const int t_batch, const int t_count) {
    m_inputNum = t_row;
    m_outputNum = t_col;
    m_batch = t_batch;
    m_count = t_count;
    m_output = std::vector<Eigen::MatrixXd>(t_count, Eigen::MatrixXd(t_row, t_col).setZero());
    m_error = std::vector<Eigen::MatrixXd>(t_count, Eigen::MatrixXd(t_row, t_col).setZero());
}


Eigen::MatrixXd& Layer::getError() {
    return m_error[0];
}

Eigen::MatrixXd& Layer::getOutput() {
    return m_output[0];
}

std::vector<Eigen::MatrixXd>& Layer::getErrorVec() {
    return m_error;
}

std::vector<Eigen::MatrixXd>& Layer::getOutputVec() {
    return m_output;
}

const int Layer::getBatch() {
    return m_batch;
}

const int Layer::getOutputNum() {
    return m_outputNum;
}

const int Layer::getInputNum() {
    return m_inputNum;
}

const int Layer::getCount() {
    return m_count;
}
