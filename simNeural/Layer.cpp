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
    m_error = Eigen::MatrixXd(m_outputNum, t_batch);
    m_output = Eigen::MatrixXd(m_outputNum, t_batch);
}

Eigen::MatrixXd& Layer::getError() {
    return m_error;
}

Eigen::MatrixXd& Layer::getOutput() {
    return m_output;
}

int& Layer::getBatch() {
    return m_batch;
}
