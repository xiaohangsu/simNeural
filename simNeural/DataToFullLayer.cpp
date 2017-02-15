//
//  DataToFullLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 2/12/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "DataToFullLayer.hpp"

void DataToFullLayer::forward(std::vector<Eigen::MatrixXd> &t_input) {
    Eigen::MatrixXd &output = getOutput();
    for (int i = 0; i < m_inputNum; i++) {
        output.block(i * m_inputRow * m_inputCol, 0, m_inputRow * m_inputCol, m_outputCol) << Eigen::MatrixXd::Map(t_input[i].data(), m_inputRow * m_inputCol, m_outputCol);
    }
}

void DataToFullLayer::backward(Eigen::MatrixXd &t_preError, Eigen::MatrixXd& t_preTheta, std::vector<Eigen::MatrixXd>& t_preOutput, ActivateLayer& t_activateLayer) {
    std::vector<Eigen::MatrixXd> &error = getErrorVec();
    Eigen::MatrixXd tempError = Eigen::MatrixXd();
    tempError = (t_preTheta.leftCols(t_preTheta.cols() - FCL_BIAS_NUM).transpose()) * t_preError;
    
    for (int i = 0; i < m_inputNum; i++) {
        error[i] = Eigen::MatrixXd::Map(tempError.block(i * m_inputRow * m_inputCol, 0, m_inputRow * m_outputCol, m_outputCol).data(), m_inputRow, m_inputCol);
    }
    t_activateLayer.deactivate(t_preOutput, error);
}

void DataToFullLayer::descentGradient() {
    
}
