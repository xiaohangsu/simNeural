//
//  DataToFullLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 2/12/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef DataToFullLayer_hpp
#define DataToFullLayer_hpp

#include <stdio.h>
#include "Layer.hpp"
#include "preDefine.h"
#include "ActivateLayer.hpp"
class DataToFullLayer : public Layer {
public:
    DataToFullLayer(int t_inputRow, int t_intputCol, int t_inputNum, int t_outputRow, int t_outputCol) : Layer() {
        std::vector<Eigen::MatrixXd> &output = getOutputVec();
        std::vector<Eigen::MatrixXd> &error = getErrorVec();
        m_inputRow = t_inputRow;
        m_inputCol = t_intputCol;
        m_inputNum = t_inputNum;
        m_outputRow = t_outputRow;
        m_outputCol = t_outputCol;
        output.push_back(Eigen::MatrixXd(m_outputRow, m_outputCol));
        error = std::vector<Eigen::MatrixXd>(m_inputNum, MatrixXd(m_inputRow, m_inputCol));
    }
    
    void forward(std::vector<Eigen::MatrixXd> &t_output);
    void backward(Eigen::MatrixXd& t_preError, Eigen::MatrixXd &t_preTheta, std::vector<Eigen::MatrixXd>& t_preOutput, ActivateLayer &t_activateLayer);
    void descentGradient();
private:
    int m_inputRow = 0;
    int m_inputCol = 0;
    int m_inputNum = 0;
    int m_outputRow = 0;
    int m_outputCol = 0;
};

#endif /* DataToFullLayer_hpp */
