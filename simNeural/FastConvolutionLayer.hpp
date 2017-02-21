//
//  FastConvolutionLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 2/20/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef FastConvolutionLayer_hpp
#define FastConvolutionLayer_hpp

#include <stdio.h>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "preDefine.h"
#include "ActivateLayer.hpp"
class FastConvolutionLayer {
public:
    FastConvolutionLayer();
    FastConvolutionLayer(const int t_inputRow, const int t_outputRow, const int t_kernel_number, const int t_kernel_row, const int t_kernel_col, double t_lr, int t_batch);
    virtual void forward(Eigen::MatrixXd& t_input);
    virtual void backward(Eigen::MatrixXd& t_preError, std::vector<Eigen::MatrixXd>& t_lastTheta);
    virtual void descentGradient(Eigen::MatrixXd&);
    void setLearningRate(const double t_lr);
    void setActivateLayer(ACTIVATE_TYPE t_TYPE);
    ActivateLayer getActivateLayer();
    
private:
    int m_row;
    int m_col;
    int m_kernel_row;
    int m_kernel_col;
    Eigen::MatrixXd m_kernel;
    double m_learningRate;
    ActivateLayer *m_activateLayer;
    Eigen::MatrixXd m_output;
    Eigen::MatrixXd m_error;
};

#endif /* FastConvolutionLayer_hpp */
