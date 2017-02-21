//
//  FastConvolutionLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 2/20/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "FastConvolutionLayer.hpp"
#include "ReluLayer.hpp"
#include "SigmoidLayer.hpp"
FastConvolutionLayer::FastConvolutionLayer(){};

FastConvolutionLayer::FastConvolutionLayer(const int t_input_row, const int t_input_col, const int t_kernel_number, const int t_kernel_row, const int t_kernel_col, const double t_lr, const int batch) {
    
}

void FastConvolutionLayer::forward(Eigen::MatrixXd &t_input) {
    
}

void FastConvolutionLayer::setActivateLayer(ACTIVATE_TYPE t_TYPE) {
    switch (t_TYPE) {
        case SIGMOID:
            m_activateLayer = new SigmoidLayer();
            break;
        case RELU:
            m_activateLayer = new ReluLayer();
            break;
        default:
            break;
    }
}

void FastConvolutionLayer::setLearningRate(const double t_lr) {
    m_learningRate = t_lr;
}

ActivateLayer FastConvolutionLayer::getActivateLayer() {
    return *m_activateLayer;
}
