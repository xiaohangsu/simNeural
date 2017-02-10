//
//  ConvolutionalLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 1/5/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef ConvolutionalLayer_hpp
#define ConvolutionalLayer_hpp

#include <stdio.h>
#include <vector>
#include "Layer.hpp"
#include "Neural_Algorithms.h"
#include "ActivateLayer.hpp"
#include "preDefine.h"

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(const int t_row, const int t_col, const int t_inputNumber, const int t_kernel_number, const int t_kernel_row, const int t_kernel_col, double t_lr, int t_batch);
    virtual void forward(std::vector<Eigen::MatrixXd>&, int);
    virtual void backward(std::vector<Eigen::MatrixXd>& preError, Eigen::MatrixXd& lastTheta);
    virtual void descentGradient(std::vector<Eigen::MatrixXd>&);
    const int getRow();
    const int getCol();
    const int getKernelNum();
    std::vector<std::vector<Eigen::MatrixXd>>& getKernels();
    void setLearningRate(double lr) {m_learningRate = lr;};
    void setActivateLayer(ACTIVATE_TYPE t_TYPE);
private:
    int m_row;
    int m_col;
    int m_kernel_row;
    int m_kernel_col;
    std::vector<std::vector<Eigen::MatrixXd>> m_kernel;
    std::vector<double> m_bias;
    double m_learningRate;
    ActivateLayer *m_activateLayer;
};

#endif /* ConvolutionalLayer_hpp */
