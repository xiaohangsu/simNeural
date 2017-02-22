//
//  FullConnectionLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 1/3/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef FullConnectionLayer_hpp
#define FullConnectionLayer_hpp

#include <stdio.h>
#include "Layer.hpp"
#include "predefine.h"
#include "Neural_Algorithms.h"
class FullConnectionLayer : public Layer {
public:
    FullConnectionLayer(const int t_input_row, const int t_input_col, const int t_output_row, const int t_output_col, const double t_learningRate, const int t_batch);
    virtual void forward(const Eigen::MatrixXd&);
    virtual void backward(const Eigen::MatrixXd& t_preError);
    virtual void descentGradient(const Eigen::MatrixXd& t_preError);
    const int getInputRow();
    const int getInputCol();
    const int getOutputRow();
    const int getOutputCol();
    const int getBatch();
    const double getLearningRate();
    void setLearningRate(const double t_lr);
    const Eigen::MatrixXd& getTheta();
    const Eigen::MatrixXd& getOutput();
    const Eigen::MatrixXd& getError();
private:
    Eigen::MatrixXd m_theta;
    Eigen::MatrixXd m_error;
    Eigen::MatrixXd m_output;
    Eigen::MatrixXd m_input;
    int m_input_row;
    int m_input_col;
    int m_output_row;
    int m_output_col;
    int m_batch;
    double m_learningRate;
};

#endif /* FullConnectionLayer_hpp */
