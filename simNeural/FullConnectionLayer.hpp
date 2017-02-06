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
#include "Neural_Algorithms.h"
class FullConnectionLayer : public Layer {
public:
    FullConnectionLayer();
    FullConnectionLayer(const int t_inNumber, const int t_outNumber, const double t_learningRate, const int t_batch);
    Eigen::MatrixXd& getTheta();
    virtual void forward(Eigen::MatrixXd&);
    virtual void backward(Eigen::MatrixXd& t_preError, Eigen::MatrixXd& t_lastTheta);
    virtual void descentGradient(Eigen::MatrixXd&);
    void backwardForOutputLayer(Eigen::MatrixXd& standOutput);
    const int getRow();
    const int getCol();
private:
    Eigen::MatrixXd m_theta;
    int m_row;
    int m_col;
    double m_learningRate;
};

#endif /* FullConnectionLayer_hpp */
