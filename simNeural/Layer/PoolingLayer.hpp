//
//  PoolingLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 1/19/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef PoolingLayer_hpp
#define PoolingLayer_hpp

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>

#include "Layer.hpp"

class PoolingLayer : public Layer {
public:
    PoolingLayer(int t_kernelRow, int t_kernelCol, int t_inputRow, int t_inputCol, int t_intputNumber, int t_batch);
    
    virtual void forward(const std::vector<Eigen::MatrixXd>&);
    virtual void backward(const std::vector<Eigen::MatrixXd>& preError, const std::vector<std::vector<Eigen::MatrixXd>>& lastTheta);
    virtual void descentGradient();
    
    int getKernelCol();
    int getKernelRow();
private:
    virtual void forwardCaculateForPoolingLayer(const std::vector<Eigen::MatrixXd>&){};
    virtual void backwardCaculateForPoolingLayer(const std::vector<Eigen::MatrixXd>& preError, const std::vector<std::vector<Eigen::MatrixXd>>& lastTheta){};
    int m_kernel_col;
    int m_kernel_row;
};
#endif /* PoolingLayer_hpp */
