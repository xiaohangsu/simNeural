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
    
    virtual void forward(std::vector<Eigen::MatrixXd>&);
    virtual void backward(std::vector<Eigen::MatrixXd>& preError, std::vector<Eigen::MatrixXd>& lastTheta, int);
    virtual void descentGradient();
    
    virtual void forwardCaculateForPoolingLayer(){};
    virtual void backwardCaculateForPoolingLayer(){};
    
    int getKernelCol();
    int getKernelRow();
private:
    int m_kernel_col;
    int m_kernel_row;
};
#endif /* PoolingLayer_hpp */
