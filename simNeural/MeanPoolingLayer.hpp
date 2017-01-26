//
//  MeanPoolingLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 1/25/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef MeanPoolingLayer_hpp
#define MeanPoolingLayer_hpp

#include <stdio.h>
#include "PoolingLayer.hpp"

class MeanPoolingLayer : public PoolingLayer {
public:
    MeanPoolingLayer(int t_kernel_row, int t_kernel_col, int t_inputRow, int t_inputCol, int t_inputNum, int t_batch);
    
    virtual void forwardCaculateForPoolingLayer(std::vector<Eigen::MatrixXd>&);
    virtual void backwardCaculateForPoolingLayer(std::vector<Eigen::MatrixXd>& preError, std::vector<Eigen::MatrixXd>& lastTheta, int);

private:
    Eigen::MatrixXd theta;
};

#endif /* MeanPoolingLayer_hpp */
