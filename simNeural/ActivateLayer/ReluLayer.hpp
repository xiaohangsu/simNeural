//
//  ReluLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 2/9/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef ReluLayer_hpp
#define ReluLayer_hpp

#include <stdio.h>
#include "Layer.hpp"

class ReluLayer : public Layer {
public:
    ReluLayer(const int t_input_row, const int t_input_col, const int t_output_row, const int t_output_col);
    virtual void forward(const Eigen::MatrixXd&);
    virtual void backward(const Eigen::MatrixXd&);
    const Eigen::MatrixXd& getError();
    const Eigen::MatrixXd& getOutput();
    const int getInputRow();
    const int getInputCol();
    const int getOutputRow();
    const int getOutputCol();
private:
    int m_input_row;
    int m_input_col;
    int m_output_row;
    int m_output_col;
    Eigen::MatrixXd m_error;
    Eigen::MatrixXd m_output;
};

#endif /* ReluLayer_hpp */
