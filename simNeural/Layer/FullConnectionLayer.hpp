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
class FullConnectionLayer : public Layer {
public:
    FullConnectionLayer(
                        const int t_input_row,
                        const int t_input_col,
                        const int t_output_row,
                        const int t_output_col,
                        const double t_learningRate,
                        const int t_batch);
    virtual void forward(Matrix_cr input);
    virtual void backward(Matrix_cr preError);
    virtual void descentGradient(Matrix_cr preError);
    const int getInputRow();
    const int getInputCol();
    const int getOutputRow();
    const int getOutputCol();
    const int getBatch();
    const double getLearningRate();
    void setLearningRate(const double t_lr);
    Matrix_cr getTheta();
    Matrix_cr getOutput();
    Matrix_cr getError();
private:
    Matrix m_theta;
    Matrix m_error;
    Matrix m_output;
    Matrix m_input;
    int m_input_row;
    int m_input_col;
    int m_output_row;
    int m_output_col;
    int m_batch;
    double m_learningRate;
};

#endif /* FullConnectionLayer_hpp */
