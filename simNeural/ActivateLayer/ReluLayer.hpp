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
    ReluLayer(
              const int t_input_row,
              const int t_input_col,
              const int t_output_row,
              const int t_output_col);
    virtual void forward(Matrix_cr);
    virtual void backward(Matrix_cr);
    Matrix_cr getError();
    Matrix_cr getOutput();
    const int getInputRow();
    const int getInputCol();
    const int getOutputRow();
    const int getOutputCol();
private:
    int m_input_row;
    int m_input_col;
    int m_output_row;
    int m_output_col;
    Matrix m_error;
    Matrix m_output;
};

#endif /* ReluLayer_hpp */
