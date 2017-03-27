//
//  OutputLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 2/22/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef OutputLayer_hpp
#define OutputLayer_hpp

#include <stdio.h>
#include "Layer.hpp"
class OutputLayer : public Layer {
public:
    OutputLayer(
                const int t_input_row,
                const int t_input_col,
                const int t_output_row,
                const int t_output_col);
    virtual void forward(Matrix_cr);
    virtual void backward(Matrix_cr);
    virtual void gradientDescent();
    const int getInputRow();
    const int getInputCol();
    const int getOutputRow();
    const int getOutputCol();
    Matrix_cr getOutput();
    Matrix_cr getError();
private:
    int m_input_row;
    int m_input_col;
    int m_output_row;
    int m_output_col;
    Matrix m_output;
    Matrix m_error;
};

#endif /* OutputLayer_hpp */
