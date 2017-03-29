//
//  SigmoidLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 2/3/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef SigmoidLayer_hpp
#define SigmoidLayer_hpp

#include <stdio.h>
#include "Layer.hpp"

class SigmoidLayer : public Layer {
public:
    SigmoidLayer(
                 const int t_input_row,
                 const int t_input_col,
                 const int t_output_row,
                 const int t_output_col);
    virtual void forward(Matrix_cr);
    virtual void backward(Matrix_cr);
    virtual LAYER_TYPE getType() {
        return ACTIVATE;
    }
    
    const int getInputRow();
    const int getInputCol();
    const int getOutputRow();
    const int getOutputCol();
private:
    int m_input_row;
    int m_input_col;
    int m_output_row;
    int m_output_col;
};

#endif /* SigmoidLayer_hpp */
