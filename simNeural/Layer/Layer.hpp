//
//  Layer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 12/21/16.
//
//

#ifndef Layer_hpp
#define Layer_hpp

#include "preDefine.h"
using namespace simNeural;
class Layer {
public:
    Layer();
    virtual void forward(Matrix_cr){};
    virtual void backward(Matrix_cr preError){};
    virtual void descentGradient(Matrix_cr t_preError){};
    
    // batch virtual function
    virtual void forward(vec_Matrix_cr){};
    virtual void backward(vec_Matrix_cr preError){};
    virtual void descentGradient(vec_Matrix_cr){};
    virtual void descentGradient(){};
    virtual LAYER_TYPE getType() = 0;
    
    Matrix_cr getTheta(){
        return m_theta;
    };
    Matrix_cr getOutput(){
        return m_output;
    };
    Matrix_cr getError(){
        return m_error;
    };

protected:
    Matrix m_theta;
    Matrix m_error;
    Matrix m_output;
    Matrix m_input;
};

#endif /* Layer_hpp */
