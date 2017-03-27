//
//  Layer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 12/21/16.
//
//

#ifndef Layer_hpp
#define Layer_hpp
#include <stdio.h>
#include "preDefine.h"

class Layer {
public:
    Layer();
    virtual void forward(Matrix_cr){};
    virtual void backward(Matrix_cr preError, Matrix_cr lastTheta){};
    virtual void descentGradient(Matrix_cr t_preError){};
    
    // batch virtual function
    virtual void forward(vec_Matrix_cr){};
    virtual void backward(vec_Matrix_cr preError, vec_Matrix_cr lastTheta){};
    virtual void descentGradient(vec_Matrix_cr){};
    virtual void descentGradient(){};

};

#endif /* Layer_hpp */
