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
#include <Eigen/Dense>
#include <vector>

class Layer {
public:
    Layer();
    virtual void forward(const Eigen::MatrixXd&){};
    virtual void backward(const Eigen::MatrixXd& preError, const Eigen::MatrixXd& lastTheta){};
    virtual void descentGradient(const Eigen::MatrixXd& t_preError){};
    
    // batch virtual function
    virtual void forward(const std::vector<Eigen::MatrixXd>&){};
    virtual void backward(const std::vector<Eigen::MatrixXd>& preError, const Eigen::MatrixXd& lastTheta){};
    virtual void descentGradient(const std::vector<Eigen::MatrixXd>&){};
    virtual void descentGradient(){};

};

#endif /* Layer_hpp */
