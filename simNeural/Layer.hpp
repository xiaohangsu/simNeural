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


class Layer {
public:
    Layer();
    Layer(const int t_input, const int t_output, const int t_bacth);

    Eigen::MatrixXd& getError();
    Eigen::MatrixXd& getOutput();
    int& getBatch();
    
    virtual void forward(Eigen::MatrixXd&, int){};
    virtual void backward(Eigen::MatrixXd& preError, Eigen::MatrixXd& lastTheta, int){};
    virtual void descentGradient(Eigen::MatrixXd&){};
private:
    int m_inputNum;
    int m_outputNum;
    int m_batch;

    Eigen::MatrixXd m_output;
    Eigen::MatrixXd m_error;
    
};

#endif /* Layer_hpp */
