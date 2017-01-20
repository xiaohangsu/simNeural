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
    Layer(const int t_input, const int t_output, const int t_bacth);
    Layer(const int t_input, const int t_output, const int t_bacth, const int t_count);
    Eigen::MatrixXd& getError();
    Eigen::MatrixXd& getOutput();
    
    std::vector<Eigen::MatrixXd>& getErrorVec();
    std::vector<Eigen::MatrixXd>& getOutputVec();
    int& getBatch();
    
    virtual void forward(Eigen::MatrixXd&, int){};
    virtual void backward(Eigen::MatrixXd& preError, Eigen::MatrixXd& lastTheta, int){};
    virtual void forward(std::vector<Eigen::MatrixXd>&){};
    virtual void backward(std::vector<Eigen::MatrixXd>& preError, Eigen::MatrixXd& lastTheta, int){};
    virtual void descentGradient(Eigen::MatrixXd&){};
    virtual void descentGradient(std::vector<Eigen::MatrixXd>&){};
private:
    int m_inputNum;
    int m_outputNum;
    int m_batch;
    std::vector<Eigen::MatrixXd> m_output;
    std::vector<Eigen::MatrixXd> m_error;
    
};

#endif /* Layer_hpp */
