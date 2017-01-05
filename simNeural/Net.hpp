//
//  Net.hpp
//  simNeural
//
//  Created by Xiaohang Su on 12/21/16.
//
//

#ifndef Net_hpp
#define Net_hpp

#include <stdio.h>
#include <vector>
#include "FullConnectionLayer.hpp"

class Net {
public:
    Net();
    Net(const std::vector<int>&, const int, const double);
    void forward(const Eigen::MatrixXd& t_input);
    void backward(const Eigen::MatrixXd& t_standardOutput);
    void gradientDescend();
    
    void setBatch(const int t_batch);
    void setLearningRate(const double t_rate);
private:
    int m_batch;
    double m_learningRate;
    std::vector<Layer> m_layers;
};

#endif /* Net_hpp */
