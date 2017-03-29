//
//  Net.hpp
//  simNeural
//
//  Created by Xiaohang Su on 12/21/16.
//
//

#ifndef Net_hpp
#define Net_hpp

#include <vector>
#include "FullConnectionLayer.hpp"
#include "ReluLayer.hpp"
#include "SigmoidLayer.hpp"
#include "OutputLayer.hpp"

class Net {
public:
    Net();
    Net(const std::vector<int>&, const int, const double);
    void forward(Matrix_crr t_input);
    void backward(Matrix_crr t_standardOutput);
    void descendGraident();
    Matrix_cr getOutput();
    void setBatch(const int t_batch);
    void setLearningRate(const double t_rate);
private:
    int m_batch;
    double m_learningRate;
    std::vector<Layer*> m_layers;
};

#endif /* Net_hpp */
