//
//  Net.cpp
//  simNeural
//
//  Created by Xiaohang Su on 12/21/16.
//
//

#include "Net.hpp"

Net::Net() {
    
};

Net::Net(const std::vector<int>& t_params, const int t_batch, const double t_lr) {
    for (int i = 0; i < t_params.size() - 1; i++) {
        m_layers.push_back(FullConnectionLayer(t_params[i], t_params[i + 1], t_lr, t_batch));
    }
}


void Net::forward(const Eigen::MatrixXd& t_input) {
    assert(m_layers.size() > 2);

}

void Net::backward(const Eigen::MatrixXd& t_standardOutput) {
    
}

void Net::setBatch(const int t_batch) {
    m_batch = t_batch;
}

void Net::setLearningRate(const double t_rate) {
    m_learningRate = t_rate;
}



