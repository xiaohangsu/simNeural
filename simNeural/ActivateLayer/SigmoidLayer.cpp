//
//  SigmoidLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 2/3/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "SigmoidLayer.hpp"

SigmoidLayer::SigmoidLayer() {
    
}

void SigmoidLayer::activate(Eigen::MatrixXd &t_input) {
    t_input = (1 / (1 + (-1 * t_input.array()).exp())).matrix();
}

void SigmoidLayer::activate(std::vector<Eigen::MatrixXd> & t_input) {
    for (int i = 0; i < t_input.size(); i++) {
        SigmoidLayer::activate(t_input[i]);
    }
}

void SigmoidLayer::deactivate(Eigen::MatrixXd &t_output, Eigen::MatrixXd &t_error) {
    Eigen::MatrixXd sigmoidReverseValue = (t_output.array() * (1 - t_output.array())).matrix();
    t_error = ((t_error.array() * (sigmoidReverseValue.array()))).matrix();
}

void SigmoidLayer::deactivate(std::vector<Eigen::MatrixXd> & t_output, std::vector<Eigen::MatrixXd> & t_error) {
    assert(t_error.size() == t_output.size());
    for (int i = 0; i < t_error.size(); i++) {
        SigmoidLayer::deactivate(t_output[i], t_error[i]);
    }
}

void SigmoidLayer::deactivate(std::vector<Eigen::MatrixXd> &t_error) {
    for (int i = 0; i < t_error.size(); i++)
        t_error[i] = (t_error[i].array() * (1 - t_error[i].array())).matrix();
}
