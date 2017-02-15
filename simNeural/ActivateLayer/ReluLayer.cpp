//
//  ReluLayer.cpp
//  simNeural
//
//  Created by Xiaohang Su on 2/9/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#include "ReluLayer.hpp"
#include <iostream>
ReluLayer::ReluLayer() {
    
}

void ReluLayer::activate(Eigen::MatrixXd & t_input) {
    t_input = t_input.array().max(0).matrix();

}

void ReluLayer::activate(std::vector<Eigen::MatrixXd> &t_input) {
    for (int i = 0; i < t_input.size(); i++) {
        ReluLayer::activate(t_input[i]);
    }
}

void ReluLayer::deactivate(Eigen::MatrixXd &t_output, Eigen::MatrixXd &t_error) {
    Eigen::MatrixXd reluReverseValue = t_output.array().max(0).ceil().min(1).matrix();
    t_error = (reluReverseValue.array() * t_error.array()).matrix();
}

void ReluLayer::deactivate(std::vector<Eigen::MatrixXd> & t_output, std::vector<Eigen::MatrixXd> &t_error) {
    for (int i = 0; i < t_error.size(); i++) {
        ReluLayer::deactivate(t_output[i], t_error[i]);
    }
}

void ReluLayer::deactivate(std::vector<Eigen::MatrixXd> &t_error) {
    for (int i = 0; i < t_error.size(); i++) {
        t_error[i] = t_error[i].array().max(0).ceil().max(1).matrix();
    }
}
