//
//  Net.cpp
//  simNeural
//
//  Created by Xiaohang Su on 12/21/16.
//
//


// to do
#include "Net.hpp"

Net::Net() {
    m_layers.push_back(new FullConnectionLayer(28*28, 1, 100, 1, 0.001, 1));
    m_layers.push_back(new ReluLayer(100, 1, 100, 1));
    m_layers.push_back(new FullConnectionLayer(100, 1, 10, 1, 0.001, 1));
    m_layers.push_back(new ReluLayer(10, 1, 10, 1));
    m_layers.push_back(new  OutputLayer(10, 1, 10, 1));
};

Net::Net(const std::vector<int>& t_params, const int t_batch, const double t_lr) {
}


void Net::forward(Matrix_crr t_input) {
    assert(m_layers.size() > 2);
    int i = 0;
    Matrix input = t_input;
    while (i < m_layers.size()) {
        switch (m_layers[i]->getType()) {
            case FULL_CONNECTION:
                m_layers[i]->forward(input);
                break;
            case ACTIVATE:
                m_layers[i]->forward(input);
                break;
            case OUT_LAYER:
                m_layers[i]->forward(input);
                break;
            default:
                break;
        }
        input = m_layers[i]->getOutput();
        i++;
    }
}

void Net::backward(Matrix_crr t_standardOutput) {
    int i = m_layers.size() - 1;
    Matrix error = t_standardOutput;
    while (i >= 0) {
        switch (m_layers[i]->getType()) {
            case FULL_CONNECTION:
                m_layers[i]->backward(error);
                break;
            case ACTIVATE:
                m_layers[i]->backward(error);
                break;
            case OUT_LAYER:
                m_layers[i]->backward(error);
                break;
            default:
                break;
        }
        error = m_layers[i]->getError();
        i--;
    }
}

void Net::descendGraident() {
    int i = m_layers.size() - 1;
    while (i >= 0) {
        switch (m_layers[i]->getType()) {
            case FULL_CONNECTION:
                m_layers[i]->descentGradient(m_layers[i + 1]->getError());
                break;
            default:
                break;
        }
        i--;
    }
}

Matrix_cr Net::getOutput() {
    return m_layers.back()->getOutput();
}

void Net::setBatch(const int t_batch) {
    m_batch = t_batch;
}

void Net::setLearningRate(const double t_rate) {
    m_learningRate = t_rate;
}



