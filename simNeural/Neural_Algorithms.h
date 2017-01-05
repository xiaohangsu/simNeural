//
//  Neural_Algorithms.h
//  simNeural
//
//  Created by Xiaohang Su on 12/21/16.
//
//

#ifndef Neural_Algorithms_h
#define Neural_Algorithms_h

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <Eigen/Dense>

namespace neu_alg {
    static double d_sigmoid(double z) {
        return 1 / (1 + exp(-z));
    }
    
    static double d_tanh(double z) {
        return (exp(z) - exp(-z)) / (exp(z) + exp(-z));
    }
    
    static double randomDouble(double downBound, double upBound) {
        int randNum = (double) rand() / RAND_MAX;
        return downBound + randNum * (upBound - downBound);
    }
    
    static void sigmoid(Eigen::VectorXd& input) {
        input = (1 / (1 + (-1 * input.array()).exp())).matrix();
    }
    
    static Eigen::VectorXd sigmoidReverse(Eigen::VectorXd& input) {
        return (input.array() * (1 - input.array())).matrix();
    }
    
    static void sigmoid(Eigen::MatrixXd& input) {
        input = (1 / (1 + (-1 * input.array()).exp())).matrix();
    }
    
    static Eigen::MatrixXd sigmoidReverse(Eigen::MatrixXd& input) {
        return (input.array() * (1 - input.array())).matrix();
    }
    
}

#endif /* Neural_Algorithms_h */
