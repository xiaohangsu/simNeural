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
#include <iostream>
#include <Eigen/Dense>

namespace neu_alg {
    static double d_sigmoid(double z) {
        return 1 / (1 + exp(-z));
    }
    
    static double d_tanh(double z) {
        return (exp(z) - exp(-z)) / (exp(z) + exp(-z));
    }
    
    static double randomDouble(double downBound, double upBound) {
        double randNum = (double) rand() / RAND_MAX;
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
    
    static void convolution(Eigen::MatrixXd& in1, Eigen::MatrixXd& in2, Eigen::MatrixXd& result) {
        long resultRow = result.rows();
        long resultCol = result.cols();
        
        for (int r = 0; r < resultRow; r++) {
            for (int c = 0; c < resultCol; c++) {
                result(r, c) += (in1.array() * in2.block(r, c, in1.rows(), in1.cols()).array()).sum();
            }
        }
    }
    
    static void conv_descent_gradient(double lr, Eigen::MatrixXd& in1, Eigen::MatrixXd& in2, Eigen::MatrixXd& result, int kernel_size) {
        int resultRow = static_cast<int>(result.rows()),
        resultCol = static_cast<int>(result.cols());

        for (int r = 0; r < resultRow; r++) {
            for (int c = 0; c < resultCol; c++) {
                result(r, c) -= lr * (in1.array() * in2.block(r, c, in1.rows(), in1.cols()).array()).sum() / kernel_size;
            }
        }
    }
}

#endif /* Neural_Algorithms_h */
