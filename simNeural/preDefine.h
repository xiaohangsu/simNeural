//
//  preDefine.h
//  simNeural
//
//  Created by Xiaohang Su on 1/3/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef preDefine_h
#define preDefine_h
#include <Eigen/Dense>
#include <vector>

namespace simNeural {
    typedef Eigen::MatrixXd                                   Matrix;
    typedef Eigen::MatrixXd&                                  Matrix_r;
    typedef Eigen::MatrixXd&&                                 Matrix_rr;
    typedef const Eigen::MatrixXd                             Matrix_c;
    typedef const Eigen::MatrixXd&                            Matrix_cr;
    typedef const Eigen::MatrixXd&&                           Matrix_crr;
    typedef std::vector<Eigen::MatrixXd>                      vec_Matrix;
    typedef std::vector<Eigen::MatrixXd>&                     vec_Matrix_r;
    typedef std::vector<Eigen::MatrixXd>&&                    vec_Matrix_rr;
    typedef const std::vector<Eigen::MatrixXd>                vec_Matrix_c;
    typedef const std::vector<Eigen::MatrixXd>&               vec_Matrix_cr;
    typedef const std::vector<Eigen::MatrixXd>&&              vec_Matrix_crr;
    typedef std::vector<std::vector<Eigen::MatrixXd>>         vec2_Matrix;
    typedef std::vector<std::vector<Eigen::MatrixXd>>&        vec2_Matrix_r;
    typedef std::vector<std::vector<Eigen::MatrixXd>>&&       vec2_Matrix_rr;
    typedef const std::vector<std::vector<Eigen::MatrixXd>>   vec2_Matrix_c;
    typedef const std::vector<std::vector<Eigen::MatrixXd>>&  vec2_Matrix_cr;
    typedef const std::vector<std::vector<Eigen::MatrixXd>>&& vec2_Matrix_crr;
    typedef Eigen::Map<Eigen::MatrixXd>                       Map_Matrix;
}

const static int FCL_BIAS_NUM = 1;
const static int FCL_THETA_RANDOM_DIV = 10;
const static double FCL_BIAS_VALUE = 1.0;
const static double CONV_BIAS_UPPERBOUND = 1.0;
const static double CONV_BIAS_LOWERBOUND = -1.0;
enum ACTIVATE_TYPE {SIGMOID, RELU};

enum LAYER_TYPE { ACTIVATE, FULL_CONNECTION, OUT_LAYER, LAYER };

#endif /* preDefine_h */
