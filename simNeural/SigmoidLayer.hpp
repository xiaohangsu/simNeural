//
//  SigmoidLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 2/3/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef SigmoidLayer_hpp
#define SigmoidLayer_hpp

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
class SigmoidLayer{
public:
    static void activate(Eigen::MatrixXd&);
    static void activate(std::vector<Eigen::MatrixXd>&);
    
    static void deactivate(Eigen::MatrixXd&, Eigen::MatrixXd&);
    static void deactivate(std::vector<Eigen::MatrixXd>&, std::vector<Eigen::MatrixXd>&);
    static void deactivate(std::vector<Eigen::MatrixXd>&);
};

#endif /* SigmoidLayer_hpp */
