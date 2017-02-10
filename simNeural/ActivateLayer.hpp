//
//  ActivateLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 2/9/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef ActivateLayer_hpp
#define ActivateLayer_hpp

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>

class ActivateLayer {
public:
    ActivateLayer(){};
    virtual void activate(Eigen::MatrixXd&) = 0;
    virtual void activate(std::vector<Eigen::MatrixXd>&) = 0;
    
    virtual void deactivate(Eigen::MatrixXd&, Eigen::MatrixXd&) = 0;
    virtual void deactivate(std::vector<Eigen::MatrixXd>&, std::vector<Eigen::MatrixXd>&) = 0;
    virtual void deactivate(std::vector<Eigen::MatrixXd>&) = 0;
};
#endif /* ActivateLayer_hpp */
