//
//  ReluLayer.hpp
//  simNeural
//
//  Created by Xiaohang Su on 2/9/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef ReluLayer_hpp
#define ReluLayer_hpp

#include <stdio.h>
#include "ActivateLayer.hpp"

class ReluLayer : public ActivateLayer {
public:
    ReluLayer();
    virtual void activate(Eigen::MatrixXd&);
    virtual void activate(std::vector<Eigen::MatrixXd>&);
    
    virtual void deactivate(Eigen::MatrixXd&, Eigen::MatrixXd&);
    virtual void deactivate(std::vector<Eigen::MatrixXd>&, std::vector<Eigen::MatrixXd>&);
    virtual void deactivate(std::vector<Eigen::MatrixXd>&);
};

#endif /* ReluLayer_hpp */
