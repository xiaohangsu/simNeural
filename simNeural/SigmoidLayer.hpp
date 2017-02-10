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
#include "ActivateLayer.hpp"

class SigmoidLayer : public ActivateLayer {
public:
    SigmoidLayer();
    virtual void activate(Eigen::MatrixXd&);
    virtual void activate(std::vector<Eigen::MatrixXd>&);
    
    virtual void deactivate(Eigen::MatrixXd&, Eigen::MatrixXd&);
    virtual void deactivate(std::vector<Eigen::MatrixXd>&, std::vector<Eigen::MatrixXd>&);
    virtual void deactivate(std::vector<Eigen::MatrixXd>&);
};

#endif /* SigmoidLayer_hpp */
