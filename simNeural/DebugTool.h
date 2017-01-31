
//
//  DebugTool.h
//  simNeural
//
//  Created by Xiaohang Su on 1/31/17.
//  Copyright © 2017 Xiaohang Su. All rights reserved.
//

#ifndef DebugTool_h
#define DebugTool_h
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"
using namespace std;
using namespace Eigen;
namespace debug {
    void printError(Layer layer) {
        vector<MatrixXd> error = layer.getErrorVec();
        for (int i = 0; i < error.size(); i++) {
            cout << "Error :" << i + 1 << endl;
            cout << error[i] << endl << endl;
        }
    }
    
    void printOutput(Layer layer) {
        vector<MatrixXd> output = layer.getOutputVec();
        for (int i = 0; i < output.size(); i++) {
            cout << "Output :" << i + 1 << endl;
            cout << output[i] << endl << endl;
        }
    }
    
    void printBasicInfo(Layer layer) {
        cout << "Layer info :\t" << "InputNum:" << layer.getInputNum() << "\t|\tOutputNum:" << layer.getOutputNum() << endl;
        cout << "\tBatch:" << layer.getBatch() << "\t|\tCount:" << layer.getCount() << endl;
        cout << endl;
    }
}

#endif /* DebugTool_h */
