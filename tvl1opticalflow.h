#pragma once

#include "opencv.hpp"

namespace cv {

class tvl1flow : public DenseOpticalFlow
{
public:
    tvl1flow();
    
    void calc(InputArray I0, InputArray I1, InputOutputArray flow);
    void collectGarbage();
    
    void setTau(double t) { tau = t; }
    void setLambda(double l) { lambda = l; }
    void setTheta(double t) { theta = t; }
    void setScales(int s) { nscales = s; }
    void setWarps(int w) { warps = w; }
    void setEpsilon(double e) { epsilon = e; }
    void setIterations(int i) { iterations = i; }
    void setUseInitialFlow(bool b) { useInitialFlow = b; }
    
protected:
    double tau;
    double lambda;
    double theta;
    int nscales;
    int warps;
    double epsilon;
    int iterations;
    bool useInitialFlow;
    
private:
    void procOneScale(const Mat_<float>& I0, const Mat_<float>& I1, Mat_<float>& u1, Mat_<float>& u2);
    
    std::vector<Mat_<float> > I0s;
    std::vector<Mat_<float> > I1s;
    std::vector<Mat_<float> > u1s;
    std::vector<Mat_<float> > u2s;
    
    Mat_<float> I1x_buf;
    Mat_<float> I1y_buf;
    
    Mat_<float> flowMap1_buf;
    Mat_<float> flowMap2_buf;
    
    Mat_<float> I1w_buf;
    Mat_<float> I1wx_buf;
    Mat_<float> I1wy_buf;
    
    Mat_<float> grad_buf;
    Mat_<float> rho_c_buf;
    
    Mat_<float> v1_buf;
    Mat_<float> v2_buf;
    
    Mat_<float> p11_buf;
    Mat_<float> p12_buf;
    Mat_<float> p21_buf;
    Mat_<float> p22_buf;
    
    Mat_<float> div_p1_buf;
    Mat_<float> div_p2_buf;
    
    Mat_<float> u1x_buf;
    Mat_<float> u1y_buf;
    Mat_<float> u2x_buf;
    Mat_<float> u2y_buf;
};

}