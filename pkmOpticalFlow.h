/*
 
 © Parag K Mital, parag@pkmital.com
 
 The Software is and remains the property of Parag K Mital
 ("pkmital") The Licensee will ensure that the Copyright Notice set
 out above appears prominently wherever the Software is used.
 
 The Software is distributed under this Licence:
 
 - on a non-exclusive basis,
 
 - solely for non-commercial use in the hope that it will be useful,
 
 - "AS-IS" and in order for the benefit of its educational and research
 purposes, pkmital makes clear that no condition is made or to be
 implied, nor is any representation or warranty given or to be
 implied, as to (i) the quality, accuracy or reliability of the
 Software; (ii) the suitability of the Software for any particular
 use or for use under any specific conditions; and (iii) whether use
 of the Software will infringe third-party rights.
 
 pkmital disclaims:
 
 - all responsibility for the use which is made of the Software; and
 
 - any liability for the outcomes arising from using the Software.
 
 The Licensee may make public, results or data obtained from, dependent
 on or arising out of the use of the Software provided that any such
 publication includes a prominent statement identifying the Software as
 the source of the results or the data, including the Copyright Notice
 and stating that the Software has been made available for use by the
 Licensee under licence from pkmital and the Licensee provides a copy of
 any such publication to pkmital.
 
 The Licensee agrees to indemnify pkmital and hold them
 harmless from and against any and all claims, damages and liabilities
 asserted by third parties (including claims for negligence) which
 arise directly or indirectly from the use of the Software or any
 derivative of it or the sale of any products based on the
 Software. The Licensee undertakes to make no liability claim against
 any employee, student, agent or appointee of pkmital, in connection
 with this Licence or the Software.
 
 
 No part of the Software may be reproduced, modified, transmitted or
 transferred in any form or by any means, electronic or mechanical,
 without the express permission of pkmital. pkmital's permission is not
 required if the said reproduction, modification, transmission or
 transference is done without financial return, the conditions of this
 Licence are imposed upon the receiver of the product, and all original
 and amended source code is included in any transmitted product. You
 may be held legally responsible for any copyright infringement that is
 caused or encouraged by your failure to abide by these terms and
 conditions.
 
 You are not permitted under this Licence to use this Software
 commercially. Use for which any financial return is received shall be
 defined as commercial use, and includes (1) integration of all or part
 of the source code or the Software into a product for sale or license
 by or on behalf of Licensee to third parties or (2) use of the
 Software or any derivative of it for research with the final aim of
 developing software products for sale or license to a third party or
 (3) use of the Software or any derivative of it for research with the
 final aim of developing non-software products for sale or license to a
 third party, or (4) use of the Software to provide any service to an
 external organisation for which payment is received. If you are
 interested in using the Software commercially, please contact pkmital to
 negotiate a licence. Contact details are: parag@pkmital.com
 
 */


#pragma once

//#define WITH_HDF5

#include "ofxOpenCv.h"
#include "opencv.hpp"

#ifdef WITH_HDF5
#include <hdf5.h>
#if (H5_VERS_MINOR==6)
#include "H5LT.h"
#else
#include "hdf5_hl.h"
#endif
#endif

namespace cv
{
    // from opencv 2.4.8 source!  only repeated here to allow interfacingw ith certain parameters that are not native to the opencv interface.
    
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

class pkmOpticalFlow
{
public:
    pkmOpticalFlow()
    {
        b_allocated = false;
    }
    
    void allocate(int w, int h, int nframes = 0)
    {
        if(b_allocated && width == w && height == h)
        {
            return;
        }
        width = w;
        height = h;
        num_prev_imgs = 2;
        
        width_rsz = min(64, width);
        ratio = (float)width_rsz / (float)width;
        height_rsz = height * ratio;
        
        hist_size = 50;
        
        printf("[pkmOpticalFlow]::allocate - resized: %d x %d -> %d x %d\n", width, height, width_rsz, height_rsz);
        
        color_img.allocate(width, height);
        color_img_rsz.allocate(width_rsz, height_rsz);
        
        gray_img.allocate(width_rsz, height_rsz);
        
        prev_gray_imgs.resize(num_prev_imgs);
        for(int i = 0; i < num_prev_imgs; i++)
            prev_gray_imgs[i].allocate(width_rsz, height_rsz);
        
        flow_color_img.allocate(width_rsz, height_rsz);
        mag_img.allocate(width_rsz, height_rsz);
        mag_img_crop.allocate(width_rsz - 10, height_rsz - 10);
        
        dev_img.allocate(width_rsz, height_rsz);
        
        mag_max_avg = 1.0;
        
        flow_img = cv::Mat(height_rsz, width_rsz, CV_32FC2);
        
        if (nframes == 0)
            num_spectra = 500;
        else
            num_spectra = nframes;
        
        frame_i = 0;
        
        num_freq = 128;  // <= than 360
        flow_entropy = cv::Mat::zeros(1, num_spectra, CV_32FC1);
        spec_HOMG = cv::Mat::zeros(num_freq, num_spectra, CV_32FC1);
        p_spec_HOMG = cv::Mat::zeros(num_freq, num_spectra, CV_32FC1);
        spec_HOMG_img8 = cv::Mat::zeros(num_freq, num_spectra, CV_8UC1);
        
//        flow.setIterations(100);
//        flow.setWarps(2);
//        flow.setScales(2);
//        flow.setEpsilon(0.08);
//        flow.setTau(0.25);
//        flow.setLambda(0.1);
//        flow.setUseInitialFlow(false);
        
        b_allocated = true;
    }
    
    float getMaxMotionMagnitude()
    {
        return mag_max_avg;
    }
    
    void update(const ofPixelsRef &pixels)
    {
        color_img.setFromPixels(pixels);
        color_img_rsz.scaleIntoMe(color_img);
        color_img_rsz.convertRgbToHsv();
        color_img_rsz.convertToGrayscalePlanarImage(gray_img, 2);
        
        cv::Mat I0(gray_img.getCvImage()), I1(prev_gray_imgs[0].getCvImage());
        
        flow.calc(I0, I1, flow_img);
        
        // convert to color, thanks to: http://stackoverflow.com/questions/7693561/opencv-displaying-a-2-channel-image-optical-flow
        //extraxt x and y channels
        split(flow_img, xy);
        
        //calculate angle and magnitude
        cv::Mat magnitude_crop(mag_img_crop.getCvImage());
        magnitude = cv::Mat(mag_img.getCvImage());
        cartToPolar(xy[0], xy[1], magnitude, angle, true);
        cv::Rect roi(5, 5, mag_img.getWidth() - 10, mag_img.getHeight() - 10);
        cv::Mat(magnitude, roi).copyTo(magnitude_crop);
        cv::copyMakeBorder(magnitude_crop, magnitude,
                           5, 5,
                           5, 5, cv::BORDER_CONSTANT,
                           cvScalar( 0, 0, 0 ));
        mag_img.flagImageChanged();
        
        //translate magnitude to range [0;1]
        minMaxLoc(magnitude, NULL, &mag_max, NULL, &pt_max);
//        mag_max_avg = mag_max * 0.2 + mag_max_avg * 0.8;
        mag_max_avg = 8.0;
        magnitude.convertTo(magnitude, -1, 1.0/mag_max_avg);
        
        //build hsv image
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magnitude;
        merge(_hsv, 3, hsv);
        
        //convert to BGR and show
        cv::Mat ofxrgb(flow_color_img.getCvImage());
        cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
        
        rgb = rgb * 255.0;
        rgb.convertTo(ofxrgb, CV_8U);
        flow_color_img.flagImageChanged();
        
        prev_gray_imgs.push_back(gray_img);
        if(prev_gray_imgs.size() > num_prev_imgs)
        {
            prev_gray_imgs.erase(prev_gray_imgs.begin());
        }
    }
    
    void computeHistogramOfOrientedMotionGradients(bool b_normalize = true)
    {
        hist_OMG = cv::Mat::zeros(1, num_freq, CV_32F);
        for(int i = 0; i < magnitude.rows; i++)
        {
            for(int j = 0; j < magnitude.cols; j++)
            {
                hist_OMG.at<float>(0, floor(angle.at<float>(i,j)) * ((float)num_freq/360.0)) += magnitude.at<float>(i,j)/sqrtf(magnitude.rows*magnitude.cols);
            }
        }
        
        if(b_normalize)
            hist_OMG = hist_OMG / cv::sum(hist_OMG)[0];
//        cv::Mat hist_OMGLog = cv::Mat::zeros(num_freq, 1, CV_32F);
//        cv::log(hist_OMG, hist_OMGLog);
    }
    
    cv::Mat getHistOMG()
    {
        return hist_OMG;
    }
    
    
    void computeSpectrumOfOrientedMotionGradients()
    {
        computeHistogramOfOrientedMotionGradients();

//        float heightScalar = 100.0;
//        cv::Mat img = cv::Mat::zeros(heightScalar, num_freq, CV_8UC3);
//        
//        for(int i = 1; i < num_freq; i++)
//        {
//            cv::line(img,
//                     cv::Point(i-1, hist_OMG.at<float>(i-1,0)*heightScalar),
//                     cv::Point(i, hist_OMG.at<float>(i-1,0)*heightScalar),
//                     cv::Scalar(255,255,255));//, 3, 4);
//        }
//        
//        cv::namedWindow("HOMG", CV_WINDOW_FREERATIO);
//        cv::imshow("HOMG", img);
        
        
        cv::Mat source_ROI = p_spec_HOMG(cv::Rect(1,0,num_spectra-1,num_freq));
        source_ROI.copyTo(spec_HOMG(cv::Rect(0,0,num_spectra-1,num_freq)));
        hist_OMG.copyTo(spec_HOMG(cv::Rect(num_spectra-1,0,1,num_freq)));
        
        spec_HOMG.convertTo(spec_HOMG_img8,CV_8UC1,255.0);
        
        cv::Mat cmImg(spec_HOMG.rows, spec_HOMG.cols, CV_8UC3);
        cv::applyColorMap(spec_HOMG_img8, cmImg, cv::COLORMAP_JET);
        
        cv::resize(cmImg, cmImg, cv::Size(512, spec_HOMG.cols), 0, 0, cv::INTER_NEAREST);
        
        cv::namedWindow("HOMG Spectra", CV_WINDOW_FREERATIO);
        cv::imshow("HOMG Spectra", cmImg);
        
        spec_HOMG.copyTo(p_spec_HOMG);
        
//        // convert to probabilistic interpretation
//        hist_OMG = hist_OMG / cv::sum(hist_OMG)[0];
//        cv::Mat hist_OMGLog = cv::Mat::zeros(num_freq, 1, CV_32F);
//        cv::log(hist_OMG, hist_OMGLog);
//        
//        flow_entropy.at<float>(0, frame_i) = cv::sum(hist_OMG.mul(hist_OMGLog))[0];
//        frame_i++;
//        
//        
//        cv::Mat img2 = cv::Mat::zeros(heightScalar, num_spectra, CV_8UC3);
//        
//        for(int i = 1; i < num_spectra; i++)
//        {
//            cv::line(img2,
//                     cv::Point(i-1, flow_entropy.at<float>(0,i-1)*heightScalar),
//                     cv::Point(i, flow_entropy.at<float>(0,i-1)*heightScalar),
//                     cv::Scalar(255,255,255));//, 3, 4);
//            
//            cout << flow_entropy.at<float>(0,i-1) << endl;
//        }
//        
//        cv::namedWindow("Flow Entropy", CV_WINDOW_FREERATIO);
//        cv::imshow("Flow Entropy", img2);
    }
    
#ifdef WITH_HDF5
    void exportHOMGToHDF5()
    {
        hid_t file_id;
        string dataset_name = ofToDataPath("homg.h5", true);
        ofLog(OF_LOG_NOTICE, "Writing to %s", dataset_name.c_str());
        
        // H5F_ACC_TRUNC specifies that if the file already exists, the current contents will be deleted so that the application can rewrite the file with new data.
        // H5F_ACC_EXCL specifies that the open will fail if the file already exists. If the file does not already exist, the file access parameter is ignored.
        file_id = H5Fcreate(dataset_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hid_t native_dtype = H5T_NATIVE_FLOAT;
        
        // get dims. Just 2D for now.
        hsize_t dims[2];
        dims[0] = spec_HOMG.rows;
        dims[1] = spec_HOMG.cols;
        
        // do it.
        if (H5LTfind_dataset(file_id, dataset_name.c_str())==1) {
            std::string error_msg("Error: ");
            error_msg += dataset_name.c_str();
            error_msg += " exists.";
            ofLog(OF_LOG_ERROR, error_msg);
        }
        herr_t status = H5LTmake_dataset(file_id,
                                         dataset_name.c_str(),
                                         2,
                                         dims,
                                         native_dtype,
                                         spec_HOMG.ptr());

        if (status < 0) {
            std::string error_msg("Error making dataset: ");
            error_msg += dataset_name.c_str();
            ofLog(OF_LOG_ERROR, error_msg);
        }
        status = H5Fclose(file_id);
        if (status < 0) {
            std::string error_msg("Error closing dataset: ");
            error_msg += dataset_name.c_str();
            ofLog(OF_LOG_ERROR, error_msg);
        }
        
    }
#endif
    
    void exportHOMGToPPM(string path = "", string token = "")
    {
        cv::imwrite(ofToDataPath(path + token + "homg.ppm", true), spec_HOMG);
    }
    
    void exportHOMGToYML(string path = "", string token = "")
    {
        cv::FileStorage fs(path + token + "homg.yml", cv::FileStorage::WRITE );
        fs << "homg" << spec_HOMG;
//        fs << "flowentropy" << flow_entropy;
        fs.release();
    }
    
    void drawColorImg(int x, int y, int w, int h)
    {
        color_img.draw(x,y,w,h);
    }
    
    void drawGrayImg(int x, int y, int w, int h)
    {
        gray_img.draw(x,y,w,h);
    }
    
    void drawPrevGrayImg(int x, int y, int w, int h)
    {
        prev_gray_imgs[0].draw(x,y,w,h);
    }
    
    void drawMagnitude(int x, int y, int w, int h)
    {
        mag_img.draw(x, y, w, h);
    }
    
    ofPixels& getFlowPixelsRef()
    {
        return mag_img.getPixels();
    }
    
    ofPixels& getColorFlowPixelsRef()
    {
        return flow_color_img.getPixels();
    }
    
    void drawColorFlow(int x, int y, int w, int h)
    {
        flow_color_img.draw(x, y, w, h);
    }
    
    double getMaximum()
    {
        return mag_max;
    }
    
    cv::Point getMaximumPoint()
    {
        return cv::Point(pt_max.x/ratio, pt_max.y/ratio);
    }
    
    double getFlowEntropy()
    {
        cv::Mat magnitude(mag_img.getCvImage());
        return getEntropy(magnitude);
    }
    
    double getFlowMaxForROI(int x, int y, int w, int h)
    {
        x*=ratio;
        y*=ratio;
        w*=ratio;
        h*=ratio;
        IplImage *img = mag_img.getCvImage();
        CvRect old_roi = cvGetImageROI(img);
        cvSetImageROI(img, cvRect(x,y,w,h));
        double minval, maxval;
        cvMinMaxLoc(img, &minval, &maxval);
        cvSetImageROI(img,old_roi);
        return maxval;
    }

    
    double getFlowMeanForROI(int x, int y, int w, int h)
    {
        x*=ratio;
        y*=ratio;
        w*=ratio;
        h*=ratio;
        IplImage *img = mag_img.getCvImage();
        CvRect old_roi = cvGetImageROI(img);
        cvSetImageROI(img, cvRect(x,y,w,h));
        CvScalar c = cvAvg(img);
        cvSetImageROI(img,old_roi);
        return c.val[0];
    }
    
    double getFlowMean()
    {
        cv::Mat magnitude(mag_img.getCvImage());
        return cv::mean(magnitude)[0];
    }
    
    double getFlowDeviance()
    {
        cv::Scalar mean, dev;
        cv::Mat magnitude(mag_img.getCvImage());
        cv::meanStdDev(magnitude, mean, dev);
        return dev[0];
    }
    
    double getFlowDevianceForROI(int x, int y, int w, int h)
    {
        x*=ratio;
        y*=ratio;
        w*=ratio;
        h*=ratio;
        
        cv::Mat magnitude(mag_img.getCvImage());
        cv::Mat roi = magnitude(cv::Rect(x,y,w,h));
        cv::Scalar mean, dev;
        cv::meanStdDev(roi, mean, dev);
        return dev[0];
    }
    
    void computeFlowDevianceImage(int kernel_size = 15)
    {
        assert(kernel_size < width_rsz && kernel_size < height_rsz);
        int radius = ceil(kernel_size / 2.0);
        
        cv::Mat devMat(dev_img.getCvImage());
        
        for (int i = radius; i < width_rsz - radius; i++) {
             for (int j = radius; j < height_rsz - radius; j++) {
                 devMat.at<float>(i,j) = getFlowDevianceForROI(i - radius, j - radius, kernel_size, kernel_size);
             }
        }
        
        dev_img.flagImageChanged();
        
    }
    
    void drawFlowDeviance(int x, int y, int w, int h)
    {
        dev_img.draw(x, y, w, h);
    }
    
private:
    
    
    
    float getEntropy(cv::Mat img)
    {
        cv::Mat hist;
        
        int channels[] = {0};
        int hist_size[] = {32};
        float range[] = { 0, 1 };
        const float* ranges[] = { range };
        
        calcHist( &img, 1, channels, cv::Mat(), // do not use mask
                 hist, 1, hist_size, ranges,
                 true, // the histogram is uniform
                 false );
        
        cv::Mat hist_norm = hist / (img.rows * img.cols);
        
        float entropy = 0.0;
        for( int i = 0; i < hist_norm.rows; i++ )
        {
            float Hc = hist_norm.at<float>(i,0);
            entropy += -Hc * log10(Hc + 0.0001);
        }
        
        return entropy;
    }
    
    ofxCvColorImage         color_img, color_img_rsz, flow_color_img;
    ofxCvGrayscaleImage     gray_img;
    vector<ofxCvGrayscaleImage>     prev_gray_imgs;
    ofxCvFloatImage         mag_img, mag_img_crop, dev_img, dev_imgRsz;
    cv::Mat                 flow_img;
    cv::Mat                 _hsv[3], hsv;
    cv::Mat                 rgb, rgb8;
    cv::Mat                 xy[2], angle, magnitude;
    cv::Mat                 hist_OMG, spec_HOMG, p_spec_HOMG, spec_HOMG_img8, flow_entropy;
    int                     num_spectra, num_freq;
    cv::tvl1flow            flow;
    cv::Point               pt_max;
    int                     num_prev_imgs;
    double                  mag_max;
    double                  mag_max_avg;
    int                     hist_size, frame_i;
    
    double                  ratio;
    
    int                     width, height, width_rsz, height_rsz;
    
    bool                    b_allocated;
};

