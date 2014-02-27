#pragma once

#include "tvl1opticalflow.h"
#include "ofxOpenCv.h"

class pkmOpticalFlow
{
public:
    pkmOpticalFlow()
    {
        
    }
    
    void allocate(int w, int h)
    {
        width = w;
        height = h;
        numPrevImgs = 2;
        
        widthRsz = min(256, width);
        ratio = (float)widthRsz / (float)width;
        heightRsz = height * ratio;
        
        histSize = 50;
        
        printf("[pkmOpticalFlow]::allocate - resized: %d x %d -> %d x %d\n", width, height, widthRsz, heightRsz);
        
        colorImg.allocate(width, height);
        colorImgRsz.allocate(widthRsz, heightRsz);
        
        grayImg.allocate(widthRsz, heightRsz);
        
        prevGrayImgs.resize(numPrevImgs);
        for(int i = 0; i < numPrevImgs; i++)
            prevGrayImgs[i].allocate(widthRsz, heightRsz);
        
        flowColorImg.allocate(widthRsz, heightRsz);
        magImg.allocate(widthRsz, heightRsz);
        magImgCrop.allocate(widthRsz - 10, heightRsz - 10);
        
        mag_max_avg = 1.0;
        
        flowImg = cv::Mat(heightRsz, widthRsz, CV_32FC2);
        
        numSpectra = 500;
        numFreq = 180;  // <= than 360
        specHOMG = cv::Mat::zeros(numFreq, numSpectra, CV_32FC1);
        pSpecHOMG = cv::Mat::zeros(numFreq, numSpectra, CV_32FC1);
        
//        flow.setIterations(100);
//        flow.setWarps(2);
//        flow.setScales(2);
//        flow.setEpsilon(0.08);
//        flow.setTau(0.25);
//        flow.setLambda(0.1);
//        flow.setUseInitialFlow(false);
    }
    
    void update(const ofPixelsRef &pixels)
    {
        colorImg.setFromPixels(pixels);
        colorImgRsz.scaleIntoMe(colorImg);
        colorImgRsz.convertRgbToHsv();
        colorImgRsz.convertToGrayscalePlanarImage(grayImg, 2);
        
        cv::Mat I0(grayImg.getCvImage()), I1(prevGrayImgs[0].getCvImage());
        
        flow.calc(I0, I1, flowImg);
        
        // convert to color, thanks to: http://stackoverflow.com/questions/7693561/opencv-displaying-a-2-channel-image-optical-flow
        //extraxt x and y channels
        split(flowImg, xy);
        
        //calculate angle and magnitude
        cv::Mat magnitudeCrop(magImgCrop.getCvImage());
        magnitude = cv::Mat(magImg.getCvImage());
        cartToPolar(xy[0], xy[1], magnitude, angle, true);
        cv::Rect roi(5, 5, magImg.getWidth() - 10, magImg.getHeight() - 10);
        cv::Mat(magnitude, roi).copyTo(magnitudeCrop);
        cv::copyMakeBorder(magnitudeCrop, magnitude,
                           5, 5,
                           5, 5, cv::BORDER_CONSTANT,
                           cvScalar( 0, 0, 0 ));
        magImg.flagImageChanged();
        
        //translate magnitude to range [0;1]
        minMaxLoc(magnitude, NULL, &mag_max, NULL, &pt_max);
        mag_max_avg = mag_max * 0.1 + mag_max_avg * 0.9;
        magnitude.convertTo(magnitude, -1, 1.0/mag_max_avg);
        
        //build hsv image
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magnitude;
        merge(_hsv, 3, hsv);
        
        //convert to BGR and show
        cv::Mat ofxrgb(flowColorImg.getCvImage());
        cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
        
        rgb = rgb * 255.0;
        rgb.convertTo(ofxrgb, CV_8U);
        flowColorImg.flagImageChanged();
        
        prevGrayImgs.push_back(grayImg);
        if(prevGrayImgs.size() > numPrevImgs)
        {
            prevGrayImgs.erase(prevGrayImgs.begin());
        }
    }
    
    void computeHistogramOfOrientedMotionGradients()
    {
        histOMG = cv::Mat::zeros(numFreq, 1, CV_32F);
        for(int i = 0; i < magnitude.rows; i++)
        {
            for(int j = 0; j < magnitude.cols; j++)
            {
                histOMG.at<float>(floor(angle.at<float>(i,j)) * ((float)numFreq/360.0), 0) += magnitude.at<float>(i,j)/sqrtf(magnitude.rows*magnitude.cols);
            }
        }
        
        float heightScalar = 100.0;
        cv::Mat img = cv::Mat::zeros(heightScalar, numFreq, CV_8UC3);
        
        for(int i = 1; i < numFreq; i++)
        {
            cv::line(img,
                     cv::Point(i-1, histOMG.at<float>(i-1,0)*heightScalar),
                     cv::Point(i, histOMG.at<float>(i-1,0)*heightScalar),
                     cv::Scalar(255,255,255));//, 3, 4);
        }
        
        cv::namedWindow("HOMG", CV_WINDOW_FREERATIO);
        cv::imshow("HOMG", img);
        
        
        cv::Mat sourceROI = pSpecHOMG(cv::Rect(1,0,numSpectra-1,numFreq));
        sourceROI.copyTo(specHOMG(cv::Rect(0,0,numSpectra-1,numFreq)));
        histOMG.copyTo(specHOMG(cv::Rect(numSpectra-1,0,1,numFreq)));
        
        cv::Mat img8(specHOMG.rows, specHOMG.cols, CV_8UC1), cmImg(specHOMG.rows, specHOMG.cols, CV_8UC3);
        specHOMG.convertTo(img8,CV_8UC1,255.0);
        cv::applyColorMap(img8, cmImg, cv::COLORMAP_JET);
        
        cv::resize(cmImg, cmImg, cv::Size(512, specHOMG.cols), 0, 0, cv::INTER_NEAREST);
        
        cv::namedWindow("HOMG Spectra", CV_WINDOW_FREERATIO);
        cv::imshow("HOMG Spectra", cmImg);
        
        specHOMG.copyTo(pSpecHOMG);
    }
    
    void drawColorImg(int x, int y, int w, int h)
    {
        colorImg.draw(x,y,w,h);
    }
    
    void drawGrayImg(int x, int y, int w, int h)
    {
        grayImg.draw(x,y,w,h);
    }
    
    void drawPrevGrayImg(int x, int y, int w, int h)
    {
        prevGrayImgs[0].draw(x,y,w,h);
    }
    
    void drawMagnitude(int x, int y, int w, int h)
    {
        magImg.draw(x, y, w, h);
    }
    
    ofPixelsRef getFlowPixelsRef()
    {
        return magImg.getPixelsRef();
    }
    
    ofPixelsRef getColorFlowPixelsRef()
    {
        return flowColorImg.getPixelsRef();
    }
    
    void drawColorFlow(int x, int y, int w, int h)
    {
        flowColorImg.draw(x, y, w, h);
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
        cv::Mat magnitude(magImg.getCvImage());
        return getEntropy(magnitude);
    }
    
    double getFlowMeanForROI(int x, int y, int w, int h)
    {
        x*=ratio;
        y*=ratio;
        w*=ratio;
        h*=ratio;
        IplImage *img = magImg.getCvImage();
        CvRect old_roi = cvGetImageROI(img);
        cvSetImageROI(img, cvRect(x,y,w,h));
        CvScalar c = cvAvg(img);
        cvSetImageROI(img,old_roi);
        return c.val[0];
    }
    
    double getFlowMean()
    {
        cv::Mat magnitude(magImg.getCvImage());
        return cv::mean(magnitude)[0];
    }
    
    double getFlowDeviance()
    {
        cv::Scalar mean, dev;
        cv::Mat magnitude(magImg.getCvImage());
        cv::meanStdDev(magnitude, mean, dev);
        return dev[0];
    }
    
private:
    
    
    
    float getEntropy(cv::Mat img)
    {
        cv::Mat hist;
        
        int channels[] = {0};
        int histSize[] = {32};
        float range[] = { 0, 1 };
        const float* ranges[] = { range };
        
        calcHist( &img, 1, channels, cv::Mat(), // do not use mask
                 hist, 1, histSize, ranges,
                 true, // the histogram is uniform
                 false );
        
        cv::Mat histNorm = hist / (img.rows * img.cols);
        
        float entropy = 0.0;
        for( int i = 0; i < histNorm.rows; i++ )
        {
            float Hc = histNorm.at<float>(i,0);
            entropy += -Hc * log10(Hc + 0.0001);
        }
        
        return entropy;
    }
    
    ofxCvColorImage         colorImg, colorImgRsz, flowColorImg;
    ofxCvGrayscaleImage     grayImg;
    vector<ofxCvGrayscaleImage>     prevGrayImgs;
    ofxCvFloatImage         magImg, magImgCrop;
    cv::Mat                 flowImg;
    cv::Mat                 _hsv[3], hsv;
    cv::Mat                 rgb, rgb8;
    cv::Mat                 xy[2], angle, magnitude;
    cv::Mat                 histOMG, specHOMG, pSpecHOMG;
    int                     numSpectra, numFreq;
    cv::tvl1flow            flow;
    cv::Point               pt_max;
    int                     numPrevImgs;
    double                  mag_max;
    double                  mag_max_avg;
    int                     histSize;
    
    double                  ratio;
    
    int                     width, height, widthRsz, heightRsz;
};

