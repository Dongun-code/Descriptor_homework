#include <opencv2/opencv.hpp>
#include <iostream>
#include "feature.hpp"


int main()
{
    std::cout << "Press 'r' to change reference frame," << std::endl
            << "'u' to increase min inlier ratio," << std::endl
            << "'d' to decrease min inlier ratio," << std::endl
            << "and 'q' to quit." << std::endl;

    cv::VideoCapture cap(0);
    if(!cap.isOpened())
        std::cout<<"camera out!"<<std::endl;
        // return -1;
    
    cv::Mat frame;
    cap >> frame;

    MatchHandler matcher({"sift","surf", "orb"}, {"bf","flann", "flann"});

    matcher.SetRefImage(frame.clone());
    while(1)
    {
        cap >> frame;
        matcher.MatchImage(frame);
        int key = cv::waitKey(10);
        if(key==int('f') || key==int('F'))
        {
            std::cout << "change reference image" << std::endl;
            matcher.SetRefImage(frame.clone());
        }
        else if(key==int('u') || key==int('U'))
            matcher.ChangeAcceptRatio(0.1f);
        else if(key==int('d') || key==int('D'))
            matcher.ChangeAcceptRatio(-0.1f);
        else if(key==int('q') || key==int('Q'))
            break;
        cv::Mat result = matcher.DrawMatchResult();
        cv::imshow("matches", result);
        cv::waitKey(10);
    }
    return 0;
}
