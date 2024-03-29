#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

typedef cv::Ptr<cv::Feature2D> FeaturePtr;
typedef cv::Ptr<cv::DescriptorMatcher> MatcherPtr;


struct DetectResult
{
    const std::string name;
    cv::Mat image;
    const std::vector<cv::KeyPoint>& keypts;
    cv::Mat descriptors;
};

// Detector holds keypoint detector, descriptor computer and their results
class Detector
{
    FeaturePtr feature;
    std::string name;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypts;
    cv::Mat descriptors;

public:
    Detector(const std::string _name, FeaturePtr _feature)
    {
        name = _name;
        feature = _feature;
    }
    
    static Detector Factory(const std::string name)
    {
        FeaturePtr desc;
        if(name=="sift")
            return Detector(name, cv::SIFT::create());
        else if(name=="surf")
            return Detector(name, cv::xfeatures2d::SURF::create());
        else if(name == "orb")
            return Detector(name, cv::ORB::create());
        else if(name == "kaze")
            return Detector(name, cv::KAZE::create());
        else if(name == "brisk")
            return Detector(name, cv::BRISK::create());
        else
            throw std::string("error");
            
    }
    
    void DetectAndCompute(cv::Mat _image)
    {
        image = _image;
        feature->detectAndCompute(image, cv::Mat(), keypts, descriptors);
    }
    
    DetectResult getResult()
    {
        return {name, image, keypts, descriptors};
    }

    std::string GetName() { return name; }
};


struct MatcherResult
{
    const std::string name;
    const std::vector<cv::DMatch>& matches;
};


class Matcher
{
    MatcherPtr matcher;
    std::string name;
    std::vector<cv::DMatch> matches;
    const int minMathces = 10;

public:
    Matcher(const std::string _name, MatcherPtr _matcher)
    {
        name = _name;
        matcher = _matcher;
    }
    
    static Matcher Factory(const std::string name, const std::string descName)
    {
        MatcherPtr match;
        if(name == "flann")
        {
            if(descName == "orb")
                return Matcher(name, new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(12, 20, 2)));
            else
                return Matcher(name, cv::FlannBasedMatcher::create());        
        }
        else if(name == "bf")
        {
            if(descName=="orb")
                return Matcher(name, cv::BFMatcher::create(cv::NORM_HAMMING));
            else
                return Matcher(name, cv::BFMatcher::create(cv::NORM_L1));
        }
        else
            throw std::string("error");
    }
    
    std::vector<cv::DMatch>& MatchDescriptors(cv::Mat referDesc, cv::Mat inputDesc)
    {

        matcher->match(inputDesc, referDesc, matches);

        std::sort(matches.begin(), matches.end());
        const int numGoodMatches = matches.size() * AcceptRatio();
        matches.erase(matches.begin()+numGoodMatches, matches.end());
        
    }
    
    static float& AcceptRatio()
    {
        static float acceptRatio = 0.5f;
        return acceptRatio;
    }

    MatcherResult getResult()
    {
        return {name, matches};
    }
};

class MatchHandler
{
    std::vector<Detector> referDets;
    std::vector<Detector> inputDets;
    std::vector<Matcher> matchers;
    float acceptRatio;

public:
    // create feature detectors and matchers depending on string inputs
    MatchHandler(const std::vector<std::string> features, 
                 const std::vector<std::string> matcher)
                 : acceptRatio(0.5f)
    {
        assert(features.size() == matcher.size());
        for(const std::string& feat : features)
            referDets.push_back( Detector::Factory(feat) );
        for(const std::string& feat : features)
            inputDets.push_back( Detector::Factory(feat) );
        for(int i=0; i<matcher.size(); i++)
            matchers.push_back(Matcher::Factory(matcher[i], inputDets[i].getResult().name));
    }

    // detect features and compute descriptors on reference image for all feature types
    void SetRefImage(cv::Mat refimg)
    {
        for(auto& det: referDets)
            det.DetectAndCompute(refimg);
    }

    // detect features and compute descriptors on input image for all feature types
    // match input descriptors with reference descriptors
    void MatchImage(cv::Mat inpimg)
    {
        for(auto& det: inputDets)
            det.DetectAndCompute(inpimg);

        for(size_t i=0; i<matchers.size(); i++)
        {
            matchers[i].MatchDescriptors(referDets[i].getResult().descriptors, 
                                         inputDets[i].getResult().descriptors);
        }
    }

    // change minimum inlier ratio in Matcher class
    void ChangeAcceptRatio(float change)
    {
        std::cout<<" Change number: "<<change<<std::endl;
        acceptRatio += change;
        acceptRatio = std::max(std::min(acceptRatio, 1.f), 0.f);
        std::cout<<"Ratio:"<<acceptRatio<<std::endl;
        Matcher::AcceptRatio() = acceptRatio;
    }

    // draw match
    cv::Mat DrawMatchResult(int maxHeight=1000)
    {
        std::vector<cv::Mat> resultImgs;
        for(size_t i=0; i<matchers.size(); i++)
        {
            cv::Mat result = DrawSingleResult(
                referDets[i].getResult(), inputDets[i].getResult(), matchers[i].getResult()
            );
            resultImgs.push_back(result);
        }
        cv::Mat stackedResult;
        cv::vconcat(resultImgs, stackedResult);
        if(stackedResult.rows > maxHeight)
            cv::resize(stackedResult, stackedResult, cv::Size(stackedResult.cols/2, stackedResult.rows/2),0,0, CV_NEON );
        return stackedResult;
    }

    cv::Mat DrawSingleResult(DetectResult refDet, DetectResult inpDet, MatcherResult match)
    {
        cv::Mat result_;
        cv::Mat matchimg;
        int maxheight = 0;
        try
        {
            // The drawMatches Fusion has a high probability of error occurring. 
            // So, I use try, catch function
            cv::drawMatches(inpDet.image, inpDet.keypts, refDet.image, refDet.keypts, match.matches, matchimg );    
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        


        cv::putText(matchimg, inpDet.name, cv::Point(10,30),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar::all(0), 2);

        if(result_.empty())
            result_ = matchimg;
        else
        {
            std::vector<cv::Mat> himgs = {result_, matchimg};
            cv::vconcat(himgs, result_);
        }

        cv::Mat resimg = result_.clone();
        result_.release();
        if(maxheight > 100 && maxheight < resimg.rows)
        {
            float scale = float(maxheight) / float(resimg.rows);
            cv::Size neosize(int(resimg.cols * scale), int(resimg.rows * scale));
            cv::resize(resimg, resimg, neosize);
        }

        return resimg;

    }

};
