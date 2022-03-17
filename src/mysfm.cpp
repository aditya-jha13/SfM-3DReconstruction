#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include <cassert>

#define dbgLine std::cerr<<"LINE:"<<__LINE__<<"\n"
#define dbg(x) std::cerr<<(#x)<<" is "<< x <<"\n"


const int DS_FACTOR = 10;
const double FOCAL_LENGTH = 4308/DS_FACTOR ; 
const int MIN_LANDMARK_SEEN = 3;

const std::string IMAGE_PATH="/home/aditya/SfM/desk/";

const std::vector<std::string> IMAGES = {
    "DSC02638.JPG",
    "DSC02639.JPG",
    "DSC02640.JPG",
    "DSC02641.JPG",
    "DSC02642.JPG"
};

class ImgPose
{
    public:
        cv::Mat img, desc;
        std::vector<cv::KeyPoint> kp;
        
        cv::Mat T;
        cv::Mat P;

        using kp_idx_t = size_t;
        using landmark_idx_t = size_t;
        using img_idx_t = size_t;

        std::map<kp_idx_t, std::map<img_idx_t, kp_idx_t>> kp_matches; // keypoint matches in other images
        std::map<kp_idx_t, landmark_idx_t> kp_landmark; // keypoint to 3d points

        // helper
        kp_idx_t& kp_match_idx(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx][img_idx]; };
        bool kp_match_exist(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx].count(img_idx) > 0; };

        landmark_idx_t& kp_3d(size_t kp_idx) { return kp_landmark[kp_idx]; };
        bool kp_3d_exist(size_t kp_idx) { return kp_landmark.count(kp_idx) > 0; };

};

struct Landmark
{
    cv::Point3f pt;
    int seen = 0;
};
std::vector<ImgPose> imgPoses;
std::vector<Landmark> landmark;

int main(int argc, char **argv)
{
    
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        for(auto image:IMAGES){
            ImgPose imgPose;
            std::cout<<"Here";
            cv::Mat img = cv::imread(IMAGE_PATH + image,cv::IMREAD_GRAYSCALE);
            assert(!img.empty());
            // cv::imshow("img",img);
            // cv::waitKey();
            //cv::resize(img, img, img.size()/DS_FACTOR);

            imgPose.img = img;

            detector->detect(imgPose.img,imgPose.kp);
            detector->compute(imgPose.img,imgPose.kp,imgPose.desc);
        
            imgPose.desc.convertTo(imgPose.desc,CV_32F);
        
            imgPoses.emplace_back(imgPose);
        }
    
    // cv::Ptr<cv::ORB> detector = cv::ORB::create();
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
 
        size_t N=imgPoses.size();
        // std::cout<<N;
        // int cnt=0;
        for( size_t i = 0; i<(N-1);i++){
            ImgPose& imgPose_i = imgPoses[i];

            for(size_t j = i+1; j<N; j++){
                ImgPose& imgPose_j = imgPoses[j];

                std::vector<std::vector<cv::DMatch>> matches;
                std::vector<cv::Point2f> src,dst;
                std::vector<uchar> mask;
                std::vector<int> i_kp, j_kp;

                matcher->knnMatch(imgPose_i.desc,imgPose_j.desc, matches,2);

                std::vector<cv::DMatch> good_matches;

                for(auto & match:matches){
                    if(match[0].distance<0.7*match[1].distance){

                        src.push_back(imgPose_i.kp[match[0].queryIdx].pt);
                        dst.push_back(imgPose_j.kp[match[0].trainIdx].pt);

                        i_kp.push_back(match[0].queryIdx);
                        j_kp.push_back(match[0].trainIdx);
                        good_matches.push_back(match[0]);
                    }
                }
                
                cv::Mat img_match;
                cv::drawMatches(imgPose_i.img,imgPose_i.kp,imgPose_j.img,imgPose_j.kp,good_matches,img_match,cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
                cv::findFundamentalMat(src, dst, cv::FM_RANSAC, 3.0, 0.99, mask);
                
                cv::Mat canvas = imgPose_i.img.clone();
                canvas.push_back(imgPose_j.img.clone());

                for(size_t k=0; k<mask.size();k++){
                    if(mask[k]){
                        imgPose_i.kp_match_idx(i_kp[k],j) = j_kp[k];
                        imgPose_j.kp_match_idx(j_kp[k], i) = i_kp[k];

                        line(canvas, src[k],dst[k] + cv::Point2f(0,imgPose_i.img.rows),cv::Scalar(0,0,255),2);
                    }
                }
                int good_ = cv::sum(mask)[0];
                assert(good_>=10);
                std::cout<<"Feature Match"<< i <<" "<<j<< " =>" <<good_<<"/"<<matches.size()<<std::endl;
                cv::resize(canvas,canvas,canvas.size()/2);
                cv::imshow("img", canvas);
                cv::waitKey(0);
                
            }
        }
    


        double cx = imgPoses[0].img.size().width/2;
        double cy = imgPoses[0].img.size().height/2;
        double fx = FOCAL_LENGTH;
        double fy = FOCAL_LENGTH;

        cv::Point2d pp(cx,cy);

        cv::Mat K = cv::Mat::eye(3,3, CV_64F);
        // std::cout<<K.rows;
        K.at<double>(0,0) = fx;
        K.at<double>(1,1) = fy;
        K.at<double>(0,2) = cx;
        K.at<double>(1,2) = cy;

        std::cout<<std::endl;
        std::cout<<"Inital Camera Matrix K "<< std::endl << K << std::endl;
      
        imgPoses[0].T = cv::Mat::eye(4,4, CV_64F);
        imgPoses[0].P = K*(cv::Mat::eye(3,4, CV_64F));
        //   dbgLine;
        int check = 0;
        for(size_t i=0; i<(imgPoses.size()-1); i++){

            check ++;
            ImgPose& prev = imgPoses[i];
            ImgPose& curr = imgPoses[i+1];
            //   dbgLine;
            std::vector<cv::Point2f> src, dst;
            std::vector<size_t> kp_used;
            for(size_t k=0; k<prev.kp.size();k++){
                if(prev.kp_match_exist(k,i+1)){
                    size_t match_idx = prev.kp_match_idx(k,i+1);
                    src.push_back(prev.kp[k].pt);
                    dst.push_back(curr.kp[match_idx].pt);
                    kp_used.push_back(k);
                }

            }
            //   dbgLine;
            cv::Mat mask;
            //   dbgLine;
            cv::Mat E =cv::findEssentialMat(dst, src, FOCAL_LENGTH, pp, cv::RANSAC, 0.999,1.0,mask);
            //   dbgLine;
            cv::Mat local_R, local_t;
            cv::recoverPose(E, dst, src, local_R, local_t, FOCAL_LENGTH ,pp, mask);

            cv::Mat T = cv::Mat::eye(4,4, CV_64F);
            //   dbgLine;
            local_R.copyTo(T(cv::Range(0,3), cv::Range(0,3)));
            local_t.copyTo(T(cv::Range(0,3), cv::Range(3,4)));
            
            curr.T = prev.T*T;

            cv::Mat R = curr.T(cv::Range(0,3), cv::Range(0,3));
            cv::Mat t = curr.T(cv::Range(0,3), cv::Range(3,4));


            cv::Mat P(3,4,CV_64F);
            //   dbgLine;
            P(cv::Range(0,3),cv::Range(0,3)) = R.t();
            P(cv::Range(0,3),cv::Range(0,3)) = -R.t()*t;
            P = K*P;

            curr.P = P;

            cv::Mat points4D; //4D homogneous coordinate
            
            cv::triangulatePoints(prev.P, curr.P, src, dst, points4D);
            //   dbgLine;
            if(i>0){
                double scale = 0;
                int count =0;
            

                cv::Point3f prev_camera; 

                prev_camera.x = prev.T.at<double>(0,3);
                prev_camera.y = prev.T.at<double>(1,3);
                prev_camera.z = prev.T.at<double>(2,3);

                std::vector<cv::Point3f> new_points;
                std::vector<cv::Point3f> existing_points;
                int N = kp_used.size();
                for(size_t j=0; j<kp_used.size(); j++){
                    size_t k = kp_used[j];
                    if(mask.at<uchar>(j) && prev.kp_match_exist(k,i+1) && prev.kp_3d_exist(k)){
                        cv::Point3f pt3d;
                        float w = points4D.at<float>(3, j);
                        pt3d.x = points4D.at<float>(0,j) /w;
                        pt3d.y = points4D.at<float>(1,j) /w;
                        pt3d.z = points4D.at<float>(2,j) /w;

                        size_t idx = prev.kp_3d(k);
                        cv::Point3f avg_landmark = landmark[idx].pt / (landmark[idx].seen - 1);

                        new_points.push_back(pt3d);
                        existing_points.push_back(avg_landmark);
                    }

                }
                int Z = new_points.size();
                for(size_t j=0 ; j<(Z-1); j++){
                    for(size_t k = j+1; k<Z; k++){
                        double s = cv::norm(existing_points[j] - existing_points[k]) / cv::norm(new_points[j] - new_points[k]);

                        scale += s;
                        count ++;
                    }
                }
                assert(count > 0);
                scale /= count;

                std::cout <<"image " << (i+1) << "+>" << i << "scale=" << scale << "count=" << count << std::endl;

                local_t *= scale;

                cv::Mat T = cv::Mat::eye(4,4, CV_64F);

                local_R.copyTo(T(cv::Range(0,3), cv::Range(0,3)));
                local_t.copyTo(T(cv::Range(0,3), cv::Range(3,4)));

                curr.T = prev.T*T;

                R = curr.T(cv::Range(0,3), cv::Range(0,3)); 
                t = curr.T(cv::Range(0,3), cv::Range(3,4));

                cv::Mat P(3, 4, CV_64F);
                P(cv::Range(0,3), cv::Range(0,3)) = R.t();
                P(cv::Range(0,3), cv::Range(3,4)) = -R.t()*t;
                P = K*P;

                curr.P = P;

                cv::triangulatePoints(prev.P, curr.P, src,dst , points4D);
            }
              dbgLine;
            for (size_t j=0; j < kp_used.size(); j++) {
                if (mask.at<uchar>(j)) {
                    size_t k = kp_used[j];
                    size_t match_idx = prev.kp_match_idx(k, i+1);

                    cv::Point3f pt3d;

                    pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                    pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                    pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);
                    //  dbgLine;
                    if (prev.kp_3d_exist(k)){
                        curr.kp_3d(match_idx) = prev.kp_3d(k);

                        landmark[prev.kp_3d(k)].pt += pt3d;
                        landmark[curr.kp_3d(match_idx)].seen++;
                    } 
                    else{
                        Landmark landmk;

                        landmk.pt = pt3d;
                        landmk.seen = 2;

                        landmark.push_back(landmk);

                        prev.kp_3d(k) =landmark.size() - 1;
                        curr.kp_3d(match_idx) = landmark.size() - 1;
                    }
                    
                }
            }
        }
        //   dbgLine;
    

}
