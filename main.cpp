#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

using cv::Mat;
using cv::KeyPoint;
using cv::Ptr;
using cv::ORB;

using namespace std;
using namespace cv;

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;

inline void detect_and_compute(Mat& img, vector<KeyPoint>& keypoints, Mat& desc)
{
  Ptr<ORB> orb = ORB::create(10000);
  orb->detectAndCompute(img, Mat(), keypoints, desc);
}

inline void match(string type, Mat& desc1, Mat& desc2, vector<DMatch>& matches) {
    matches.clear();
    if (type == "bf") {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        desc_matcher.match(desc1, desc2, matches, Mat());
    }
    if (type == "knn") {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        vector< vector<DMatch> > vmatches;
        desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
        for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
            if (!vmatches[i].size()) {
                continue;
            }
            matches.push_back(vmatches[i][0]);
        }
    }
    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }
}

inline void findKeyPointsHomography(vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2,
        vector<DMatch>& matches, vector<char>& match_mask) {
    if (static_cast<int>(match_mask.size()) < 3) {
        return;
    }
    vector<Point2f> pts1;
    vector<Point2f> pts2;
    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        pts1.push_back(kpts1[matches[i].queryIdx].pt);
        pts2.push_back(kpts2[matches[i].trainIdx].pt);
    }
    findHomography(pts1, pts2, cv::RANSAC, 4, match_mask);
}

// inline void findKeyPointsHomography(vector<Keypoint>)

int main(int argc, char** argv){

  if(argc != 3)
   {
    std::cout <<" Usage: ./bin/association Image1 Image2" << std::endl;
    return -1;
   }

  std::string imgPath = argv[1];
  std::string fragTestPath = argv[2];
  cv::Mat matImg = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
  cv::Mat matFrag = cv::imread(fragTestPath, cv::IMREAD_UNCHANGED);
  cv::Mat descriptor1;
  cv::Mat descriptor2;
  vector<KeyPoint> keypoints1;
  vector<KeyPoint> keypoints2;
  vector<DMatch> matches;

  detect_and_compute(matImg, keypoints1, descriptor1);
  detect_and_compute(matFrag, keypoints2, descriptor2);

  int nbKP1 = 0;
  for (auto kp : keypoints1)
  {
    Vec3b color = matImg.at<Vec3b>(Point(kp.pt.x,kp.pt.y));
    color[0] = 0;
    color[1] = 0;
    color[2] = 255;
    matImg.at<Vec3b>(Point(kp.pt.x,kp.pt.y)) = color;
    nbKP1++;
  }
  std::cout << "Nb KP pour img1 : "<<nbKP1<< '\n';

  int nbKP2 = 0;
  for (auto kp : keypoints2)
  {
    Vec3b color = matFrag.at<Vec3b>(Point(kp.pt.x,kp.pt.y));
    color[0] = 0;
    color[1] = 0;
    color[2] = 255;
    matFrag.at<Vec3b>(Point(kp.pt.x,kp.pt.y)) = color;
    nbKP2++;
  }
  std::cout << "Nb KP pour img2 : "<<nbKP2<< '\n';

  match("bf", descriptor1, descriptor2, matches);

  int nbMatch=0;
  for(auto match : matches)
  {
    nbMatch++;
  }
  std::cout <<"nb Match : "<<nbMatch<< '\n';

  vector<char> match_mask(matches.size(), 1);
  findKeyPointsHomography(keypoints1, keypoints2, matches, match_mask);

   Mat res;
   cv::drawMatches(matImg, keypoints1, matFrag, keypoints2, matches, res, Scalar::all(-1),Scalar::all(-1), match_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imshow("Img1", matImg);
  cv::imshow("Img2", matFrag);
  cv::imshow("Resultat", res);
  cv::waitKey(0);

	return 0;
}
