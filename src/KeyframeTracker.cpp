#include "optical_flow/KeyframeTracker.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

// ###################################################################### 
KeyframeTracker::KeyframeTracker() :
  num_features_(0),
  key_eig_image_(NULL),
  flags_(0),
  termination_criteria_(cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3)),
  param_track_window_(cvSize(15, 15)),
  param_level_(3),
  param_backward_thresh_(10)
{ }

// ###################################################################### 
void KeyframeTracker::setKeyframe(cv::Mat keyframe)
{
  // Keep a copy of the keyframe (for refcounting purposes), and convert it to an IplImage
  keyframe_ = keyframe;
  keyframe_ipl_ = keyframe_;

  // Allocate all of our storage if needed
  CvSize frame_size = cvGetSize(&keyframe_ipl_);
  allocateOnDemand(&key_eig_image_, frame_size, IPL_DEPTH_32F, 1);
  allocateOnDemand(&key_tmp_image_, frame_size, IPL_DEPTH_32F, 1);
  allocateOnDemand(&key_pyramid_,   frame_size, IPL_DEPTH_8U, 1 );
  allocateOnDemand(&new_pyramid_,   frame_size, IPL_DEPTH_8U, 1 );

  // Find the desired features in the keyframe
  num_features_ = param_num_features_;
  cvGoodFeaturesToTrack(&keyframe_ipl_, key_eig_image_, key_tmp_image_,
      key_features_, &num_features_, .01, 10, NULL);

  // Reset the LK flags
  flags_ = 0;
}

// ###################################################################### 
void KeyframeTracker::trackFrame(cv::Mat frame)
{
  IplImage frame_ipl = frame;
  cvCalcOpticalFlowPyrLK(&keyframe_ipl_, &frame_ipl,
      key_pyramid_, new_pyramid_,
      key_features_, new_features_,
      num_features_, param_track_window_, param_level_,
      feature_status_, feature_error_, termination_criteria_, flags_);

  // No need to recalculate the keyframe pyramid until we set a new keyframe
  flags_ = CV_LKFLOW_PYR_A_READY;

  // Filter out any features with a bad status
  for(int i=0; i<num_features_; ++i)
  {
    if(feature_status_[i] != 0) continue;
    std::swap(feature_status_[i], feature_status_[num_features_-1]);
    std::swap(key_features_[i], key_features_[num_features_-1]);
    std::swap(new_features_[i], new_features_[num_features_-1]);
    num_features_--;
  }

  // Perform the backwards LK step
  cvCalcOpticalFlowPyrLK(&frame_ipl, &keyframe_ipl_,
      new_pyramid_, key_pyramid_,
      new_features_, backwards_features_,
      num_features_, param_track_window_, param_level_,
      feature_status_, feature_error_, termination_criteria_, CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY);

  // Filter out any features with a bad status
  for(int i=0; i<num_features_; ++i)
  {
    if(feature_status_[i] == 0) continue;

    double distance = sqrt(
        pow(backwards_features_[i].x - key_features_[i].x, 2) + 
        pow(backwards_features_[i].y - key_features_[i].y, 2));

    if(distance < param_backward_thresh_) continue;

    std::swap(feature_status_[i], feature_status_[num_features_-1]);
    std::swap(key_features_[i], key_features_[num_features_-1]);
    std::swap(new_features_[i], new_features_[num_features_-1]);
    num_features_--;
  }
}

// ######################################################################
void KeyframeTracker::getFeatures(std::vector<cv::Point2f> & key_features, std::vector<cv::Point2f> & new_features)
{
  key_features.resize(num_features_);
  new_features.resize(num_features_);
  for(int i=0; i<num_features_; ++i)
  {
    key_features[i] = cv::Point2f(key_features_[i].x, key_features_[i].y);
    new_features[i] = cv::Point2f(new_features_[i].x, new_features_[i].y);
  }
}

// ######################################################################
void KeyframeTracker::allocateOnDemand(IplImage **img, CvSize size, int depth, int channels)
{
	if(*img == NULL) *img = cvCreateImage(size, depth, channels);
}

