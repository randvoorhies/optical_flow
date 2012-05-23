#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

class OpticalFlow
{
  public:
    OpticalFlow();
    ~OpticalFlow();

  protected:
    void imageCallback(sensor_msgs::ImageConstPtr const & input_img_ptr);

  private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    cv::Mat key_image_;
    std::vector<cv::Point2f> key_corners_;
};

// ######################################################################
OpticalFlow::OpticalFlow() :
  it_(nh_)
{
  // Subscriptions/Advertisements
  image_sub_ = it_.subscribe("image", 1, &OpticalFlow::imageCallback, this);
  cv::namedWindow("input");
}

// ######################################################################
OpticalFlow::~OpticalFlow() 
{ 
}

// ######################################################################
void drawFeatures(cv::Mat & image, std::vector<cv::Point2f> const & old_corners, std::vector<cv::Point2f> const & new_corners)
{
  assert(old_corners.size() == new_corners.size());
  for(size_t i=0; i<new_corners.size(); ++i)
  {
    cv::rectangle(image, old_corners[i]-cv::Point2f(3,3), old_corners[i]+cv::Point2f(2,2), 255, 5);
    cv::rectangle(image, old_corners[i]-cv::Point2f(3,3), old_corners[i]+cv::Point2f(3,3), 0,   1);

    cv::rectangle(image, new_corners[i]-cv::Point2f(3,3), new_corners[i]+cv::Point2f(2,2), 0,   5);
    cv::rectangle(image, new_corners[i]-cv::Point2f(3,3), new_corners[i]+cv::Point2f(3,3), 255, 1);

    cv::line(image, old_corners[i], new_corners[i], 0,   5); 
    cv::line(image, old_corners[i], new_corners[i], 255, 1); 
  }
}

// ######################################################################
void trackFeatures(cv::Mat key_image, cv::Mat curr_image, std::vector<cv::Point2f> & corners, std::vector<cv::Point2f> & new_corners)
{
  new_corners.resize(corners.size());

  if(corners.size() == 0) return;

  std::vector<unsigned char> status(corners.size());
  std::vector<float> error(corners.size());
  calcOpticalFlowPyrLK(key_image, curr_image, corners, new_corners, status, error);

  //std::vector<cv::Point2f> corners_filt;     corners_filt.reserve(corners.size());
  //std::vector<cv::Point2f> new_corners_filt; new_corners_filt.reserve(corners.size());

  //std::vector<cv::Point2f>::iterator corners_it = corners.begin();
  //std::vector<cv::Point2f>::iterator new_corners_it = new_corners.begin();
  //std::vector<unsigned char>::iterator status_it = status.begin();

  //// Filter out bad tracks
  //while(status_it != status.end())
  //{
  //  if(*status_it > 0)
  //  {
  //    corners_filt.push_back(*corners_it);
  //    new_corners_filt.push_back(*new_corners_it);
  //  }
  //  ++corners_it;
  //  ++new_corners_it;
  //  ++status_it;
  //}
  //corners     = corners_filt;
  //new_corners = new_corners_filt;

}

// ######################################################################
void OpticalFlow::imageCallback(sensor_msgs::ImageConstPtr const & input_img_ptr)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(input_img_ptr, sensor_msgs::image_encodings::MONO8);

    cv::Mat input_image = cv_ptr->image;

    // First frame is the keyframe for now
    if(key_corners_.size() == 0)
    {
      cv::goodFeaturesToTrack(input_image, key_corners_, 100, 0.01, 15);
      key_image_ = input_image;
    }

    std::vector<cv::Point2f> new_corners;
    trackFeatures(key_image_, input_image, key_corners_, new_corners);

    drawFeatures(input_image, key_corners_, new_corners);

    cv::imshow("input", input_image);
    cv::waitKey(2);
  }
  catch(cv_bridge::Exception & e)
  {
    ROS_ERROR("%s:%d: Exception: %s", __FILE__, __LINE__, e.what());  
    return;
  }
}

// ######################################################################
int main(int argc, char ** argv)
{
  ros::init(argc, argv, "contrast_enhancer");
  OpticalFlow of;
  ros::spin();
  return 0;
}

