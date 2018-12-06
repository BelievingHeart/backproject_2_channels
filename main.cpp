#include <math.h>                        // for floor
#include <opencv2/core/cvdef.h>          // for MAX
#include <iostream>                      // for operator<<, basic_ostream, cout
#include <opencv2/core.hpp>              // for merge, normalize
#include <opencv2/highgui.hpp>           // for imshow, namedWindow, createT...
#include <opencv2/imgproc.hpp>           // for calcHist, calcBackProject

const char *winName_1 = "histogram";
const char *winName_2 = "backprojection";
cv::Mat hsv, hist, hist_color, hsv_back, show_back, show_back_1, show_back_2,
    blank;
std::vector<cv::Mat> images;
std::vector<cv::Mat> images_back;
std::vector<cv::Mat> show_back_images;
int bins = 10;
static void drawHist_and_backproject(int, void *);

int main(const int argc, const char *argv[]) {
  cv::CommandLineParser parser(argc, argv,
                               "{@image_1|/home/afterburner/Pictures/"
                               "pikachu1.png|image to draw histogram}"
                               "{@image_2|/home/afterburner/Pictures/"
                               "pikachu2.jpeg|image to be backprojected}");
  const auto image_name_1 = parser.get<cv::String>("@image_1");
  const auto image_name_2 = parser.get<cv::String>("@image_2");
  const cv::Mat origin = cv::imread(image_name_1, cv::IMREAD_COLOR);
  const cv::Mat backprojected = cv::imread(image_name_2, cv::IMREAD_COLOR);
  if (origin.empty()) {
    std::cout << "Error loading " << image_name_1 << '\n';
    return 1;
  }
  if (backprojected.empty()) {
    std::cout << "Error loading " << image_name_2 << '\n';
    return 1;
  }

  cv::namedWindow(winName_1, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(winName_2, cv::WINDOW_AUTOSIZE);
  cv::createTrackbar("bins", winName_1, &bins, 20, drawHist_and_backproject);
  cv::cvtColor(origin, hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(backprojected, hsv_back, cv::COLOR_BGR2HSV);
  images.push_back(std::move(hsv));
  images_back.push_back(std::move(hsv_back));
  drawHist_and_backproject(0, nullptr);
  cv::waitKey(0);
  return 0;
}

void drawHist_and_backproject(int, void *) {
  //    draw hist image
  bins = MAX(2, bins);
  hist.create(cv::Size(255, 180), CV_8U);
  const std::vector<int> channels{0, 1};
  const std::vector<int> histSize{bins, bins};
  const std::vector<float> ranges{0, 179, 0, 255};
  cv::Mat histMap;
  cv::calcHist(images, channels, cv::Mat(), histMap, histSize, ranges, false);
  cv::normalize(histMap, histMap, 0, 255, cv::NORM_MINMAX, CV_8U,
                cv::noArray());
  const int bin_width = floor(255.0 / bins);
  const int bin_height = floor(180.0 / bins);
  for (int i = 0; i < bins; i++) {
    auto roi = hist(cv::Range(i * bin_height, (i + 1) * bin_height),
                    cv::Range(i * bin_width, (i + 1) * bin_width));
    roi = histMap.at<uchar>(i, i);
  }
  cv::applyColorMap(hist, hist_color, cv::COLORMAP_JET);
  //  backprojection, NOTE: this is irrelevant with the above hist image
  const std::vector<int> channels_1{0};
  const std::vector<int> histSize_1{bins};
  const std::vector<float> ranges_1{0, 179};
  cv::Mat histMap_1;
  cv::calcHist(images, channels_1, cv::Mat(), histMap_1, histSize_1, ranges_1,
               false);
  cv::calcBackProject(images_back, channels_1, histMap_1, show_back_1, ranges_1,
                      1.0);

  const std::vector<int> channels_2{1};
  const std::vector<int> histSize_2{bins};
  const std::vector<float> ranges_2{0, 255};
  cv::Mat histMap_2;
  cv::calcHist(images, channels_2, cv::Mat(), histMap_2, histSize_2, ranges_2,
               false);
  cv::calcBackProject(images_back, channels_2, histMap_2, show_back_2, ranges_2,
                      1.0);
  blank.create(show_back_1.size(), show_back_1.type());
  show_back_images = {show_back_1, show_back_2, blank};
  cv::merge(show_back_images, show_back);
  cv::imshow(winName_2, show_back);
  cv::imshow(winName_1, hist_color);
}