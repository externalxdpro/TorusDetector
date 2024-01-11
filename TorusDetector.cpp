#include <fmt/format.h>
#include <fmt/ranges.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

int p8Bit(int num) { return num * 255 / 100; }

class TorusDetectorOptions {
  public:
    cv::Scalar hsvStart = cv::Scalar(0, p8Bit(40), p8Bit(80));
    cv::Scalar hsvEnd = cv::Scalar(15, p8Bit(100), p8Bit(100));

    int blurKernelSize = 7;
};

class TorusDetector {
  public:
    TorusDetectorOptions config;

    TorusDetector(const TorusDetectorOptions &config = TorusDetectorOptions())
        : config(config) {}

    cv::Mat generateColourMask(cv::Mat imageBGR) {
        cv::Mat hsv;
        cv::cvtColor(imageBGR, hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        cv::inRange(hsv, config.hsvStart, config.hsvEnd, mask);
        return mask;
    }

    std::vector<cv::Vec3f> operator()(const cv::Mat &inputImage) {
        cv::Mat mask = generateColourMask(inputImage);
        cv::Mat result;
        cv::bitwise_and(inputImage, inputImage, result, mask);
        cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
        cv::medianBlur(result, result, config.blurKernelSize);

        std::vector<cv::Vec3f> circles;

        cv::HoughCircles(result, circles, cv::HOUGH_GRADIENT, 1,
                         result.rows / 8, 100, 30, 10, 100);

        return circles;
    }
};

int main(int argc, char *argv[]) {
    auto vid = cv::VideoCapture(0);
    auto td = TorusDetector();

    while (true) {
        cv::Mat image;
        vid.read(image);

        std::vector<cv::Vec3f> circles = td(image);

        cv::Scalar blobColour = {255, 0, 0};
        cv::Mat blobs = image.clone();
        int count = 0;

        if (!circles.empty()) {
            count = circles.size();
            for (auto circle : circles) {
                cv::Point center(circle[0], circle[1]);
                cv::circle(blobs, center, 1, blobColour, 3);
                int radius = circle[2];
                cv::circle(blobs, center, radius, blobColour, 3);
            }
        }

        std::string text = fmt::format("Number of Detected Circles: {}", count);
        cv::putText(blobs, text, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1,
                    blobColour, 3);

        cv::imshow("Filtering Circular Blobs Only", blobs);
        if (cv::waitKey(10) == 'q') {
            break;
        }
    }

    vid.release();
    cv::destroyAllWindows();
}
