#ifndef FIND_MARKER_H
#define FIND_MARKER_H

#include <QWidget>
#include <QDebug>
#include <QPoint>
#include <QApplication>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <math.h>
#include <sys/time.h>

#define USEC_PER_SEC 1000000L

using namespace cv;
using namespace std;

class find_marker : public QWidget
{
    Q_OBJECT
    
public:
    find_marker(QWidget *parent = 0);
    ~find_marker();
    Mat img, imgGrayScale, gray;
    cv::Mat Marker_Matrix;
    std::vector<Point2f> m_markerCorners2d;
    std::vector<vector<Point> > contours;
    std::vector<vector<Point> > contours0;
    std::vector<Vec4i> hierarchy;
    std::vector<Point> approxCurve;
    std::vector<Point> m;
    std::vector<Point> m1;
    std::vector<Point> m2;
    std::vector<Point> marker;
    std::vector<vector<Point> > possibleMarkers;
    std::vector<vector<Point> > detectedMarkers;
    std::vector<vector<Point> > goodMarkers;

    cv::Mat camMatrix, viewMatrix, distCoeff;
    cv::Mat_<float> viewPosition;
    std::vector<Point3f> m_markerCorners3d;
    std::vector<Point3f> objectPoints;
    double markerCorners3d_scale;
    int viewCamera_position_z;

    int minContourPointsAllowed, m_minContourLengthAllowed;

    Point2i set_point;
    Scalar colors[8];

    VideoCapture cap;
    long fps;
    char fps_counter, fps_dev;

    struct timeval start_time;
    struct timeval end_time;

    struct result_pr{
        cv::Mat frame;
        QPoint Point;
    };

    struct result_pr result;

    result_pr detect(cv::Mat frame);

protected slots:
    void timerEvent(QTimerEvent *te);

protected:
    long time_elapsed (struct timeval &t1, struct timeval &t2);
    void set_time();
    long get_time();
};

#endif // FIND_MARKER_H
