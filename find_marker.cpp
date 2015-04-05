#include "find_marker.h"

find_marker::find_marker(QWidget *parent)
    : QWidget(parent)
{    
//    img = imread("/home/ameen/img0.jpg", CV_LOAD_IMAGE_COLOR);
//    if(img.empty())
//    {
//        qDebug() << "No Image!";
//        qApp->quit();
//    }    

    colors[0] = Scalar(255,0,0);
    colors[1] = Scalar(0,255,0);
    colors[2] = Scalar(0,0,255);
    colors[3] = Scalar(0,255,255);
    colors[4] = Scalar(255,0,255);
    colors[5] = Scalar(255,255,0);
    colors[6] = Scalar(255,255,255);
    colors[7] = Scalar(0,0,0);

    /*    SolvePnP    */
    markerCorners3d_scale = 50;
    viewCamera_position_z = 10;

    viewMatrix.create(4, 4, CV_32F);
    camMatrix.create(3, 3, CV_32F);
    camMatrix.at<float>(0, 0) = 5.4083532801474939e+02;
    camMatrix.at<float>(0, 1) = 0.0;
    camMatrix.at<float>(0, 2) = 3.5417484645509603e+02;
    camMatrix.at<float>(1, 0) = 0.0;
    camMatrix.at<float>(1, 1) = 5.4101317864311306e+02;
    camMatrix.at<float>(1, 2) = 2.3036713891762045e+02;
    camMatrix.at<float>(2, 0) = 0.0;
    camMatrix.at<float>(2, 1) = 0.0;
    camMatrix.at<float>(2, 2) = 1.0;

    distCoeff.create(5, 1, CV_32F);
    distCoeff.at<float>(0, 0) = -2.8270601352054947e-01;
    distCoeff.at<float>(1, 0) = 5.9650854462160097e-01;
    distCoeff.at<float>(2, 0) = -1.0979399821211730e-04;
    distCoeff.at<float>(3, 0) = 9.2313816287902689e-04;
    distCoeff.at<float>(4, 0) = -4.5340155404527733e+00;    

    m_markerCorners3d.push_back(Point3f(markerCorners3d_scale, markerCorners3d_scale, 0.0));
    m_markerCorners3d.push_back(Point3f(markerCorners3d_scale, -markerCorners3d_scale, 0.0));
    m_markerCorners3d.push_back(Point3f(-markerCorners3d_scale, -markerCorners3d_scale, 0.0));
    m_markerCorners3d.push_back(Point3f(-markerCorners3d_scale, markerCorners3d_scale, 0.0));

    double objectPoints_scale = markerCorners3d_scale;
    objectPoints.push_back(Point3f(objectPoints_scale, objectPoints_scale, 2*objectPoints_scale));
    objectPoints.push_back(Point3f(objectPoints_scale, -objectPoints_scale, 2*objectPoints_scale));
    objectPoints.push_back(Point3f(-objectPoints_scale, -objectPoints_scale, 2*objectPoints_scale));
    objectPoints.push_back(Point3f(-objectPoints_scale, objectPoints_scale, 2*objectPoints_scale));
    objectPoints.push_back(Point3f(objectPoints_scale, objectPoints_scale, 0));
    objectPoints.push_back(Point3f(objectPoints_scale, -objectPoints_scale, 0));
    objectPoints.push_back(Point3f(-objectPoints_scale, -objectPoints_scale, 0));
    objectPoints.push_back(Point3f(-objectPoints_scale, objectPoints_scale, 0));


    cv::Mat marker_detected = imread("marker_detected.jpg", IMREAD_GRAYSCALE);
    if(marker_detected.empty())
    {
        qDebug() << "Defualt Marker Image Error (No Image Loaded)" << endl;
        qApp->quit();
    }
    Marker_Matrix = cv::Mat::zeros(7, 7, CV_8UC1);
    for(int i = 0; i < 7; i++)
        for(int j = 0; j < 7; j++)
        {
            Rect marker_size(i*60, j*60, 60, 60);
            cv::Mat subrect = marker_detected(marker_size);
            int nZ = cv::countNonZero(subrect);
            if (nZ > (30 * 60))
                Marker_Matrix.at<uchar>(j, i) = 1;
        }

    m_markerCorners2d.push_back(cv::Point2f(0, 0));
    m_markerCorners2d.push_back(cv::Point2f(420, 0));
    m_markerCorners2d.push_back(cv::Point2f(420, 420));
    m_markerCorners2d.push_back(cv::Point2f(0, 420));

    cap.open(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    cap.set(CV_CAP_PROP_FPS,30);
//    cap.open("/home/ameen/left_1.avi");
    fps = 0; fps_counter = 0; fps_dev = 50;
    if(!cap.isOpened()) qApp->quit();
    cap >> img;
    set_point.x = 0; set_point.y = 0;
    startTimer(1);
}

void find_marker::timerEvent(QTimerEvent *te)
{
    Q_UNUSED(te);
    //set_time();
    cap >> img;
    imshow("Video", detect(img).frame);
    waitKey(1);
}

find_marker::result_pr find_marker::detect(cv::Mat frame)
{
    cvtColor(frame, gray, COLOR_BGR2GRAY); //CV_BGR2GRAY);
    detectedMarkers.clear();
    possibleMarkers.clear();
    goodMarkers.clear();
    contours.clear();
    contours0.clear();

    /*  Filteration  */
    medianBlur(gray, imgGrayScale, 3);
    adaptiveThreshold(imgGrayScale, imgGrayScale, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV,77 , 7);
//    imshow("gray", imgGrayScale);

    findContours( imgGrayScale, contours0, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
    //qDebug() << "contours0:" << QString::number(contours0.size());
    minContourPointsAllowed = 100;
    for (size_t i = 0; i < contours0.size(); i++)
    {
        int contourSize = contours0[i].size();
        if (contourSize > minContourPointsAllowed)
            contours.push_back(contours0[i]);
    }
    //qDebug() << "contours:" << QString::number(contours.size());
    m_minContourLengthAllowed = 100;
    for (size_t i=0; i < contours.size(); i++)
    {
        // Approximate to a polygon
        double eps = contours[i].size() * 0.05;
        approxPolyDP(Mat(contours[i]), approxCurve, eps, true);
        // We interested only in polygons that contains only four points
        if (approxCurve.size() != 4)
            continue;
        // And they have to be convex
        if (!cv::isContourConvex(approxCurve))
            continue;
        // Ensure that the distance between consecutive points is large enough        
        float minDist = 50000;//std::numeric_limits<float>::max();
        for (int i = 0; i < 4; i++)
        {
            cv::Point side = approxCurve[i] - approxCurve[(i+1)%4];
            float squaredSideLength = side.dot(side);
            minDist = std::min(minDist, squaredSideLength);
        }
        // Check that distance is not very small
        if (minDist < m_minContourLengthAllowed)
            continue;
        // Sort the points in anti-clockwise order
        // Trace a line between the first and second point.
        // If the third point is at the right side, then the points are anti-clockwise
        cv::Point v1 = approxCurve[1] - approxCurve[0];
        cv::Point v2 = approxCurve[2] - approxCurve[0];
        double o = (v1.x * v2.y) - (v1.y * v2.x);
        if (o < 0.0)   //if the third point is in the left side, then sort in anti-clockwise order
            std::swap(approxCurve[1], approxCurve[3]);
        possibleMarkers.push_back(approxCurve);
    }
    //    qDebug() << "possibleMarkers:" << QString::number(possibleMarkers.size());
    // Remove these elements which corners are too close to each other.
    // First detect candidates for removal:
    std::vector< std::pair<int,int> > tooNearCandidates;
    for (size_t i = 0; i < possibleMarkers.size(); i++)
    {
        m1 = possibleMarkers[i];
        //  calculate the average distance of each corner to the nearest corner of the other marker candidate
        for (size_t j = i+1; j < possibleMarkers.size(); j++)
        {
            m2 = possibleMarkers[j];
            float distSquared = 0;
            for (int c = 0; c < 4; c++)
            {
                cv::Point v = m1[c] - m2[c];
                distSquared += v.dot(v);
            }
            distSquared /= 4;
            if (distSquared < 100)
            {
                tooNearCandidates.push_back(std::pair<int,int>(i,j));
            }
        }
    }

    // Mark for removal the element of the pair with smaller perimeter
    std::vector<bool> removalMask (possibleMarkers.size(), false);
    //qDebug() << QString::number(tooNearCandidates.size());
    for (size_t i = 0; i < tooNearCandidates.size(); i++)
    {
        float p1 = 0;
        for(int sum_counter = 0; sum_counter < 4; sum_counter++)
           p1 += norm( possibleMarkers[tooNearCandidates[i].first][sum_counter]- possibleMarkers[tooNearCandidates[i].first][(sum_counter+1)%4]);
        float p2 = 0;
        for(int sum_counter = 0; sum_counter < 4; sum_counter++)
           p1 += norm( possibleMarkers[tooNearCandidates[i].second][sum_counter]- possibleMarkers[tooNearCandidates[i].second][(sum_counter+1)%4]);

        size_t removalIndex;
        if (p1 > p2)
            removalIndex = tooNearCandidates[i].second;
        else
            removalIndex = tooNearCandidates[i].first;
        removalMask[removalIndex] = true;
    }
    // Return candidates
    detectedMarkers.clear();
    for (size_t i = 0; i < possibleMarkers.size(); i++)
        if (!removalMask[i])
            detectedMarkers.push_back(possibleMarkers[i]);

    cv::Mat canonical[detectedMarkers.size()];
    for(size_t i = 0; i < detectedMarkers.size(); i++)
    {
        canonical[i].create(420, 420, CV_8UC3);

        std::vector<Point2f> detectedMarkerCorners2d1;
        detectedMarkerCorners2d1.push_back(cv::Point2f(detectedMarkers[i][0].x, detectedMarkers[i][0].y));
        detectedMarkerCorners2d1.push_back(cv::Point2f(detectedMarkers[i][1].x, detectedMarkers[i][1].y));
        detectedMarkerCorners2d1.push_back(cv::Point2f(detectedMarkers[i][2].x, detectedMarkers[i][2].y));
        detectedMarkerCorners2d1.push_back(cv::Point2f(detectedMarkers[i][3].x, detectedMarkers[i][3].y));

        cv::Mat M = cv::getPerspectiveTransform(detectedMarkerCorners2d1, m_markerCorners2d);
        // Transform image to get a canonical marker image
        cvtColor(frame,imgGrayScale, COLOR_BGR2GRAY);
        cv::warpPerspective(imgGrayScale, canonical[i], M, Size(420, 420));
        //threshold image
        cv::threshold(canonical[i], canonical[i], 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        //QString name_img = QString::fromAscii("img") + QString::number(i);
        //imshow(name_img.toLatin1().data(), canonical[i]);
    }
    //imwrite("/home/amin/marker_detected_test.jpg", canonical[22]);

    for(size_t num = 0; num < detectedMarkers.size(); num++)
    {
        cv::Mat bitMatrix = cv::Mat::zeros(7, 7, CV_8UC1);  /* get information(for each inner square, determine if it is or white) */
        for(int i = 0; i < 7; i++)
            for(int j = 0; j < 7; j++)
            {
                Rect marker_size(i*60, j*60, 60, 60);
                cv::Mat subrect = canonical[num](marker_size);
                int nZ = cv::countNonZero(subrect);
                if (nZ > (30 * 60))
                    bitMatrix.at<uchar>(j, i) = 1;
            }

        int marker_flag = 1;
        for(int i = 0; i < 7; i++)
            for(int j = 0; j < 7; j++)
            {
                if(Marker_Matrix.at<uchar>(j, i) != bitMatrix.at<uchar>(j, i))
                    marker_flag = 0;
            }
        if(!marker_flag)
        {
            marker_flag = 1;
            for(int i = 0; i < 7; i++)
                for(int j = 0; j < 7; j++)
                {
                    if(Marker_Matrix.at<uchar>(i, 6-j) != bitMatrix.at<uchar>(j, i))
                        marker_flag = 0;
                }
            if(!marker_flag)
            {
                marker_flag = 1;
                for(int i = 0; i < 7; i++)
                    for(int j = 0; j < 7; j++)
                    {
                        if(Marker_Matrix.at<uchar>(6-i, j) != bitMatrix.at<uchar>(j, i))
                            marker_flag = 0;
                    }
                if(!marker_flag)
                {
                    marker_flag = 1;
                    for(int i = 0; i < 7; i++)
                        for(int j = 0; j < 7; j++)
                        {
                            if(Marker_Matrix.at<uchar>(6-j, 6-i) != bitMatrix.at<uchar>(j, i))
                                marker_flag = 0;
                        }
                }
            }
        }
        if(marker_flag)
            goodMarkers.push_back(detectedMarkers[num]);
    }
    //qDebug() << "Detected Parallelepipeds:" << QString::number(goodMarkers.size());

    std::vector<cv::Point2f> preciseCorners(4 * goodMarkers.size());
    cvtColor(frame,imgGrayScale, COLOR_BGR2GRAY);
    for (size_t i = 0; i < goodMarkers.size(); i++)
    {
        marker = goodMarkers[i];
        for (int c = 0; c < 4; c++)
            preciseCorners[i * 4 + c] = marker[c];
    }
    if(goodMarkers.size() > 0)
        cornerSubPix(imgGrayScale, preciseCorners, cvSize(5,5), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_ITER,30,0.1));
    /*  copy back  */
    for (size_t i = 0; i < goodMarkers.size(); i++)
    {
        for (int c = 0; c < 4; c++)
            marker[c] = preciseCorners[i * 4 + c];
        goodMarkers[i] =  marker;
    }

    for(size_t k = 0; k < goodMarkers.size(); k++ )           /* Draw the Contours */
        drawContours( frame, goodMarkers, k, colors[k], 3, CV_AA);//, hierarchy, std::abs(0));
    //    qDebug() << QString::number(goodMarkers.size());
    if(goodMarkers.size() > 0)
        qDebug() << sqrt((goodMarkers[0][0].x-goodMarkers[0][1].x)*(goodMarkers[0][0].x-goodMarkers[0][1].x)+
            (goodMarkers[0][0].y-goodMarkers[0][1].y)*(goodMarkers[0][0].y-goodMarkers[0][1].y));
    if(goodMarkers.size() > 0)
    {

        int num = 0;

        /* Law of Cosines */
        double center_x = 0, center_y = 0;
        for(int i = 0; i < 4; i++)
            center_x += goodMarkers[num][i].x;
        center_x /= 4;
        for(int i = 0; i < 4; i++)
            center_y += goodMarkers[num][i].y;
        center_y /= 4;

        //double beta = 0, gama = 0, a = 0, b = 0, c = 0, d = 0, e = 0, u = 0, alfa = 0;
        //int first_corner = 0;
        //for(int i = 0; i < 4; i++)
        //    if(goodMarkers[num][i].x > center_x && goodMarkers[num][i].y)
        //        first_corner = i;
        //for(int j = 0; j < 4; j++)
        //    circle(img, Point(goodMarkers[num][(first_corner+j)%4].x, goodMarkers[num][(first_corner+j)%4].y), 3, colors[j], 2);
        /*     3 __________ 0
                / c      e \
            a  /   \    /   \  d
              /______________\
             2       b        1      */
        //a = sqrt((goodMarkers[num][(first_corner+3)%4].x - goodMarkers[num][(first_corner+2)%4].x) * (goodMarkers[num][(first_corner+3)%4].x - goodMarkers[num][(first_corner+2)%4].x)
        //        + (goodMarkers[num][(first_corner+3)%4].y - goodMarkers[num][(first_corner+2)%4].y) * (goodMarkers[num][(first_corner+3)%4].y - goodMarkers[num][(first_corner+2)%4].y));
        //b = sqrt((goodMarkers[num][(first_corner+2)%4].x - goodMarkers[num][(first_corner+1)%4].x) * (goodMarkers[num][(first_corner+2)%4].x - goodMarkers[num][(first_corner+1)%4].x)
        //        + (goodMarkers[num][(first_corner+2)%4].y - goodMarkers[num][(first_corner+1)%4].y) * (goodMarkers[num][(first_corner+2)%4].y - goodMarkers[num][(first_corner+1)%4].y));
        //c = sqrt((goodMarkers[num][(first_corner+3)%4].x - goodMarkers[num][(first_corner+1)%4].x) * (goodMarkers[num][(first_corner+3)%4].x - goodMarkers[num][(first_corner+1)%4].x)
        //        + (goodMarkers[num][(first_corner+3)%4].y - goodMarkers[num][(first_corner+1)%4].y) * (goodMarkers[num][(first_corner+3)%4].y - goodMarkers[num][(first_corner+1)%4].y));
        //d = sqrt((goodMarkers[num][(first_corner+1)%4].x - goodMarkers[num][first_corner].x) * (goodMarkers[num][(first_corner+1)%4].x - goodMarkers[num][first_corner].x)
        //        + (goodMarkers[num][(first_corner+1)%4].y - goodMarkers[num][first_corner].y) * (goodMarkers[num][(first_corner+1)%4].y - goodMarkers[num][first_corner].y));
        //e = sqrt((goodMarkers[num][(first_corner+2)%4].x - goodMarkers[num][first_corner].x) * (goodMarkers[num][(first_corner+2)%4].x - goodMarkers[num][first_corner].x)
        //        + (goodMarkers[num][(first_corner+2)%4].y - goodMarkers[num][first_corner].y) * (goodMarkers[num][(first_corner+2)%4].y - goodMarkers[num][first_corner].y));
        //qDebug() << QString::number(b);
        //qDebug() << QString::number(a) << QString::number(b) << QString::number(c);
        //beta = acos((a * a + b * b - c * c)/(2 * a * b));
        //gama = acos((d * d + b * b - e * e)/(2 * d * b));
        //gama = 3.1415 - gama;
        //qDebug() << QString::number(beta * 180 / 3.1415) << ","  << QString::number(gama * 180 / 3.1415);
        //beta = (beta + gama) / 2;
        //qDebug() << QString::number(beta * 180 / 3.1415);
        //u = sqrt((goodMarkers[num][3].x - goodMarkers[num][0].x) * (goodMarkers[num][3].x - goodMarkers[num][0].x) +
        //        (goodMarkers[num][3].y - goodMarkers[num][0].y) * (goodMarkers[num][3].y - goodMarkers[num][0].y));

        //double coeff = 0.00133;
        //qDebug() << QString::number(coeff);
        //alfa = atan(coeff * ((a * b * sin(beta)) / (b - u)));
        //qDebug() << QString::number(alfa * 180 / 3.1415);


        /*      SolvePnP       */
        std::vector<Point2f> m;
        m.push_back(Point2f(goodMarkers[num][0].x, goodMarkers[num][0].y));
        m.push_back(Point2f(goodMarkers[num][1].x, goodMarkers[num][1].y));
        m.push_back(Point2f(goodMarkers[num][2].x, goodMarkers[num][2].y));
        m.push_back(Point2f(goodMarkers[num][3].x, goodMarkers[num][3].y));
        cv::Mat Rvec;
        cv::Mat_<float> Tvec(3, 1);
        cv::Mat raux,taux;

        cv::solvePnP(m_markerCorners3d, m, camMatrix, distCoeff,raux,taux);
        raux.convertTo(Rvec,CV_32F);
        taux.convertTo(Tvec ,CV_32F);
        cv::Mat_<float> rotMat(3,3);
        cv::Rodrigues(Rvec, rotMat);

        //qDebug() << QString::number(beta*180/(2*3.14)) << endl;
        //qDebug() << QString::number(Tvec.rows) << QString::number(Tvec.cols);
        //qDebug() << QString::number(Rvec.rows) << QString::number(Rvec.cols);
        //qDebug() << QString::number(cv::determinant(rotMat));
        //qDebug() << QString::number(Rvec.at<float>(0,0)) << QString::number(Rvec.at<float>(1,0))
        //         << QString::number(Rvec.at<float>(2,0));
        //qDebug() << QString::number(rotMat.at<float>(0,0)) << QString::number(rotMat.at<float>(0,1)) << QString::number(rotMat.at<float>(0,2)) << endl
        //         << QString::number(rotMat.at<float>(1,0)) << QString::number(rotMat.at<float>(1,1)) << QString::number(rotMat.at<float>(1,2)) << endl
        //         << QString::number(rotMat.at<float>(2,0)) << QString::number(rotMat.at<float>(2,1)) << QString::number(rotMat.at<float>(2,2)) << endl;
        //qDebug() << QString::number(Tvec.at<float>(0,0)) << QString::number(Tvec.at<float>(1,0))
        //         << QString::number(Tvec.at<float>(2,0));

        for(int row = 0; row < 3; row++)
        {
           for(int col = 0; col < 3; col++)
              viewMatrix.at<float>(row, col) = rotMat.at<float>(row, col);
           viewMatrix.at<float>(row, 3) = Tvec.at<float>(row, 0);
        }
        viewMatrix.at<float>(3, 3) = 1.0f;

        //qDebug() << QString::number(viewMatrix.at<float>(0,0)) << QString::number(viewMatrix.at<float>(0,1))
        //         << QString::number(viewMatrix.at<float>(0,2)) << QString::number(viewMatrix.at<float>(0,3)) << endl
        //         << QString::number(viewMatrix.at<float>(1,0)) << QString::number(viewMatrix.at<float>(1,1))
        //         << QString::number(viewMatrix.at<float>(1,2)) << QString::number(viewMatrix.at<float>(1,3)) << endl
        //         << QString::number(viewMatrix.at<float>(2,0)) << QString::number(viewMatrix.at<float>(2,1))
        //         << QString::number(viewMatrix.at<float>(2,2)) << QString::number(viewMatrix.at<float>(2,3)) << endl
        //         << QString::number(viewMatrix.at<float>(3,0)) << QString::number(viewMatrix.at<float>(3,1))
        //         << QString::number(viewMatrix.at<float>(3,2)) << QString::number(viewMatrix.at<float>(3,3)) << endl;

        cv::Mat_<float> position(4, 1);
        position.at<float>(0,0) = 0;
        position.at<float>(1,0) = 0;
        position.at<float>(2,0) = viewCamera_position_z;
        position.at<float>(3,0) = 1;
        viewPosition = viewMatrix.inv() * position;
//        qDebug() << QString::number(position.at<float>(0,0)) << QString::number(position.at<float>(1,0))
//                 << QString::number(position.at<float>(2,0)) << QString::number(position.at<float>(3,0)) << endl;

        /*   Beta   */
        double Theta = 0, Phi = 0, viewDistance = 0;
        float x=0,y=0,z=0;
        x = viewPosition.at<float>(0,0); y = viewPosition.at<float>(1,0); z = viewPosition.at<float>(2,0);
        Theta = acos(z / (sqrt((x*x)+(y*y)+(z*z))));
        Phi = atan(x / y);
        viewDistance = sqrt((x*x)+(y*y)+(z*z));
//        qDebug() << "Theta:" << QString::number(Theta * 180 / 3.1415f)
//                 << "Phi:" << QString::number(Phi * 180 / 3.1415f)
//                 << "Distance:" << QString::number(viewDistance);

        std::vector<Point2f> imagepoints;
        cv::projectPoints(objectPoints, raux, taux, camMatrix, distCoeff, imagepoints);
        line(frame, Point(imagepoints[0].x, imagepoints[0].y), Point(imagepoints[1].x, imagepoints[1].y), colors[5]);
        line(frame, Point(imagepoints[1].x, imagepoints[1].y), Point(imagepoints[2].x, imagepoints[2].y), colors[5]);
        line(frame, Point(imagepoints[2].x, imagepoints[2].y), Point(imagepoints[3].x, imagepoints[3].y), colors[5]);
        line(frame, Point(imagepoints[3].x, imagepoints[3].y), Point(imagepoints[0].x, imagepoints[0].y), colors[5]);

        line(frame, Point(imagepoints[4].x, imagepoints[4].y), Point(imagepoints[5].x, imagepoints[5].y), colors[5]);
        line(frame, Point(imagepoints[5].x, imagepoints[5].y), Point(imagepoints[6].x, imagepoints[6].y), colors[5]);
        line(frame, Point(imagepoints[6].x, imagepoints[6].y), Point(imagepoints[7].x, imagepoints[7].y), colors[5]);
        line(frame, Point(imagepoints[7].x, imagepoints[7].y), Point(imagepoints[4].x, imagepoints[4].y), colors[5]);

        line(frame, Point(imagepoints[0].x, imagepoints[0].y), Point(imagepoints[4].x, imagepoints[4].y), colors[5]);
        line(frame, Point(imagepoints[1].x, imagepoints[1].y), Point(imagepoints[5].x, imagepoints[5].y), colors[5]);
        line(frame, Point(imagepoints[2].x, imagepoints[2].y), Point(imagepoints[6].x, imagepoints[6].y), colors[5]);
        line(frame, Point(imagepoints[3].x, imagepoints[3].y), Point(imagepoints[7].x, imagepoints[7].y), colors[5]);

        //circle(img, Point(imagepoints[0].x, imagepoints[0].y), 3, colors[4]);
        //circle(img, Point(imagepoints[1].x, imagepoints[1].y), 3, colors[4]);
        //circle(img, Point(imagepoints[2].x, imagepoints[2].y), 3, colors[4]);
        //circle(img, Point(imagepoints[3].x, imagepoints[3].y), 3, colors[4]);
        //circle(img, Point(imagepoints[4].x, imagepoints[4].y), 3, colors[4]);
        //circle(img, Point(imagepoints[5].x, imagepoints[5].y), 3, colors[4]);
        //circle(img, Point(imagepoints[6].x, imagepoints[6].y), 3, colors[4]);
        //circle(img, Point(imagepoints[7].x, imagepoints[7].y), 3, colors[4]);

        set_point.x = 0; set_point.y = 0;
        for(int i = 0 ; i < 4; i++)
        {
            set_point.x += goodMarkers[num][i].x;
            set_point.y += goodMarkers[num][i].y;
        }
        set_point.x /= 4; set_point.y /= 4;
        circle(frame, Point(set_point.x, set_point.y), 4, Scalar(255,0,0), 5);
        set_point.x -= frame.cols/2;
        set_point.y = -set_point.y + frame.rows/2;
    }


//    qDebug() << "Set Point:" << QString::number(set_point.x) << QString::number(set_point.y);
    line(frame, Point(frame.cols/2, 0), Point(frame.cols/2, frame.rows), Scalar(0,255,0));
    line(frame, Point(0, frame.rows/2), Point(frame.cols, frame.rows/2), Scalar(0,255,0));
//    imshow("Contour", frame);
    frame.copyTo(result.frame);
    result.Point = QPoint(set_point.x, set_point.y);
    return result;
    //if(waitKey(27) > 0) qApp->quit();
    //fps += get_time();
    //fps_counter++;
    //if(fps_counter == fps_dev)
    //{
    //    qDebug() << "FPS" << QString::number((double)(1000000/(fps/fps_dev)));
    //    fps_counter = 0; fps = 0;
    //}
}

find_marker::~find_marker()
{    
    qDebug() << "Bye Bye ..." << endl;
}


long find_marker::time_elapsed(timeval &t1, timeval &t2)
{
    long sec, usec;
    sec = t2.tv_sec - t1.tv_sec;
    usec = t2.tv_usec - t1.tv_usec;
    if (usec < 0) {
        --sec;
        usec = usec + USEC_PER_SEC;
    }
    return sec*USEC_PER_SEC + usec;
}

void find_marker::set_time()
{
    struct timezone tz;
    gettimeofday (&start_time, &tz);
}

long find_marker::get_time()
{
    struct timezone tz;
    gettimeofday (&end_time, &tz);
    return time_elapsed(start_time, end_time);
}
