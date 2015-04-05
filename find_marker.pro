#-------------------------------------------------
#
# Project created by QtCreator 2014-01-08T22:05:30
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = find_marker
TEMPLATE = app


SOURCES += main.cpp\
        find_marker.cpp

HEADERS  += find_marker.h

INCLUDEPATH += /usr/local/include \

LIBS +=   /usr/local/lib/libopencv_videostab.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_video.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_ts.a
LIBS +=   /usr/local/lib/libopencv_superres.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_stitching.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_photo.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_objdetect.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_nonfree.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_ml.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_legacy.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_imgproc.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_highgui.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_gpu.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_flann.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_features2d.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_core.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_contrib.so.2.4.9
LIBS +=   /usr/local/lib/libopencv_calib3d.so.2.4.9
