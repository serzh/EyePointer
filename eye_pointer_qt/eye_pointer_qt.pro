#-------------------------------------------------
#
# Project created by QtCreator 2012-02-11T19:03:08
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = eye_pointer_qt
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp

#INCLUDEPATH += /usr/local/include/

#LIBS += -L/usr/local/lib \
#    -lopencv_core \
#    -lopencv_highgui \
#    -lopencv_imgproc \
#    -lopencv_features2d \
#    -lopencv_calib3d

unix {
    CONFIG += link_pkgconfig
    PKGCONFIG += opencv
}
