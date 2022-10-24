# Copyright (C) 

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_demo(NAME face_detection_yunet_demo
    SOURCES ${SOURCES}
    HEADERS ${HEADERS}
    DEPENDENCIES monitors models pipelines opencv_objdetect
    OPENCV_VERSION_REQUIRED 4.5.4)
