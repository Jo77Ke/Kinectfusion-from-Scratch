#pragma once

#include "utils.h"

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f position = {MINF, MINF, MINF, MINF};

    // color stored as 4 unsigned char (RGBX)
    cv::Vec4b color = cv::Vec4b(0, 0, 0, 0);
};