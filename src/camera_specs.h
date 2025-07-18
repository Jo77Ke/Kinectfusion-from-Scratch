#pragma once

#include "utils.h"

struct CameraSpecifications {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    unsigned int imageWidth;
    unsigned int imageHeight;

    Matrix3f intrinsics;
    Matrix3f intrinsicsInverse;

    float minDepthRange = 0.4f;
    float maxDepthRange = 8.0f;

    CameraSpecifications() :
            imageWidth(0),
            imageHeight(0),
            intrinsics(Matrix3f::Identity()),
            intrinsicsInverse(Matrix3f::Identity()) {}

    CameraSpecifications(
            const unsigned int imageWidth, const unsigned int imageHeight,
            const Matrix3f &intrinsics
    ) : imageWidth(imageWidth), imageHeight(imageHeight),
        intrinsics(intrinsics), intrinsicsInverse(intrinsics.inverse()) {}
};