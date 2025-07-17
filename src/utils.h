#pragma once

#include <iostream>
#include <omp.h>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <optional>

typedef Eigen::Matrix<unsigned char, 4, 1> Vector4uc;

using namespace Eigen;

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector2f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector4f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Vector4uc)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::VectorXf)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix4f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::MatrixXf)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Quaternionf)

#define MINF (-std::numeric_limits<float>::infinity())

constexpr int sgn(float val) noexcept {
    return (0.0f < val) - (val < 0.0f);
}