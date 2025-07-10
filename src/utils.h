#pragma once

#include <omp.h>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

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