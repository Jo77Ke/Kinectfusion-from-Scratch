#pragma once

#include "utils.h"
#include "model.h"
#include "frame_data.h"
#include "vertex.h"
#include <cmath> // For std::isnan, std::isinf

struct IcpParameters {
    int maxIterations;
    float terminationThreshold;
    float maxCorrespondenceDistance;

    IcpParameters(const int maxIter, const float termThresh, const float maxCorrDist)
        : maxIterations(maxIter), terminationThreshold(termThresh), maxCorrespondenceDistance(maxCorrDist) {}
};

Matrix4f estimateCameraPoseICP(
        const cv::Mat &currentVertexMap,
        const CameraSpecifications &cameraSpecs,
        const cv::Mat &previousVertexMap,
        const cv::Mat &previousNormalMap,
        const Matrix4f &initialGuessPose,
        const IcpParameters &params
) {
    const unsigned int imageWidth = cameraSpecs.imageWidth;
    const unsigned int imageHeight = cameraSpecs.imageHeight;

    // 1. Start with current estimated pose
    Matrix4f currentEstimatedPose = initialGuessPose;

    std::cout << "ICP input: currentVertexMap type: " << currentVertexMap.type() << ", rows: " << currentVertexMap.rows << ", cols: " << currentVertexMap.cols << std::endl;
    std::cout << "ICP input: previousVertexMap type: " << previousVertexMap.type() << ", rows: " << previousVertexMap.rows << ", cols: " << previousVertexMap.cols << std::endl;
    std::cout << "ICP input: previousNormalMap type: " << previousNormalMap.type() << ", rows: " << previousNormalMap.rows << ", cols: " << previousNormalMap.cols << std::endl;

    if (initialGuessPose.array().isNaN().any() || initialGuessPose.array().isInf().any()) {
        std::cerr << "ERROR: initialGuessPose contains NaN or Inf values. Returning identity pose." << std::endl;
        return Matrix4f::Identity();
    }
    std::cout << "Initial Guess Pose:\n" << initialGuessPose << std::endl;

    if (currentVertexMap.empty() || previousVertexMap.empty() || previousNormalMap.empty()) {
        std::cerr << "ERROR: One or more input maps to ICP are empty!" << std::endl;
        return currentEstimatedPose;
    }
    if (currentVertexMap.type() != CV_32FC4 || previousVertexMap.type() != CV_32FC4 || previousNormalMap.type() != CV_32FC4) {
        std::cerr << "ERROR: Input maps to ICP have incorrect types. Expected CV_32FC4 (29)." << std::endl;
        std::cerr << "currentVertexMap type: " << currentVertexMap.type() << std::endl;
        std::cerr << "previousVertexMap type: " << previousVertexMap.type() << std::endl;
        std::cerr << "previousNormalMap type: " << previousNormalMap.type() << std::endl;
        return currentEstimatedPose;
    }
    if (currentVertexMap.size() != previousVertexMap.size() || currentVertexMap.size() != previousNormalMap.size()) {
        std::cerr << "ERROR: Input maps to ICP have inconsistent sizes." << std::endl;
        return currentEstimatedPose;
    }


    for (int iter = 0; iter < params.maxIterations; ++iter) {
        std::vector<Matrix<float, 6, 6>> perThreadJtJ(omp_get_max_threads(), Matrix<float, 6, 6>::Zero());
        std::vector<Matrix<float, 6, 1>> perThreadJtE(omp_get_max_threads(), Matrix<float, 6, 1>::Zero());
        std::vector<int> perThreadValid(omp_get_max_threads(), 0);

        long debug_print_counter = 0;
        const long DEBUG_PRINT_LIMIT = 10;

#pragma omp parallel for
        for (int v = 0; v < imageHeight; ++v) {
            int threadId = omp_get_thread_num();

            if (!currentVertexMap.ptr(v) || !previousVertexMap.ptr(v) || !previousNormalMap.ptr(v)) {
                #pragma omp critical
                {
                    if (debug_print_counter < DEBUG_PRINT_LIMIT) {
                        std::cerr << "ERROR: Null row pointer for v=" << v << " in ICP loop. Skipping row." << std::endl;
                        debug_print_counter++;
                    }
                }
                continue;
            }

            const auto *currentVertexRow = currentVertexMap.ptr<const cv::Vec4f>(v);
            const auto *previousVertexRow = previousVertexMap.ptr<const cv::Vec4f>(v);
            const auto *previousNormalRow = previousNormalMap.ptr<const cv::Vec4f>(v);

            for (int u = 0; u < imageWidth; ++u) {
                const cv::Vec4f p_k_curr_hom_cv = currentVertexRow[u];

                bool current_point_invalid = false;
                if (p_k_curr_hom_cv[0] == MINF || std::isnan(p_k_curr_hom_cv[0]) || std::isinf(p_k_curr_hom_cv[0])) current_point_invalid = true;
                if (std::isnan(p_k_curr_hom_cv[1]) || std::isinf(p_k_curr_hom_cv[1])) current_point_invalid = true;
                if (std::isnan(p_k_curr_hom_cv[2]) || std::isinf(p_k_curr_hom_cv[2])) current_point_invalid = true;

                if (current_point_invalid) {
                    #pragma omp critical
                    {
                        if (debug_print_counter < DEBUG_PRINT_LIMIT) {
                            std::cerr << "DEBUG: Invalid current point data at (u,v)=(" << u << "," << v << "): " << p_k_curr_hom_cv << std::endl;
                            debug_print_counter++;
                        }
                    }
                    continue;
                }

                const Vector4f p_k_curr_hom(p_k_curr_hom_cv[0], p_k_curr_hom_cv[1], p_k_curr_hom_cv[2], p_k_curr_hom_cv[3]);

                const Vector3f p_k_aligned = (currentEstimatedPose * p_k_curr_hom).head<3>();

                if (p_k_aligned.array().isNaN().any() || p_k_aligned.array().isInf().any()) {
                    #pragma omp critical
                    {
                        if (debug_print_counter < DEBUG_PRINT_LIMIT) {
                            std::cerr << "ERROR: p_k_aligned is NaN/Inf at (u,v) = (" << u << "," << v << "). Current Point (Homogeneous): " << p_k_curr_hom.transpose() << std::endl;
                            debug_print_counter++;
                        }
                    }
                    continue;
                }

                const cv::Vec4f q_k_prev_hom_cv = previousVertexRow[u];
                // --- DEBUGGING ---
                bool prev_point_invalid = false;
                if (q_k_prev_hom_cv[0] == MINF || std::isnan(q_k_prev_hom_cv[0]) || std::isinf(q_k_prev_hom_cv[0])) prev_point_invalid = true;
                if (std::isnan(q_k_prev_hom_cv[1]) || std::isinf(q_k_prev_hom_cv[1])) prev_point_invalid = true;
                if (std::isnan(q_k_prev_hom_cv[2]) || std::isinf(q_k_prev_hom_cv[2])) prev_point_invalid = true;

                if (prev_point_invalid) {
                    #pragma omp critical
                    {
                        if (debug_print_counter < DEBUG_PRINT_LIMIT) {
                            std::cerr << "DEBUG: Invalid previous point data at (u,v)=(" << u << "," << v << "): " << q_k_prev_hom_cv << std::endl;
                            debug_print_counter++;
                        }
                    }
                    continue;
                }
                // --- END DEBUGGING ---
                const Vector4f q_k_prev_hom(q_k_prev_hom_cv[0], q_k_prev_hom_cv[1], q_k_prev_hom_cv[2], q_k_prev_hom_cv[3]);


                const cv::Vec4f n_k_prev_cv = previousNormalRow[u];
                Vector3f n_k_prev = {n_k_prev_cv[0], n_k_prev_cv[1], n_k_prev_cv[2]};

                // --- DEBUGGING ---
                bool normal_invalid = false;
                if (std::isnan(n_k_prev.x()) || std::isinf(n_k_prev.x())) normal_invalid = true;
                if (std::isnan(n_k_prev.y()) || std::isinf(n_k_prev.y())) normal_invalid = true;
                if (std::isnan(n_k_prev.z()) || std::isinf(n_k_prev.z())) normal_invalid = true;

                if (normal_invalid) {
                    #pragma omp critical
                    {
                        if (debug_print_counter < DEBUG_PRINT_LIMIT) {
                            std::cerr << "DEBUG: Invalid previous normal data at (u,v)=(" << u << "," << v << "): " << n_k_prev.transpose() << std::endl;
                            debug_print_counter++;
                        }
                    }
                    continue;
                }

                float normal_norm = n_k_prev.norm();
                if (normal_norm < 1e-6) {
                    #pragma omp critical
                    {
                        if (debug_print_counter < DEBUG_PRINT_LIMIT) {
                            std::cerr << "DEBUG: Near-zero normal norm at (u,v)=(" << u << "," << v << "). Skipping. Norm: " << normal_norm << std::endl;
                            debug_print_counter++;
                        }
                    }
                    continue;
                }
                n_k_prev.normalize();

                if (n_k_prev.array().isNaN().any() || n_k_prev.array().isInf().any()) {
                    #pragma omp critical
                    {
                        if (debug_print_counter < DEBUG_PRINT_LIMIT) {
                            std::cerr << "ERROR: Normal became NaN/Inf after normalization at (u,v)=(" << u << "," << v << "). Original norm: " << normal_norm << std::endl;
                            debug_print_counter++;
                        }
                    }
                    continue;
                }
                // --- END DEBUGGING ---

                if ((p_k_aligned - q_k_prev_hom.head<3>()).norm() > params.maxCorrespondenceDistance) {
                    #pragma omp critical
                    {
                        if (debug_print_counter < DEBUG_PRINT_LIMIT) {
                            std::cerr << "DEBUG: Correspondence distance too large at (u,v)=(" << u << "," << v << "). Skipping." << std::endl;
                            debug_print_counter++;
                        }
                    }
                    continue;
                }

                ++perThreadValid[threadId];

                const Vector3f error_vec = p_k_aligned - q_k_prev_hom.head<3>();
                const float e_k = error_vec.dot(n_k_prev);

                Matrix<float, 1, 6> J_k;
                J_k.block<1, 3>(0, 0) = n_k_prev.transpose();
                J_k.block<1, 3>(0, 3) = -n_k_prev.transpose() * (Matrix3f() <<
                                                            0, -p_k_aligned.z(), p_k_aligned.y(),
                                                            p_k_aligned.z(), 0, -p_k_aligned.x(),
                                                            -p_k_aligned.y(), p_k_aligned.x(), 0).finished();

                perThreadJtJ[threadId] += J_k.transpose() * J_k;
                perThreadJtE[threadId] += J_k.transpose() * e_k;
            }
        }

        Matrix<float, 6, 6> JtJ = Matrix<float, 6, 6>::Zero();
        Matrix<float, 6, 1> JtE = Matrix<float, 6, 1>::Zero();
        int validCorrespondences = 0;

        for (int i = 0; i < omp_get_max_threads(); i++) {
            JtJ += perThreadJtJ[i];
            JtE += perThreadJtE[i];
            validCorrespondences += perThreadValid[i];
        }

        if (validCorrespondences < 10) {
            std::cerr << "Too few valid correspondences in ICP (Valid: " << validCorrespondences << "). Returning current pose." << std::endl;
            return currentEstimatedPose;
        }

        if (JtJ.array().isNaN().any() || JtJ.array().isInf().any()) {
            std::cerr << "ERROR: JtJ contains NaN or Inf values. Returning current pose." << std::endl;
            return currentEstimatedPose;
        }
        if (JtE.array().isNaN().any() || JtE.array().isInf().any()) {
            std::cerr << "ERROR: JtE contains NaN or Inf values. Returning current pose." << std::endl;
            return currentEstimatedPose;
        }

        if (JtJ.determinant() < 1e-9 && JtJ.determinant() > -1e-9) { // Check if determinant is close to zero
            std::cerr << "WARNING: JtJ matrix is near-singular. Could lead to unstable solution. Determinant: " << JtJ.determinant() << std::endl;
        }

        Matrix<float, 6, 1> dx = JtJ.ldlt().solve(-JtE);

        if (dx.array().isNaN().any() || dx.array().isInf().any()) {
            std::cerr << "ICP: dx contains NaN or Inf values. Returning current pose." << std::endl;
            return currentEstimatedPose;
        }

        Matrix4f dT = Matrix4f::Identity();
        Vector3f delta_translation = dx.head<3>();
        Vector3f delta_rotation_axis_angle = dx.tail<3>();

        Matrix3f R_inc = AngleAxisf(
                delta_rotation_axis_angle.norm(),
                delta_rotation_axis_angle.normalized()
        ).toRotationMatrix();

        if (delta_rotation_axis_angle.norm() < 1e-6) {
            R_inc = Matrix3f::Identity();
        }

        dT.block<3, 3>(0, 0) = R_inc;
        dT.block<3, 1>(0, 3) = delta_translation;

        currentEstimatedPose = currentEstimatedPose * dT;

        if (dx.norm() < params.terminationThreshold) {
            break;
        }
    }

    return currentEstimatedPose;
}
