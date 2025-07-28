/* 
1. Start with current estimated pose
2. Find correspondences
-- For each point in current frame, find a matching point in the previous frame
-- To do this, first transform the current frame points using current estimated pose and
bring them into previous frame's coordinate system
3. Calculate error and Jacobian
-- For each pait of matched points, calculate the point-to-plane and its Jacobian
4. Sum up contributions from all matches to form the JtJ and JtE vector
5. Solve JtJ * dx = -JtE for dx
6. Apply dx to current estimated pose
7. Repeat until dx too small
*/

#pragma once

constexpr size_t MINIMUM_CORRESPONDENCES = 10;

#include "utils.h"
#include "model.h"
#include "frame_data.h"
#include "vertex.h"

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

    for (int iter = 0; iter < params.maxIterations; ++iter) {
        // Iterate over pixels, shared across threads
        std::vector<Matrix<float, 6, 6>> perThreadJtJ(omp_get_max_threads(), Matrix<float, 6, 6>::Zero());
        std::vector<Matrix<float, 6, 1>> perThreadJtE(omp_get_max_threads(), Matrix<float, 6, 1>::Zero());
        std::vector<int> perThreadValid(omp_get_max_threads(), 0);

#pragma omp parallel for
        for (int v = 0; v < imageHeight; ++v) {
            int threadId = omp_get_thread_num();

            const auto *currentVertexRow = reinterpret_cast<const Vertex *>(currentVertexMap.ptr(v));
            const auto *previousVertexRow = reinterpret_cast<const Vertex *>(previousVertexMap.ptr(v));
            const auto *previousNormalRow = reinterpret_cast<const cv::Vec4f *>(previousNormalMap.ptr(v));
            for (int u = 0; u < imageWidth; ++u) {
                // Current frames point and normal
                const Vertex &currentVertex = currentVertexRow[u];
                const Vector4f p_k_curr_hom = currentVertex.position;

                // Check if valid
                if (p_k_curr_hom.x() == MINF || p_k_curr_hom.y() == MINF || p_k_curr_hom.z() == MINF) continue;

                // Move the point to the previousFrame's coordinate system
                const Vector3f p_k_aligned = (currentEstimatedPose * p_k_curr_hom).head<3>();
                if (!p_k_aligned.allFinite() || p_k_aligned.norm() > 1e6) {
                    continue;
                }

                // Find correspondences via projective data association
                // Assume point at pixel (u,v) in the current frame corresponds to point at the same pixel (u,v) in previous frame.
                const Vertex &previousVertex = previousVertexRow[u];
                const Vector4f q_k_prev_hom = previousVertex.position; // Target point q_k

                const cv::Vec4f n_k_prev_cv = previousNormalRow[u];
                const Vector3f n_k_prev = {n_k_prev_cv[0], n_k_prev_cv[1], n_k_prev_cv[2]}; // Target normal n_k

                // Check if they are valid
                if (q_k_prev_hom.x() == MINF || q_k_prev_hom.y() == MINF || q_k_prev_hom.z() == MINF) continue;
                if (n_k_prev.x() == MINF || n_k_prev.y() == MINF || n_k_prev.z() == MINF || n_k_prev.norm() < 1e-6) continue;
                if ((p_k_aligned - q_k_prev_hom.head<3>()).norm() > params.maxCorrespondenceDistance) continue;

                if (std::isnan(p_k_aligned.x()) || std::isnan(n_k_prev.x())) {
                    std::cerr << "Invalid p_k_aligned or n_k_prev at (" << u << ", " << v << ")" << std::endl;
                    std::cerr << "p_k_aligned: " << p_k_aligned.transpose() << std::endl;
                    std::cerr << "n_k_prev: " << n_k_prev.transpose() << std::endl;
                    continue;
                }

                // Valid match
                ++perThreadValid[threadId];

                // 3. Calculate point-to-plane error (e_k) and Jacobian (J_k) for each correspondence
                // e_k = ((p_k_aligned) - q_k_prev) . n_k_prev
                const Vector3f error_vec = p_k_aligned - q_k_prev_hom.head<3>();
                const float e_k = error_vec.dot(n_k_prev);

                // Jacobian
                // J_k = (n_k_prev)^T * [ I | -[p_k_aligned]x ]
                Matrix<float, 1, 6> J_k;
                // Change in error when we apply translation (dx, dy, dz)
                J_k.block<1, 3>(0, 0) = n_k_prev.transpose();
                // Change in error when we apply rotation (rx, ry, rz)
                J_k.block<1, 3>(0, 3) = -n_k_prev.transpose() * (Matrix3f() <<
                        0, -p_k_aligned.z(), p_k_aligned.y(),
                        p_k_aligned.z(), 0, -p_k_aligned.x(),
                        -p_k_aligned.y(), p_k_aligned.x(), 0).finished();

                if (!J_k.allFinite()) {
                    std::cerr << "Invalid Jacobian at (" << u << ", " << v << ")" << std::endl;
                    std::cerr << "J_k: " << J_k << std::endl;
                    continue;
                }

                if (J_k.hasNaN()) {
                    std::cerr << "NaN in J_k at (" << u << ", " << v << ")" << std::endl;
                    std::cerr << "J_k: " << J_k << std::endl;
                    continue;
                }

                // 4. Sum up contributions from all matches to form the JtJ and JtE vector
                perThreadJtJ[threadId] += J_k.transpose() * J_k;
                perThreadJtE[threadId] += J_k.transpose() * e_k;
            }
        }

        // Combine results from all threads
        //JtJ * dx = JtE
        Matrix<float, 6, 6> JtJ = Matrix<float, 6, 6>::Zero(); // 6DoF
        Matrix<float, 6, 1> JtE = Matrix<float, 6, 1>::Zero();
        int validCorrespondences = 0;

        for (int i = 0; i < omp_get_max_threads(); i++) {
            JtJ += perThreadJtJ[i];
            JtE += perThreadJtE[i];
            validCorrespondences += perThreadValid[i];
        }

        // Require minimum correspondences
        if (validCorrespondences < MINIMUM_CORRESPONDENCES) {
            std::cerr << "Too few valid correspondences in ICP" << std::endl;
            return currentEstimatedPose;
        }

        // 5. Solve JtJ * dx = -JtE for dx
        JtJ += Matrix<float, 6, 6>::Identity() * 1e-6; // Regularization to avoid singularity
        Matrix<float, 6, 1> dx = JtJ.bdcSvd(ComputeThinU | ComputeThinV).solve(-JtE);

        // 6. Apply dx to current estimated pose
        Matrix4f dT = Matrix4f::Identity();
        Vector3f delta_translation = dx.head<3>();
        Vector3f delta_rotation_axis_angle = dx.tail<3>();

        Matrix3f R_inc = AngleAxisf(
                delta_rotation_axis_angle.norm(),
                delta_rotation_axis_angle.normalized()
        ).toRotationMatrix();

        // Handle very small rotations
        if (delta_rotation_axis_angle.norm() < 1e-6) {
            R_inc = Matrix3f::Identity();
        }

        dT.block<3, 3>(0, 0) = R_inc;
        dT.block<3, 1>(0, 3) = delta_translation;

        // Apply the transformation to the current estimated pose
        currentEstimatedPose = currentEstimatedPose * dT;

        // Repeat until below threshold
        if (dx.norm() < params.terminationThreshold) {
            std::cout << "ICP converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    return currentEstimatedPose;
}
