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

#include "frame_data.h"
#include <Eigen/Geometry>
#include <vector>

struct IcpParameters {
    int maxIterations;
    float terminationThreshold;
    float maxCorrespondenceDistance;
};

Eigen::Matrix4f estimateCameraPoseICP(
    const FrameData& currentFrame,
    const FrameData& previousFrame, //TODO SWITCH WITH RAYCASTED MODEL
    const Eigen::Matrix4f& initialGuessPose,
    const IcpParameters& params
) {

  // 1. Start with current estimated pose
  Eigen::Matrix4f currentEstimatedPose = initialGuessPose;
  
  for (int iter = 0; iter < params.maxIterations; ++iter) {

    //JtJ * dx = JtE
    Eigen::Matrix<float, 6, 6> JtJ = Eigen::Matrix<float, 6, 6>::Zero(); // 6DoF
    Eigen::Matrix<float, 6, 1> JtE = Eigen::Matrix<float, 6, 1>::Zero();


    // TODO ADD GETTERS?
    // Map data
    const auto* currentVertexMapData = reinterpret_cast<const Vertex*>(currentFrame.getVertexMap().data);
    const auto* currentNormalMapData = reinterpret_cast<const cv::Vec4f*>(currentFrame.getNormalMap().data);

    const auto* previousVertexMapData = reinterpret_cast<const Vertex*>(previousFrame.getVertexMap().data);
    const auto* previousNormalMapData = reinterpret_cast<const cv::Vec4f*>(previousFrame.getNormalMap().data);

    int validCorrespondences = 0;

    // Iterate over pixels
    for (int v = 0; v < currentFrame.getImageHeight(); ++v) {
      for (int u = 0; u < currentFrame.getImageWidth(); ++u) {

        // Current frames point and normal
        const Vertex& currentVertex = currentVertexMapData[v * currentFrame.getImageWidth() + u];
        const Eigen::Vector4f p_k_curr_hom = currentVertex.position; 

        // Check if valid
        if (p_k_curr_hom.x() == MINF) continue;

        // Move the point to the previousFrame's coordinate system
        const Eigen::Vector3f p_k_aligned = (currentEstimatedPose * p_k_curr_hom).head<3>();

        // 2. Find correspondences 
        // Projective data association
        // Assume point at pixel (u,v) in the current frame corresponds to point at the same pixel (u,v) in previous frame. 
        const Vertex& previousVertex = previousVertexMapData[v * previousFrame.getImageWidth() + u];
        const Eigen::Vector4f q_k_prev_hom = previousVertex.position; // Target point q_k
        
        const cv::Vec4f n_k_prev_cv = previousNormalMapData[v * previousFrame.getImageWidth() + u];
        const Eigen::Vector3f n_k_prev = Eigen::Vector3f(n_k_prev_cv[0], n_k_prev_cv[1], n_k_prev_cv[2]); // Target normal n_k

        // Check if they are valid
        if (q_k_prev_hom.x() == MINF || n_k_prev.x() == MINF || n_k_prev.norm() < 1e-6) continue;
        if ((p_k_aligned - q_k_prev_hom.head<3>()).norm() > params.maxCorrespondenceDistance) continue;
        
        // Valid match
        validCorrespondences++;

        // 3. Calculate point-to-plane error (e_k) and Jacobian (J_k) for each correspondence
        // e_k = ((p_k_aligned) - q_k_prev) . n_k_prev
        const Eigen::Vector3f error_vec = p_k_aligned - q_k_prev_hom.head<3>();
        const float e_k = error_vec.dot(n_k_prev);

        // Jacobian
        // J_k = (n_k_prev)^T * [ I | -[p_k_aligned]x ]
        Eigen::Matrix<float, 1, 6> J_k;
        // Change in error when we apply translation (dx, dy, dz)
        J_k.block<1,3>(0,0) = n_k_prev.transpose();  
        // Change in error when we apply rotation (rx, ry, rz)
        J_k.block<1,3>(0,3) = -n_k_prev.transpose() * (Eigen::Matrix3f() <<
                                                             0, -p_k_aligned.z(), p_k_aligned.y(),
                                                             p_k_aligned.z(), 0, -p_k_aligned.x(),
                                                             -p_k_aligned.y(), p_k_aligned.x(), 0).finished();

        // 4. Sum up contributions from all matches to form the JtJ and JtE vector
        JtJ += J_k.transpose() * J_k;
        JtE += J_k.transpose() * e_k;
      }
    }

    // Require minimum correspondences
    if (validCorrespondences < 10) {
      std::cerr << "Too few valid correspondences in ICP" << std::endl;
      return currentEstimatedPose;
    }

    // 5. Solve JtJ * dx = -JtE for dx
    Eigen::Matrix<float, 6, 1> dx = JtJ.ldlt().solve(-JtE);

    // 6. Apply dx to current estimated pose
    Eigen::Matrix4f dT = Eigen::Matrix4f::Identity();
    Eigen::Vector3f delta_translation = dx.head<3>();
    Eigen::Vector3f delta_rotation_axis_angle = dx.tail<3>();

    Eigen::Matrix3f R_inc = Eigen::AngleAxisf(delta_rotation_axis_angle.norm(), delta_rotation_axis_angle.normalized()).toRotationMatrix();

    // Handle very small rotations
    if (delta_rotation_axis_angle.norm() < 1e-6) { 
        R_inc = Eigen::Matrix3f::Identity();
    }

    dT.block<3,3>(0,0) = R_inc;
    dT.block<3,1>(0,3) = delta_translation;

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
