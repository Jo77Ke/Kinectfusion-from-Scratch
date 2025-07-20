#pragma once

#include "utils.h"

constexpr float BILATERAL_EPSILON = 1e-5f;

float gaussian(float t, float sigma) {
    return std::exp(-(t * t) / (2 * sigma * sigma));
}

void bilateralFilter(
        const cv::Mat &depthMap,
        cv::Mat &denoisedDepthMap,
        const float sigma_s, const float sigma_r
) {

    CV_DbgAssert(depthMap.type() == CV_32FC1);
    CV_DbgAssert(denoisedDepthMap.type() == CV_32FC1);
    CV_DbgAssert(depthMap.size() == denoisedDepthMap.size());

    const int imageHeight = depthMap.rows;
    const int imageWidth = depthMap.cols;

    int halfWidth = static_cast<int>(3 * sigma_s);

#pragma omp parallel for schedule(dynamic)
    for (int v1 = 0; v1 < imageHeight; ++v1) {
        const float* depthRow = depthMap.ptr<float>(v1);
        float* denoisedRow = denoisedDepthMap.ptr<float>(v1);

        for (int u1 = 0; u1 < imageWidth; ++u1) {
            const Eigen::Vector2i pixel1(u1, v1);
            float depth1 = depthRow[u1];

            if (depth1 == MINF) { // Skip pixels with no depth information
                denoisedRow[u1] = MINF;
                continue;
            }

            float denoisedValue = 0.0f;
            float normalization = 0.0f;

            for (int v2 = std::max(v1 - halfWidth, 0); v2 <= std::min(v1 + halfWidth, (int) imageHeight - 1); ++v2) {
                const float* depthRow2 = depthMap.ptr<float>(v2);
                for (int u2 = std::max(u1 - halfWidth, 0); u2 <= std::min(u1 + halfWidth, (int) imageWidth - 1); ++u2) {
                    const Eigen::Vector2i pixel2(u2, v2);
                    float depth2 = depthRow2[u2];

                    if (depth2 == MINF) { // Skip pixels with no depth information
                        continue;
                    }

                    const auto spatialDistance = static_cast<float>((pixel1 - pixel2).lpNorm<2>());
                    const auto intensityDistance = static_cast<float>(std::abs(depth1 - depth2));

                    const float spatialWeight = gaussian(spatialDistance, (float) sigma_s);
                    const float intensityWeight = gaussian(intensityDistance, (float) sigma_r);
                    const float weight = spatialWeight * intensityWeight;
                    denoisedValue += weight * depth2;
                    normalization += weight;
                }
            }

            denoisedRow[u1] = (normalization > BILATERAL_EPSILON) ? (denoisedValue / normalization)
                                                                                     : depth1;
        }
    }
}
