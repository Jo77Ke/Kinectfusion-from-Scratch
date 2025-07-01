#pragma once

#include <cmath>
#include "Eigen.h"

float gaussian(float t, float sigma) {
    return std::exp(-(t * t) / (2 * sigma * sigma));
}

void bilateralFilter(float *const depthMap,
                     const unsigned int imageWidth, const unsigned int imageHeight,
                     float *const denoisedDepthMap,
                     const int sigma_s, const int sigma_r
) {
    for (int u1 = 0; u1 < imageWidth; ++u1) {
        for (int v1 = 0; v1 < imageHeight; ++v1) {
            const Eigen::Vector2i pixel1(u1, v1);
            const auto idx1 = u1 + v1 * imageWidth;

            if (depthMap[idx1] == MINF) { // Skip pixels with no depth information
                denoisedDepthMap[idx1] = depthMap[idx1];
                continue;
            }

            float denoisedValue = 0.0f;
            float normalization = 0.0f;

            int halfWidth = static_cast<int>(3 * sigma_s);
            for (int u2 = std::max(u1 - halfWidth, 0); u2 <= std::min(u1 + halfWidth, (int) imageWidth - 1); ++u2) {
                for (int v2 = std::max(v1 - halfWidth, 0);
                     v2 <= std::min(v1 + halfWidth, (int) imageHeight - 1); ++v2) {
                    const Eigen::Vector2i pixel2(u2, v2);
                    const auto idx2 = u2 + v2 * imageWidth;

                    if (depthMap[idx2] == MINF) { // Skip pixels with no depth information
                        continue;
                    }

                    float spatialWeight = gaussian(static_cast<float>((pixel1 - pixel2).lpNorm<2>()), (float) sigma_s);
                    float intensityWeight = gaussian(static_cast<float>(std::abs(depthMap[idx1] - depthMap[idx2])),
                                                     (float) sigma_r);
                    denoisedValue += spatialWeight * intensityWeight * depthMap[idx2];
                    normalization += spatialWeight * intensityWeight;
                }
            }

            denoisedDepthMap[idx1] = (normalization > 1e-5f) ? (denoisedValue / normalization) : depthMap[idx1];
        }
    }
}