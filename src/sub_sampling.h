#pragma once

#include "utils.h"
#include "frame_data.h"
#include "bilateral_filter.h"
#include "sub_sampling.h"

void buildDepthPyramid(
        int levels, const float sigma_s, float sigma_r,
        std::vector<cv::Mat> &depthPyramid,
        const cv::Mat &rawDepthMap
) {
    depthPyramid.clear();
    depthPyramid.resize(levels);

    depthPyramid[0] = cv::Mat(rawDepthMap.size(), CV_32FC1);
    // Level 0 = filtered depth map
    bilateralFilter(rawDepthMap, depthPyramid[0], sigma_s, sigma_r);

    for (int l = 1; l < levels; ++l) {
        const cv::Mat &prev = depthPyramid[l - 1];
        const int w = prev.cols / 2;
        const int h = prev.rows / 2;

        cv::Mat current = cv::Mat::zeros(h, w, CV_32F);
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                std::vector<float> validValues;
                float center = prev.at<float>(y * 2, x * 2);

                if (center == MINF) continue;

                for (int dy = 0; dy <= 1; ++dy) {
                    for (int dx = 0; dx <= 1; ++dx) {
                        float val = prev.at<float>(y * 2 + dy, x * 2 + dx);
                        if (val != MINF && std::abs(val - center) <= 3.0f * sigma_r) {
                            validValues.push_back(val);
                        }
                    }
                }

                if (!validValues.empty()) {
                    float avg = std::accumulate(validValues.begin(), validValues.end(), 0.0f) /
                                static_cast<float>(validValues.size());
                    current.at<float>(y, x) = avg;
                } else {
                    current.at<float>(y, x) = MINF;
                }
            }
        }

        depthPyramid[l] = current;
    }
}

void buildVertexPyramid(
        const std::vector<cv::Mat> &depthPyramid,
        const CameraSpecifications &cameraSpecs,
        std::vector<cv::Mat> &vertexPyramid
) {
    vertexPyramid.clear();
    vertexPyramid.resize(depthPyramid.size());

    for (size_t l = 0; l < depthPyramid.size(); ++l) {
        const cv::Mat &depth = depthPyramid[l];
        const int h = depth.rows;
        const int w = depth.cols;
        cv::Mat vertexMapLevel = cv::Mat(h, w, CV_32FC4);

#pragma omp parallel for
        for (int v = 0; v < h; ++v) {
            for (int u = 0; u < w; ++u) {
                float d = depth.at<float>(v, u);
                if (d == MINF) {
                    vertexMapLevel.at<cv::Vec4f>(v, u) = cv::Vec4f(MINF, MINF, MINF, MINF);
                    continue;
                }

                Vector3f pos = cameraSpecs.intrinsicsInverse * d * Vector2i(u, v).cast<float>().homogeneous();
                vertexMapLevel.at<cv::Vec4f>(v, u) = cv::Vec4f(pos.x(), pos.y(), pos.z(), 1.0f);
            }
        }

        vertexPyramid[l] = vertexMapLevel;
    }
}

void buildNormalPyramid(
        const std::vector<cv::Mat> &vertexPyramid,
        std::vector<cv::Mat> &normalPyramid
) {
    normalPyramid.clear();
    normalPyramid.resize(vertexPyramid.size());

    for (size_t l = 0; l < vertexPyramid.size(); ++l) {
        const cv::Mat &vertexMap = vertexPyramid[l];
        const int h = vertexMap.rows;
        const int w = vertexMap.cols;

        cv::Mat normalMap = cv::Mat(h, w, CV_32FC4);

#pragma omp parallel for
        for (int v = 0; v < h - 1; ++v) {
            for (int u = 0; u < w - 1; ++u) {
                const auto &center = vertexMap.at<cv::Vec4f>(v, u);
                const auto &right = vertexMap.at<cv::Vec4f>(v, u + 1);
                const auto &down = vertexMap.at<cv::Vec4f>(v + 1, u);

                if (center[0] == MINF || right[0] == MINF || down[0] == MINF) {
                    normalMap.at<cv::Vec4f>(v, u) = cv::Vec4f(MINF, MINF, MINF, MINF);
                    continue;
                }

                Vector3f du = Vector3f(right[0], right[1], right[2]) - Vector3f(center[0], center[1], center[2]);
                Vector3f dv = Vector3f(down[0], down[1], down[2]) - Vector3f(center[0], center[1], center[2]);

                Vector3f n = du.cross(dv).normalized();
                normalMap.at<cv::Vec4f>(v, u) = cv::Vec4f(n.x(), n.y(), n.z(), 1.0f);
            }
        }

        normalPyramid[l] = normalMap;
    }
}

void buildVertexPyramidFromVertexMap(int levels, float sigma_r,  const cv::Mat &vertexMap, std::vector<cv::Mat> &vertexPyramid) {
    vertexPyramid.clear();
    vertexPyramid.resize(levels);

    vertexPyramid[0] = vertexMap.clone();

    for (int l = 1; l < levels; ++l) {
        const cv::Mat &prev = vertexPyramid[l - 1];
        const int w = prev.cols / 2;
        const int h = prev.rows / 2;

        cv::Mat current = cv::Mat(h, w, CV_32FC4);

#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                cv::Vec4f center = prev.at<cv::Vec4f>(y * 2, x * 2);
                if (center[2] == MINF) {
                    current.at<cv::Vec4f>(y, x) = cv::Vec4f(MINF, MINF, MINF, MINF);
                    continue;
                }

                std::vector<cv::Vec3f> validPoints;

                for (int dy = 0; dy <= 1; ++dy) {
                    for (int dx = 0; dx <= 1; ++dx) {
                        int yy = y * 2 + dy;
                        int xx = x * 2 + dx;
                        if (yy >= prev.rows || xx >= prev.cols) continue;

                        cv::Vec4f sample = prev.at<cv::Vec4f>(yy, xx);
                        if (sample[2] != MINF && std::abs(sample[2] - center[2]) < 3.0f * sigma_r) {
                            validPoints.push_back(cv::Vec3f(sample[0], sample[1], sample[2]));
                        }
                    }
                }

                if (!validPoints.empty()) {
                    cv::Vec3f avg(0, 0, 0);
                    for (const auto &pt : validPoints) avg += pt;
                    avg /= static_cast<float>(validPoints.size());
                    current.at<cv::Vec4f>(y, x) = cv::Vec4f(avg[0], avg[1], avg[2], 1.0f);
                } else {
                    current.at<cv::Vec4f>(y, x) = cv::Vec4f(MINF, MINF, MINF, MINF);
                }
            }
        }

        vertexPyramid[l] = current;
    }
}