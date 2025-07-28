#pragma once

#include "utils.h"
#include "frame_data.h"
#include "bilateral_filter.h"
#include "camera_specs.h"
#include "output.h"


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
        const cv::Mat &depthMapAtPreviousLevel = depthPyramid[l - 1];
        const int w = depthMapAtPreviousLevel.cols / 2;
        const int h = depthMapAtPreviousLevel.rows / 2;

        cv::Mat depthMapAtLevel = cv::Mat(h, w, CV_32FC1, cv::Scalar(MINF));
#pragma omp parallel for
        for (int v = 0; v < h; ++v) {
            float *depthRow = depthMapAtLevel.ptr<float>(v);
            const float *depthRowPrev = depthMapAtPreviousLevel.ptr<float>(v * 2);
            for (int u = 0; u < w; ++u) {
                std::vector<float> validValues;
                float center = depthRowPrev[u * 2];

                if (center == MINF) {
                    depthRow[u] = MINF;
                    continue;
                }

                for (int dy = 0; dy <= 1; ++dy) {
                    for (int dx = 0; dx <= 1; ++dx) {
                        int yy = v * 2 + dy;
                        int xx = u * 2 + dx;
                        if (yy >= depthMapAtPreviousLevel.rows || xx >= depthMapAtPreviousLevel.cols) continue;

                        float val = depthMapAtPreviousLevel.at<float>(yy, xx);
                        if (val != MINF && std::abs(val - center) <= 3.0f * sigma_r) {
                            validValues.push_back(val);
                        }
                    }
                }

                if (!validValues.empty()) {
                    float avg = std::accumulate(validValues.begin(), validValues.end(), 0.0f) /
                                static_cast<float>(validValues.size());
                    depthRow[u] = avg;
                } else {
                    depthRow[u] = MINF;
                }
            }
        }

        depthPyramid[l] = depthMapAtLevel;
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
        const cv::Mat &depthMapAtLevel = depthPyramid[l];
        const int h = depthMapAtLevel.rows;
        const int w = depthMapAtLevel.cols;
        cv::Mat vertexMapAtLevel = cv::Mat(h, w, CV_8UC(sizeof(Vertex)), cv::Mat::AUTO_STEP);

        float scale = 1.0f / (1.0f * (1 << l));
        Matrix3f scaledIntrinsicsInverse = cameraSpecs.intrinsicsInverse;
        scaledIntrinsicsInverse(0, 0) *= scale;
        scaledIntrinsicsInverse(1, 1) *= scale;
        scaledIntrinsicsInverse(0, 2) = (scaledIntrinsicsInverse(0, 2) + 0.5f) * scale - 0.5f;
        scaledIntrinsicsInverse(1, 2) = (scaledIntrinsicsInverse(1, 2) + 0.5f) * scale - 0.5f;

#pragma omp parallel for
        for (int v = 0; v < h; ++v) {
            const float *depthRow = depthMapAtLevel.ptr<float>(v);
            Vertex *vertexRow = reinterpret_cast<Vertex *>(vertexMapAtLevel.ptr(v));
            for (int u = 0; u < w; ++u) {
                float d = depthRow[u];

                if (d == MINF) {
                    vertexRow[u].position = Vector4f(MINF, MINF, MINF, MINF);
                    continue;
                }

                Vector2i pixel(u, v);
                Vector3f pos = scaledIntrinsicsInverse * (d * pixel.cast<float>().homogeneous());

                vertexRow[u].position = pos.homogeneous();
            }
        }
        vertexPyramid[l] = vertexMapAtLevel;
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
            cv::Vec4f *normalRow = normalMap.ptr<cv::Vec4f>(v);
            const Vertex *vertexRow = reinterpret_cast<const Vertex *>(vertexMap.ptr(v));
            const Vertex *vertexRowBelow = reinterpret_cast<const Vertex *>(vertexMap.ptr(v + 1));
            for (int u = 0; u < w - 1; ++u) {
                const auto &center = vertexRow[u];
                const auto &right = vertexRow[u + 1];
                const auto &down = vertexRowBelow[u];

                if (center.position.x() == MINF || right.position.x() == MINF || down.position.x() == MINF) {
                    normalRow[u] = cv::Vec4f(MINF, MINF, MINF, MINF);
                    continue;
                }

                Vector3f du = right.position.head<3>() - center.position.head<3>();
                Vector3f dv = down.position.head<3>() - center.position.head<3>();

                Vector3f n = du.cross(dv);
                if (n.norm() < 1e-6) {
                    normalRow[u] = cv::Vec4f(MINF, MINF, MINF, MINF);
                    continue;
                }
                n.normalize();

                normalRow[u] = cv::Vec4f(n.x(), n.y(), n.z(), 1.0f);
            }
        }

        // Handle borders
#pragma omp parallel for
        for (int v = 0; v < h; ++v) {
            normalMap.at<cv::Vec4f>(v, w - 1) = cv::Vec4f(MINF, MINF, MINF, MINF);
        }

#pragma omp parallel for
        for (int u = 0; u < w; ++u) {
            normalMap.at<cv::Vec4f>(h - 1, u) = cv::Vec4f(MINF, MINF, MINF, MINF);
        }
        normalPyramid[l] = normalMap;

    }
}

void buildVertexPyramidFromVertexMap(int levels, float sigma_r, const cv::Mat &vertexMap,
                                     std::vector<cv::Mat> &vertexPyramid) {
    vertexPyramid.clear();
    vertexPyramid.resize(levels);

    vertexPyramid[0] = vertexMap.clone();

    for (int l = 1; l < levels; ++l) {
        const cv::Mat &vertexMapAtPreviousLevel = vertexPyramid[l - 1];
        const int w = vertexMapAtPreviousLevel.cols / 2;
        const int h = vertexMapAtPreviousLevel.rows / 2;

        cv::Mat vertexMapAtLevel = cv::Mat(h, w, CV_8UC(sizeof(Vertex)), cv::Mat::AUTO_STEP);

#pragma omp parallel for
        for (int v = 0; v < h; ++v) {
            const Vertex *centerVertexRowPrev = reinterpret_cast<const Vertex *>(vertexMapAtPreviousLevel.ptr(v * 2));
            Vertex *vertexRow = reinterpret_cast<Vertex *>(vertexMapAtLevel.ptr(v));
            for (int u = 0; u < w; ++u) {
                Vertex center = centerVertexRowPrev[u * 2];
                if (center.position.x() == MINF) {
                    vertexRow[u].position = Vector4f(MINF, MINF, MINF, MINF);
                    continue;
                }

                std::vector<Vector3f> validPoints;

                for (int dy = 0; dy <= 1; ++dy) {
                    for (int dx = 0; dx <= 1; ++dx) {
                        int yy = v * 2 + dy;
                        int xx = u * 2 + dx;
                        if (yy >= vertexMapAtPreviousLevel.rows || xx >= vertexMapAtPreviousLevel.cols) continue;

                        const Vertex &sampleVertex = vertexMapAtPreviousLevel.at<const Vertex>(yy, xx);
                        if (
                                sampleVertex.position.x() != MINF && sampleVertex.position.y() != MINF
                                && sampleVertex.position.z() != MINF
                                && (sampleVertex.position - center.position).norm() < 3.0f * sigma_r
                                ) {
                            validPoints.emplace_back(sampleVertex.position.head<3>());
                        }
                    }
                }

                if (!validPoints.empty()) {
                    Vector3f avg(0, 0, 0);
                    for (const auto &pt: validPoints) avg += pt;
                    avg /= static_cast<float>(validPoints.size());
                    vertexRow[u].position = avg.homogeneous();
                } else {
                    vertexRow[u].position = Vector4f(MINF, MINF, MINF, MINF);
                }
            }
        }
        vertexPyramid[l] = vertexMapAtLevel;
    }
}