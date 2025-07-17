#pragma once

#include <utility>
#include <numeric>

#include "utils.h"
#include "camera_specs.h"
#include "vertex.h"
#include "sub_sampling.h"

constexpr float TRUNCATION_RADIUS = 0.05f; // 5cm

class FrameData {
public:
    FrameData() = default;

    FrameData(
            int frameNumber,
            CameraSpecifications cameraSpecs,
            cv::Mat &&depthMap,
            cv::Mat &&colorMap,
            const Matrix4f &trajectory
    ) : frameNumber(frameNumber),
        cameraSpecs(std::move(cameraSpecs)),
        imageWidth(static_cast<int>(cameraSpecs.imageWidth)),
        imageHeight(static_cast<int>(cameraSpecs.imageHeight)),
        depthMap(std::move(depthMap)),
        colorMap(std::move(colorMap)) {
        validityMap = cv::Mat(imageHeight, imageWidth, CV_8UC1);
        validityMap.setTo(0, depthMap == MINF);
        validityMap.setTo(1, depthMap != MINF);
    }

    int getFrameNumber() const { return frameNumber; }

    const CameraSpecifications &getCameraSpecifications() const { return cameraSpecs; }

    const cv::Mat &getRawDepthMap() const { return depthMap; }

    const cv::Mat &getRawColorMap() const { return colorMap; }

    const std::vector<cv::Mat> &getDepthPyramid() const {
        return depthPyramid;
    }

    const std::vector<cv::Mat> &getVertexPyramid() const {
        return vertexPyramid;
    }

    const std::vector<cv::Mat> &getNormalPyramid() const {
        return normalPyramid;
    }

    void setPose(const Matrix4f &newPose) {
        pose = newPose;
    }

    const Matrix4f &getPose() const { return pose; }

    const cv::Mat &getVertexMap() const { return vertexMap; }

    const cv::Mat &getNormalMap() const { return normalMap; }

    void computeVertexMap() {
        vertexMap = cv::Mat(imageHeight, imageWidth, CV_8UC(sizeof(Vertex)), cv::Mat::AUTO_STEP);
        validityMap = cv::Mat(imageHeight, imageWidth, CV_8UC1);

#pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < imageHeight; ++v) {
            const float *depthRow = depthMap.ptr<float>(v);
            const cv::Vec4b *colorRow = colorMap.ptr<cv::Vec4b>(v);
            Vertex *vertexRow = reinterpret_cast<Vertex *>(vertexMap.ptr(v));
            uint8_t *maskRow = validityMap.ptr<uint8_t>(v);
            for (int u = 0; u < imageWidth; ++u) {
                Vertex &vertex = vertexRow[u];

                if (maskRow[u] == 0) {
                    // invalid depth value -> invalid vertex
                    vertex.position = Vector4f(MINF, MINF, MINF, MINF);
                    vertex.color = cv::Vec4b(0, 0, 0, 0);
                    continue;
                }

                Vector3f posInCameraSpace = cameraSpecs.intrinsicsInverse * depthRow[u]
                                            * Vector3f(static_cast<float>(u), static_cast<float>(v), 1.0f);
                vertex.position = pose * posInCameraSpace.homogeneous();
                vertex.color = colorRow[u];

            }
        }
    }

    void computeNormalMap() {
        normalMap = cv::Mat(imageHeight, imageWidth, CV_32FC4);

        // Debug assertions
        CV_DbgAssert(vertexMap.type() == CV_8UC(sizeof(Vertex)));
        CV_DbgAssert(vertexMap.size() == normalMap.size());

#pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < imageHeight - 1; ++v) {
            const cv::Vec4f *vertexRow = vertexMap.ptr<cv::Vec4f>(v);
            const cv::Vec4f *vertexRowBelow = vertexMap.ptr<cv::Vec4f>(v + 1);
            const uint8_t *maskRow = validityMap.ptr<uint8_t>(v);
            const uint8_t *maskRowBelow = validityMap.ptr<uint8_t>(v + 1);

            cv::Vec4f *normalRow = normalMap.ptr<cv::Vec4f>(v);
            for (int u = 0; u < imageWidth - 1; ++u) {
                if (maskRow[u] == 0 || maskRow[u + 1] == 0 || maskRowBelow[u] == 0) {
                    // at least one point invalid -> normal vector invalid
                    normalRow[u] = cv::Vec4f(MINF, MINF, MINF, MINF);
                    continue;
                }

                const Vector3f vertex(
                        vertexRow[u][0],
                        vertexRow[u][1],
                        vertexRow[u][2]
                );
                const Vector3f vertexRight(
                        vertexRow[u + 1][0],
                        vertexRow[u + 1][1],
                        vertexRow[u + 1][2]
                );
                const Vector3f vertexBelow(
                        vertexRowBelow[u][0],
                        vertexRowBelow[u][1],
                        vertexRowBelow[u][2]
                );

                Vector3f du = vertexRight - vertex; // du = V(u+1, v) - V(u,v)
                Vector3f dv = vertexBelow - vertex; // dv = V(u, v+1) - V(u,v)

                Vector3f n = du.cross(dv).normalized();
                normalRow[u] = cv::Vec4f(n.x(), n.y(), n.z(), 1.0f);
            }
        }
    }

    void buildPyramids(int levels, float sigmaS, float sigmaR) {
        buildDepthPyramid(levels, sigmaS, sigmaR, depthPyramid, depthMap);
        buildVertexPyramid(depthPyramid, cameraSpecs, vertexPyramid);
        buildNormalPyramid(vertexPyramid, normalPyramid);
    }

    void computeCameraCenterInGlobalSpace() {
        cameraCenterInGlobalSpace = pose.col(3).head<3>();
    }

    float tsdfValueAt(const Vector3f &globalPoint) const {
        const Vector3f pointInCameraSpace = pose.topLeftCorner<3, 4>() * globalPoint.homogeneous();

        Vector3f projectedPoint = cameraSpecs.intrinsics * pointInCameraSpace;
        projectedPoint /= projectedPoint.z();

        // Check if within image bounds
        if (projectedPoint.x() < 0 || projectedPoint.x() >= static_cast<float>(imageWidth) ||
            projectedPoint.y() < 0 || projectedPoint.y() >= static_cast<float>(imageHeight)) {
            return MINF;
        }

        const Vector3i pixel = projectedPoint.cast<int>(); // Nearest neighbor projection via rounding down

        const float depthValue = depthMap.at<float>(pixel.y(), pixel.x());
        // Check if depth value is valid
        if (depthValue == MINF) {
            return MINF;
        }

        const float normalizationFactor = (cameraSpecs.intrinsicsInverse * pixel.cast<float>()).norm();
        // Check if normalization factor is valid
        if (normalizationFactor == 0.0f) {
            return MINF;
        }

        const float rayDepth = (cameraCenterInGlobalSpace - globalPoint).norm() / normalizationFactor;
        // Check if ray depth is valid
        if (rayDepth < 0.0f) {
            return MINF;
        }

        const float deltaDepth = rayDepth - depthValue;
        // Truncate points that are too far away
        if (deltaDepth < -TRUNCATION_RADIUS) {
            return MINF;
        }

        return std::min(1.0f, deltaDepth / TRUNCATION_RADIUS) * static_cast<float>(sgn(deltaDepth));
    }

private:
    int frameNumber{};

    CameraSpecifications cameraSpecs;
    int imageWidth{};
    int imageHeight{};

    cv::Mat depthMap;
    cv::Mat colorMap;

    Matrix4f pose;

    cv::Mat vertexMap;
    cv::Mat validityMap;
    cv::Mat normalMap;

    Vector3f cameraCenterInGlobalSpace;
    std::vector<cv::Mat> depthPyramid;
    std::vector<cv::Mat> vertexPyramid;
    std::vector<cv::Mat> normalPyramid;

};