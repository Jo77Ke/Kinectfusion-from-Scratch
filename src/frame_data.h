#pragma once

#include <iostream>
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
            const CameraSpecifications &cameraSpecs,
            cv::Mat &&depthMap,
            cv::Mat &&colorMap,
            const Matrix4f &groundTruthPose
    ) : frameNumber(frameNumber),
        cameraSpecs(cameraSpecs),
        imageWidth(static_cast<int>(cameraSpecs.imageWidth)),
        imageHeight(static_cast<int>(cameraSpecs.imageHeight)),
        depthMap(std::move(depthMap)),
        colorMap(std::move(colorMap)),
        groundTruthPose(groundTruthPose) {
        validityMap = cv::Mat(imageHeight, imageWidth, CV_8UC1, cv::Scalar(0));
        validityMap.setTo(1, this->depthMap != MINF);
    }

    int getFrameNumber() const { return frameNumber; }

    const CameraSpecifications &getCameraSpecifications() const { return cameraSpecs; }

    const cv::Mat &getRawDepthMap() const { return depthMap; }

    const cv::Mat &getRawColorMap() const { return colorMap; }

    const cv::Mat &getDepthPyramidAtLevel(int level) const {
        return depthPyramid[level];
    }

    const cv::Mat &getVertexPyramidAtLevel(int level) const {
        return vertexPyramid[level];
    }

    const cv::Mat &getNormalPyramidAtLevel(int level) const {
        return normalPyramid[level];
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
            uchar *maskRow = validityMap.ptr<uchar>(v);
            for (int u = 0; u < imageWidth; ++u) {
                Vertex &vertex = vertexRow[u];

                if (maskRow[u] == 0) {
                    // invalid depth value -> invalid vertex
                    vertex.position = Vector4f(MINF, MINF, MINF, MINF);
                    vertex.color = cv::Vec4b(0, 0, 0, 0);
                } else {
                    Vector3f posInCameraSpace = cameraSpecs.intrinsicsInverse * depthRow[u]
                                                * Vector3f(static_cast<float>(u), static_cast<float>(v), 1.0f);
                    vertex.position = pose * posInCameraSpace.homogeneous();
                    vertex.color = colorRow[u];
                }
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
            const uchar *maskRow = validityMap.ptr<uchar>(v);
            const uchar *maskRowBelow = validityMap.ptr<uchar>(v + 1);

            cv::Vec4f *normalRow = normalMap.ptr<cv::Vec4f>(v);
            for (int u = 0; u < imageWidth - 1; ++u) {
                if (maskRow[u] == 0 || maskRow[u + 1] == 0 || maskRowBelow[u] == 0) {
                    // at least one point invalid -> normal vector invalid
                    normalRow[u] = cv::Vec4f(MINF, MINF, MINF, MINF);
                } else {
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
    }

    void buildPyramids(int levels, float sigmaS, float sigmaR) {
        std::cout << "frame_data.h buildPyramids" << std::endl;
        buildDepthPyramid(levels, sigmaS, sigmaR, depthPyramid, depthMap);
        buildVertexPyramid(depthPyramid, cameraSpecs, vertexPyramid);
        buildNormalPyramid(vertexPyramid, normalPyramid);
    }

    void computeCameraCenterInGlobalSpace() {
        cameraCenterInGlobalSpace = pose.col(3).head<3>();
    }

    float tsdfValueAt(const Vector3f &globalPoint) const {
        // Transform the 3D point from global space to the camera's local space
        const Vector3f pointInCameraSpace = (pose.inverse().eval()).topLeftCorner<3, 4>() * globalPoint.homogeneous();

        // Ignore voxels that are behind the camera
        if (pointInCameraSpace.z() <= 0) {
            return MINF;
        }

        // Project the 3D point into the 2D image plane
        Vector3f projectedPoint = cameraSpecs.intrinsics * pointInCameraSpace;
        const Vector2i pixel(round(projectedPoint.x() / projectedPoint.z()),
                             round(projectedPoint.y() / projectedPoint.z()));

        // Ignore voxels that project outside the image bounds
        if (pixel.x() < 0 || pixel.x() >= imageWidth || pixel.y() < 0 || pixel.y() >= imageHeight) {
            return MINF;
        }

        // Get the measured depth from the depth map at the projected pixel
        const float measuredDepth = depthMap.at<float>(pixel.y(), pixel.x());

        // Ignore pixels with invalid depth measurements
        if (measuredDepth <= 0.0f || measuredDepth == MINF) {
            return MINF;
        }

        // Calculate the difference between the measured depth and the point's actual depth
        const float sdf = measuredDepth - pointInCameraSpace.z();

        // TRUNCATE the signed distance and scale it to be between -1.0 and 1.0.
        return std::max(-1.0f, std::min(1.0f, sdf / TRUNCATION_RADIUS));
    }

private:
    int frameNumber{};

    CameraSpecifications cameraSpecs;
    int imageWidth{};
    int imageHeight{};

    cv::Mat depthMap;
    cv::Mat colorMap;

    Matrix4f pose;
    Matrix4f groundTruthPose;

    cv::Mat vertexMap;
    cv::Mat validityMap; // 1 for valid pixels, 0 for invalid
    cv::Mat normalMap;

    Vector3f cameraCenterInGlobalSpace;
    std::vector<cv::Mat> depthPyramid;
    std::vector<cv::Mat> vertexPyramid;
    std::vector<cv::Mat> normalPyramid;

};
