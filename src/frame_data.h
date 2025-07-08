#pragma once

#include <Eigen/Geometry>
#include <utility>
#include <fstream>

#include "utils.h"
#include "bilateral_filter.h"

constexpr float TRIANGULATION_EDGE_THRESHOLD = 0.01f; // 1cm

struct CameraSpecifications {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    unsigned int imageWidth;
    unsigned int imageHeight;

    Matrix3f intrinsics;
    Matrix3f intrinsicsInverse;
    Matrix4f extrinsics;
    Matrix4f extrinsicsInverse;

    CameraSpecifications() :
            imageWidth(0),
            imageHeight(0),
            intrinsics(Matrix3f::Identity()),
            intrinsicsInverse(Matrix3f::Identity()),
            extrinsics(Matrix4f::Identity()),
            extrinsicsInverse(Matrix4f::Identity()) {}

    CameraSpecifications(
            const unsigned int imageWidth, const unsigned int imageHeight,
            const Matrix3f &intrinsics, const Matrix4f &extrinsics
    ) : imageWidth(imageWidth), imageHeight(imageHeight),
        intrinsics(intrinsics), intrinsicsInverse(intrinsics.inverse()),
        extrinsics(extrinsics), extrinsicsInverse(extrinsics.inverse()) {}
};


struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f position;

    // color stored as 4 unsigned char (RGBX)
    cv::Vec4b color;
};


class FrameData {
public:
    FrameData() = default;

    FrameData(
            int frameNumber,
            CameraSpecifications cameraSpecs,
            cv::Mat &&depthMap,
            cv::Mat &&colorMap,
            const Eigen::Matrix4f &trajectory
    ) : frameNumber(frameNumber),
        cameraSpecs(std::move(cameraSpecs)),
        imageWidth(static_cast<int>(cameraSpecs.imageWidth)),
        imageHeight(static_cast<int>(cameraSpecs.imageHeight)),
        depthMap(std::move(depthMap)),
        colorMap(std::move(colorMap)),
        trajectory(trajectory) {
        CV_DbgAssert(!this->depthMap.empty() && !this->colorMap.empty());
    }

    int getFrameNumber() const { return frameNumber; }

    const CameraSpecifications &getCameraSpecifications() const { return cameraSpecs; }

    const cv::Mat &getDepthMap() const { return depthMap; }

    const cv::Mat &getColorMap() const { return colorMap; }

    const Eigen::Matrix4f &getTrajectory() const { return trajectory; }

    void applyBilateralFilter(
            const float sigma_s, const float sigma_r
    ) {
        cv::Mat filteredDepthMap = cv::Mat(imageHeight, imageWidth, CV_32F);;
        bilateralFilter(depthMap, filteredDepthMap, sigma_s, sigma_r);
        depthMap = std::move(filteredDepthMap);
    }

    void computeVertexMap() {
        vertexMap = cv::Mat(imageHeight, imageWidth, CV_8UC(sizeof(Vertex)), cv::Mat::AUTO_STEP);
        for (int v = 0; v < imageHeight; ++v) {
            const auto *depthRow = depthMap.ptr<float>(v);
            const auto *colorRow = colorMap.ptr<cv::Vec4b>(v);
            auto *vertexRow = reinterpret_cast<Vertex *>(vertexMap.ptr(v));
            for (int u = 0; u < imageWidth; ++u) {
                float depthValue = depthRow[u];
                Vertex &vertex = vertexRow[u];

                if (depthValue == MINF) { // invalid depth value -> invalid vertex
                    vertex.position = Vector4f(MINF, MINF, MINF, MINF);
                    vertex.color = cv::Vec4b(0, 0, 0, 0);
                } else {
                    Vector3f posInCameraSpace = cameraSpecs.intrinsicsInverse * depthValue
                                                * Vector3f(static_cast<float>(u), static_cast<float>(v), 1.0f);
                    vertex.position =
                            trajectory.inverse() * cameraSpecs.extrinsicsInverse * posInCameraSpace.homogeneous();

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

        for (int v = 0; v < imageHeight - 1; ++v) {
            const auto *vertexRow = vertexMap.ptr<cv::Vec4f>(v);
            const auto *vertexRowBelow = vertexMap.ptr<cv::Vec4f>(v);
            auto *normalRow = normalMap.ptr<cv::Vec4f>(v);
            for (int u = 0; u < imageWidth - 1; ++u) {
                const Eigen::Vector4f vertex(
                        vertexRow[u][0],
                        vertexRow[u][1],
                        vertexRow[u][2],
                        vertexRow[u][3]
                );
                const Eigen::Vector4f vertexRight(
                        vertexRow[u + 1][0],
                        vertexRow[u + 1][1],
                        vertexRow[u + 1][2],
                        vertexRow[u + 1][3]
                );
                const Eigen::Vector4f vertexBelow(
                        vertexRowBelow[u][0],
                        vertexRowBelow[u][1],
                        vertexRowBelow[u][2],
                        vertexRowBelow[u][3]
                );

                // check for invalid points
                if (vertex.x() != MINF && vertexRight.x() != MINF && vertexBelow.x() != MINF) {
                    // du = V(u+1, v) - V(u,v), dv = V(u, v+1) - V(u,v)
                    Vector3f du = vertexRight.head<3>() - vertex.head<3>();
                    Vector3f dv = vertexBelow.head<3>() - vertex.head<3>();

                    Vector3f n = du.cross(dv).normalized();
                    normalRow[u] = cv::Vec4f(n.x(), n.y(), n.z(), 1.0f);
                } else {
                    // at least one point invalid -> normal vector invalid
                    normalRow[u] = cv::Vec4f(MINF, MINF, MINF, MINF);
                }
            }
        }
    }

    void writeMesh(const std::string &filename) {
        // Get number of vertices
        const unsigned int nVertices = imageWidth * imageHeight;

        // Debug assertions
        CV_DbgAssert(vertexMap.type() == CV_8UC(sizeof(Vertex)));
        CV_DbgAssert(vertexMap.rows == imageHeight &&
                     vertexMap.cols == imageWidth &&
                     vertexMap.isContinuous());
        CV_DbgAssert(vertexMap.total() * vertexMap.elemSize() == nVertices * sizeof(Vertex));

        // Get number of faces
        const float squaredEdgeThreshold = TRIANGULATION_EDGE_THRESHOLD * TRIANGULATION_EDGE_THRESHOLD;
        std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> faces;
        for (int v = 0; v < imageHeight - 1; ++v) {
            const auto *verticesRow = reinterpret_cast<const Vertex *>(vertexMap.ptr(v));
            const auto *verticesRowBelow = reinterpret_cast<const Vertex *>(vertexMap.ptr(v + 1));

            for (int u = 0; u < imageWidth - 1; ++u) {
                /*
                 * Simple neighborhood-based triangulation:
                 *
                 * a - b
                 * | \ |
                 * c - d
                 *
                 * Only write triangles with valid vertices and an edge length smaller than the threshold.
                 */
                const Vertex &a = verticesRow[u];
                const Vertex &b = verticesRow[u + 1];
                const Vertex &c = verticesRowBelow[u];
                const Vertex &d = verticesRowBelow[u + 1];

                if (a.position.x() == MINF || d.position.x() == MINF) {
                    continue;
                }

                unsigned int idxA = u + v * imageWidth;
                unsigned int idxD = u + 1 + (v + 1) * imageWidth;

                // Check triangle ABD
                if (b.position.x() != MINF
                    && (a.position - b.position).squaredNorm() < squaredEdgeThreshold
                    && (a.position - d.position).squaredNorm() < squaredEdgeThreshold
                    && (b.position - d.position).squaredNorm() < squaredEdgeThreshold) {
                    unsigned int idxB = u + 1 + v * imageWidth;
                    faces.emplace_back(idxA, idxB, idxD);
                }

                // Check triangle ACD
                if (c.position.x() != MINF
                    && (a.position - c.position).squaredNorm() < squaredEdgeThreshold
                    && (a.position - d.position).squaredNorm() < squaredEdgeThreshold
                    && (c.position - d.position).squaredNorm() < squaredEdgeThreshold) {
                    unsigned int idxC = u + (v + 1) * imageWidth;
                    faces.emplace_back(idxA, idxC, idxD);
                }
            }
        }
        unsigned nFaces = faces.size();


        // Write off file
        std::ofstream outFile(filename);
        if (!outFile.is_open()) throw std::runtime_error("Could not open file for writing: " + filename);

        // Write header
        outFile << "COFF\n" << nVertices << " " << nFaces << " 0\n";

        // Save vertices
        auto *vertex = reinterpret_cast<Vertex *>(vertexMap.data);
        for (int i = 0; i < nVertices; ++i, ++vertex) {
            Vertex &v = *vertex;
            if (v.position.x() != MINF) {
                outFile << v.position.x() << " " << v.position.y() << " " << v.position.z()
                        << " " << static_cast<int>(v.color(0))
                        << " " << static_cast<int>(v.color(1))
                        << " " << static_cast<int>(v.color(2))
                        << " " << static_cast<int>(v.color(3))
                        << "\n";
            } else {
                outFile << "0 0 0\n";
            }
        }

        // save faces
        for (const auto &face: faces) {
            outFile << "3 "
                    << std::get<0>(face) << " "
                    << std::get<1>(face) << " "
                    << std::get<2>(face) << "\n";
        }

        // close file
        outFile.close();
    }

private:
    int frameNumber{};
    CameraSpecifications cameraSpecs;
    int imageWidth{};
    int imageHeight{};
    cv::Mat depthMap;
    cv::Mat colorMap;
    Eigen::Matrix4f trajectory;
    cv::Mat vertexMap;
    cv::Mat normalMap;
};