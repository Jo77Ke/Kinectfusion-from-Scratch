#pragma once

#include <Eigen/Geometry>
#include <utility>
#include <fstream>
#include <numeric>

#include "utils.h"
#include "bilateral_filter.h"

constexpr float TRIANGULATION_EDGE_THRESHOLD = 0.01f; // 1cm

constexpr float TRUNCATION_RADIUS = 0.05f; // 5cm

struct CameraSpecifications {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    unsigned int imageWidth;
    unsigned int imageHeight;

    Matrix3f intrinsics;
    Matrix3f intrinsicsInverse;
    Matrix4f extrinsics;
    Matrix4f extrinsicsInverse;

    float minDepthRange = 0.4f;
    float maxDepthRange = 8.0f;

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
    }

    int getFrameNumber() const { return frameNumber; }

    const CameraSpecifications &getCameraSpecifications() const { return cameraSpecs; }

    const cv::Mat &getDepthMap() const { return depthMap; }

    const cv::Mat &getColorMap() const { return colorMap; }

    const Eigen::Matrix4f &getTrajectory() const { return trajectory; }

    void applyBilateralFilter(
            const float sigma_s, const float sigma_r
    ) {
        cv::Mat filteredDepthMap = cv::Mat(imageHeight, imageWidth, CV_32F);
        bilateralFilter(depthMap, filteredDepthMap, sigma_s, sigma_r);
        depthMap = std::move(filteredDepthMap);
    }

    void computeVertexMap() {
        vertexMap = cv::Mat(imageHeight, imageWidth, CV_8UC(sizeof(Vertex)), cv::Mat::AUTO_STEP);

#pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < imageHeight; ++v) {
            const float *depthRow = depthMap.ptr<float>(v);
            const cv::Vec4b *colorRow = colorMap.ptr<cv::Vec4b>(v);
            Vertex *vertexRow = reinterpret_cast<Vertex *>(vertexMap.ptr(v));
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

#pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < imageHeight - 1; ++v) {
            const cv::Vec4f *vertexRow = vertexMap.ptr<cv::Vec4f>(v);
            const cv::Vec4f *vertexRowBelow = vertexMap.ptr<cv::Vec4f>(v);
            cv::Vec4f *normalRow = normalMap.ptr<cv::Vec4f>(v);
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

        // Compute faces
        const float squaredEdgeThreshold = TRIANGULATION_EDGE_THRESHOLD * TRIANGULATION_EDGE_THRESHOLD;

        std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> faces;
        size_t nFaces = 0;

        // Share work among threads by dividing the image into horizontal strips
        std::vector<std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>> threadFaces(omp_get_max_threads());
        std::vector<size_t> threadCounts(omp_get_max_threads(), 0);

#pragma omp parallel
        {
            const int threadID = omp_get_thread_num();
            const int numberOfRows = imageHeight - 1;
            const int chunkSize = (numberOfRows + omp_get_num_threads() - 1) / omp_get_num_threads();
            const int startRow = std::min(threadID * chunkSize, numberOfRows);
            const int endRow = std::min((threadID + 1) * chunkSize, numberOfRows);

            threadFaces[threadID].resize(2 * (imageWidth - 1) * (endRow - startRow));
            size_t localCount = 0;

            computeFacesForRegion(startRow, endRow, squaredEdgeThreshold,
                                  threadFaces[threadID], localCount);

            threadCounts[threadID] = localCount;
            threadFaces[threadID].resize(localCount);
        }

        // Combine results from all threads
        nFaces = std::accumulate(threadCounts.begin(), threadCounts.end(), size_t(0));
        faces.reserve(nFaces);
        for (auto& tf : threadFaces) {
            faces.insert(faces.end(), tf.begin(), tf.end());
        }

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

        // Save faces
        for (size_t i = 0; i < nFaces; ++i) {
            const auto face = faces[i];
            outFile << "3 "
                    << std::get<0>(face) << " "
                    << std::get<1>(face) << " "
                    << std::get<2>(face) << "\n";
        }

        outFile.close();
    }

    void computeCameraCenterInGlobalSpace() {
        cameraCenterInGlobalSpace = (trajectory.inverse() * cameraSpecs.extrinsicsInverse).col(3).head<3>();
    }

    float tsdfValueAt(const Vector3f& globalPoint) const {
        const Vector3f pointInCameraSpace = (cameraSpecs.extrinsics * trajectory).topLeftCorner<3, 4>() * globalPoint.homogeneous();

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
    void computeFacesForRegion(
            int startRow, int endRow,
            float squaredEdgeThreshold,
            std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>& faces,
            size_t& faceCount)
    {
        for (int v = startRow; v < endRow; ++v) {
            const auto* verticesRow = reinterpret_cast<const Vertex*>(vertexMap.ptr(v));
            const auto* verticesRowBelow = reinterpret_cast<const Vertex*>(vertexMap.ptr(v + 1));

            for (int u = 0; u < imageWidth - 1; ++u) {
                const Vertex& a = verticesRow[u];
                const Vertex& d = verticesRowBelow[u + 1];
                if (a.position.x() == MINF || d.position.x() == MINF) continue;

                unsigned int idxA = u + v * imageWidth;
                unsigned int idxD = u + 1 + (v + 1) * imageWidth;
                const Vertex& b = verticesRow[u + 1];
                const Vertex& c = verticesRowBelow[u];

                // Triangle ABD
                if (b.position.x() != MINF
                    && (a.position - b.position).squaredNorm() < squaredEdgeThreshold
                    && (a.position - d.position).squaredNorm() < squaredEdgeThreshold
                    && (b.position - d.position).squaredNorm() < squaredEdgeThreshold) {
                    faces[faceCount++] = {idxA, u + 1 + v * imageWidth, idxD};
                }

                // Triangle ACD
                if (c.position.x() != MINF
                    && (a.position - c.position).squaredNorm() < squaredEdgeThreshold
                    && (a.position - d.position).squaredNorm() < squaredEdgeThreshold
                    && (c.position - d.position).squaredNorm() < squaredEdgeThreshold) {
                    faces[faceCount++] = {idxA, u + (v + 1) * imageWidth, idxD};
                }
            }
        }
    }

    int frameNumber{};
    CameraSpecifications cameraSpecs;
    int imageWidth{};
    int imageHeight{};
    cv::Mat depthMap;
    cv::Mat colorMap;
    Eigen::Matrix4f trajectory;
    cv::Mat vertexMap;
    cv::Mat normalMap;
    Vector3f cameraCenterInGlobalSpace;
};