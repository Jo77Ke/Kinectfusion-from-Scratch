#include <iostream>
#include <fstream>

#include <Eigen/Geometry>

#include "utils.h"
#include "rgbd_frame_stream.h"
#include "bilateral_filter.h"

#define BILATERAL_FILTERING false

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f position;

    // color stored as 4 unsigned char (RGBX)
    cv::Vec4b color;
};

bool writeMesh(
        const cv::Mat &vertexMap,
        const std::string &filename
) {
    CV_Assert(vertexMap.type() == CV_8UC(sizeof(Vertex)));

    float edgeThreshold = 0.01f; // 1cm

    // Get number of vertices
    const int imageHeight = vertexMap.rows;
    const int imageWidth = vertexMap.cols;
    const unsigned int nVertices = imageWidth * imageHeight;

    // Get number of faces
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
            unsigned int idxA = u + v * imageWidth;
            unsigned int idxB = u + 1 + v * imageWidth;
            unsigned int idxC = u + (v + 1) * imageWidth;
            unsigned int idxD = u + 1 + (v + 1) * imageWidth;

            Vertex a = verticesRow[u];
            Vertex b = verticesRow[u+1];
            Vertex c = verticesRowBelow[u];
            Vertex d = verticesRowBelow[u+1];

            if (a.position.x() == MINF || d.position.x() == MINF) {
                continue;
            }

            if (b.position.x() != MINF
                && (a.position - b.position).norm() < edgeThreshold
                && (a.position - d.position).norm() < edgeThreshold
                && (b.position - d.position).norm() < edgeThreshold) {
                faces.emplace_back(idxA, idxB, idxD);
            }

            if (c.position.x() != MINF
                && (a.position - c.position).norm() < edgeThreshold
                && (a.position - d.position).norm() < edgeThreshold
                && (c.position - d.position).norm() < edgeThreshold) {
                faces.emplace_back(idxA, idxC, idxD);
            }
        }
    }
    unsigned nFaces = faces.size();


    // Write off file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) return false;

    // write header
    outFile << "COFF" << std::endl;
    outFile << nVertices << " " << nFaces << " 0" << std::endl;

    // save vertices
    // Iterate over all vertices in vertexMap without using u, v
    auto* vertex = reinterpret_cast<Vertex *>(vertexMap.data);
    for (int i = 0; i < nVertices; ++i, ++vertex) {
        Vertex& v = *vertex;
        if (v.position.x() != MINF) {
            outFile << v.position.x() << " " << v.position.y() << " " << v.position.z()
                    << " " << static_cast<int>(v.color(0))
                    << " " << static_cast<int>(v.color(1))
                    << " " << static_cast<int>(v.color(2))
                    << " " << static_cast<int>(v.color(3))
                    << std::endl;
        } else {
            outFile << "0 0 0" << std::endl;
        }
    }

    // save faces
    for (const auto &face: faces) {
        outFile << "3 "
                << std::get<0>(face) << " "
                << std::get<1>(face) << " "
                << std::get<2>(face)
                << std::endl;
    }

    // close file
    outFile.close();

    return true;
}

bool computeNormals(const cv::Mat &vertexMap, cv::Mat &normalMap) {
    CV_Assert(vertexMap.type() == CV_8UC(sizeof(Vertex)));
    CV_Assert(normalMap.type() == CV_32FC4);
    CV_Assert(vertexMap.size() == normalMap.size());

    const int imageHeight = vertexMap.rows;
    const int imageWidth = vertexMap.cols;

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

    return true;
}

int main() {
    std::string filenameIn = "./data/rgbd_dataset_freiburg1_xyz/";
    std::string outputDirectory = "./results/";
    std::string filenameBaseOut = "mesh_";

    // load video
    std::cout << "Initialize frame stream..." << std::endl;
    RGBDFrameStream stream;
    if (!stream.init(filenameIn)) {
        std::cerr << "Failed to initialize the frame stream!" << std::endl;
        return -1;
    }

    // convert video to meshes
    while (stream.processNextFrame()) {
        const int imageWidth = static_cast<int>(stream.getDepthImageWidth());
        const int imageHeight = static_cast<int>(stream.getDepthImageHeight());

        // get depth map of current frame
        const cv::Mat &depthMap = stream.getCurrentDepthMap();

        cv::Mat denoisedDepthMap = cv::Mat(imageHeight, imageWidth, CV_32F);;
        if (BILATERAL_FILTERING) {
            const float sigma_s = 3; // controls filter region: the larger -> the more distant pixels contribute -> more smoothing
            const float sigma_r = 0.1; // controls allowed depth difference: the larger -> smooths higher contrasts -> edges may be blurred

            bilateralFilter(depthMap, denoisedDepthMap, sigma_s, sigma_r);
            filenameBaseOut = "smoothedMesh_s" + std::to_string(sigma_s) + "_r" + std::to_string(sigma_r) + "_";
        }

        // get color map of current frame (stored as RGBX)
        const cv::Mat &colorMap = stream.getCurrentColorMap();

        // get depth intrinsics
        const Matrix3f &depthIntrinsics = stream.getDepthIntrinsics();
        Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();

        // compute inverse depth extrinsics
        Matrix4f depthExtrinsicsInv = stream.getDepthExtrinsics().inverse();
        Matrix4f trajectoryInv = stream.getCurrentTrajectory().inverse();

        cv::Mat vertexMap(imageHeight, imageWidth, CV_8UC(sizeof(Vertex)), cv::Mat::AUTO_STEP);
        for (int v = 0; v < imageHeight; ++v) {
            const auto *depthRow = BILATERAL_FILTERING ? denoisedDepthMap.ptr<float>(v) : depthMap.ptr<float>(v);
            const auto *colorRow = colorMap.ptr<cv::Vec4b>(v);
            auto *vertexRow = reinterpret_cast<Vertex *>(vertexMap.ptr(v));
            for (int u = 0; u < imageWidth; ++u) {
                float depthValue = depthRow[u];
                Vertex &vertex = vertexRow[u];

                if (depthValue == MINF) { // invalid depth value -> invalid vertex
                    vertex.position = Vector4f(MINF, MINF, MINF, MINF);
                    vertex.color = cv::Vec4b(0, 0, 0, 0);
                } else {
                    Vector3f posInCameraSpace = depthIntrinsicsInv * depthValue
                                                * Vector3f(static_cast<float>(u), static_cast<float>(v), 1.0f);
                    vertex.position = trajectoryInv * depthExtrinsicsInv * posInCameraSpace.homogeneous();

                    vertex.color = colorRow[u];
                }
            }
        }

        cv::Mat normalMap(imageHeight, imageWidth, CV_32FC4);
        computeNormals(vertexMap, normalMap);

        // Write mesh file
        std::stringstream ss;
        ss << outputDirectory << filenameBaseOut << stream.getCurrentFrameIndex() << ".off";
        if (!writeMesh(vertexMap, ss.str())) {
            std::cerr << "Failed to write mesh!" << std::endl;
            return -1;
        }
    }

    return 0;
}