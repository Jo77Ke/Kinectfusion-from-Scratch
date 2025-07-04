#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "BilateralFilter.h"

#define BILATERAL_FILTERING true

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f position;

    // normals stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f normals;

    // color stored as 4 unsigned char
    Vector4uc color;
};

bool WriteMesh(Vertex *vertices, unsigned int width, unsigned int height, const std::string &filename) {
    float edgeThreshold = 0.01f; // 1cm

    // Get number of vertices
    unsigned int nVertices = width * height;

    // Get number of faces
    std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> faces;
    for (int u = 0; u < width - 1; ++u) {
        for (int v = 0; v < height - 1; ++v) {
            /*
             * Simple neighborhood-based triangulation:
             *
             * a - b
             * | \ |
             * c - d
             *
             * Only write triangles with valid vertices and an edge length smaller than the threshold.
             */
            unsigned int idxA = u + v * width;
            unsigned int idxB = u + 1 + v * width;
            unsigned int idxC = u + (v + 1) * width;
            unsigned int idxD = u + 1 + (v + 1) * width;

            Vertex a = vertices[idxA];
            Vertex b = vertices[idxB];
            Vertex c = vertices[idxC];
            Vertex d = vertices[idxD];

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
    for (int idx = 0; idx < nVertices; ++idx) {
        Vertex v = vertices[idx];
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

void computeNormals(Vertex *const vertexMap, const unsigned int imageWidth, const unsigned int imageHeight) {
    for (unsigned int v = 0; v < imageHeight; ++v) {
        for (unsigned int u = 0; u < imageWidth; ++u) {
            const unsigned int idx = u + v * imageWidth;

            // check if u+1, v+1 are in the image
            if (u < imageWidth - 1 && v < imageHeight - 1) {
                const unsigned int idxRight = (u + 1) + v * idx+1;
                const unsigned int idxDown  = u + (v + 1) * imageHeight;

                const Vector4f& p     = vertexMap[idx].position;
                const Vector4f& pRight = vertexMap[idxRight].position;
                const Vector4f& pDown  = vertexMap[idxDown].position;

                // check for MINF
                if (p.x() != MINF && pRight.x() != MINF && pDown.x() != MINF) {
                    // du = V(u+1, v) - V(u,v), dv = V(u, v+1) - V(u,v)
                    Vector3f du = pRight.head<3>() - p.head<3>();
                    Vector3f dv = pDown.head<3>() - p.head<3>();

                    vertexMap[idx].normals = du.cross(dv).normalized();
                } else {
                    // at least one point invalid -> normal vector invalid
                    vertexMap[idx].normals = Vector4f(MINF, MINF, MINF, MINF);
                }
            } else {
                // neighbours dont exist -> invalid
                vertexMap[idx].normals = Vector4f(MINF, MINF, MINF, MINF);
            }
        }
    }
}

int main() {
    std::string filenameIn = "./data/rgbd_dataset_freiburg1_xyz/";
    std::string outputDirectory = "./results/";
    std::string filenameBaseOut = BILATERAL_FILTERING ? "smoothMesh_" : "mesh_";

    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.Init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    // convert video to meshes
    while (sensor.ProcessNextFrame()) {
        const auto imageWidth = sensor.GetDepthImageWidth();
        const auto imageHeight = sensor.GetDepthImageHeight();

        // get ptr to the current depth frame
        // depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
        float *depthMap = sensor.GetDepth();

        float filteredDepthMap[imageWidth * imageHeight];
        if (BILATERAL_FILTERING) {
            const int sigma_s = 8; // controls filter region: the larger -> the more distant pixels contribute -> more smoothing
            const int sigma_r = 25; // controls allowed depth difference: the larger -> smooths higher contrasts -> edges may be blurred
            bilateralFilter(depthMap, imageWidth, imageHeight, filteredDepthMap, sigma_s, sigma_r);
        }

        // get ptr to the current color frame
        // color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
        BYTE *colorMap = sensor.GetColorRGBX();

        // get depth intrinsics
        Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();

        // compute inverse depth extrinsics
        Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();

        Matrix4f trajectory = sensor.GetTrajectory();
        Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();

        Vertex *vertices = new Vertex[sensor.GetDepthImageWidth() * sensor.GetDepthImageHeight()];

        Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();

        for (int u = 0; u < imageWidth; ++u) {
            for (int v = 0; v < imageHeight; ++v) {
                const auto idx = u + v * imageWidth;
                float depthValue = BILATERAL_FILTERING ? filteredDepthMap[idx] : depthMap[idx];
                if (depthValue == MINF) { // invalid depth value -> invalid vertex
                    vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
                    vertices[idx].color = Vector4uc(0, 0, 0, 0);
                } else {
                    Vector3f posInCameraSpace = depthIntrinsicsInv * depthValue
                                                * Vector3f(static_cast<float>(u), static_cast<float>(v), 1.0f);
                    vertices[idx].position = trajectoryInv * depthExtrinsicsInv * posInCameraSpace.homogeneous();
                    vertices[idx].color = Vector4uc(
                            colorMap[4 * idx],
                            colorMap[4 * idx + 1],
                            colorMap[4 * idx + 2],
                            colorMap[4 * idx + 3]
                    );
                }
            }
        }

        computeNormals(vertices, imageWidth, imageHeight);

        // write mesh file
        std::stringstream ss;
        ss << outputDirectory << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
        if (!WriteMesh(vertices, sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str())) {
            std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
            return -1;
        }


        // free mem
        delete[] vertices;
    }

    return 0;
}