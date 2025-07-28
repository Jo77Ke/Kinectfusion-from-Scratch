#pragma once

#include <fstream>

#include "utils.h"
#include "vertex.h"

constexpr float TRIANGULATION_EDGE_THRESHOLD = 0.01f; // 1cm


size_t computeFacesForRegion(
        const cv::Mat &vertexMap,
        unsigned int imageWidth,
        int startRow, int endRow,
        float squaredEdgeThreshold,
        std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>& faces
) {
    size_t faceCount = 0;
    for (int v = startRow; v < endRow; ++v) {
        const auto *verticesRow = reinterpret_cast<const Vertex *>(vertexMap.ptr(v));
        const auto *verticesRowBelow = reinterpret_cast<const Vertex *>(vertexMap.ptr(v + 1));

        for (int u = 0; u < imageWidth - 1; ++u) {
            const Vertex &a = verticesRow[u];
            const Vertex &d = verticesRowBelow[u + 1];
            if (a.position.x() == MINF || a.position.y() == MINF || a.position.z() == MINF ||
                d.position.x() == MINF || d.position.y() == MINF || d.position.z() == MINF) {
                continue;
            }


            unsigned int idxA = u + v * imageWidth;
            unsigned int idxD = u + 1 + (v + 1) * imageWidth;
            const Vertex &b = verticesRow[u + 1];
            const Vertex &c = verticesRowBelow[u];

            // Triangle ABD
            if (b.position.x() != MINF && b.position.y() != MINF && b.position.z() != MINF
                && (a.position - b.position).squaredNorm() < squaredEdgeThreshold
                && (a.position - d.position).squaredNorm() < squaredEdgeThreshold
                && (b.position - d.position).squaredNorm() < squaredEdgeThreshold) {
                faces[faceCount] = {idxA, u + 1 + v * imageWidth, idxD};
                ++faceCount;
            }

            // Triangle ACD
            if (c.position.x() != MINF && c.position.y() != MINF && c.position.z() != MINF
                && (a.position - c.position).squaredNorm() < squaredEdgeThreshold
                && (a.position - d.position).squaredNorm() < squaredEdgeThreshold
                && (c.position - d.position).squaredNorm() < squaredEdgeThreshold) {
                faces[faceCount] = {idxA, u + (v + 1) * imageWidth, idxD};
                ++faceCount;
            }
        }
    }

    return faceCount;
}


void writeMesh(
        const std::string &filename, const cv::Mat &vertexMap
) {
    const int imageWidth = vertexMap.cols;
    const int imageHeight = vertexMap.rows;
    if (imageWidth < 2 && imageHeight < 2) throw std::runtime_error("No enough pixels to form faces");

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

    // Share work among threads by dividing the image into horizontal strips
#pragma omp parallel
    {
        const int threadID = omp_get_thread_num();
        const int numberOfRows = static_cast<int>(imageHeight) - 1;
        const int rowsPerThread = (numberOfRows + omp_get_num_threads() - 1) / omp_get_num_threads();
        const int startRow = std::min(threadID * rowsPerThread, numberOfRows);
        const int endRow = std::min(startRow + rowsPerThread, numberOfRows);

        std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > localFaces;
        localFaces.reserve(2 * (imageWidth - 1) * (endRow - startRow));
        const size_t localCount = computeFacesForRegion(vertexMap, imageWidth,
                                                  startRow, endRow,
                                                  squaredEdgeThreshold,
                                                  localFaces);

#pragma omp critical
        {
            faces.insert(faces.end(), localFaces.begin(), localFaces.begin() + static_cast<int>(localCount));
        }
    }

    const int nFaces = static_cast<int>(faces.size());

    // Write off file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) throw std::runtime_error("Could not open file for writing: " + filename);

    // Write header
    outFile << "COFF\n" << nVertices << " " << nFaces << " 0\n";

    // Save vertices
    for (int v = 0; v < imageHeight; ++v) {
        const auto * vertexRow = reinterpret_cast<const Vertex *>(vertexMap.ptr(v));
        for (int u = 0; u < imageWidth; ++u) {
            const auto &vertex = vertexRow[u];
            if (vertex.position.x() != MINF && vertex.position.y() != MINF && vertex.position.z() != MINF) {
                outFile << vertex.position.x() << " " << vertex.position.y() << " " << vertex.position.z()
                        << " " << static_cast<int>(vertex.color(0))
                        << " " << static_cast<int>(vertex.color(1))
                        << " " << static_cast<int>(vertex.color(2))
                        << " " << static_cast<int>(vertex.color(3))
                        << "\n";
            } else {
                outFile << "0 0 0\n";
            }
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

void writeMeshWithNormals(
        const std::string &filename, const cv::Mat &vertexMap, const cv::Mat &normalMap
) {
    const int imageWidth = vertexMap.cols;
    const int imageHeight = vertexMap.rows;

    if (imageWidth < 2 && imageHeight < 2) throw std::runtime_error("No enough pixels to form faces");

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

    // Share work among threads by dividing the image into horizontal strips
#pragma omp parallel
    {
        const int threadID = omp_get_thread_num();
        const int numberOfRows = static_cast<int>(imageHeight) - 1;
        const int rowsPerThread = (numberOfRows + omp_get_num_threads() - 1) / omp_get_num_threads();
        const int startRow = std::min(threadID * rowsPerThread, numberOfRows);
        const int endRow = std::min(startRow + rowsPerThread, numberOfRows);

        std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > localFaces;
        localFaces.reserve(2 * (imageWidth - 1) * (endRow - startRow));
        const size_t localCount = computeFacesForRegion(vertexMap, imageWidth,
                                                        startRow, endRow,
                                                        squaredEdgeThreshold,
                                                        localFaces);

#pragma omp critical
        {
            faces.insert(faces.end(), localFaces.begin(), localFaces.begin() + static_cast<int>(localCount));
        }
    }

    const int nFaces = static_cast<int>(faces.size());

    // Write off file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) throw std::runtime_error("Could not open file for writing: " + filename);

    // Write header
    outFile << "COFF\n" << nVertices << " " << nFaces << " 0\n";

    // Save vertices
    for (int v = 0; v < imageHeight; ++v) {
        const auto * vertexRow = reinterpret_cast<const Vertex *>(vertexMap.ptr(v));
        const auto * normalRow = reinterpret_cast<const cv::Vec4f *>(normalMap.ptr(v));
        for (int u = 0; u < imageWidth; ++u) {
            const auto &vertex = vertexRow[u];
            if (vertex.position.x() != MINF && vertex.position.y() != MINF && vertex.position.z() != MINF) {
                outFile << vertex.position.x() << " " << vertex.position.y() << " " << vertex.position.z() << " ";

                const auto& normal = normalRow[u];
                if (normal[0] == MINF && normal[1] == MINF && normal[2] == MINF) {
                    outFile << static_cast<int>(vertex.color(0))
                            << " " << static_cast<int>(vertex.color(1))
                            << " " << static_cast<int>(vertex.color(2))
                            << " " << static_cast<int>(vertex.color(3))
                            << "\n";
                } else {
                    int r = static_cast<int>((normal[0] * 0.5f + 0.5f) * 255);
                    int g = static_cast<int>((normal[1] * 0.5f + 0.5f) * 255);
                    int b = static_cast<int>((normal[2] * 0.5f + 0.5f) * 255);

                    r = std::max(0, std::min(255, r));
                    g = std::max(0, std::min(255, g));
                    b = std::max(0, std::min(255, b));

                    outFile << r << " " << g << " " << b << " 0\n";
                }

            } else {
                outFile << "0 0 0\n";
            }
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