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
            if (a.position.x() == MINF || d.position.x() == MINF) continue;

            unsigned int idxA = u + v * imageWidth;
            unsigned int idxD = u + 1 + (v + 1) * imageWidth;
            const Vertex &b = verticesRow[u + 1];
            const Vertex &c = verticesRowBelow[u];

            // Triangle ABD
            if (b.position.x() != MINF
                && (a.position - b.position).squaredNorm() < squaredEdgeThreshold
                && (a.position - d.position).squaredNorm() < squaredEdgeThreshold
                && (b.position - d.position).squaredNorm() < squaredEdgeThreshold) {
                faces[faceCount] = {idxA, u + 1 + v * imageWidth, idxD};
                ++faceCount;
            }

            // Triangle ACD
            if (c.position.x() != MINF
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
        const std::string &filename, const cv::Mat &vertexMap,
        unsigned int imageWidth, unsigned int imageHeight
) {
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
    auto *vertex = reinterpret_cast<Vertex *>(vertexMap.data);
    for (size_t i = 0; i < nVertices; ++i, ++vertex) {
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