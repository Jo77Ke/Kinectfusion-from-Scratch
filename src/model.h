#pragma once

#include "utils.h"
#include "frame_data.h"
#include "sub_sampling.h"

struct TSDFVoxel {
    float sdf = MINF;
//    cv::Vec4b color = {0, 0, 0, 0};
    unsigned char weight = 0;
};

class TSDFVolume {
public:
    TSDFVolume(const float voxelSize, const Vector3f &volumeSize)
            : voxelSize(voxelSize), volumeSize(volumeSize) {
        volumeOrigin = -volumeSize / 2.0f; // Center volume around camera by setting origin to the lower corner
        gridResolution = (volumeSize / voxelSize).array().cast<int>();

        voxelGrid = cv::Mat(
                gridResolution.z(),
                gridResolution.y() * gridResolution.x(),
                CV_8UC(sizeof(TSDFVoxel)),
                cv::Scalar(0)
        );

        // Mark all voxels as invalid by default
#pragma omp parallel for
        for (int z = 0; z < gridResolution.z(); ++z) {
            TSDFVoxel *slice = voxelGrid.ptr<TSDFVoxel>(z);
            for (int i = 0; i < gridResolution.y() * gridResolution.x(); ++i) {
                slice[i].sdf = MINF;
            }
        }
    }

    void integrate(const FrameData &frameData) {
#pragma omp parallel for
        for (int z = 0; z < gridResolution.z(); ++z) {
            TSDFVoxel *slice = voxelGrid.ptr<TSDFVoxel>(z);
            for (int y = 0; y < gridResolution.y(); ++y) {
                for (int x = 0; x < gridResolution.x(); ++x) {
                    Vector3f voxelWorld = volumeOrigin + Vector3i(x, y, z).cast<float>() * voxelSize;

                    float tsdf = frameData.tsdfValueAt(voxelWorld);

                    // Skip invalid TSDF values
                    if (tsdf == MINF) {
                        continue;
                    }

                    TSDFVoxel &voxel = slice[y * gridResolution.x() + x];


                    if (voxel.sdf == MINF) {
                        // Initialize voxel with the first valid value pair
                        voxel.sdf = tsdf * static_cast<float>(weightUpdate());
                        voxel.weight = weightUpdate();
                    } else {
                        unsigned char newWeight = static_cast<unsigned char>(
                                std::min( // Ensure weight does not overflow by truncation
                                        static_cast<int>(voxel.weight) + static_cast<int>(weightUpdate()),
                                        static_cast<int>(maxWeight())
                                )
                        );

                        voxel.sdf =
                                (voxel.sdf * static_cast<float>(voxel.weight) +
                                 tsdf * static_cast<float>(weightUpdate())) /
                                (static_cast<float>(newWeight));
                        voxel.weight = newWeight;
                    }
                }
            }
        }
    }


    void predictSurface(const CameraSpecifications &cameraSpecs, const Matrix4f &cameraPose) {
        const int imageWidth = static_cast<int>(cameraSpecs.imageWidth);
        const int imageHeight = static_cast<int>(cameraSpecs.imageHeight);

        vertexMap = cv::Mat(imageHeight, imageWidth, CV_8UC(sizeof(Vertex)), cv::Mat::AUTO_STEP);
        normalMap = cv::Mat(imageHeight, imageWidth, CV_32FC4);

        const Matrix3f Rcw = cameraPose.block<3, 3>(0, 0);
        const Vector3f cameraOrigin = cameraPose.block<3, 1>(0, 3);

#pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < imageHeight; ++v) {
            Vertex *vertexRow = reinterpret_cast<Vertex *>(vertexMap.ptr(v));
            cv::Vec4f *normalRow = normalMap.ptr<cv::Vec4f>(v);
            for (int u = 0; u < imageWidth; ++u) {
                Vector3f pixel(static_cast<float>(u), static_cast<float>(v), 1.0f);
                Vector3f rayDirection = Rcw * (cameraSpecs.intrinsicsInverse * pixel).normalized();

                castRay(
                        cameraOrigin, rayDirection,
                        vertexRow[u],
                        cameraSpecs.minDepthRange, cameraSpecs.maxDepthRange
                );

                computeNormalFromTSDF(vertexRow[u].position.head<3>(), normalRow[u]);
            }
        }
    }

    const cv::Mat &getVertexMap() const {
        return vertexMap;
    }

    const cv::Mat &getNormalMap() const {
        return normalMap;
    }

private:
    static unsigned char weightUpdate() {
        return 1;
    }

    static unsigned char maxWeight() {
        return std::numeric_limits<unsigned char>::max();
    }

    bool isInBounds(const Vector3f &worldPoint) const {
        Vector3f local_point = worldPoint - volumeOrigin;
        return (local_point.x() >= 0 && local_point.x() < volumeSize.x() &&
                local_point.y() >= 0 && local_point.y() < volumeSize.y() &&
                local_point.z() >= 0 && local_point.z() < volumeSize.z());
    }

    bool castRay(
            const Vector3f &origin, const Vector3f &direction,
            Vertex &intersectionVertex,
            const float tMin, const float tMax
    ) const {
        // Default result to invalid
        intersectionVertex.position = Vector4f(MINF, MINF, MINF, MINF);
        intersectionVertex.color = cv::Vec4b(0, 0, 0, 0);

        float t = tMin;
        Vector3f p = origin + t * direction;
        // Ray starts outside the volume bounds
        if (!isInBounds(p)) {
            return false;
        }

        TSDFVoxel voxel = getVoxel(p);

        float sdf = voxel.sdf;
        float step = (sdf > TRUNCATION_RADIUS) ? TRUNCATION_RADIUS : std::max(sdf, voxelSize);
        t += step;
        float prevSDF = sdf;


        while (t < tMax) {
            p = origin + t * direction;

            // Ray left the volume bounds
            if (!isInBounds(p)) {
                return false;
            }

            voxel = getVoxel(p);
            sdf = voxel.sdf;

            // Found intersection from inside to outside the surface, thus occluded by the object
            if (prevSDF != MINF && prevSDF <= 0.0f && sdf >= 0.0f) {
                return false;
            }

            // Found intersection from outside to inside the surface
            if (sdf != MINF && sdf <= 0.0f && prevSDF >= 0.0f) {
                float tInterpolated = t - step * prevSDF / (prevSDF - sdf);
                intersectionVertex.position = (origin + tInterpolated * direction).homogeneous();
//                intersectionVertex.color = voxel.color;
                return true;
            }

            // Set step size based on the truncated SDF value while ensuring to land in a new voxel
            step = (sdf > TRUNCATION_RADIUS) ? TRUNCATION_RADIUS : std::max(sdf, voxelSize);
            t += step;
            prevSDF = sdf;
        }

        return false;
    }

    void computeNormalFromTSDF(const Vector3f &point, cv::Vec4f &normal) const {
        normal = {MINF, MINF, MINF, MINF}; // Default to undefined normal

        // Normal is undefined outside the volume
        if (!isInBounds(point)) {
            return;
        }

        Vector3f dx(voxelSize, 0, 0);
        Vector3f dy(0, voxelSize, 0);
        Vector3f dz(0, 0, voxelSize);

        // Normal is undefined if any of the neighboring voxels are out of bounds
        if (!isInBounds(point + dx) || !isInBounds(point - dx) ||
            !isInBounds(point + dy) || !isInBounds(point - dy) ||
            !isInBounds(point + dz) || !isInBounds(point - dz)) {
            return;
        }

        float Fx1 = getVoxel(point + dx).sdf;
        float Fx2 = getVoxel(point - dx).sdf;
        float Fy1 = getVoxel(point + dy).sdf;
        float Fy2 = getVoxel(point - dy).sdf;
        float Fz1 = getVoxel(point + dz).sdf;
        float Fz2 = getVoxel(point - dz).sdf;

        if (Fx1 == MINF || Fx2 == MINF || Fy1 == MINF || Fy2 == MINF || Fz1 == MINF || Fz2 == MINF) {
            return;
        }

        Vector3f n = Vector3f(Fx1 - Fx2, Fy1 - Fy2, Fz1 - Fz2).normalized();
        normal = {n.x(), n.y(), n.z(), 1.0f};
    }

    TSDFVoxel getVoxel(const Vector3f &worldPoint) const {
        Vector3f localPoint = worldPoint - volumeOrigin;

        Vector3i voxelIndices = (localPoint / voxelSize).array()
                .floor()
                .cast<int>();
        voxelIndices = voxelIndices
                .cwiseMax(Vector3i::Zero())
                .cwiseMin(gridResolution - Vector3i::Ones());


        const int rowStride = gridResolution.x() * gridResolution.y();

        int index = voxelIndices.z() * rowStride
                    + voxelIndices.y() * gridResolution.x()
                    + voxelIndices.x();
        return voxelGrid.at<TSDFVoxel>(voxelIndices.z(), index % rowStride);
    }

    float voxelSize;
    Vector3f volumeSize;
    Vector3i gridResolution;
    Vector3f volumeOrigin;

    cv::Mat voxelGrid; // 3D grid of voxels, flattened into 2D

    cv::Mat vertexMap;
    cv::Mat normalMap;

    std::vector<cv::Mat> vertexPyramid;
    std::vector<cv::Mat> normalPyramid;
};
