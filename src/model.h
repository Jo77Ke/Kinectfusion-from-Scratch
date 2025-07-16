#pragma once

#include "utils.h"
#include "frame_data.h"

struct TSDFVoxel {
    float sdf;
    cv::Vec4b color;
    float weight;
};

class TSDFVolume {
public:
    TSDFVolume(const float voxelSize, const FrameData &firstFrame)
        : voxelSize(voxelSize) {
        integrate(firstFrame);
    }

    void integrate(const FrameData &frameData) {
        boundingBox.update(frameData.getBoundingBox());
        const CameraSpecifications &cameraSpecs = frameData.getCameraSpecifications();

        // TODO: Fusion of the frame data into the TSDF volume
    }


    void predictSurface(const CameraSpecifications &cameraSpecs) {
        const int imageWidth = static_cast<int>(cameraSpecs.imageWidth);
        const int imageHeight = static_cast<int>(cameraSpecs.imageHeight);

        vertexMap = cv::Mat(imageHeight, imageWidth, CV_8UC(sizeof(Vertex)), cv::Mat::AUTO_STEP);
        normalMap = cv::Mat(imageHeight, imageWidth, CV_32FC4);

        const Matrix3f Rcw = cameraSpecs.extrinsics.block<3, 3>(0, 0);
        const Vector3f cameraOrigin = cameraSpecs.extrinsics.block<3, 1>(0, 3);

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
    bool isInBounds(const Vector3f &point) const {
        return boundingBox.contains(point);
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

        TSDFVoxel voxel = getInterpolatedVoxel(p);

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

            voxel = getInterpolatedVoxel(p);
            sdf = voxel.sdf;

            // Found intersection from inside to outside the surface, thus occluded by the object
            if (prevSDF != MINF && prevSDF <= 0.0f && sdf >= 0.0f) {
                return false;
            }

            // Found intersection from outside to inside the surface
            if (sdf != MINF && sdf <= 0.0f && prevSDF >= 0.0f) {
                float tInterpolated = t - step * prevSDF / (prevSDF - sdf);
                intersectionVertex.position = (origin + tInterpolated * direction).homogeneous();
                intersectionVertex.color = voxel.color;
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

        float Fx1 = getInterpolatedVoxel(point + dx).sdf;
        float Fx2 = getInterpolatedVoxel(point - dx).sdf;
        float Fy1 = getInterpolatedVoxel(point + dy).sdf;
        float Fy2 = getInterpolatedVoxel(point - dy).sdf;
        float Fz1 = getInterpolatedVoxel(point + dz).sdf;
        float Fz2 = getInterpolatedVoxel(point - dz).sdf;

        if (Fx1 == MINF || Fx2 == MINF || Fy1 == MINF || Fy2 == MINF || Fz1 == MINF || Fz2 == MINF) {
            return;
        }

        Vector3f n = Vector3f(Fx1 - Fx2, Fy1 - Fy2, Fz1 - Fz2).normalized();
        normal = {n.x(), n.y(), n.z(), 1.0f};
    }

    TSDFVoxel getInterpolatedVoxel(const Vector3f &pos) const {
        // TODO: Mapping from 3D point to voxel
        return {0.0f, cv::Vec4b(0, 0, 0, 0), 1.0f};
    }

    float voxelSize;
    BoundingBox boundingBox;
    Vector3f volumeOrigin;
    cv::Mat vertexMap;
    cv::Mat normalMap;
};
