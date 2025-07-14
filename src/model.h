#pragma once

#include "utils.h"
#include "frame_data.h"
#include <vector>
#include <cmath>

struct TSDFVoxel {
    float sdf;
    float weight;
};

class TSDFVolume {
public:
    Eigen::Vector3f volumeOrigin;
    float voxelSize;
    int dimX, dimY, dimZ;

    TSDFVoxel getInterpolatedVoxel(const Eigen::Vector3f &pos) const {
        // Dummy: hier sollte trilineare Interpolation passieren
        return {0.0f, 1.0f};
    }

    bool isInBounds(const Eigen::Vector3f &pos) const {
        return (pos.x() >= volumeOrigin.x() && pos.x() < volumeOrigin.x() + (float) dimX * voxelSize &&
                pos.y() >= volumeOrigin.y() && pos.y() < volumeOrigin.y() + (float) dimY * voxelSize &&
                pos.z() >= volumeOrigin.z() && pos.z() < volumeOrigin.z() + (float) dimZ * voxelSize);
    }
};

class SurfaceRayCaster {
public:
    SurfaceRayCaster(
            const TSDFVolume &volume,
            const Eigen::Matrix3f &K,
            const Eigen::Matrix4f &Tcw,
            int width, int height,
            float mu
    ) : tsdf(volume), K(K), Tcw(Tcw), width(width), height(height), mu(mu) {}

    void castRay(
            std::vector<Eigen::Vector3f> &vertexMap,
            std::vector<Eigen::Vector3f> &normalMap
    ) {
        vertexMap.resize(width * height, Eigen::Vector3f::Zero());
        normalMap.resize(width * height, Eigen::Vector3f::Zero());

        Eigen::Matrix3f Rcw = Tcw.block<3, 3>(0, 0);
        Eigen::Vector3f tcw = Tcw.block<3, 1>(0, 3);

        for (int v = 0; v < height; ++v) {
            for (int u = 0; u < width; ++u) {
                Eigen::Vector3f pixel((float) u, (float) v, 1.0f);
                Eigen::Vector3f rayDir = K.inverse() * pixel;
                rayDir.normalize();

                Eigen::Vector3f origin = tcw;
                Eigen::Vector3f dir = Rcw * rayDir;

                Eigen::Vector3f point;
                int idx = v * width + u;
                if (findSurfaceIntersection(origin, dir, point)) {
                    vertexMap[idx] = point;
                    normalMap[idx] = computeNormal(point);
                } else {
                    vertexMap[idx] = Vector3f(MINF, MINF, MINF);
                    normalMap[idx] = Vector3f(MINF, MINF, MINF);
                }
            }
        }
    }

private:
    const TSDFVolume &tsdf;
    Eigen::Matrix3f K;
    Eigen::Matrix4f Tcw;
    int width, height;
    float mu;

    bool findSurfaceIntersection(const Eigen::Vector3f &origin,
                                 const Eigen::Vector3f &direction,
                                 Eigen::Vector3f &surfacePoint) {
        float t = 0.4f;
        float t_max = 8.0f;

        Eigen::Vector3f p = origin + t * direction;
        TSDFVoxel voxel = tsdf.getInterpolatedVoxel(p);

        if (!tsdf.isInBounds(p)) { // Ray outside the volume bounds
            return false;
        }

        float sdf = voxel.sdf;
        float step = (sdf > mu) ? mu : std::max(sdf, tsdf.voxelSize);
        t += step;
        float prevSDF = sdf;


        while (t < t_max) {
            p = origin + t * direction;
            voxel = tsdf.getInterpolatedVoxel(p);
            sdf = voxel.sdf;


            if (!tsdf.isInBounds(p)) { // Ray outside the volume bounds
                return false;
            }

            if (prevSDF != MINF && prevSDF <= 0.0f && sdf >= 0.0f) { // Intersection from inside to outside the volume
                return false;
            }

            if (sdf != MINF && sdf <= 0.0f && prevSDF >= 0.0f) {
                float tInterp = t - step * prevSDF / (prevSDF - sdf);
                surfacePoint = origin + tInterp * direction;
                return true;
            }

            step = (sdf > mu) ? mu : std::max(sdf, tsdf.voxelSize);
            t += step;
            prevSDF = sdf;
        }

        return false;
    }

    Eigen::Vector3f computeNormal(const Eigen::Vector3f &p) {
        float delta = tsdf.voxelSize;

        Eigen::Vector3f dx(delta, 0, 0);
        Eigen::Vector3f dy(0, delta, 0);
        Eigen::Vector3f dz(0, 0, delta);

        float Fx1 = tsdf.getInterpolatedVoxel(p + dx).sdf;
        float Fx2 = tsdf.getInterpolatedVoxel(p - dx).sdf;
        float Fy1 = tsdf.getInterpolatedVoxel(p + dy).sdf;
        float Fy2 = tsdf.getInterpolatedVoxel(p - dy).sdf;
        float Fz1 = tsdf.getInterpolatedVoxel(p + dz).sdf;
        float Fz2 = tsdf.getInterpolatedVoxel(p - dz).sdf;

        Eigen::Vector3f n(Fx1 - Fx2, Fy1 - Fy2, Fz1 - Fz2);
        return n.normalized();
    }
};