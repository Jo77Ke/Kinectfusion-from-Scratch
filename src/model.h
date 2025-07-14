#pragma once
#include <Eigen/Dense>
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

    TSDFVoxel getInterpolatedVoxel(const Eigen::Vector3f& pos) const {
        // Dummy: hier sollte trilineare Interpolation passieren
        return {0.0f, 1.0f};
    }

    bool inBounds(const Eigen::Vector3f& pos) const {
        // Dummy: prüft ob Punkt im Volumen liegt
        return true;
    }
};

class SurfaceRaycaster {
public:
    SurfaceRaycaster(const TSDFVolume& volume,
                     const Eigen::Matrix3f& K,
                     const Eigen::Matrix4f& Tcw,
                     int width, int height,
                     float mu)
        : tsdf(volume), K(K), Tcw(Tcw), width(width), height(height), mu(mu) {}

    void raycast(std::vector<Eigen::Vector3f>& vertexMap,
                 std::vector<Eigen::Vector3f>& normalMap)
    {
        vertexMap.resize(width * height, Eigen::Vector3f::Zero());
        normalMap.resize(width * height, Eigen::Vector3f::Zero());

        Eigen::Matrix3f Rcw = Tcw.block<3,3>(0,0);
        Eigen::Vector3f tcw = Tcw.block<3,1>(0,3);

        for (int v = 0; v < height; ++v) {
            for (int u = 0; u < width; ++u) {
                Eigen::Vector3f pixel((float)u, (float)v, 1.0f);
                Eigen::Vector3f rayDir = K.inverse() * pixel;
                rayDir.normalize();

                Eigen::Vector3f origin = tcw;
                Eigen::Vector3f dir = Rcw * rayDir;

                Eigen::Vector3f point;
                if (findSurfaceIntersection(origin, dir, point)) {
                    int idx = v * width + u;
                    vertexMap[idx] = point;
                    normalMap[idx] = computeNormal(point);
                }
            }
        }
    }

private:
    const TSDFVolume& tsdf;
    Eigen::Matrix3f K;
    Eigen::Matrix4f Tcw;
    int width, height;
    float mu;

    bool findSurfaceIntersection(const Eigen::Vector3f& origin,
                                 const Eigen::Vector3f& direction,
                                 Eigen::Vector3f& surfacePoint)
    {
        float t = 0.5f;
        float t_max = 3.0f;
        float step = tsdf.voxelSize;
        float prevSDF = 1.0f;

        while (t < t_max) {
            Eigen::Vector3f p = origin + t * direction;
            if (!tsdf.inBounds(p)) return false;

            TSDFVoxel voxel = tsdf.getInterpolatedVoxel(p);
            float sdf = voxel.sdf;

            if (sdf < 0.0f && prevSDF > 0.0f) {
                float tInterp = t - step * prevSDF / (prevSDF - sdf);
                surfacePoint = origin + tInterp * direction;
                return true;
            }

            step = (sdf > mu) ? mu : std::max(0.1f * sdf, tsdf.voxelSize);
            t += step;
            prevSDF = sdf;
        }

        return false;
    }

    Eigen::Vector3f computeNormal(const Eigen::Vector3f& p)
    {
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