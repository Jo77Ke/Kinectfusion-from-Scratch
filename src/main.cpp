#include <iostream>

#include "rgbd_frame_stream.h"
#include "model.h"
#include "pose_estimation.h"
#include "output.h"

// Parameters
// Bilateral filter
constexpr float SIGMA_S = 3.0f; // controls filter region: the larger -> the more distant pixels contribute -> more smoothing
constexpr float SIGMA_R = 0.1f; // controls allowed depth difference: the larger -> smooths higher contrasts -> edges may be blurred

// Subsampling
constexpr int LEVELS = 1;

// TSDF volumetric fusion
constexpr float TSDF_VOXEL_SIZE = 0.02f;
const Vector3f TSDF_VOLUME_SIZE(5.12f, 5.12f, 5.12f);// in m

// Pose estimation
const std::vector<int> MAX_ITERATIONS_PER_LEVEL = {10, 5, 4}; // corresponding to the levels 3, 2, 1
constexpr float TERMINATION_THRESHOLD = 1e-4f; // threshold for the change in pose estimation
constexpr float MAX_CORRESPONDENCE_DISTANCE = 0.1f; // maximum distance for correspondences in the point cloud

int main() {
#ifdef _OPENMP
    std::cout << "Running with OpenMP (" << omp_get_max_threads() << " threads)\n";
#endif

    // I/O paths
    std::string filenameIn = "./data/rgbd_dataset_freiburg1_xyz/";
    std::string outputDirectory = "./results/";
    std::string filenameBaseOut = "mesh_";

    // Load video
    RGBDFrameStream stream;
    if (!stream.init(filenameIn)) {
        throw std::runtime_error("Failed to initialize the frame stream!");
    }

    TSDFVolume model(TSDF_VOXEL_SIZE, TSDF_VOLUME_SIZE);

    FrameData firstFrame = stream.processNextFrame();
    CameraSpecifications cameraSpecs = firstFrame.getCameraSpecifications();

    // Requirements for computing the tsdf values
    firstFrame.setPose(Matrix4f::Identity());
    firstFrame.computeVertexMap();
    firstFrame.computeNormalMap();
    firstFrame.computeWorldToCameraCenter();

    // Initialize the model with the first frame
    model.integrate(firstFrame);
    Matrix4f previousPose = firstFrame.getPose();

    // Predict current model surface and output as mesh
    model.predictSurface(cameraSpecs, previousPose);
    std::stringstream ss;
    ss << outputDirectory << filenameBaseOut << stream.getCurrentFrameIndex() << ".off";
    writeMesh(
            ss.str(),
            model.getVertexMap()
    );

    // Convert video to meshes
    while (stream.hasNextFrame()) {
        FrameData frameData = stream.processNextFrame();
        cameraSpecs = frameData.getCameraSpecifications();

        // Filter and subsample raw data
        frameData.buildPyramids(LEVELS, SIGMA_S, SIGMA_R);

        // Estimate pose
        Matrix4f newPose = previousPose;
        for (int level = LEVELS - 1; level >= 0; --level) {
            const auto &frameVertexMap = frameData.getDepthMapAtPyramidLevel(level);
            const auto &modelVertexMap = model.getVertexMap();
            const auto &modelNormalMap = model.getNormalMap();

            const IcpParameters icpParams(MAX_ITERATIONS_PER_LEVEL[level], TERMINATION_THRESHOLD,
                                          MAX_CORRESPONDENCE_DISTANCE);

            newPose = estimateCameraPoseICP(
                    frameVertexMap,
                    cameraSpecs,
                    modelVertexMap,
                    modelNormalMap,
                    newPose,
                    icpParams
            );
        }

        frameData.setPose(newPose);
        frameData.computeVertexMap();
        frameData.computeNormalMap();
        frameData.computeWorldToCameraCenter();

        model.integrate(frameData);

        previousPose = frameData.getPose();

        // Predict current model surface and output as mesh
        model.predictSurface(frameData.getCameraSpecifications(), previousPose);

        ss.str("");
        ss.clear();
        ss << outputDirectory << filenameBaseOut << stream.getCurrentFrameIndex() << ".off";
        writeMesh(
                ss.str(),
                model.getVertexMap()
        );
    }

    return 0;
}