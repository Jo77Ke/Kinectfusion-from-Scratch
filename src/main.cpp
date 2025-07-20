#include <iostream>
#include <sstream>
#include <limits>

#include "rgbd_frame_stream.h"
#include "model.h"
#include "pose_estimation.h"
#include "output.h"

// Parameters
// Bilateral filter
constexpr float SIGMA_S = 3.0f; // controls filter region: the larger -> the more distant pixels contribute -> more smoothing
constexpr float SIGMA_R = 0.1f; // controls allowed depth difference: the larger -> smooths higher contrasts -> edges may be blurred

// Subsampling
constexpr int LEVELS = 3;

// TSDF volumetric fusion
constexpr float TSDF_VOXEL_SIZE = 0.02f;
const Vector3f TSDF_VOLUME_SIZE(5.12, 5.12, 5.12);// in m

// Pose estimation
const std::vector<int> MAX_ITERATIONS_PER_LEVEL = {10, 5, 4}; // corresponding to the levels 3, 2, 1
constexpr float TERMINATION_THRESHOLD = 1e-4f; // threshold for the change in pose estimation
constexpr float MAX_CORRESPONDENCE_DISTANCE = 0.05f; // maximum distance for correspondences in the point cloud

int main() {
#ifdef _OPENMP
    std::cout << "Running with OpenMP (" << omp_get_max_threads() << " threads)\n";
#endif

    std::string filenameIn = "./data/rgbd_dataset_freiburg1_xyz/";
    std::string outputDirectory = "./results/";
    std::string filenameBaseOut = "mesh_";

    // Load video
    std::cout << "Initialize frame stream..." << std::endl;
    RGBDFrameStream stream;
    if (!stream.init(filenameIn)) {
        throw std::runtime_error("Failed to initialize the frame stream!");
    }

    TSDFVolume model(TSDF_VOXEL_SIZE, TSDF_VOLUME_SIZE);
    std::cout << "Initialized model volume" << std::endl;

    FrameData firstFrame = stream.processNextFrame();
    CameraSpecifications cameraSpecs = firstFrame.getCameraSpecifications();

    // Requirements for computing the tsdf values (TODO: documentation on that in the class)
    firstFrame.setPose(Matrix4f::Identity());
    firstFrame.computeVertexMap();
    firstFrame.computeNormalMap();
    firstFrame.computeCameraCenterInGlobalSpace();
    std::cout << "Prepared first frame for integration" << std::endl;

    // Initialize the model with the first frame
    model.integrate(firstFrame);
    std::cout << "Integrated first frame into model" << std::endl;

    // Predict current model surface and output as mesh
    Matrix4f previousPose = firstFrame.getPose();
    model.predictSurface(cameraSpecs, previousPose);
    std::cout << "Predicted model surface after the first frame" << std::endl;

    std::stringstream ss;
    ss << outputDirectory << filenameBaseOut << stream.getCurrentFrameIndex() << ".off";
    writeMesh(
            ss.str(),
            model.getVertexMap(),
            cameraSpecs.imageWidth,
            cameraSpecs.imageHeight
    );
    std::cout << "Wrote first mesh to " << ss.str() << std::endl;

    // Convert video to meshes
    while (stream.hasNextFrame()) {
        FrameData frameData = stream.processNextFrame();
        cameraSpecs = frameData.getCameraSpecifications();

        // --- DEBUGGING ---
        std::cout << "\n--- Debugging Raw Depth Map for Current Frame (" << stream.getCurrentFrameIndex() << ") ---" << std::endl;
        const cv::Mat& rawDepth = frameData.getRawDepthMap();
        std::cout << "Raw Depth Map size: " << rawDepth.cols << "x" << rawDepth.rows << ", type: " << rawDepth.type() << std::endl;
        std::cout << "First 5x5 pixels of Raw Depth Map:" << std::endl;
        for (int i = 0; i < std::min(5, rawDepth.rows); ++i) {
            const float* rowPtr = rawDepth.ptr<const float>(i);
            for (int j = 0; j < std::min(5, rawDepth.cols); ++j) {
                std::cout << "(" << i << "," << j << "): " << rowPtr[j] << " | ";
            }
            std::cout << std::endl;
        }
        std::cout << "--- End Debugging Raw Depth Map ---" << std::endl;
        // --- END DEBUGGING ---


        // Filter and subsample raw data
        frameData.buildPyramids(LEVELS, SIGMA_S, SIGMA_R);
        std::cout << "Build pyramids for current frame" << std::endl;

        // Subsample it
        model.buildPyramids(LEVELS, SIGMA_R);
        std::cout << "BuilT pyramids for model" << std::endl;

        // Estimate pose
        Matrix4f newPose = previousPose;
        std::cout << "Estimating pose..." << std::endl;
        for (int level = LEVELS - 1; level >= 0; --level) {
            const auto &frameVertexMap = frameData.getVertexPyramidAtLevel(level);
            const auto &modelVertexMap = model.getVertexPyramidAtLevel(level);
            const auto &modelNormalMap = model.getNormalPyramidAtLevel(level);

            // --- DEBUGGING  ---
            std::cout << "\n--- Debugging ICP Inputs for Level " << level << " ---" << std::endl;
            std::cout << "Frame Vertex Map (Level " << level << ") size: " << frameVertexMap.cols << "x" << frameVertexMap.rows << ", type: " << frameVertexMap.type() << std::endl;
            std::cout << "Model Vertex Map (Level " << level << ") size: " << modelVertexMap.cols << "x" << modelVertexMap.rows << ", type: " << modelVertexMap.type() << std::endl;
            std::cout << "Model Normal Map (Level " << level << ") size: " << modelNormalMap.cols << "x" << modelNormalMap.rows << ", type: " << modelNormalMap.type() << std::endl;

            std::cout << "First 5 pixels of Frame Vertex Map (Level " << level << "):" << std::endl;
            for (int i = 0; i < std::min(5, frameVertexMap.rows); ++i) {
                const cv::Vec4f* rowPtr = frameVertexMap.ptr<const cv::Vec4f>(i);
                for (int j = 0; j < std::min(5, frameVertexMap.cols); ++j) {
                    std::cout << "(" << i << "," << j << "): " << rowPtr[j] << " | ";
                }
                std::cout << std::endl;
            }

            std::cout << "First 5 pixels of Model Vertex Map (Level " << level << "):" << std::endl;
            for (int i = 0; i < std::min(5, modelVertexMap.rows); ++i) {
                const cv::Vec4f* rowPtr = modelVertexMap.ptr<const cv::Vec4f>(i);
                for (int j = 0; j < std::min(5, modelVertexMap.cols); ++j) {
                    std::cout << "(" << i << "," << j << "): " << rowPtr[j] << " | ";
                }
                std::cout << std::endl;
            }

            std::cout << "First 5 pixels of Model Normal Map (Level " << level << "):" << std::endl;
            for (int i = 0; i < std::min(5, modelNormalMap.rows); ++i) {
                const cv::Vec4f* rowPtr = modelNormalMap.ptr<const cv::Vec4f>(i);
                for (int j = 0; j < std::min(5, modelNormalMap.cols); ++j) {
                    std::cout << "(" << i << "," << j << "): " << rowPtr[j] << " | ";
                }
                std::cout << std::endl;
            }
            std::cout << "--- End Debugging ICP Inputs ---" << std::endl;
            // --- END DEBUGGING


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
        std::cout << "Pose estimation completed" << std::endl;

        frameData.setPose(newPose);
        frameData.computeVertexMap();
        frameData.computeNormalMap();
        frameData.computeCameraCenterInGlobalSpace();
        std::cout << "Prepared current frame for integration" << std::endl;

        model.integrate(frameData);
        std::cout << "Integrated current frame into model" << std::endl;


        // Predict current model surface and output as mesh
        previousPose = frameData.getPose();
        model.predictSurface(frameData.getCameraSpecifications(), previousPose);
        std::cout << "Predicted model surface after current frame" << std::endl;

        ss.str("");
        ss.clear();
        ss << outputDirectory << filenameBaseOut << stream.getCurrentFrameIndex() << ".off";
        writeMesh(
                ss.str(),
                model.getVertexMap(),
                cameraSpecs.imageWidth,
                cameraSpecs.imageHeight
        );
        std::cout << "Wrote mesh to " << ss.str() << std::endl;
    }

    return 0;
}
