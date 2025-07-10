#include <iostream>

#include "rgbd_frame_stream.h"

#define BILATERAL_FILTERING true


int main() {
#ifdef _OPENMP
    std::cout << "Running with OpenMP (" << omp_get_max_threads() << " threads)\n";
#endif

    std::string filenameIn = "./data/rgbd_dataset_freiburg1_xyz/";
    std::string outputDirectory = "./results/";

    // Bilateral filtering parameters
    const float sigma_s = 3; // controls filter region: the larger -> the more distant pixels contribute -> more smoothing
    const float sigma_r = 0.1; // controls allowed depth difference: the larger -> smooths higher contrasts -> edges may be blurred

    std::string filenameBaseOut = BILATERAL_FILTERING ? "smoothedMesh_s" + std::to_string(sigma_s) + "_r" +
                                                        std::to_string(sigma_r) + "_" : "mesh_";

    // Load video
    std::cout << "Initialize frame stream..." << std::endl;
    RGBDFrameStream stream;
    if (!stream.init(filenameIn)) {
        throw std::runtime_error("Failed to initialize the frame stream!");
    }

    // Convert video to meshes
    while (stream.hasNextFrame()) {
        FrameData frameData = stream.processNextFrame();

        if (BILATERAL_FILTERING) {
            frameData.applyBilateralFilter(sigma_s, sigma_r);
        }

        frameData.computeVertexMap();
        frameData.computeNormalMap();

        std::stringstream ss;
        ss << outputDirectory << filenameBaseOut << stream.getCurrentFrameIndex() << ".off";
        frameData.writeMesh(ss.str());

    }

    return 0;
}