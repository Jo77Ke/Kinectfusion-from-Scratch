#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

#include "utils.h"

class RGBDFrameStream {
public:
    RGBDFrameStream() : currentFrameIndex(-1), frameStride(10) {}

    ~RGBDFrameStream() = default;

    bool init(const fs::path &datasetDirectory) {
        baseDirectory = datasetDirectory;

        // Read input file names and trajectories
        if (!readFileList(datasetDirectory / "depth.txt", filenamesDepthImages, depthImagesTimeStamps)) return false;
        if (!readFileList(datasetDirectory / "rgb.txt", filenamesColorImages, colorImagesTimeStamps)) return false;
        if (!readTrajectoryFile(datasetDirectory / "groundtruth.txt")) return false;

        if (filenamesDepthImages.size() != filenamesColorImages.size()) return false;
        numberOfFrames = filenamesDepthImages.size();


        // Set camera intrinsics and extrinsics
        depthImageWidth = colorImageWidth = 640;
        depthImageHeight = colorImageHeight = 480;

        depthIntrinsics << 525.0f, 0.0f, 319.5f,
                            0.0f, 525.0f, 239.5f,
                            0.0f, 0.0f, 1.0f;
        colorIntrinsics = depthIntrinsics;

        depthExtrinsics.setIdentity();
        colorExtrinsics.setIdentity();


        // Reset index
        currentFrameIndex = -1;
        return true;
    }

    bool processNextFrame() {
        currentFrameIndex = currentFrameIndex == -1 ? 0 : currentFrameIndex + frameStride;
        if ((unsigned int) currentFrameIndex >= numberOfFrames) return false;

        std::cout << "Processing frame [" << currentFrameIndex << " | " << numberOfFrames << "]" << std::endl;

        // Load depth image
        std::string depthFilename = baseDirectory / filenamesDepthImages[currentFrameIndex];
        cv::Mat rawDepthMap = cv::imread(depthFilename, cv::IMREAD_UNCHANGED);
        if (rawDepthMap.empty() || rawDepthMap.type() != CV_16UC1) {
            std::cerr << "Failed to load depth image: " << depthFilename << std::endl;
            return false;
        }

        // Convert and scale depth map to meters (see https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
        rawDepthMap.convertTo(currentDepthMap, CV_32FC1, 1.0f / 5000.0f);
        // Mark invalid depth values (0.0f)
        currentDepthMap.setTo(MINF, rawDepthMap == 0.0f);

        // Load color image
        std::string colorFilename = baseDirectory / filenamesColorImages[currentFrameIndex];
        cv::Mat rawColorMap = cv::imread(colorFilename, cv::IMREAD_COLOR);
        if (rawColorMap.empty()) {
            std::cerr << "Failed to load color image: " << colorFilename << std::endl;
            return false;
        }

        cv::cvtColor(rawColorMap, currentColorMap, cv::COLOR_BGR2RGBA); // Convert to RGBA format


        // Find the closest trajectory
        double timestamp = depthImagesTimeStamps[currentFrameIndex];
        double min = std::numeric_limits<double>::max();
        size_t idx = 0;
        for (size_t i = 0; i < trajectory.size(); ++i)
        {
            double d = abs(trajectoryTimeStamps[i] - timestamp);
            if (min > d)
            {
                min = d;
                idx = i;
            }
        }
        currentTrajectory = trajectory[idx];

        return true;
    }

    unsigned int getCurrentFrameIndex() const {
        return (unsigned int) currentFrameIndex;
    }

    unsigned int getNumberOfFrames() const {
        return numberOfFrames;
    }

    const cv::Mat& getCurrentDepthMap() const {
        return currentDepthMap;
    }

    const cv::Mat& getCurrentColorMap() const {
        return currentColorMap;
    }

    const Eigen::Matrix4f& getCurrentTrajectory() const {
        return currentTrajectory;
    }

    const Eigen::Matrix3f& getDepthIntrinsics() const {
        return depthIntrinsics;
    }

    const Eigen::Matrix3f& getColorIntrinsics() const {
        return colorIntrinsics;
    }

    const Eigen::Matrix4f& getDepthExtrinsics() const {
        return depthExtrinsics;
    }

    const Eigen::Matrix4f& getColorExtrinsics() const {
        return colorExtrinsics;
    }

    unsigned int getDepthImageWidth() const {
        return depthImageWidth;
    }

    unsigned int getDepthImageHeight() const {
        return depthImageHeight;
    }

    unsigned int getColorImageWidth() const {
        return colorImageWidth;
    }

    unsigned int getColorImageHeight() const {
        return colorImageHeight;
    }

private:
    static bool readFileList(
            const fs::path &pathToFilenameList,
            std::vector<std::string> &filenames,
            std::vector<double> &timestamps
    ) {
        std::ifstream fileDepthList(pathToFilenameList, std::ios::in);
        if (!fileDepthList.is_open()) return false;

        filenames.clear();
        timestamps.clear();

        std::string line;
        while (fileDepthList.good() && std::getline(fileDepthList, line)) {
            if (line.empty() || line[0] == '#') continue; // Skip comments and empty lines

            std::istringstream ss(line);
            double timestamp;
            std::string filename;

            ss >> timestamp >> filename;
            if (filename.empty()) break;

            timestamps.push_back(timestamp);
            filenames.push_back(filename);
        }
        fileDepthList.close();
        return true;
    }

    bool readTrajectoryFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        trajectory.clear();
        trajectoryTimeStamps.clear();
        std::string line;
        while (file.good() && std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue; // Skip comments and empty lines

            std::istringstream ss(line);
            double timestamp;
            float tx, ty, tz, qx, qy, qz, qw;
            ss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

            Eigen::Quaternionf q(qw, qx, qy, qz);
            if (q.norm() == 0) {
                std::cerr << "Invalid quaternion in trajectory file: " << line << std::endl;
            }

            Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
            transformation.block<3,3>(0,0) = q.toRotationMatrix();
            transformation.block<3,1>(0,3) = Eigen::Vector3f(tx, ty, tz);
            transformation = transformation.inverse().eval(); // Invert to camera pose

            trajectoryTimeStamps.push_back(timestamp);
            trajectory.push_back(transformation);
        }
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Current frame index and state
    int currentFrameIndex;
    unsigned int numberOfFrames{};
    int frameStride; // only every frameStride-th frame is used

    cv::Mat currentDepthMap;
    cv::Mat currentColorMap;
    Eigen::Matrix4f currentTrajectory;

    // Camera intrinsics
    Eigen::Matrix3f depthIntrinsics;
    Eigen::Matrix3f colorIntrinsics;

    unsigned int depthImageWidth{};
    unsigned int depthImageHeight{};
    unsigned int colorImageWidth{};
    unsigned int colorImageHeight{};

    // Camera extrinsics
    Eigen::Matrix4f depthExtrinsics;
    Eigen::Matrix4f colorExtrinsics;

    // Paths to the files and their timestamps
    fs::path baseDirectory;

    std::vector<std::string> filenamesDepthImages;
    std::vector<double> depthImagesTimeStamps;

    std::vector<std::string> filenamesColorImages;
    std::vector<double> colorImagesTimeStamps;

    std::vector<Eigen::Matrix4f> trajectory;
    std::vector<double> trajectoryTimeStamps;


};
