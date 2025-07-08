#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

#include "utils.h"
#include "frame_data.h"

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
        int imageWidth = 640;
        int imageHeight = 480;

        Matrix3f intrinsics = (Matrix3f() << 525.0f, 0.0f, 319.5f, 0.0f, 525.0f, 239.5f, 0.0f, 0.0f, 1.0f).finished();
        auto extrinsics = Matrix4f::Identity();

        cameraSpecifications = CameraSpecifications(
                imageWidth,
                imageHeight,
                intrinsics,
                extrinsics
        );


        // Reset index
        currentFrameIndex = -1;
        return true;
    }

    FrameData processNextFrame() {
        currentFrameIndex = currentFrameIndex == -1 ? 0 : currentFrameIndex + frameStride;
        if ((unsigned int) currentFrameIndex >= numberOfFrames) throw std::runtime_error("No more frames available");

        std::cout << "Processing frame [" << currentFrameIndex << " | " << numberOfFrames << "]" << std::endl;
        return {
                currentFrameIndex,
                cameraSpecifications,
                loadDepthMap(),
                loadColorMap(),
                findClosestTrajectory()
        };
    }

    unsigned int getCurrentFrameIndex() const {
        return currentFrameIndex;
    }

    bool hasNextFrame() const {
        return currentFrameIndex == -1 && numberOfFrames > 0 || (unsigned int) currentFrameIndex + frameStride <= numberOfFrames;
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

    bool readTrajectoryFile(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        trajectory.clear();
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
            transformation.block<3, 3>(0, 0) = q.toRotationMatrix();
            transformation.block<3, 1>(0, 3) = Eigen::Vector3f(tx, ty, tz);
            transformation = transformation.inverse().eval(); // Invert to camera pose

            trajectory.emplace_back(timestamp, transformation);
        }
        return true;
    }

    cv::Mat loadDepthMap() {
        const auto depthFilename = baseDirectory / filenamesDepthImages[currentFrameIndex];
        cv::Mat rawDepthMap = cv::imread(depthFilename, cv::IMREAD_UNCHANGED);
        if (rawDepthMap.empty()) {
            throw std::runtime_error("Empty depth image: " + depthFilename.string());
        }
        if (rawDepthMap.type() != CV_16UC1) {
            throw std::runtime_error("Invalid depth format in: " + depthFilename.string());
        }

        // Convert and scale depth map to meters (see https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
        cv::Mat depthMap;
        rawDepthMap.convertTo(depthMap, CV_32FC1, 1.0f / 5000.0f);
        // Mark invalid depth values (0.0f)
        depthMap.setTo(MINF, rawDepthMap == 0.0f);

        return depthMap;
    }

    cv::Mat loadColorMap() {
        const auto colorFilename = baseDirectory / filenamesColorImages[currentFrameIndex];
        cv::Mat colorMap = cv::imread(colorFilename, cv::IMREAD_COLOR);
        if (trajectory.empty()) {
            throw std::runtime_error("No trajectory data available");
        }

        cv::cvtColor(colorMap, colorMap, cv::COLOR_BGR2RGBA); // Convert to RGBA format

        return colorMap;
    }

    Matrix4f findClosestTrajectory() {
        double timestamp = depthImagesTimeStamps[currentFrameIndex];
        if (trajectory.empty()) {
            throw std::runtime_error("No trajectory data available");
        }
        auto closestTrajectory = std::min_element(
                trajectory.begin(),
                trajectory.end(),
                [timestamp](const auto &a, const auto &b) {
                    return std::abs(a.first - timestamp) < std::abs(b.first - timestamp);
                }
        );

        return closestTrajectory->second;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Current frame index and state
    int currentFrameIndex;
    unsigned int numberOfFrames{};
    int frameStride; // only every frameStride-th frame is used


    cv::Mat currentDepthMap;
    cv::Mat currentColorMap;
    Eigen::Matrix4f currentTrajectory;

    // Camera specifications
    CameraSpecifications cameraSpecifications;

    // Paths to the files and their timestamps
    fs::path baseDirectory;

    std::vector<std::string> filenamesDepthImages;
    std::vector<double> depthImagesTimeStamps;

    std::vector<std::string> filenamesColorImages;
    std::vector<double> colorImagesTimeStamps;

    // Timestamp-trajectory pairs
    std::vector<std::pair<double, Eigen::Matrix4f>> trajectory;
};
