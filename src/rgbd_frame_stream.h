#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <filesystem>
#include <numeric>
#include <limits>

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

        cameraSpecifications = CameraSpecifications(
                imageWidth,
                imageHeight,
                intrinsics
        );

        std::cout << "RGBDFrameStream initialized with " << numberOfFrames << " frames." << std::endl;
        return true;
    }

    FrameData processNextFrame() {
        currentFrameIndex += frameStride;
        if (currentFrameIndex >= numberOfFrames) {
            throw std::runtime_error("No more frames to process.");
        }

        // Load depth map
        fs::path depthPath = baseDirectory / filenamesDepthImages[currentFrameIndex];
        cv::Mat loadedDepthMap = cv::imread(depthPath.string(), cv::IMREAD_UNCHANGED);

        if (loadedDepthMap.empty()) {
            std::cerr << "ERROR: Failed to load depth map: " << depthPath << std::endl;
            currentDepthMap = cv::Mat(cameraSpecifications.imageHeight, cameraSpecifications.imageWidth, CV_32FC1, cv::Scalar(MINF));
        } else {
            currentDepthMap.create(cameraSpecifications.imageHeight, cameraSpecifications.imageWidth, CV_32FC1);

            if (loadedDepthMap.type() == CV_16UC1) { // Typical for TUM datasets (depth in mm, unsigned short)
                for (int r = 0; r < loadedDepthMap.rows; ++r) {
                    const ushort* srow = loadedDepthMap.ptr<ushort>(r);
                    float* drow = currentDepthMap.ptr<float>(r);
                    for (int c = 0; c < loadedDepthMap.cols; ++c) {
                        ushort depth_val_mm = srow[c];
                        if (depth_val_mm == 0) { // 0 indicates invalid depth in TUM datasets
                            drow[c] = MINF;
                        } else {
                            drow[c] = static_cast<float>(depth_val_mm) / 5000.0f; // Convert mm to meters (5000 for Kinect v1)
                        }
                    }
                }
            } else if (loadedDepthMap.type() == CV_32FC1) { // If already float, copy it and ensure 0s are MINF
                loadedDepthMap.copyTo(currentDepthMap); // Use copyTo to ensure deep copy
                currentDepthMap.setTo(MINF, currentDepthMap == 0.0f);
            } else {
                std::cerr << "WARNING: Unexpected depth map type: " << loadedDepthMap.type() << " for " << depthPath << ". Attempting generic conversion to CV_32FC1." << std::endl;
                loadedDepthMap.convertTo(currentDepthMap, CV_32FC1);
                currentDepthMap.setTo(MINF, currentDepthMap == 0.0f); // Set exact zeros to MINF after conversion
            }

            double minVal, maxVal;
            cv::minMaxLoc(currentDepthMap, &minVal, &maxVal);
            std::cout << "Loaded & Processed Depth Map " << depthPath.filename() << ": Type=" << currentDepthMap.type()
                      << ", Dims=" << currentDepthMap.cols << "x" << currentDepthMap.rows
                      << ", Min=" << minVal << ", Max=" << maxVal << std::endl;

            if (minVal == MINF && maxVal == MINF) {
                std::cerr << "WARNING: Processed depth map is entirely MINF! Data issue suspected." << std::endl;
            }
        }

        fs::path colorPath = baseDirectory / filenamesColorImages[currentFrameIndex];
        currentColorMap = cv::imread(colorPath.string(), cv::IMREAD_COLOR);
        if (currentColorMap.empty()) {
            std::cerr << "ERROR: Failed to load color map: " << colorPath << std::endl;
            currentColorMap = cv::Mat(cameraSpecifications.imageHeight, cameraSpecifications.imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        cv::cvtColor(currentColorMap, currentColorMap, cv::COLOR_BGR2RGBA);

        currentTrajectory = findClosestTrajectory();

        return FrameData(
                currentFrameIndex,
                cameraSpecifications,
                std::move(currentDepthMap),
                std::move(currentColorMap),
                currentTrajectory
        );
    }

    bool hasNextFrame() const {
        return (currentFrameIndex + frameStride) < numberOfFrames;
    }

    int getCurrentFrameIndex() const {
        return currentFrameIndex;
    }

private:
    bool readFileList(const fs::path &filepath, std::vector<std::string> &filenames, std::vector<double> &timestamps) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open file: " << filepath << std::endl;
            return false;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue; // Skip comments and empty lines
            std::stringstream ss(line);
            double timestamp;
            std::string filename;
            ss >> timestamp >> filename;
            timestamps.push_back(timestamp);
            filenames.push_back(filename);
        }
        std::cout << "Read " << filenames.size() << " entries from " << filepath.filename() << std::endl;
        return true;
    }

    bool readTrajectoryFile(const fs::path &filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open trajectory file: " << filepath << std::endl;
            return false;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue; // Skip comments and empty lines
            std::stringstream ss(line);
            double timestamp;
            float tx, ty, tz, qx, qy, qz, qw;
            ss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

            // Convert quaternion to rotation matrix
            Eigen::Quaternionf q(qw, qx, qy, qz);
            Eigen::Matrix3f R = q.normalized().toRotationMatrix();

            // Create 4x4 transformation matrix
            Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
            T.block<3, 3>(0, 0) = R;
            T.block<3, 1>(0, 3) = Eigen::Vector3f(tx, ty, tz);

            trajectory.emplace_back(timestamp, T);
        }
        std::cout << "Read " << trajectory.size() << " trajectory entries from " << filepath.filename() << std::endl;
        return true;
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
