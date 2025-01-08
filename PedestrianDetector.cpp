#include <opencv2/opencv.hpp> // OpenCV for image processing and point handling
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include "Utilities.h" 

// Ground truth for pedestrian crossings. Each row contains
// 1. the image number (PC?.jpg)
// 2. the coordinates of the line at the top of the pedestrian crossing 
//    (left column, left row, right column, right row)
// 3. the coordinates of the line at the bottom of the pedestrian crossing 
//    (left column, left row, right column, right row)


int pedestrian_crossing_ground_truth[][9] = {
    { 10,0,132,503,113,0,177,503,148},
    { 11,0,131,503,144,0,168,503,177},
    { 12,0,154,503,164,0,206,503,213},
    { 13,0,110,503,110,0,156,503,144},
    { 14,0,95,503,104,0,124,503,128},
    { 15,0,85,503,91,0,113,503,128},
    { 16,0,65,503,173,0,79,503,215},
    { 17,0,43,503,93,0,89,503,146},
    { 18,0,122,503,117,0,169,503,176},
    {19, 0, 0, 0, 0, 0, 0, 0, 0},
    { 20,0,157,503,131,0,223,503,184},
    { 21,0,140,503,136,0,190,503,183},
    { 22,0,114,503,97,0,140,503,123},
    { 23,0,133,503,122,0,198,503,186},
    { 24,0,107,503,93,0,146,503,118},
    { 25,0,58,503,164,0,71,503,204},
    { 26,0,71,503,131,0,106,503,199},
    { 27,0,138,503,151,0,179,503,193},
    {28, 0, 0, 0, 0, 0, 0, 0, 0},
    {29, 0, 0, 0, 0, 0, 0, 0, 0}
};


using namespace cv;
using namespace std;

double calculateIoU(const std::vector<cv::Point>& groundPoly, const std::vector<cv::Point>& obtainedPoly) {
    // Create two masks for the ground truth and obtained polygons

    cv::Mat image = cv::imread("Media/PC10.jpg");


    // Get the dimensions of the image
    cv::Size imageSize = image.size();

    // Create two masks for the ground truth and obtained polygons with the same size as the image
    cv::Mat groundMask = cv::Mat::zeros(imageSize, CV_8UC1);

    cv::Mat obtainedMask = cv::Mat::zeros(imageSize, CV_8UC1);


    // Draw the polygons on the masks
    cv::fillConvexPoly(groundMask, groundPoly.data(), groundPoly.size(), cv::Scalar(255));
    cv::fillConvexPoly(obtainedMask, obtainedPoly.data(), obtainedPoly.size(), cv::Scalar(255));

    // Calculate the intersection and union areas
    cv::Mat intersection;
    cv::bitwise_and(groundMask, obtainedMask, intersection);
    double intersectionArea = cv::countNonZero(intersection);
    double areaGround = cv::countNonZero(groundMask);
    double areaObtained = cv::countNonZero(obtainedMask);
	double unionArea = areaGround + areaObtained - intersectionArea;


    // Calculate and return the IoU
    return intersectionArea / unionArea;
}

std::vector<std::string> evaluateGroundTruth(const int groundTruth[][9], const std::vector<std::vector<int>>& obtainedResults, int size) {
    std::vector<std::string> errors;

    for (int i = 0; i < size; ++i) {

        // Define ground truth and obtained points as polygons
        std::vector<cv::Point> groundPoly = {
            cv::Point(groundTruth[i][1], groundTruth[i][2]),
            cv::Point(groundTruth[i][3], groundTruth[i][4]),
            cv::Point(groundTruth[i][7], groundTruth[i][8]),
            cv::Point(groundTruth[i][5], groundTruth[i][6])
        };

        std::vector<cv::Point> obtainedPoly = {
            cv::Point(obtainedResults[i][1], obtainedResults[i][2]),
            cv::Point(obtainedResults[i][3], obtainedResults[i][4]),
            cv::Point(obtainedResults[i][7], obtainedResults[i][8]),
            cv::Point(obtainedResults[i][5], obtainedResults[i][6])
        };


        // Calculate Intersection over Union (IoU)
        double iou = calculateIoU(groundPoly, obtainedPoly);



        // Store error and IoU for each image
        errors.push_back("img " + std::to_string(groundTruth[i][0]) + " -> :  IoU: " + std::to_string(iou));
    }

    return errors;
}




std::vector<int> extractPedestrianCrossingData(const std::string& image_path,
    const cv::Point& upperPoint1,
    const cv::Point& upperPoint2,
    const cv::Point& lowerPoint1,
    const cv::Point& lowerPoint2) {

    std::string fileName = image_path.substr(image_path.find_last_of("/\\") + 1);
    std::string imageNumberStr = fileName.substr(2, fileName.find('.') - 2);
    int imageNumber = std::stoi(imageNumberStr);

    std::vector<int> result;

    result.push_back(imageNumber);

    result.push_back(upperPoint1.x);
    result.push_back(upperPoint1.y);
    result.push_back(upperPoint2.x);
    result.push_back(upperPoint2.y);

    result.push_back(lowerPoint1.x);
    result.push_back(lowerPoint1.y);
    result.push_back(lowerPoint2.x);
    result.push_back(lowerPoint2.y);

    return result;
}



std::pair<cv::Point, cv::Point> drawLineOfBestFit(cv::Mat& img, const std::vector<cv::Point2f>& points) {
    if (points.size() < 2) {
        std::cout << "Insufficient points to calculate the line." << std::endl;
        return { cv::Point(0, 0), cv::Point(0, 0) };
    }

    // Calculate the mean of the points
    double sumX = 0, sumY = 0;
    for (const auto& point : points) {
        sumX += point.x;
        sumY += point.y;
    }
    double meanX = sumX / points.size();
    double meanY = sumY / points.size();

    // Calculate slope (m) and intercept (b) for line of best fit
    double num = 0; 
    double den = 0; 
    for (const auto& point : points) {
        num += (point.x - meanX) * (point.y - meanY);
        den += (point.x - meanX) * (point.x - meanX);
    }
    double m = num / den; 
    double b = meanY - m * meanX; 

    // Define the line points: first at x = 0
    int x1 = 0;
    int y1 = static_cast<int>(m * x1 + b);
    y1 = std::max(0, std::min(y1, img.rows - 1)); // Ensure y1 is within image bounds
    cv::Point point1(x1, y1);

    // Define the second point at x = 503
    int x2 = 503;
    int y2 = static_cast<int>(m * x2 + b);
    y2 = std::max(0, std::min(y2, img.rows - 1)); 
    cv::Point point2(x2, y2);

    cv::line(img, point1, point2, cv::Scalar(255, 0, 0), 2); // Draw a red line

    return { point1, point2 };
}

void saveImageWithSuffix(const cv::Mat& image, const std::string& originalPath, const std::string& suffix) {
    std::string directory = originalPath.substr(0, originalPath.find_last_of("/\\"));
    std::string filename = originalPath.substr(originalPath.find_last_of("/\\") + 1);

    std::string baseName = filename.substr(0, filename.find_last_of('.'));

    std::string newFilename = directory + "/" + baseName + "_" + suffix;  

    cv::imwrite(newFilename + ".png", image); 
}

// Function to process an image and extract pedestrian crossing data
std::pair<cv::Mat, std::vector<int>> DetectCrossing(const std::string& imagePath) {
    // Load the image
    cv::Mat originalImage = cv::imread(imagePath);

    // Convert the loaded image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(originalImage, grayImage, cv::COLOR_BGR2GRAY);

    // Normalize the histogram of the grayscale image
    cv::Mat normalizedImage;
    cv::equalizeHist(grayImage, normalizedImage);

    // Apply median blur to reduce noise
    cv::Mat filteredImage;
    cv::medianBlur(normalizedImage, filteredImage, 5);

    // Binarize the filtered image
    cv::Mat binaryImage;
    cv::threshold(filteredImage, binaryImage, 230, 255, cv::THRESH_BINARY);

    // Detect contours in the binary image
    std::vector<std::vector<cv::Point>> detectedContours;
    cv::findContours(binaryImage, detectedContours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);


    //save image with contours
    cv::Mat contoursImage2 = originalImage.clone();
    cv::drawContours(contoursImage2, detectedContours, -1, cv::Scalar(255, 0, 0), 2);

    // Approximate the detected contours to reduce the number of points
    std::vector<std::vector<cv::Point>> approximatedContours(detectedContours.size());
    for (size_t i = 0; i < detectedContours.size(); i++) {
        double epsilon = 0.02 * cv::arcLength(detectedContours[i], true);
        cv::approxPolyDP(detectedContours[i], approximatedContours[i], epsilon, true);
    }

    // Filter for contours with four or five sides
    std::vector<std::vector<cv::Point>> fourSidedContours;
    for (const auto& approx : approximatedContours) {
        if (approx.size() == 4) {
            fourSidedContours.push_back(approx);
        }
    }



    // Define thresholds for rectangularity and elongation
    double rectangularityThreshold = 0.5;
    double elongationThreshold = 1.85;

    // Create a blank image to draw good contours
    std::vector<std::vector<cv::Point>> goodContours;

    // Analyze rectangularity and elongation of each contour
    for (const auto& contour : fourSidedContours) {
        cv::RotatedRect minRect = cv::minAreaRect(contour); 
        double contourArea = cv::contourArea(contour); 
        double rectangleArea = minRect.size.width * minRect.size.height; 
        double rectangularity = contourArea / rectangleArea; 

        // Calculate inertia moments
        cv::Moments moments = cv::moments(contour);
        double mu20 = moments.mu20 / moments.m00;
        double mu02 = moments.mu02 / moments.m00;
        double mu11 = moments.mu11 / moments.m00;

        // Calculate eigenvalues from covariance matrix
        double trace = mu20 + mu02;
        double det = mu20 * mu02 - mu11 * mu11;
        double eigenvalue1 = trace / 2 + std::sqrt(trace * trace / 4 - det);
        double eigenvalue2 = trace / 2 - std::sqrt(trace * trace / 4 - det);

        // Calculate elongation as the ratio of eigenvalues
        double elongation = eigenvalue1 / eigenvalue2;

        // If the rectangularity and elongation exceed the thresholds, save the contour
        if (rectangularity > rectangularityThreshold && elongation > elongationThreshold && contourArea > 200) {
            goodContours.push_back(contour);
        }
    }


    // Define grid dimensions based on image size
    int gridWidth = originalImage.cols; 
    int gridHeight = originalImage.rows / 6;  

    // Map to group contours based on their position
    std::map<std::pair<int, int>, std::vector<std::vector<cv::Point>>> contourGroups;

    // Group good contours into the grid
    for (const auto& contour : goodContours) {
        cv::Rect boundingRect = cv::boundingRect(contour);

        // Determine the grid position of the contour
        int xKey = static_cast<int>(boundingRect.x / gridWidth); 
        int yKey = static_cast<int>(boundingRect.y / gridHeight); 

        contourGroups[{yKey, xKey}].push_back(contour);
    }

    std::vector<std::vector<cv::Point>> pedestrianCrossings;

    // Collect groups with at least 3 contours
    for (const auto& pair : contourGroups) {
        const auto& alignedContours = pair.second;
        if (alignedContours.size() >= 3) {
            pedestrianCrossings.insert(pedestrianCrossings.end(), alignedContours.begin(), alignedContours.end());
        }
    }


    // Clone the original image for drawing purposes
    cv::Mat resultImage = originalImage.clone();

    std::vector<cv::Point2f> lowerPoints;
    std::vector<cv::Point2f> upperPoints;

    // Iterate over each detected pedestrian crossing
    for (const auto& contour : pedestrianCrossings) {
        // Find the points with minimum and maximum y coordinates
        auto minYPoint = *std::min_element(contour.begin(), contour.end(), [](const cv::Point& a, const cv::Point& b) {
            return a.y < b.y;
            });
        auto maxYPoint = *std::max_element(contour.begin(), contour.end(), [](const cv::Point& a, const cv::Point& b) {
            return a.y < b.y;
            });

        // Add the highest and lowest points
        lowerPoints.push_back(maxYPoint);
        upperPoints.push_back(minYPoint);
    }




    // Draw the line of best fit for the lower points
    auto lowerLinePoints = drawLineOfBestFit(resultImage, lowerPoints);
    cv::Point lowerPoint1 = lowerLinePoints.first;
    cv::Point lowerPoint2 = lowerLinePoints.second;

    // Draw the line of best fit for the upper points
    auto upperLinePoints = drawLineOfBestFit(resultImage, upperPoints);
    cv::Point upperPoint1 = upperLinePoints.first;
    cv::Point upperPoint2 = upperLinePoints.second;

    // Extract pedestrian crossing data using the defined points
    std::vector<int> pedestrianCrossingData = extractPedestrianCrossingData(imagePath, upperPoint1, upperPoint2, lowerPoint1, lowerPoint2);

    // Annotate the result image with the image number from the filename
    cv::putText(resultImage, imagePath.substr(imagePath.size() - 6, 2), cv::Point(450, 450), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    saveImageWithSuffix(resultImage, imagePath, "result");



    return { resultImage, pedestrianCrossingData };
}

cv::Mat drawGroundTruth(const string& imagePath, int groundTruth[][9], int i) {
    	Mat image = imread(imagePath);


    Point topLeft(groundTruth[i][1], groundTruth[i][2]);
    Point topRight(groundTruth[i][3], groundTruth[i][4]);
    Point bottomLeft(groundTruth[i][5], groundTruth[i][6]);
    Point bottomRight(groundTruth[i][7], groundTruth[i][8]);

    line(image, topLeft, topRight, Scalar(0, 255, 0), 2); 
    line(image, bottomLeft, bottomRight, Scalar(0, 255, 0), 2);

    cv::putText(image, imagePath.substr(imagePath.size() - 6, 2), cv::Point(250, 350), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);


	return image;
    
}

cv::Mat createImageGrid(std::vector<cv::Mat>& images, int cols) {
    if (images.empty()) {
        throw std::runtime_error("No lines");
    }


    int newWidth = images[0].cols / 2 ; 
    int newHeight = images[0].rows / 2;

    for (auto& img : images) {
        cv::resize(img, img, cv::Size(newWidth, newHeight));
    }

    int n = images.size();
    int rows = (n + cols - 1) / cols; 

    cv::Mat gridImage(rows * newHeight, cols * newWidth, images[0].type(), cv::Scalar(0, 0, 0));

    for (int i = 0; i < n; ++i) {
        int row = i / cols;
        int col = i % cols;
        images[i].copyTo(gridImage(cv::Rect(col * newWidth, row * newHeight, newWidth, newHeight)));
    }

    return gridImage; 
}

void MyApplication(const std::string& input_image) {
    std::vector<cv::Mat> img;
    std::vector<std::vector<int>> myCrossingGroundTruth; 

    // Process single input image
    auto result = DetectCrossing(input_image);
    img.push_back(result.first);
    std::vector<int> pedestrianCrossingData = result.second;
    
    // Add valid data only
    if (!pedestrianCrossingData.empty()) {
        myCrossingGroundTruth.push_back(pedestrianCrossingData);
    }

    // Display the input image with results
    cv::imshow("Result", result.first);

    // Call to evaluate ground truth
    std::vector<std::string> errors = evaluateGroundTruth(pedestrian_crossing_ground_truth, myCrossingGroundTruth, 1);

    // Print errors
    for (const auto& error : errors) {
        std::cout << error << std::endl;
    }

    // Create an image to show results
    cv::Mat resultImage = cv::Mat::zeros(800, 800, CV_8UC3); 
    cv::Scalar textColor = cv::Scalar(255, 255, 255); 
    
    // Initial position for text
    int lineHeight = 30;
    int yOffset = 30; 

    // Iterate over the error vector
    for (const auto& error : errors) {
        cv::putText(resultImage, error, cv::Point(10, yOffset), cv::FONT_HERSHEY_SIMPLEX, 0.7, textColor, 1);
        yOffset += lineHeight; 
    }

    // Display the results
    cv::imshow("Stats", resultImage);
    cv::waitKey(0); 
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <input_image_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " Media/PC10.jpg" << std::endl;
        return -1;
    }

    try {
        std::string input_image = argv[1];
        
        // Verify file exists
        if (!std::filesystem::exists(input_image)) {
            std::cerr << "Error: Image file " << input_image << " does not exist." << std::endl;
            return -1;
        }

        MyApplication(input_image);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing image: " << e.what() << std::endl;
        return -1;
    }
}