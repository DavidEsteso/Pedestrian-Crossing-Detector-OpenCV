# Pedestrian Crossing Detector

## Overview
This C++ application uses computer vision techniques to detect and analyze pedestrian crossings in images. The program employs OpenCV to process images, detect crossing patterns, and evaluate the accuracy of detections against ground truth data. PLEASE SEE THE REPORT

## Key Features
- Automated detection of pedestrian crossings in images
- Ground truth comparison and evaluation using IoU (Intersection over Union)
- Contour detection and filtering
- Line of best fit calculation for crossing boundaries
- Image preprocessing pipeline
- Grid-based contour grouping
- Results visualization and statistical analysis

## Technical Skills Demonstrated

### Computer Vision & Image Processing
- Advanced contour detection and manipulation
- Image preprocessing techniques (histogram equalization, thresholding)
- Geometric shape analysis and filtering
- Custom feature detection algorithms
- Noise reduction techniques

### Mathematics & Algorithms
- Statistical analysis for line fitting
- Geometric computations
- Moment calculations
- Eigenvalue analysis
- Grid-based spatial clustering

### Software Engineering
- Object-oriented design
- Error handling and input validation
- File I/O operations
- Command-line argument processing
- Modular code structure
- Memory management

### Data Analysis
- Ground truth comparison
- IoU calculation
- Statistical reporting
- Performance metrics
- Results visualization

## Implementation Details

### Image Processing Pipeline
1. Grayscale conversion
2. Histogram normalization
3. Median blur application
4. Binary thresholding
5. Contour detection and approximation
6. Shape analysis and filtering
7. Grid-based grouping
8. Line fitting and boundary detection

### Evaluation Methods
- IoU (Intersection over Union) calculation
- Ground truth comparison
- Statistical analysis of results
- Visual result verification

### Visualization Features
- Annotated result images
- Statistical reports
- Debug visualizations
- Grid-based result display

## Usage
```bash
./PedestrianCrossingDetector <input_image_path>
Example: ./PedestrianCrossingDetector Media/PC10.jpg
```

## Technical Requirements
- C++ 17 or higher
- OpenCV 4.x
- Standard C++ libraries
- CMake for building

## Error Handling
- Input validation
- File existence checking
- Exception handling for image processing
- Robust error reporting

## Output
- Processed images with detected crossings
- Statistical analysis of detection accuracy
- IoU scores for each detection
- Visual debugging information

## Skills Context
This project demonstrates expertise in:
- Computer vision algorithms
- Image processing techniques
- Statistical analysis
- Performance evaluation
- Software architecture
- Data visualization
- Error handling
- Documentation

## Future Improvements
- Real-time video processing
- Machine learning integration
- Performance optimization
- Additional evaluation metrics
- Enhanced visualization options
- Batch processing capabilities