# Environment-Anomaly-Detection-Using-Satellite-Images
Environment Anomaly Detection using standard machine learning techniques.

## Introduction

The technique uses PCA and KMeans to detect anomaly. The code is based on the paper: https://ieeexplore.ieee.org/document/5196726. 

Blog link: https://medium.com/omdena/environmental-anomaly-detection-as-a-result-of-climate-change-using-satellite-images-34109977c09d

#### detect_anomaly.Anomaly (first_image_path = None, second_image_path = None, block_size = 5, return_diff = False)

### Parameters

#### first_image_path: String
  Image path of the first image of the particular location
#### second_image_path: String
  Image path of the second image of the same location
#### block_size: Integer
  Size of the window that will be used to create eigen space and vector space 
  * Default: 5 if not set
  * Choose it to be an odd number
#### return_diff: Bool
  If you want function to return difference image as well
  
  
### Returns

#### change_map: matrix
  Image matrix depicting the change.
  * Black region showing no change
  * White region showing change
 
#### diff_image: matrix
  Difference image is the absolute difference between two given image.

