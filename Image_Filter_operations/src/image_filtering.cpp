/*
 * Q1.cpp
 *
 *  Created on: 16-Oct-2015
 *      Author: virendra
 */


#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#define PI 3.14159265
#define root2 1.414
using namespace cv;
using namespace std;

int i,j,count=0;
int ResultantMatrix[3];
Mat createLowGaussianFilterImage(Mat image){
	Mat image3(image.cols, image.rows, CV_32F, Scalar(0, 0, 0));
	int d = 40;
	float power;
	int shift1 = image3.rows/2;
	int shift2 = image3.cols/2;
	for(int i=0;i< image3.rows;i++){
		for(int j=0;j<image3.cols;j++){
			power = -(pow(i-shift1,2)+pow(j-shift2,2))/(2*pow(d,2));
			image3.at<float>(i,j)= exp(power);
		}
	}

	return image3;
}

Mat createHighGaussianFilterImage(Mat image){
	Mat image3(image.cols, image.rows, CV_32F, Scalar(0, 0, 0));
	int d = 30;
	float power;
	int shift1 = image3.rows/2;
	int shift2 = image3.cols/2;
	for(int i=0;i< image3.rows;i++){
		for(int j=0;j<image3.cols;j++){
			power = -(pow(i-shift1,2)+pow(j-shift2,2))/(2*pow(d,2));
			image3.at<float>(i,j)= 1 - exp(power);
		}
	}

	return image3;
}

Mat butterworth2LowFilter(Mat image){
	Mat image3(image.cols, image.rows, CV_32F, Scalar(0, 0, 0));
	int d = 40;
	int order = 1;
	float power;
	int shift1 = image3.rows/2;
	int shift2 = image3.cols/2;
	for(int i=0;i< image3.rows;i++){
		for(int j=0;j<image3.cols;j++){
			power = 1+(root2 - 1)*(pow(i-shift1,2*order)+pow(j-shift2,2*order))/(pow(d,2*order));
			image3.at<float>(i,j)= 1/power;
		}
	}
	return image3;
}
Mat butterworthLowFilter(Mat image){
	Mat image3(image.cols, image.rows, CV_32F, Scalar(0, 0, 0));
	int d = 40;
	int order = 1;
	float power;
	int shift1 = image3.rows/2;
	int shift2 = image3.cols/2;
	for(int i=0;i< image3.rows;i++){
		for(int j=0;j<image3.cols;j++){
			power = 1+(pow(i-shift1,2*order)+pow(j-shift2,2*order))/(pow(d,2*order));
			image3.at<float>(i,j)= 1/power;
		}
	}
	return image3;
}
Mat butterworthHighFilter(Mat image){
	Mat image3(image.cols, image.rows, CV_32F, Scalar(0, 0, 0));
		int d = 40;
		int order = 1;
		float power;
		int shift1 = image3.rows/2;
		int shift2 = image3.cols/2;
		for(int i=0;i< image3.rows;i++){
			for(int j=0;j<image3.cols;j++){
				power = 1+(pow(i-shift1,2*order)+pow(j-shift2,2*order))/(pow(d,2*order));
				image3.at<float>(i,j)= 1 - 1/power;
			}
		}

		return image3;
}

Mat idealHighFilter(Mat image){
	Mat image3(image.cols, image.rows, CV_32F, Scalar(0, 0, 0));
		int d = 40;
		int order = 1;
		float power;
		int shift1 = image3.rows/2;
		int shift2 = image3.cols/2;
		for(int i=0;i< image3.rows;i++){
			for(int j=0;j<image3.cols;j++){
				if((pow(i-shift1,2*order)+pow(j-shift2,2*order))<= pow(d,2)){
					image3.at<float>(i,j) = 0;
				}
				else{
					image3.at<float>(i,j) = 1;
				}
				
			}
		}

		return image3;
}

Mat idealLowFilter(Mat image){
	Mat image3(image.cols, image.rows, CV_32F, Scalar(0, 0, 0));
		int d = 40;
		int order = 1;
		float power;
		int shift1 = image3.rows/2;
		int shift2 = image3.cols/2;
		for(int i=0;i< image3.rows;i++){
			for(int j=0;j<image3.cols;j++){
				if((pow(i-shift1,2*order)+pow(j-shift2,2*order))<= pow(d,2)){
					image3.at<float>(i,j) = 1;
				}
				else{
					image3.at<float>(i,j) = 0;
				}
				
			}
		}

		return image3;
}

void quadrantSwap(Mat image ){
	int cx = image.cols/2;
	int cy = image.rows/2;
	//Swapping Quadrant
	Mat tmp;
	Mat q0(image, Rect(0, 0, cx, cy));
	Mat q1(image, Rect(cx, 0, cx, cy));
	Mat q2(image, Rect(0, cy, cx, cy));
	Mat q3(image, Rect(cx, cy, cx, cy));
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
		   
}
Mat fourierTransform(Mat image){
	 int M = getOptimalDFTSize( image.rows );
	 int N = getOptimalDFTSize( image.cols );
	 Mat padded;
	 copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
	 Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	 Mat complexImg;
	 merge(planes, 2, complexImg);
     dft(complexImg, complexImg);
     split(complexImg, planes);
     magnitude(planes[0], planes[1], planes[0]);
     Mat mag = planes[0];
     mag += Scalar::all(1);
     log(mag, mag);
     mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
     int cx = mag.cols/2;
     int cy = mag.rows/2;
     Mat tmp;
     Mat q0(mag, Rect(0, 0, cx, cy));
     Mat q1(mag, Rect(cx, 0, cx, cy));
     Mat q2(mag, Rect(0, cy, cx, cy));
     Mat q3(mag, Rect(cx, cy, cx, cy));
     q0.copyTo(tmp);
     q3.copyTo(q0);
     tmp.copyTo(q3);
     q1.copyTo(tmp);
     q2.copyTo(q1);
     tmp.copyTo(q2);
     normalize(mag, mag, 0, 1, NORM_MINMAX);
    // mag.convertTo(mag, CV_8U);
     return mag;
}
// To implement gaussian blur on image in fourier domain with gaussian in fourier domain and then inverse it.

Mat gaussianBlurC(Mat image, Mat gaussian){
	Mat ch[3];
	int r = image.rows; 
	ch[0] = Mat::zeros(r, r, CV_32F);
	ch[1] = Mat::zeros(r, r, CV_32F);
	ch[2] = Mat::zeros(r, r, CV_32F);
	
	split(image, ch);
	for(int ind =0; ind<3; ind++){
	
		Mat IM;
		ch[ind].copyTo(IM);
		
			
		int M = getOptimalDFTSize( IM.rows );
		int N = getOptimalDFTSize( IM.cols );
		Mat padded;
		copyMakeBorder(IM, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
		Mat planes[]= {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
		Mat complexImage;
		merge(planes,2, complexImage);
		dft(complexImage, complexImage,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
		split(complexImage, planes);
		Mat plane1 = planes[0];
		Mat plane2 = planes[1];
		quadrantSwap(plane1);
		quadrantSwap(plane2);
		for(int i=0;i<image.rows;i++){
			 for(int j=0;j<image.cols;j++){
				 plane1.at<float>(i,j)=plane1.at<float>(i,j)*gaussian.at<float>(i,j);
				 plane2.at<float>(i,j)=plane2.at<float>(i,j)*gaussian.at<float>(i,j);
				
			 }
		}
	
	     // Swapping Coordinate Again
		quadrantSwap(plane1);
		quadrantSwap(plane2);
		merge(planes, 2,complexImage);
	//mulSpectrums(plane1,image3,image,0,0);
		cv::Mat inverseTransform;
		cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
	// Back to 8-bits
		cv::Mat finalImage;
		inverseTransform.convertTo(finalImage, CV_8U);
		ch[ind] = finalImage;
	}
	//imshow("Blurred Image", finalImage);
	merge(ch, 3, image);
	return image;
	
}

Mat gaussianBlur(Mat image, Mat gaussian){
	imshow("image",image);
	int M = getOptimalDFTSize( image.rows );
	int N = getOptimalDFTSize( image.cols );
	Mat padded;
	copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[]= {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImage;
	merge(planes,2, complexImage);
	dft(complexImage, complexImage,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
	split(complexImage, planes);
	Mat plane1 = planes[0];
	Mat plane2 = planes[1];
	quadrantSwap(plane1);
	quadrantSwap(plane2);
	for(int i=0;i<image.rows;i++){
	 	 for(int j=0;j<image.cols;j++){
	  		 plane1.at<float>(i,j)=plane1.at<float>(i,j)*gaussian.at<float>(i,j);
	   		 plane2.at<float>(i,j)=plane2.at<float>(i,j)*gaussian.at<float>(i,j);
	   		
	   	 }
	}
	
	     // Swapping Coordinate Again
	quadrantSwap(plane1);
	quadrantSwap(plane2);
	merge(planes, 2,complexImage);
	//mulSpectrums(plane1,image3,image,0,0);
	cv::Mat inverseTransform;
	cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
	// Back to 8-bits
	cv::Mat finalImage;
	inverseTransform.convertTo(finalImage, CV_8U);
	
	//imshow("Blurred Image", finalImage);
	return finalImage;
	
	
}
Mat weinerDeblur(Mat image, Mat weiner){
	float K = 0.025;
	int M = getOptimalDFTSize( image.rows );
	int N = getOptimalDFTSize( image.cols );
	Mat padded;
	copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[]= {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImage;
	merge(planes,2, complexImage);
	dft(complexImage, complexImage,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
	split(complexImage, planes);
	Mat plane1 = planes[0];
	Mat plane2 = planes[1];
	quadrantSwap(plane1);
	quadrantSwap(plane2);
	//printf(" value %f ",gaussian.at<float>(40,40));
	for(int i=0;i<image.rows;i++){
	 	 for(int j=0;j<image.cols;j++){
	 		 	 weiner.at<float>(i,j) = weiner.at<float>(i,j)/(pow(weiner.at<float>(i,j),2) + K) ;
				 plane1.at<float>(i,j) = plane1.at<float>(i,j)*weiner.at<float>(i,j);
				 plane2.at<float>(i,j) = plane2.at<float>(i,j)*weiner.at<float>(i,j);  
	   	 }
	}
		
			     // Swapping Coordinate Again
	quadrantSwap(plane1);
	quadrantSwap(plane2);
	merge(planes, 2,complexImage);
			//mulSpectrums(plane1,image3,image,0,0);
	cv::Mat inverseTransform;
	cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
		// Back to 8-bits
	cv::Mat finalImage;
	inverseTransform.convertTo(finalImage, CV_8U);
		
			//imshow("Blurred Image", finalImage);
	return finalImage;
	
}
// Inverse Filter with a threshold
Mat gaussianDeblur(Mat image, Mat gaussian){
	int M = getOptimalDFTSize( image.rows );
	int N = getOptimalDFTSize( image.cols );
	Mat padded;
	copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[]= {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImage;
	merge(planes,2, complexImage);
	dft(complexImage, complexImage,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
	split(complexImage, planes);
	Mat plane1 = planes[0];
	Mat plane2 = planes[1];
	quadrantSwap(plane1);
	quadrantSwap(plane2);
	printf(" value %f ",gaussian.at<float>(40,40));
	for(int i=0;i<image.rows;i++){
	 	 for(int j=0;j<image.cols;j++){
	 		 if(1/gaussian.at<float>(i,j) < 15){
	 			 gaussian.at<float>(i,j)= 1/gaussian.at<float>(i,j);
	 		 }
	 		 else{
	 			gaussian.at<float>(i,j) = 15;
	 		 }
				 plane1.at<float>(i,j)=plane1.at<float>(i,j)*gaussian.at<float>(i,j);
				 plane2.at<float>(i,j)=plane2.at<float>(i,j)*gaussian.at<float>(i,j);  
	   	 }
	}
	
		     // Swapping Coordinate Again
	quadrantSwap(plane1);
	quadrantSwap(plane2);
	merge(planes, 2,complexImage);
		//mulSpectrums(plane1,image3,image,0,0);
	cv::Mat inverseTransform;
	cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
	// Back to 8-bits
	cv::Mat finalImage;
	inverseTransform.convertTo(finalImage, CV_8U);
	
		//imshow("Blurred Image", finalImage);
	return finalImage;
		
}

Mat lowButterWorthOnImage(Mat image, Mat butterworth){
	imshow("image",image);
		int M = getOptimalDFTSize( image.rows );
		int N = getOptimalDFTSize( image.cols );
		Mat padded;
		copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
		Mat planes[]= {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
		Mat complexImage;
		merge(planes,2, complexImage);
		dft(complexImage, complexImage,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
		split(complexImage, planes);
		Mat plane1 = planes[0];
		Mat plane2 = planes[1];
		quadrantSwap(plane1);
		quadrantSwap(plane2);
		for(int i=0;i<image.rows;i++){
		 	 for(int j=0;j<image.cols;j++){
		  		 plane1.at<float>(i,j)=plane1.at<float>(i,j)*butterworth.at<float>(i,j);
		   		 plane2.at<float>(i,j)=plane2.at<float>(i,j)*butterworth.at<float>(i,j);
		   		
		   	 }
		}
		
		     // Swapping Coordinate Again
		quadrantSwap(plane1);
		quadrantSwap(plane2);
		merge(planes, 2,complexImage);
		//mulSpectrums(plane1,image3,image,0,0);
		cv::Mat inverseTransform;
		cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
		// Back to 8-bits
		cv::Mat finalImage;
		inverseTransform.convertTo(finalImage, CV_8U);
		
		//imshow("Blurred Image", finalImage);
		return finalImage;
}

Mat highButterWorthOnImage(Mat image, Mat butterworth){
	imshow("image",image);
		int M = getOptimalDFTSize( image.rows );
		int N = getOptimalDFTSize( image.cols );
		Mat padded;
		copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
		Mat planes[]= {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
		Mat complexImage;
		merge(planes,2, complexImage);
		dft(complexImage, complexImage,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
		split(complexImage, planes);
		Mat plane1 = planes[0];
		Mat plane2 = planes[1];
		quadrantSwap(plane1);
		quadrantSwap(plane2);
		for(int i=0;i<image.rows;i++){
		 	 for(int j=0;j<image.cols;j++){
		  		 plane1.at<float>(i,j)=plane1.at<float>(i,j)*butterworth.at<float>(i,j);
		   		 plane2.at<float>(i,j)=plane2.at<float>(i,j)*butterworth.at<float>(i,j);
		   		
		   	 }
		}
		
		     // Swapping Coordinate Again
		quadrantSwap(plane1);
		quadrantSwap(plane2);
		merge(planes, 2,complexImage);
		//mulSpectrums(plane1,image3,image,0,0);
		cv::Mat inverseTransform;
		cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
		// Back to 8-bits
		cv::Mat finalImage;
		inverseTransform.convertTo(finalImage, CV_8U);
		
		//imshow("Blurred Image", finalImage);
		return finalImage;
}

Mat highGaussianOnImage(Mat image, Mat gaussian){
	imshow("image",image);
		int M = getOptimalDFTSize( image.rows );
		int N = getOptimalDFTSize( image.cols );
		Mat padded;
		copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
		Mat planes[]= {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
		Mat complexImage;
		merge(planes,2, complexImage);
		dft(complexImage, complexImage,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
		split(complexImage, planes);
		Mat plane1 = planes[0];
		Mat plane2 = planes[1];
		quadrantSwap(plane1);
		quadrantSwap(plane2);
		for(int i=0;i<image.rows;i++){
		 	 for(int j=0;j<image.cols;j++){
		  		 plane1.at<float>(i,j)=plane1.at<float>(i,j)*gaussian.at<float>(i,j);
		   		 plane2.at<float>(i,j)=plane2.at<float>(i,j)*gaussian.at<float>(i,j);
		   		
		   	 }
		}
		
		     // Swapping Coordinate Again
		quadrantSwap(plane1);
		quadrantSwap(plane2);
		merge(planes, 2,complexImage);
		//mulSpectrums(plane1,image3,image,0,0);
		cv::Mat inverseTransform;
		cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
		// Back to 8-bits
		cv::Mat finalImage;
		inverseTransform.convertTo(finalImage, CV_8U);
		
		//imshow("Blurred Image", finalImage);
		return finalImage;
}

Mat idealHighPassOnImage(Mat image, Mat ideal){
	int M = getOptimalDFTSize( image.rows );
	int N = getOptimalDFTSize( image.cols );
	Mat padded;
	copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[]= {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImage;
	merge(planes,2, complexImage);
	dft(complexImage, complexImage,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
	split(complexImage, planes);
	Mat plane1 = planes[0];
	Mat plane2 = planes[1];
	quadrantSwap(plane1);
	quadrantSwap(plane2);
	for(int i=0;i<image.rows;i++){
	 	 for(int j=0;j<image.cols;j++){
	  		 plane1.at<float>(i,j)=plane1.at<float>(i,j)*ideal.at<float>(i,j);
	  		 plane2.at<float>(i,j)=plane2.at<float>(i,j)*ideal.at<float>(i,j);
	   		
	   	 }
	}
			
			     // Swapping Coordinate Again
	quadrantSwap(plane1);
	quadrantSwap(plane2);
	merge(planes, 2,complexImage);
			//mulSpectrums(plane1,image3,image,0,0);
	cv::Mat inverseTransform;
	cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
			// Back to 8-bits
	cv::Mat finalImage;
	inverseTransform.convertTo(finalImage, CV_8U);
			
			//imshow("Blurred Image", finalImage);
	return finalImage;
}
Mat idealLowPassOnImage(Mat image, Mat ideal){
	int M = getOptimalDFTSize( image.rows );
	int N = getOptimalDFTSize( image.cols );
	Mat padded;
	copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[]= {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImage;
	merge(planes,2, complexImage);
	dft(complexImage, complexImage,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
	split(complexImage, planes);
	Mat plane1 = planes[0];
	Mat plane2 = planes[1];
	quadrantSwap(plane1);
	quadrantSwap(plane2);
	for(int i=0;i<image.rows;i++){
	 	 for(int j=0;j<image.cols;j++){
	  		 plane1.at<float>(i,j)=plane1.at<float>(i,j)*ideal.at<float>(i,j);
	  		 plane2.at<float>(i,j)=plane2.at<float>(i,j)*ideal.at<float>(i,j);
	   		
	   	 }
	}
			
			     // Swapping Coordinate Again
	quadrantSwap(plane1);
	quadrantSwap(plane2);
	merge(planes, 2,complexImage);
			//mulSpectrums(plane1,image3,image,0,0);
	cv::Mat inverseTransform;
	cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
			// Back to 8-bits
	cv::Mat finalImage;
	inverseTransform.convertTo(finalImage, CV_8U);
			
			//imshow("Blurred Image", finalImage);
	return finalImage;
}

Mat removeWrinkles(Mat image){
	int numberOfChannels = image.channels();
	Mat image2 = Mat::zeros(image.rows, image.cols, image.type());
	for(i=1;i<image.rows-1;i++){
		for(j=1;j<image.cols-1;j++){
			for(int k=0;k<numberOfChannels;k++){
					double s = (image.at<Vec3b>(i,j).val[k])*(41)+
							image.at<Vec3b>(i-1,j-1).val[k]*20+
							image.at<Vec3b>(i-1,j).val[k]*30+
							image.at<Vec3b>(i-1,j+1).val[k]*20+
							image.at<Vec3b>(i,j-1).val[k]*30+
							image.at<Vec3b>(i,j+1).val[k]*30+
							image.at<Vec3b>(i+1,j).val[k]*26+
							image.at<Vec3b>(i+1,j-1).val[k]*20+
							image.at<Vec3b>(i+1,j+1).val[k]*20+
							image.at<Vec3b>(i+2,j).val[k]*10+
							image.at<Vec3b>(i-2,j).val[k]*10+
							image.at<Vec3b>(i+2,j+1).val[k]*7+
							image.at<Vec3b>(i-2,j+1).val[k]*7+
							image.at<Vec3b>(i+2,j-1).val[k]*7+
							image.at<Vec3b>(i-2,j-1).val[k]*7+
							image.at<Vec3b>(i-2,j+2).val[k]*4+
							image.at<Vec3b>(i-1,j+2).val[k]*10+
							image.at<Vec3b>(i,j+2).val[k]*15+
							image.at<Vec3b>(i+1,j+2).val[k]*10+
							image.at<Vec3b>(i+2,j+2).val[k]*4+
							image.at<Vec3b>(i-2,j-2).val[k]*4+
							image.at<Vec3b>(i-1,j-2).val[k]*10+
							image.at<Vec3b>(i,j-2).val[k]*15+
							image.at<Vec3b>(i+1,j-2).val[k]*10+
							image.at<Vec3b>(i+2,j-2).val[k]*4;
					s/=375;
					if(s<0){
						s=0;
					}
					else if(s>255){
						s=255;
					}
					image2.at<Vec3b>(i,j).val[k] = abs(s);
						//image4.at<Vec3b>(i,j)[k] = s;
			}
			printf("\n");
		}
	}
	return image2;	
}
Mat gaussianBlurColored(Mat image, Mat gaussian){
	imshow("image",image);
	Mat channels[3];
	channels[0] = Mat::zeros(image.rows,image.cols,CV_32F);
	channels[1] = Mat::zeros(image.rows,image.cols,CV_32F);
	channels[2] = Mat::zeros(image.rows,image.cols,CV_32F);
	split(image,channels);
	for(int p=0;p<3;p++){
		int M = getOptimalDFTSize( channels[p].rows );
		int N = getOptimalDFTSize( channels[p].cols );
		Mat padded;
		copyMakeBorder(channels[p], padded, 0, M - channels[p].rows, 0, N - channels[p].cols, BORDER_CONSTANT, Scalar::all(0));
		Mat planes[]= {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
		Mat complexImage;
		merge(planes,2, complexImage);
		dft(complexImage, complexImage,cv::DFT_SCALE|cv::DFT_COMPLEX_OUTPUT);
		split(complexImage, planes);
		Mat plane1 = planes[0];
		Mat plane2 = planes[1];
		quadrantSwap(plane1);
		quadrantSwap(plane2);
		for(int i=0;i<image.rows;i++){
			for(int j=0;j<image.cols;j++){
				plane1.at<float>(i,j)=plane1.at<float>(i,j)*gaussian.at<float>(i,j);
				plane2.at<float>(i,j)=plane2.at<float>(i,j)*gaussian.at<float>(i,j);
			}
		}
	
	     // Swapping Coordinate Again
		quadrantSwap(plane1);
		quadrantSwap(plane2);
		merge(planes, 2,complexImage);
	//mulSpectrums(plane1,image3,image,0,0);
		cv::Mat inverseTransform;
		cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
	// Back to 8-bits
		cv::Mat finalImage;
		inverseTransform.convertTo(finalImage, CV_8U);
		channels[p]=finalImage;
		//imshow("Blurred Image", channels[p]);
	}
	Mat output;
	merge(channels,3,output);
	
	//imshow("Blurred Image", output);
	return output;
	
	
}
int main(int argc, char** argv){
 char ch;
 Mat image, image2, gray, gray2, output;
 image  = imread(argv[1],1);
 image2 = imread(argv[2],1);

 if( argc != 3 || !image.data )
         {
           printf( "No image data \n" );
           return -1;
         }
 
  cvtColor(image, gray, CV_BGR2GRAY);
  cvtColor(image2, gray2, CV_BGR2GRAY);
  printf("%d\n",image.type());
  Mat image3 = Mat::zeros(image.rows, image.cols, image.type());
  printf("a. Fourier Transform of image\n");
  printf("b. Low Pass Gaussian Filter in fourier domain\n");
  printf("c. Blurred Image by Gaussian\n");
  printf("d. Deblurring Image by inverse\n");
  printf("e. Butterworth LowPass Filter\n");
  printf("f. Butterworth HighPass Filter\n");
  printf("g. High Pass Gaussian Filter \n");
  printf("h. Ideal High Pass Filter\n");
  printf("i. Ideal Low  Pass Filter\n");
  printf("j. (Bonus) Butterworth Filter2\n");
  printf("k  Weiner Deblur(Inverse) Filter\n");
  printf("l  Remove Wrinkles on Gaussian domain\n");
  printf("m  Remove Wrinkles on spatial domain\n");
  printf("Enter Your Choice\n");
  scanf("%c",&ch);
  switch(ch){
 	 case 'a':
 		//gray.convertTo(gray, CV_32F);
 		imshow("image",gray);
 		gray = fourierTransform(gray);
 		imshow("Fourier Transformed Image", gray);
 		break;
 	 case 'b':
 		image3 = createLowGaussianFilterImage(image3);
 		//namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
 		imshow( "Gaussian in Frequency Domain", image3 );
 		break;
 	 case 'c':
 		gray.convertTo(gray, CV_8U); 
 		image3 = createLowGaussianFilterImage(image3);
 		gray = gaussianBlur(gray, image3);
 		imshow("gaussian Blurred Image",gray);
 		break;
 	 case 'd':
 		gray.convertTo(gray, CV_8U); 
 		image3 = createLowGaussianFilterImage(image3);
 		gray = gaussianBlur(gray, image3);
 		imshow("Gaussian Blurred",gray);
 		gray = gaussianDeblur(gray, image3);
 		imshow("Ressurected Image", gray);
 		break;
 	 case 'e':
 		 image3 = butterworthLowFilter(image3);
 		 namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
 		 imshow("Low Butterworth in frequency Domain", image3);
 		 gray.convertTo(gray, CV_8U); 
 		 gray = lowButterWorthOnImage(gray, image3);
 		 imshow("butterworth Filtered Image",gray);
 		 break;
 	 case 'f':
 		image3 = butterworthHighFilter(image3);
 		namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
 		imshow("High Butterworth in frequency Domain", image3);
 		gray.convertTo(gray, CV_8U); 
 		gray = highButterWorthOnImage(gray, image3);
 		imshow("butterworth Filtered Image",gray);
 		break;
 	 case 'g':
 		image3 = createHighGaussianFilterImage(image3);
 		namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
 		imshow("High Pass Gaussian in frequency Domain", image3);
 		gray.convertTo(gray, CV_8U); 
 		gray = highGaussianOnImage(gray, image3);
 		imshow("High Gaussian Filtered Image",gray);
 		break;
 	 case 'h':
 		 image3 = idealHighFilter(image3);
 		 imshow("Ideal High Filter", image3);
 		 gray.convertTo(gray, CV_8U); 
 		 gray = idealHighPassOnImage(gray, image3);
 		 imshow("High Ideal Filtered Image",gray);
 		 break;
 	 case 'i':
 		 image3 = idealLowFilter(image3);
 		 imshow("Ideal High Filter", image3);
 		 gray.convertTo(gray, CV_8U); 
 		 gray = idealLowPassOnImage(gray, image3);
 		 imshow("High Ideal Filtered Image",gray);
 		 break;
 	 case 'j':
 		image3 = butterworth2LowFilter(image3);
 		namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
 		imshow("Low Butterworth in frequency Domain", image3);
 		gray.convertTo(gray, CV_8U); 
 		gray = lowButterWorthOnImage(gray, image3);
 		imshow("butterworth 2 Filtered Image",gray);
 		break;
 	 case 'k':
 		gray.convertTo(gray, CV_8U); 
 		image3 = createLowGaussianFilterImage(image3);
 		gray = gaussianBlur(gray, image3);
 		imshow("Gaussian Blurred",gray);
 		gray = weinerDeblur(gray, image3);
 		imshow("Ressurected Image", gray);
 		break;
 	 case 'l':
 		//gray2.convertTo(gray2, CV_8U); 
 		image3  = createLowGaussianFilterImage(gray2);
 		output  = gaussianBlurColored(image2, image3);
 		imshow("Gaussian Blurred",output);
 		waitKey(0);
 		break;
 	 case 'm':
 		 imshow("image",image2);
 		 image2 = removeWrinkles(image2);
 		 imshow("Wrinkles Removed", image2);
 		// waitKey(0);
 		 break;
 		 
 	 default:
 	 		 printf("Invalid Input\n");
 	 		 break;
 	     }


 int newCols = image3.cols;
 int newRows = image3.rows;
 printf("Rows %d\n",newRows);
 printf("Cols %d\n",newCols);





/* printf("printing image\n");
 namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
 imshow( "Display Image", image3 );*/
 waitKey(0);



 return 0;
}
