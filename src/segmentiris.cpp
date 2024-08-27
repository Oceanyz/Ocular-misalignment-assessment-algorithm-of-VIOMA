/**************************************************
*  This is a C translation from Masek's matlab code
*  Author:
*  Xiaomei Liu
*  xliu5@cse.nd.edu
*  Computer Vision Research Laboratory
*  Department of Computer Science & Engineering
*  U. of Notre Dame
***************************************************/

/*% segmentiris - peforms automatic segmentation of the iris region
% from an eye image. Also isolates noise areas such as occluding
% eyelids and eyelashes.
%
% Usage: 
% [circleiris, circlepupil, imagewithnoise] = segmentiris(image)
%
% Arguments:
%	eyeimage		- the input eye image
%	
% Output:
%	circleiris	    - centre coordinates and radius
%			          of the detected iris boundary
%	circlepupil	    - centre coordinates and radius
%			          of the detected pupil boundary
%	imagewithnoise	- original eye image, but with
%			          location of noise marked with
%			          NaN values
%
% Author: 
% Libor Masek
% masekl01@csse.uwa.edu.au
% School of Computer Science & Software Engineering
% The University of Western Australia
% November 2003

function [circleiris, circlepupil, imagewithnoise] = segmentiris(eyeimage)
*/
#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string.h>
#include <math.h>
#include <float.h>
#include "Masek.h"

//LEE:: segmentation for video images
void Masek::segmentiris(Masek::IMAGE *eyeimage, int *center_y, int *center_x, int *radius)
{
	int lirisradius, uirisradius;
	double scaling;	
	int rowi, coli, ri;

	//OURS DATASET
	lirisradius = 56;
	uirisradius = 80;

	//% define scaling factor to speed up Hough transform
	scaling = 0.6;

	//% find the iris boundary
	findcircle(eyeimage, lirisradius, uirisradius, scaling, 2, 0.15, 0.10, 1.00, 0.00, &rowi, &coli, &ri);
	
	*center_y = rowi;
	*center_x = coli;
	*radius = ri;

	printf("iris is %d %d %d\n", rowi, coli, ri);
	

}

