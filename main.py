# EL2320 course project
# Author: Michaela Söderström

"""
Single face detection and tracking with a particle filter.

OpenCV's built in Haar cascade detection is used to detect and choose the first ROI 
to initialize the particles. The tracking of the face is then performed with the 
particle filter algorithm.
-----------------------------------------------------------------------------------
"""

# Load libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load classes and functions
from particlefilter import particleFilter
from framefunctions import findROI
from color_model import makeHist_GRAY, makeHist_RGB

def CaptureVideo(video, method=1, color_hist=1, init_hypotheses=0, init_pf=False, frame_no=0):
    '''
    Tracking of an object with a particle filter. 
    --------------------------------
    INPUT:
        video: int or str
            Real time video capture from webcam (int=0) or a pre-recorded video (str=" ")
        color_hist: bool
            1 (=default) if tracking with color histogram, 0 if tracking with grayscale histogram
        init_hypotheses: bool
            0 (=default) if initialize in roi, 1 if initialize over the whole frame
        init_pf: bool
            returns True when the first ROI is detected
        frame_no: int
            counter for video frames
    OUTPUT: 
        Saves the tracking result as a avi-file and Bayes error plot for all played frames.
    --------------------------------
    '''

    execution_path = os.getcwd()
    
    # Find path to video frame
    try:
        if isinstance(video, str) == True:
            video_frame = os.path.join(execution_path, video) # prerecorded data
        else:
            video_frame = video # through "real time" frame capture
    except:
        print('Occuring error trying to open video. None existing file or sensor data. Request cancelled.')
        raise SystemExit

    vidCapture = cv2.VideoCapture(video_frame) # Capture video frame from input type

    width = int(vidCapture.get(3))
    height = int(vidCapture.get(4))
    frame_size_org = (width, height)

    # Save video if using webcam
    if video_frame: #video:

        videoOutput = cv2.VideoWriter('object_tracking54_1.avi',
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    30, #fps
                    (width, height), # frame size
                    isColor=True) # grayscale -> false
    # Create dynamical lists
    bayesE_l_r, bayesE_l_g, bayesE_l_b = [], [], []
    bayesE_u_r, bayesE_u_g, bayesE_u_b = [], [], []
    frame_vec = []


    while vidCapture.isOpened():

        ret, frame = vidCapture.read() # returns boolean value 1 if succeding reading and frame is the captured frame
        frame_no += 1 # counter
        frame_vec.append(frame_no)
        if ret == True:

            if init_pf == False:
                # Find region of interest of the first visible face with Haar cascade detection
                roi_coord, roi_target, frame_marked = findROI(frame)
                
            if len(roi_target) > 0 and init_pf == False:
                init_pf = True
                roi_size = (np.size(roi_target, 0), np.size(roi_target, 1)) # size of target frame
                # create a histogram  for target reference
                if color_hist:
                    hist_ref, bins = makeHist_RGB(roi_target)
                    # color = ('r', 'g', 'b')
                    # ch_id = (0, 1, 2)
                    # for i, c in zip(ch_id, color):
                    #     plt.plot(bins[0:-1], hist_ref[i], color=c)
                    #plt.hist(bins[:-1], bins, weights=hist_ref, density=True)
                    #plt.hist(hist_ref[0], bins=bins, density=True)
                    #plt.hist(hist_ref[1], bins=bins, density=True)
                    #plt.hist(hist_ref[2], bins=bins, density=True)
                    #plt.savefig('target_histogram.png')
                else:
                    hist_ref = makeHist_GRAY(roi_target)
                # initialize particle filter
                pf = particleFilter(roi_coord, frame_size_org, roi_size, method, color_hist, init_hypotheses, detect)

            elif init_pf == True:
                # particle filter algorithm
                S_hat = pf.predict()
                frame_S_est, S_est, E_l, E_u = pf.measurementUpdate(S_hat, hist_ref, frame)
                if color_hist:
                    bayesE_l_r.append(E_l[0])
                    bayesE_l_g.append(E_l[1])
                    bayesE_l_b.append(E_l[2])
                    bayesE_u_r.append(E_u[0])
                    bayesE_u_g.append(E_u[1])
                    bayesE_u_b.append(E_u[2])
                else:
                    bayesE_l_g.append(E_l)
                    bayesE_u_g.append(E_u)

                frame = frame_S_est
                # plot predicted and estimated state positions
                for i in range(len(S_hat[0])):
                    cv2.circle(frame, (S_hat[0, i],S_hat[1, i]), 5, (0, 255, 0), 3) # propagation states
                cv2.circle(frame, (S_est[0], S_est[1]), 5, (255, 0, 0), 3) # estimated state
     
            else:
                continue

            cv2.imshow('result', frame)
            #videoOutput.write(frame)

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                
                plt.plot(frame_vec[:-1], bayesE_l_r, '--r', label='lower bound')
                plt.plot(frame_vec[:-1], bayesE_l_g, '--g', label='lower bound')
                plt.plot(frame_vec[:-1], bayesE_l_b, '--b', label='lower bound')
                plt.plot(frame_vec[:-1], bayesE_u_r, 'r', label='upper bound')
                plt.plot(frame_vec[:-1], bayesE_u_g, 'g', label='upper bound')
                plt.plot(frame_vec[:-1], bayesE_u_b, 'b', label='upper bound')
                plt.xlabel('Frame number')
                plt.ylabel('Error')
                plt.title('Bayes error')
                plt.legend(loc='lower left')
                plt.show()
                plt.savefig('bayeserror_54break.png')
                break
            
        else:
            if color_hist:
                plt.plot(frame_vec[:-2], bayesE_l_r, '--r', label='lower bound')
                plt.plot(frame_vec[:-2], bayesE_l_g, '--g', label='lower bound')
                plt.plot(frame_vec[:-2], bayesE_l_b, '--b', label='lower bound')
                plt.plot(frame_vec[:-2], bayesE_u_r, 'r', label='upper bound')
                plt.plot(frame_vec[:-2], bayesE_u_g, 'g', label='upper bound')
                plt.plot(frame_vec[:-2], bayesE_u_b, 'b', label='upper bound')
                plt.title('Bayes error for RGB histogram')
            else:
                plt.plot(frame_vec[:-2], bayesE_l_g, label='lower bound')
                plt.plot(frame_vec[:-2], bayesE_u_g, label='upper bound')
                plt.title('Bayes error for grayscale')
            plt.xlabel('Frame number')
            plt.ylabel('Error')
            plt.legend(loc='lower left')
            plt.show()
            plt.savefig('bayeserror_54.png')
            break

    # release the captured video object
    vidCapture.release()

    # Close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
        video = "IMG_8754.m4v"#"object_tracking3.avi" # Input file. 0 if front camera (web camera)

        # Tracking choices/variations:
        method = 0 # 0=Bhattacharyya distance (default), 1=Euclidian distance
        color_hist = 0 # 1=rgb, 0=gray scale
        init_hypotheses = 0 # 0=in roi, 1=whole frame
        detect=0 #0=never update roi, 1=re-detection, 2=update roi

        CaptureVideo(video)