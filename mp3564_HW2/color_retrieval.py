# mp3564
import numpy as np 
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt 
import cv2 
import glob 
import math 
import re  
import csv 
import imutils 

# All references are cited where I use them. The following are  references I used
# additional to OpenCV algorithms 
# https://stackoverflow.com/questions/46498041/is-the-opencv-3d-histogram-3-axis-histogram
# https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/

def main():
    images = glob.glob("images/*.jpg") # Images does NOT have images in order 1-40
    
    # Color - System vs. Crowd Preferences 
    all_res_color = [] 
    for i in range(len(images)):
        query_im = images[i]
        res_out = get_color_results(query_im, images)
        score = get_score(res_out[0], res_out[1], res_out[2], res_out[3], res_out[4])
        res_out.append(score[0])
        res_out.append(score[1])
        res_out.append(score[2])
        res_out.append(score[3])
        res_out.append(score[4])
        all_res_color.append(res_out)

    print("color")
    all_res_color.sort(key=lambda all_res_color: all_res_color[0])
    print(all_res_color)

    # Output to CSV File color_res.csv
    with open('color_res.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Query", "T1", "T2", "T3",  "T40", "Tot Score", "T1 Score", "T2 Score", "T3 Score", "T40 Score"])
        writer.writerows(all_res_color)
    
    # Texture - System vs. Crowd Preferences 
    all_res_text = []
    for i in range(len(images)):
        query_im = images[i]
        res_out = get_texture_results(query_im, images)
        score = get_score(res_out[0], res_out[1], res_out[2], res_out[3], res_out[4])
        res_out.append(score[0])
        res_out.append(score[1])
        res_out.append(score[2])
        res_out.append(score[3])
        res_out.append(score[4])
        all_res_text.append(res_out)
    
    print("text")
    all_res_text.sort(key=lambda all_res_text: all_res_text[0])
    print(all_res_text)
    
    # Output to CSV file text_res.csv
    with open('text_res.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Query", "T1", "T2", "T3",  "T40", "Tot Score", "T1 Score", "T2 Score", "T3 Score", "T40 Score"])
        writer.writerows(all_res_text)
    
    # Shape - System vs. Crowd Preferences 
    all_res_shape = []
    for i in range(len(images)):
        query_im = images[i]
        res_out = get_shape_results(query_im, images)
        score = get_score(res_out[0], res_out[1], res_out[2], res_out[3], res_out[4])
        res_out.append(score[0])
        res_out.append(score[1])
        res_out.append(score[2])
        res_out.append(score[3])
        res_out.append(score[4])
        all_res_shape.append(res_out)
    
    print("shape")
    all_res_shape.sort(key=lambda all_res_shape: all_res_shape[0])
    print(all_res_shape)
    
    # Output to CSV file shape_res.csv
    with open('shape_res.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Query", "T1", "T2", "T3",  "T40", "Tot Score", "T1 Score", "T2 Score", "T3 Score", "T40 Score"])
        writer.writerows(all_res_shape)

    # Combined - System vs. Crowd Preferences
    all_res_combined = []
    running_score =0
    for i in range(len(images)):
        query_im = images[i]
        res_out = get_combined_results(query_im, images)
        score = get_score(res_out[0], res_out[1], res_out[2], res_out[3], res_out[4])
        res_out.append(score[0])
        res_out.append(score[1])
        res_out.append(score[2])
        res_out.append(score[3])
        res_out.append(score[4])
        all_res_combined.append(res_out)
        running_score += score[0]
   
    
    print("combined")
    all_res_combined.sort(key=lambda all_res_combined: all_res_combined[0])
    print(all_res_combined)
    
    # Output to CSV file combined_res.csv
    with open('combined_res.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Query", "T1", "T2", "T3",  "T40", "Tot Score", "T1 Score", "T2 Score", "T3 Score", "T40 Score"])
        writer.writerows(all_res_combined)
    
    # Color, Texture, Shape, Combined - System vs. User Preferences
    print("check preferences")
    print(check_preferences(all_res_color, all_res_text, all_res_shape, all_res_combined))

    get_upper_bound()
    create_my_crowd()

# Reads Crowd.txt File and returns a 2D array 
def read_file():
    data  = []
    rows, cols = (40, 40)

    with open ('Crowd.txt', 'r') as f: 
        for line in f.readlines():
            line = line.split()
            data.append(line)
    return data 

# Reads MyPreferences.txt file and returns 2D array
def read_preferences():
    data = []

    with open('MyPreferences.txt', 'r') as f:
        for line in f.readlines():
            line  = line.split()
            data.append(line)
    return data 

# Get max 3 values per query, sums them, computes upper bound for Crowd.txt
def get_upper_bound():
    crowd_scores = read_file()
    total_sum = 0 
    for i in range(len(crowd_scores)):
        targets = [int(t) for t in crowd_scores[i]]
        top_3 = sorted(targets, reverse=True)[:3]
        target_sum = sum(top_3)
        total_sum += target_sum 
    #print(total_sum)
    return total_sum
    
# Creates MyCrowd.txt, a 40 x 40 sparse matrix 
def create_my_crowd():
    my_crowd = np.zeros((40,40))
   
    for i in range(my_crowd.shape[0]):
        my_crowd[i,0] = 3
        my_crowd[i,1] = 2
        my_crowd[i,2] = 1

    return my_crowd 


# Helper function: given a query number, returns top 3 targets
def get_real_res(q_num, result):
    q = result[q_num-1][0]
    res = result[q-1]
    # only want top 3 target scores 
    out = [res[1], res[2], res[3]]
    return out 


# Counts intersection between 3 best scores and 3 user-chosen scores 
# This is for color, texture, shape, and combined
def check_preferences(color_res, text_res, shape_res, combined_res):
    data_pref = read_preferences()
    count_color = []
    count_text =  []
    count_shape = []
    count_combined = []

    for i in range(len(data_pref)):
        q = int(data_pref[i][0])
        prefs = [int(p) for p in data_pref[q-1]]
        prefs_out = [prefs[1], prefs[2], prefs[3]]
        p = set(prefs_out)
        
        color = get_real_res(q, color_res)
        text = get_real_res(q, text_res)
        shape = get_real_res(q, shape_res)
        combined = get_real_res(q, combined_res)

        c = set(color)
        t = set(text)
        s = set(shape)
        comb = set(combined)

        intersect_color = p.intersection(c)
        intersect_text = p.intersection(t)
        intersect_shape = p.intersection(s)
        intersect_combined = p.intersection(comb)

        count_color.append(len(intersect_color))
        count_text.append(len(intersect_text))
        count_shape.append(len(intersect_shape))
        count_combined.append(len(intersect_combined))

    print("Intersections")   
    print(count_color)
    print(count_text)
    print(count_shape)
    print(count_combined)
    return sum(count_color), sum(count_text), sum(count_shape), sum(count_combined)


# Returns total score and t1, t2, t3, t40 Crowd(q,t) scores
def get_score(q, t1, t2, t3, t40):
    crowd_scores = read_file()
    #crowd_scores = create_my_crowd()
    t1_score = int(crowd_scores[q-1][t1-1])
    t2_score  = int(crowd_scores[q-1][t2-1])
    t3_score = int(crowd_scores[q-1][t3-1])
    t40_score = int(crowd_scores[q-1][t40-1])
    tot_score = t1_score + t2_score + t3_score
    
    return tot_score, t1_score, t2_score, t3_score, t40_score

# Returns total scores with sparse 40 x 40 matrix 
def get_new_score(q, t1, t2, t3, t40):
    crowd_scores = create_my_crowd()
    t1_score = int(crowd_scores[q-1][t1-1])
    t2_score  = int(crowd_scores[q-1][t2-1])
    t3_score = int(crowd_scores[q-1][t3-1])
    t40_score = int(crowd_scores[q-1][t40-1])
    tot_score = t1_score + t2_score + t3_score
    
    return tot_score, t1_score, t2_score, t3_score, t40_score

# STEP 1
# OpenCV https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html
# I use cvtColor to convert from BGR to RGB because openCV automatically reads images as BGR 
# I use calcHist because it better helped me understand what a 3D histogram may look like
# It is also more efficient than np.histogramddd()
def compare_color(query_im, target_im):
    query_im = cv2.imread(query_im)
    target_im = cv2.imread(target_im)

    # Converts image from BGR to RGB 
    query_im = cv2.cvtColor(query_im, cv2.COLOR_BGR2RGB)
    target_im = cv2.cvtColor(target_im, cv2.COLOR_BGR2RGB)
    
    # Calculates histogram of query and target image
    hist_query = cv2.calcHist([query_im], [0,1,2],None, [16,16,16], [0,256,0,256,0,256])
    hist_target = cv2.calcHist([target_im], [0,1,2], None, [16,16,16], [0,256,0,256,0,256])
     
    # Gets l1 distance: SUM |Image1(r, g, b) − Image2(r, g, b)| /(2 ∗ rows ∗ cols)
    norm = 2 * 60 * 89
    diff = abs(hist_query-hist_target)
    tot = sum(sum(sum((diff))))

    res = tot/norm 
    return res 


def get_color_results(query_im, images):
    res = []
    output = []
    for i in range(len(images)):
        if query_im != images[i]:
            query_number = re.findall("\d+", query_im)[0]
            target_number = re.findall("\d+", images[i])[0]
            l1_dist = compare_color(query_im, images[i])
            res.append((l1_dist, int(target_number)))
    size = len(res)
    res.sort(key=lambda x:x[0])

    # Returns query number, top 3 targets and worst target 
    output = [int(query_number), res[0][1], res[1][1], res[2][1], res[size-1][1]]
    return output 
    

# STEP 2 
# Gets texture histogram 
# https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html
# I use cv2.GaussianBlur() to remove noise from the image
# I use cv2.Laplacian() so I could then easily compute a histogram with OpenCV's calcHist by passing in this result 
def get_texture2(query_im):
    depth = cv2.CV_16S
    kernel_size = 3

    # Reads image 
    query_im = cv2.imread(query_im)

    # Removes noise by blurring with Gaussian filter 
    query_im_smooth = cv2.GaussianBlur(query_im, (3,3), 0)

    # Converts to grayscale 
    query_im_gray = cv2.cvtColor(query_im, cv2.COLOR_BGR2GRAY)

    # Applies Laplacian function 
    dist = cv2.Laplacian(query_im_gray, depth, ksize=kernel_size)
    
    # Convert back to uint8
    abs_dist =  cv2.convertScaleAbs(dist)

    # Calculates histogram 
    hist_query = cv2.calcHist([abs_dist], [0], None, [256], [0, 256])
    return hist_query


# Sums over all pixels to get a value, and then normalize by 2 x rows x columns 
def compare_texture(query_im, target_im):
    hist_query = get_texture2(query_im)
    hist_target = get_texture2(target_im)
    norm =  2* 60 * 89
    diff = abs(hist_query-hist_target)
    tot = sum(sum(diff))

    res = tot/norm 
    return res 
    

def get_texture_results(query_im, images):
    res = []
    output = []
    for i in range(len(images)):
        if query_im != images[i]:
            query_number = re.findall("\d+", query_im)[0]
            target_number = re.findall("\d+", images[i])[0]
            l1_dist = compare_texture(query_im, images[i])
            res.append((l1_dist, int(target_number)))
    size = len(res)
    res.sort(key=lambda x:x[0])
    
    # Returns query number, top 3 targets and worst target 
    output = [int(query_number), res[0][1], res[1][1], res[2][1], res[size-1][1]]
    return output  


# STEP 3 - shape distance 
#  https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
# I used Otsu Binarization to optimize the treshold parameter. Doing this manually would have been time and cost inefficient. 
def get_shape(query_im):
    query_im = cv2.imread(query_im)
    print("testing ")
    print(query_im)

    query_im_smooth = cv2.GaussianBlur(query_im, (3,3), 0)

    # Converts to grayscale 
    query_im_gray = cv2.cvtColor(query_im_smooth, cv2.COLOR_BGR2GRAY)

    # If value is above 127, set  it to 255 
    #thresh = cv2.threshold(query_im_gray, 127, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(query_im_gray, 0, 255, cv2.THRESH_OTSU)[1]
    cv2.imwrite('test1.jpg', thresh)
    
    return thresh

# Counts the number of pixels where images disagree foreground background 
def compare_shape(query_im, target_im):
    binary1 = get_shape(query_im)
    binary2 = get_shape(target_im)
    norm = 60  * 89 
    diff = binary1-binary2

    # Instead of iterating through the binary objects, I used OpenCV's
    # countNonZero() function 
    count_diff = cv2.countNonZero(diff)
   
    res = count_diff/norm 
    return res 

def get_shape_results(query_im, images):
    res = []
    output = []
    for i in range(len(images)):
        if query_im != images[i]:
            # Uses regex to find number at end of file name 
            query_number = re.findall("\d+", query_im)[0]
            target_number = re.findall("\d+", images[i])[0]
            l1_dist = compare_shape(query_im, images[i])
            res.append((l1_dist, int(target_number)))
    size = len(res)
    res.sort(key=lambda x:x[0])
    
    # Returns query number, top 3 targets and worst target 
    output = [int(query_number), res[0][1], res[1][1], res[2][1], res[size-1][1]]
    return output  
 
# STEP 4 - overall/combined distance 
# a * color + b * text + c * shape 
def combined_distance(query_im, target_im):
    combined_dist = (0.75 * compare_color(query_im, target_im)) + (0.125 * compare_texture(query_im, target_im)) + (0.125 * compare_shape(query_im, target_im))
    return combined_dist

def get_combined_results(query_im, images):
    res = []
    output = []
    for i in range(len(images)):
        if query_im != images[i]:
            # Uses regex to find number at end of file name 
            query_number = re.findall("\d+", query_im)[0]
            target_number = re.findall("\d+", images[i])[0]
            l1_dist = combined_distance(query_im, images[i])
            res.append((l1_dist, int(target_number)))
    size = len(res)
    res.sort(key=lambda x:x[0])
    
    # Returns query number, top 3 targets and worst target 
    output = [int(query_number), res[0][1], res[1][1], res[2][1], res[size-1][1]]
    return output  
 

 



if __name__ == "__main__":
    main()