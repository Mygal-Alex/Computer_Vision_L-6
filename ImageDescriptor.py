import cv2 as cv

def sift_feature_matching(img1, img2):
    
    img1_gray = cv.imread(img1, cv.IMREAD_GRAYSCALE)
    img2_gray = cv.imread(img2, cv.IMREAD_GRAYSCALE)

    
    sift = cv.SIFT_create()

    
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    
    matches = flann.knnMatch(des1, des2, k=2)

    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    
    img_matches = cv.drawMatches(img1_gray, kp1, img2_gray, kp2, good_matches, None,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    
    cv.imshow('SIFT Feature Matching', img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    image_pairs = [('img/img1.jpg', 'img/img2.jpg'),
                   ('img/img3.jpg', 'img/img4.jpg'),
                   ('img/img5.jpg', 'img/img6.jpg')]

    for img1, img2 in image_pairs:
        sift_feature_matching(img1, img2)