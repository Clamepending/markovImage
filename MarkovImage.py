import cv2
import numpy as np

DECAYRATE = 0.002
STEPSPERDECAY = 10000

# Load the image
image = cv2.imread("./pikachu-lets-go.jpg")

# conver tot greyscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# clip the range of pixel colors to 1 to 255
gray = np.clip(gray, 0, 254)

#start in middle of image
cur_coordiante = (gray.shape[1] // 2, gray.shape[0] // 2)

canvas = image.copy().astype(float)

def getTransitionProbs(x, y):
    probs = np.ones(4)
    if x == 0:
        probs[0] = 0
    else:
        probs[0] = 255 - gray[y, x - 1]
        
    if y == 0:
        probs[1] = 0
    else:
        probs[1] = 255 - gray[y - 1, x]
        
    if x == image.shape[1] - 1:
        probs[2] = 0
    else:
        probs[2] = 255 - gray[y, x + 1]
        
    if y == image.shape[0] - 1:
        probs[3] = 0
    else:
        probs[3] = 255 - gray[y + 1, x]
    
    if sum(probs) == 0:
        return np.ones(4)*0.25

    total = sum(probs)
    return probs / total

while True:
    canvas = canvas * (1 - DECAYRATE) + np.ones_like(canvas) * 255 * DECAYRATE
    
    for _ in range(STEPSPERDECAY):
        transition_probs = getTransitionProbs(cur_coordiante[0], cur_coordiante[1])
        # print(transition_probs)
        x = np.random.choice(len(transition_probs), p=transition_probs)
        if x == 0:
            cur_coordiante = (cur_coordiante[0] - 1, cur_coordiante[1])
        elif x == 1:
            cur_coordiante = (cur_coordiante[0], cur_coordiante[1] - 1)
        elif x == 2:
            cur_coordiante = (cur_coordiante[0] + 1, cur_coordiante[1])
        elif x == 3:
            cur_coordiante = (cur_coordiante[0], cur_coordiante[1] + 1)
        canvas[cur_coordiante[1], cur_coordiante[0]] = [0, 0, 0]
        
    #update and render
    cv2.imshow("Markov Image", canvas.astype(np.uint8))
    key = cv2.waitKey(1)
    if key == 27:
        break
    
