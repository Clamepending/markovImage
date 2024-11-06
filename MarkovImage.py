import cv2
import numpy as np
import imageio

DECAYRATE = 0.015
STEPSPERDECAY = 2000

# Load the image
# image = cv2.imread("./pikachu-lets-go.jpg")
# image = cv2.imread("./Markicon.jpg")
image = cv2.imread("./markCircle.jpg").astype(float)

# image = np.clip(image, 0, 254)

# Start in the middle of the image
cur_coordiante = [(image.shape[1] // 2, image.shape[0] // 2)]*3

canvas = image.copy().astype(float)

def getTransitionProbs(x, y, color):
    probs = np.ones(4)
    if x == 0:
        probs[0] = 0
    else:
        probs[0] = 255 - image[y, x - 1, color]
        
    if y == 0:
        probs[1] = 0
    else:
        probs[1] = 255 - image[y - 1, x, color]
        
    if x == image.shape[1] - 1:
        probs[2] = 0
    else:
        probs[2] = 255 - image[y, x + 1, color]
        
    if y == image.shape[0] - 1:
        probs[3] = 0
    else:
        probs[3] = 255 - image[y + 1, x, color]
    
    if sum(probs) == 0:
        return np.ones(4)*0.25

    total = sum(probs)
    return probs / total

# Create a list to store the frames
frames = [cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB)]

while True:
    canvas = canvas * (1 - DECAYRATE) + (np.ones_like(canvas) * 205 + image*50/255) * DECAYRATE
    
    for _ in range(STEPSPERDECAY):
        for color in range(3):
            x, y = cur_coordiante[color]
            transition_probs = getTransitionProbs(x, y, color)
            choice = np.random.choice(len(transition_probs), p=transition_probs)
            if choice == 0:
                x = max(x - 1, 0)
            elif choice == 1:
                y = max(y - 1, 0)
            elif choice == 2:
                x = min(x + 1, image.shape[1] - 1)
            elif choice == 3:
                y = min(y + 1, image.shape[0] - 1)
            cur_coordiante[color] = (x, y)
            canvas[y, x, color] = 0
        
    # Add the current frame to the list
    frames.append(cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    # Update and render
    cv2.imshow("Markov Image", canvas.astype(np.uint8))
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Save the frames as a GIF
imageio.mimsave("markov_image.gif", frames, duration=0.1)