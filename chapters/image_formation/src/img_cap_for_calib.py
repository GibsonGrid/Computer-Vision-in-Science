from pynput import keyboard
import cv2
import matplotlib.pyplot as plt

# Use direct show flag if on Windows: cv2.CAP_DSHOW
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# In other case
# cap = cv2.VideoCapture(0)

# Path for saving captured frames
save_folder = "../calib_imgs"

# Define the resolution of web-camera (you can check later the shape of captured image)
width = 1920
height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# Four Character Code codec (e.g., XVID, MJPG, MP4V, X264), find what works for your system
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M","J","P","G"))  

scale = 2  # Scale image to visualize
num = 0  # Counter for calibration images
save = False
terminate = False

def on_key_press(key):
    global save, terminate
    if key == keyboard.Key.shift:
        print("Image is saved!")
        save = True
    if key == keyboard.Key.esc:
        print("###### TERMINATE APPLICATION ######")
        terminate = True

listener = keyboard.Listener(on_press=on_key_press)
listener.start()

while cap.isOpened():
    # Read the image from the video capture object
    succes, img = cap.read()

    # Visualize for proper pattern positioning
    cv2.imshow("Img", cv2.resize(
        img, (img.shape[1]//scale, img.shape[0]//scale)))
    # Wait for 1ms (necessary for visualization of OpenCV obejcts in while loop)    
    cv2.waitKey(1)

    if save:
        # Save images (.png, 300 dpi) in {save_folder} using plt.imsave
        plt.imsave(f"{save_folder}/" + str(num) + ".png",
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dpi=300)
        num += 1
        save = not save
    
    if terminate:
        break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()