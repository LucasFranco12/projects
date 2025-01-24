import cv2
import numpy as np
import pyautogui
import time


# Example usage
def main():
    

    """
    Make a move by typing it directly (e.g., 'e2e4' or 'g1f3')
    """
    # Type the move
    pyautogui.PAUSE = 5.0
    pyautogui.write("poop")
    pyautogui.write("e2e3")
    pyautogui.PAUSE = 5.0
    pyautogui.write("e3e4")
    # Press enter to submit
    #pyautogui.press('enter')
    # Small delay to let the move register
    time.sleep(0.5)


if __name__ == "__main__":
    main()