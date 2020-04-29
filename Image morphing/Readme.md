# Image Morphing Project

This project implements image morphing from a given source image to a given destination image.

## Requirements

Make sure the following libraries are present in your Python3 environment:

* numpy
* matplotlib
* ffmpeg
* cv2

## Usage

Make sure the command line is running in the directory where the project is saved.
* With number of frames included:
    ```bash
    python morphing.py Bush.jpg Clinton.jpg -frames 10
    ```

* Without number of frames included:
    ```bash
    python morphing.py Bush.jpg Clinton.jpg
    ```
After running this command, 
1. The source image will pop out in front of you.
2. Hover to the point you intend to choose. Left click on it. Pressing Enter will finalize this point and will be shown with a red dot. **The point will not be finalized till you press the ENTER key.**
3. The destination image pops out now. 
4. Repeat step 2 for the destination image by selecting the same part of the face as you did for he source image.
5. Now, alternatively, keep choosing the points between the source and the destination by following step 2.
6. When you intend to end the process, press 0.

* The morphing starts taking place and the code shall terminate once all the frames along with the final video are created and saved in the respective directories.



