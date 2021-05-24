# ML-Motion-Blur
After recording a video, you may find that movements look choppy. Rerecording it
at a higher framerate may not be an option, so what can you do to make it look
smoother? You could apply a generic motion blur filter, but that often leads to
messy images. Manually adding frames and animating them works, but that is impractical.
This application aims to create an elegant solution to this problem. With machine
learning, relevant features in the image can be detected for blurring. For example, 
the legs of someone running should be blurred, but not the bench behind them! 

## How it Works
We trained a machine learning model track objects that will likely be in motion, 
like limbs and cars. Using those detections, the application smooths motion by
creating an "in between" frame. In this frame, objects are placed in the image 
at the midpoint where it was detected in the prior and following frame.\
\
For a more detailed explanation of how everything works, see our [report](https://github.com/hhenry01/ML-Motion-Blur/blob/main/reports/final/Team%20Caranotaurus%20Final%20Report.pdf) and [video](https://youtu.be/6rk4YBmm4HI).

## Installation
1. Install [Python 3.9](https://www.python.org/downloads/) or above.
2. Clone the repository. 
*You may get errors related to a file named Model.pt. If this happens, then download it directly from Backend/main_site/Model.pt and overwrite your local version.* 

3. Install the required packages using your preferred package manager. For example, while in the project directory, enter:
```
pip install -r requirements.txt
```

## How to run
*Note, this build is intended to work for Windows, and requires modifications to be run on Linux.* \
*In Backend/main_site/tracking.py, change ``from . import sort`` to ``import sort`` to make it work on Linux.\
It has not been tested to work on Mac.*

1. In the Backend folder, enter ``python manage.py runserver runserver`` in the terminal.
2. In a browser, enter ``localhost:8000`` or ``http://127.0.0.1:8000/`` as the url.
3. You should see an option to upload a video. Select a video to upload and wait. When it is done, it will be available to download.

A visual walkthrough of this process is found in both the [video](https://youtu.be/6rk4YBmm4HI) and [report](https://github.com/hhenry01/ML-Motion-Blur/blob/main/reports/final/Team%20Caranotaurus%20Final%20Report.pdf).

Alternatively, while in the Backend/main_site directory, you can run tracking.py with the following arguments to get the same result: 
```
1: path the the Faster RCNN Model
2: file/path name, either to an image or video
3: Enter 0 for image, 1 for video
Ex.
python3 tracking.py Model.pt test_samples/person_standing.jpg 0
```
tracking.py also contains the necessary functions to use this application's features
in your own project's.
## Additional Credits
A core component of this application was the [SORT](https://github.com/abewley/sort#:~:text=SORT%20is%20a%20barebones%20implementation,object%20identities%20on%20the%20fly.) algorithm. Their work is found in Backend/main_site/sort.py.

