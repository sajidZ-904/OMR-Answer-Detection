# OMR-Answer-Detection

Bubble sheet multiple choice scanner and test grader web app using OMR, Python, OpenCV and Flask

How to install locally (assuming you have git and python> = 3.12 installed):

```console
git clone https://github.com/sajidZ-904/OMR-Answer-Detection
cd OMR-Answer-Detection
python -m venv virtualenv
source virtualenv/Scripts/activate
pip install -r requirements.txt
```

To run:

Manage the image_path variable in the test_grader.py file with any images in the images folder. Then manage the answer_sheet_structure variable based on answers in two test images, and run

```console
python test_grader.py
```

The output image is the output_image.jpg file. The web app displays the answer from 1 to 100 perfectly.