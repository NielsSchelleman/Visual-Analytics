import the requirements with pip install -r requirements.txt

For 1080*1920 screen resolution, it is recommended to use 90% zoom in the web browser for the best experience

IMPORTANT!

For shap to work, some changes need to be made to the internals of the shap library due to an error within the library.
In Lib\site-packages\shap\plots go to line 4 and add the line "import matplotlib.pyplot as plt". Do not remove the
"import matplotlib.pyplot as pl", simply add them both.
