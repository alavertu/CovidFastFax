## Covid Fast Fax

This program will evaluate PDF files in the `TARGET_DIR`. Each page of the PDF will be checked against 
the Contra Costa County CMR template and the California State CMR template. If the page matches the template
it will be extracted and based on the checkboxes, placed into one of the following directories within `OUTPUT_DIR`:

* `high_priority` - Individual covid morbidity reports for workers in healthcare setting 
* `congregate_settings` - Individual covid morbidity reports for workers in healthcare setting
* `hp_and_cong` - Individual covid morbidity reports that meet both the `high_priority` and `congregate_settings` criteria
* `uncertain` - For the Contra Costa Country Form, if the model thought both Yes and No boxes were checked for any of the above criteria, those reports are placed here.
* `other` - Reports that don't meet any of the above criteria

***NOTES***: 
* Once initiated, the program will continue to monitor the `TARGET_DIR`, any new files that appear in the 
directory will be processed as they arrive. 
* Program currently only monitors for PDF files, based on file endings. This can easily be fixed if we have a broader 
list of file formats we want to accept.
* Model is tuned to avoid False Negatives, this results in a higher rate of False Positives. If this isn't desired 
or the False Positive Rate is too high, I can adjust the prediction thresholds
* Program takes a little while (>10s) to process each page of a PDF, as the image registration step takes a while. I'm currently looking at ways to speed this up.  
* Currently can't handle upside-down reports, I'm working on a fix for this as well. 

## Install steps  
#### Built and tested with Python 3.6

**1\. Clone the repo to desired location**
> git clone https://github.com/alavertu/CovidReportReader.git

**2\. Install Poppler**

https://poppler.freedesktop.org/  
> conda install -c conda-forge poppler  

Make sure you can properly run the following conversion:
> pdftoppm -r 200 -cropbox -png ./data/test_images/test_curr_both.pdf test

If there's an issue, try using the following conda distribution instead:  

> conda install -c conda-forge/label/gcc7 poppler  
    
<br/>  
  
**3\. Install CPU version of pytorch 4.1**

https://pytorch.org/get-started/previous-versions/

For OSX:
> pip install torch==1.4.0 torchvision==0.5.0  

For Linux Windows:  
> pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html  
  
<br/>  

**4\. Install remaining python packages**

> cd CovidReportReader
> pip install -r requirements.txt


## To Run:

> cd CovidReportReader/code

> python CovidReportReader.py -t <path_to_directory_with_faxes> -O <path_to_output_directory> 

## To Test:
> cd CovidReportReader/code

> python CovidReportReader.py -t ../data/test_images/ -O ../test_out -v

This should create a directory called `test_out` in the main directory, with the following folders:
* `high_priority` - Individual covid morbidity reports for workers in healthcare setting 
* `congregate_settings` - Individual covid morbidity reports for workers in healthcare setting
* `hp_and_cong` - Individual covid morbidity reports that meet both the `high_priority` and `congregate_settings` criteria
* `uncertain` - For the Contra Costa Country Form, if the model thought both Yes and No boxes were checked for any of the above criteria, those reports are placed here.
* `other` - Reports that don't meet any of the above criteria

The files in `../data/test_images/` should be sorted accurately, except for the files that have been rotated 180 degrees. 
Still working on an accurate way to check and fix file orientation. These files will simply be skipped at the moment. 

## 

CovidReportReader with basic checkbox prioritization  

required arguments:  
  -t TARGET_DIR, --target_dir TARGET_DIR, Target directory to monitor for new PDFs  
  -O OUTPUT_DIR, --output_dir OUTPUT_DIR, Location to create output directory  
                        
optional arguments:  
  -h, --help            show this help message and exit    
  -r, --reset           Reset cache, will result in reprocessing of all files   
                        in the target directory (Not yet implemented)  
  -v, --verbose         Verbose mode  


