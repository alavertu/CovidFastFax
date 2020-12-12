## Covid Fast Fax

This program will evaluate PDF files in the `TARGET_DIR`. Each page of the PDF will be checked against 
the 5 accepted templates. PDFs will then be placed into the output directory with names that indicate priority of the enclosed reports.

* `00_vulnerable_*` - PDF contains a case with an outbreak in a vulnerable population
* `01_hcw_*` - PDF contains a case for a healthcare worker
* `02_np_*` - PDF contains a covid morbidity report with no special priority
* `NTD_*` - None of the templates were detected in this PDF, may still contain a case that the system missed

***NOTES***: 
* Program currently only supports 5 templates, see `data/references/` for examples of supported templates within template folders
* Program currently only monitors for PDF files, based on file endings. This can easily be fixed if we have a broader 
list of file formats we want to accept.  
* On Windows Powershell make sure to disable QuickEdit mode, otherwise process may hang randomly. [Solution noted here](https://stackoverflow.com/questions/39676635/a-process-running-on-powershell-freezes-randomly/39676636#39676636) 

## Install steps  
#### Built and tested with Python 3.6

**1\. Clone the repo to desired location**
> git clone https://github.com/alavertu/CovidFastFax.git

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

For Linux/Windows:  
> pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html  
  
<br/>  

**4\. Install remaining python packages**

> cd CovidFastFax
> pip install -r requirements.txt


## To Run with Recommended Settings:

> cd CovidFastFax/code

> python CovidFastFax.py -t <path_to_directory_with_faxes> -O <path_to_output_directory> -v -s

## To Test:
> cd CovidFastFax/code

> python CovidFastFax.py -t ../data/test_images/ -O ../test_out -v -s 

This should create a directory called `test_out` in the main directory:


## Arguments

```
usage: CovidFastFax.py [-h] -t TARGET_DIR -O OUTPUT_DIR [-r] [-f] [-v]
                            [-d] [-s] [-e]
                            
optional arguments:  
      -h, --help            show this help message and exit  
      -r, --reset           Reset cache, will result in reprocessing of all files  
                            in the target directory  
      -f, --forced_reset    Reformat target directory, use with extreme caution,  
                            -r flag must also be specified  
      -v, --verbose         Verbose mode  
      -d, --debug           Debugging mode  
      -s, --split_pdfs      Split PDFs if a CMR template is detected. Creates a  
                            new PDF for each detected PDF, if the evaluated PDF  
                            only contains CMR pages, plus or minus one page. Otherwise,  
                            won't split the PDF  
      -e, --email_pings    Send email staying alive pings by pinging the server in  
                            email_endpoint.json  

required arguments:  
      -t TARGET_DIR, --target_dir TARGET_DIR  
                            Target directory to monitor for new PDFs  
      -O OUTPUT_DIR, --output_dir OUTPUT_DIR  
                            Location to create output directory  
```  
## CMR template construction

CMR template forms follow a standardized directory structure for import into the system. Each template directory contains a template image stored in PNG format. Two JSON files specifying the pixel locations of the relevant checkboxes for each priority group, vulnerable populations and healthcare workers are placed in the same directory. 
```bash
├── CovidFastFax
│   ├── data
│   │   ├── references
│   │   │   ├── template_name
│   │   │   │   ├── template_name.png
│   │   │   │   ├── template_name_vulnerable_coords.json
│   │   │   │   ├── template_name_hcw_coords.json
```

The inclusion of a new template requires the manual mapping of the priority checkboxes to pixel locations in form template. In our experience this process takes less than an hour and often less than 15 minutes. We then retrain the form template classification model to detect this new template, as well as the previous template. Retraining the template classification model on a single Nvidia V100 GPU<sup>1</sup> takes a few hours and additional speed improvements could be made to this process. Overall, it is possible to set up a new template within a 24hr time period for fast deployment of the system to new forms. 

<sup>1</sup>Can be trained on GPUs with lower amounts of GPU memory. 

