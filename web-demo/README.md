# Web-based Demo
![](./assets/github_banner.png)

This directory is a web-based demo for DeepRest. It is preloaded with three types of API traffic that are unseen in the application learning phase based on common business scenarios: (1) unseen scale of application users, (2) unseen API compositions, and (3) unseen API traffic shapes. For more details, you can refer to our paper at EuroSys'22:
* Ka-Ho Chow, Umesh Deshpande, Sangeetha Seshadri, and Ling Liu, "DeepRest: Deep Resource Estimation for Interactive Microservices," *ACM European Conference on Computer Systems (EuroSys)*, Rennes, France, Apr. 5-8, 2022.


# Installation
Follow the instruction below to create a virtual environment and install necessary Python libraries.

* Create and activate a virtual environment  
```bash
virtualenv venv
source venv/bin/activate
```

* Manually install the following Python libraries or run `pip install -r requirements.txt`
```bash
dash==1.0.0
gunicorn>=19.9.0
numpy>=1.16.2
pandas>=0.24.2
datetime==4.3
pathlib==1.0.1
```

# Running the Web-based Demo
After following the instruction to install necessary libraries, you can run the web-based demo with:
```bash
python app.py
```

### Tested Environment
This demo kit is tested on the following machine and environment.

Operating System: macOS Catalina (Version 10.15.5)  
Processor: 2.3 GHz 8-Core Intel Core i9  
Memory: 64 GB 2667 MHz DDR4  
Graphics: Intel UHD Graphics 630 1536 MB  
Python Version: 3.6  
Browser: Chrome
