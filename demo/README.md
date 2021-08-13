# Deep Learning for API-aware Resource Estimation
![](./assets/github_banner.png)

Cloud applications are increasingly shifting from monoliths to multiple fine-grained, interactive, and loosely-coupled microservices. Despite their advantages in enabling fine-grained scaling, estimating resources for each component can be challenging because an API call from an end-user can traverse along the path triggering different microservices and utilizing their resources differently based on the business logic. Distributed tracing has been gaining popularity as one of the pillars of observability. It offers end-to-end visibility from an API request perspective and provides invaluable information to learn the characteristics of each API. This project utilizes quantile recurrent neural networks in deep learning with distributed traces to automatically understand how each API utilizes resources in different microservices. We use the API-aware resource estimator to predict the scaling factor for each type of resource in each component under a hypothetical workload (e.g., 10x more users sharing photos on the social network application). The results demonstrate that our API-aware resource estimator can provide accurate and fine-grained insights on strategically reprovisioning each component.


# Installation
Please follow the instruction below to create a virtual environment and install necessary Python libraries.

* Create and start a virtual environment  
```bash
virtualenv venv
source venv/bin/activate
```

* Install the following Python libraries  
```bash
pip install pandas==1.1.0
pip install numpy==1.19.1
pip install tqdm==4.48.2
pip install scipy==1.5.2
pip install statsmodels==0.11.1
pip install plotly==4.9.0
pip install dash==1.14.0
pip install matplotlib==3.3.0
pip install jupyter==1.0.0
```

### Tested Environment
This demo kit is tested within the following machine and environment.

Operating System: macOS Catalina (Version 10.15.5)  
Processor: 2.3 GHz 8-Core Intel Core i9  
Memory: 64 GB 2667 MHz DDR4  
Graphics: Intel UHD Graphics 630 1536 MB  
Python Version: 3.6  
Browser: Chrome

# How to run the demo?
After following the instruction to install necessary libraries, you can run the web-based demo with:
```bash
python app.py
```


