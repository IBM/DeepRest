# Microservice-based Social Network
![Microservices](assets/social-network.png)
This directory is a microservice-based social network modified based on the [DeathStarBench](https://github.com/delimitrou/DeathStarBench). For more details, you can refer to our paper at EuroSys'22:
* Ka-Ho Chow, Umesh Deshpande, Sangeetha Seshadri, and Ling Liu, "DeepRest: Deep Resource Estimation for Interactive Microservices," *ACM European Conference on Computer Systems (EuroSys)*, Rennes, France, Apr. 5-8, 2022.

It includes the following two components:   
* `./social-network-deploy/`: the yaml files for deploying the application on the cloud
* `./social-network-source/`: the source code of the social network

# Deploy the Social Network
This section describes how to deploy the social network on IBM cloud. It can be similarly deployed on Kubernetes or minikube with OpenEBS for experiment purposes (see `minikube-openebs`). 

1. Copy the `social-network-deploy` folder to the infrastructure node. You should see two folders in your current directory: (1) `assets` and (2) `k8s-yaml`.
2. Create PVCs and the initialization helper (a CentOS pod)
```bash
oc apply -f k8s-yaml/init/
```
3. Copy config files to two PVCs through the helper
```bash
oc cp assets/media-frontend/ social-network/centos:/media-config
oc cp assets/nginx-web-server/ social-network/centos:/nginx-config
oc cp assets/gen-lua/ social-network/centos:/nginx-config
```
4. Delete the helper pod
```bash
oc delete -f k8s-yaml/init/02-frontend-initializer.yaml
```
5. Start running all components of the social network
```bash
oc apply -f k8s-yaml/
```
6. Visit the OpenShift console, select `Networking` -> `Routes`, and find `social-network` in the project dropdown list
7. Click `Create Route` with the following configuration:
```
Name: nginx-thrift
Service: nginx-thrift
Target Port: 8080 -> 8080 (TCP)
```
8. Visit the nginx-thrift location shown in the `Routes` page, you should see:
![NGINX Thrift](assets/route-nginx-thrift.png)

9. Click `Create Route` with the following configuration:
```
Name: media-frontend
Service: media-frontend
Target Port: 8080 -> 8080 (TCP)
```
10. Visit the media-frontend location shown in the `Routes` page, you should see:
![Media Frontend](assets/route-media-frontend.png)

    
### Enable Jaeger for Distributed Tracing
1. Delete all running components of the social network
```bash
oc delete -f k8s-yaml/
```
2. Start an elasticsearch for trace storage and install Jaeger operator
```bash
oc apply -f k8s-yaml/tracing/init/
```
3. Start the distributed tracing with Jaeger
```bash
oc apply -f k8s-yaml/tracing/run.yaml
```
4. Start the ephemeral version of MongoDB components to trigger sidecar injection by Jaeger
```bash
oc apply -f k8s-yaml/ephemeral-mongodb/
```
5. Run all components of the social network
```bash
oc apply -f k8s-yaml/
```
6. Go to the `Routes` page in the OpenShift console again. You will see a new entry `jaeger-elasticsearch` created for you. Visit the location and you should see the following:
![Jaeger](assets/route-jaeger.png)
