# OpenEBS + cStor Storage Pool on minikube
This repository describes the setup of minikube with OpenEBS cStor to be the storage engine. Using pool devices on the minikube VM, this setup allows you to create multiple PVCs for your application and enjoy the per-PVC monitoring features provided by OpenEBS.

## Enabling iSCSI on minikube
Simply creating loop devices on the minikube VM does not work because OpenEBS requires iSCSI for provisioning. By default, iSCSI is not supported in the official minikube image. Hence, you need to follow one of the two options below to obtain a minikube iso with open-iscsi enabled. Please put the `minikube.iso` under this directory.

### Option 1: Download the iso from IBM Box
You can download the `minikube.iso` here: [https://ibm.ent.box.com/s/c3d6j4ovenccsa4vnsehzypvpbhvbrl9](https://ibm.ent.box.com/s/c3d6j4ovenccsa4vnsehzypvpbhvbrl9)

### Option 2: Build from the source
To build `minikube.iso`, you can first enter the `minikube-openiscsi/` directory and then execute:
```bash
sudo make buildroot-image
sudo make out/minikube.iso
```

## Creating Loop Devices on minikube
Step 1: Use the minikube.iso obtained from the above section to start minikube with open-iscsi
```bash
minikube start --cpus=8 --memory=16000 --driver=virtualbox --iso-url=file://$(pwd)/minikube.iso
```

Step 2: Enter the minikube VM with `minikube ssh` and create loop devices to be the block devices consumed by OpenEBS
```bash
dd if=/dev/zero of=loopbackfile.img bs=100M count=100
sudo losetup -fP loopbackfile.img
```

Step 3: Add the initiator name of iSCSI with `sudo vi /etc/iscsi/initiatorname.iscsi`
```bash
InitiatorName=iqn.1993-08.org.debian:01:3a1171891ef
```

Step 4: Restart `iscsid` with `sudo systemctl restart iscsid` and confirm the output of `systemctl status iscsid` matches below:
```bash
● iscsid.service - Open-iSCSI
     Loaded: loaded (/usr/lib/systemd/system/iscsid.service; enabled; vendor preset: enabled)
     Active: active (running) since Thu 2021-06-17 03:23:59 UTC; 5s ago
TriggeredBy: ● iscsid.socket
       Docs: man:iscsid(8)
             man:iscsiuio(8)
             man:iscsiadm(8)
   Main PID: 6271 (iscsid)
     Status: "Ready to process requests"
      Tasks: 1 (limit: 18957)
     Memory: 1.7M
     CGroup: /system.slice/iscsid.service
             └─6271 /sbin/iscsid -f

Jun 17 03:23:59 minikube systemd[1]: Starting Open-iSCSI...
Jun 17 03:23:59 minikube systemd[1]: Started Open-iSCSI.
```

## Installing OpenEBS
Step 1: Install OpenEBS operator using the provided YAML (Note: the official YAML file filters out loop devices)
```bash
kubectl apply -f ./openebs-operator.yaml
```

Step 2: Confirm all pods are running with `watch kubectl -n openebs get pods`
```bash
NAME                                          READY   STATUS    RESTARTS   AGE
maya-apiserver-68f4bc8578-97lh9               1/1     Running   0          74s
openebs-admission-server-574cf7c984-bzmxf     1/1     Running   0          74s
openebs-localpv-provisioner-9f598b455-bkdhr   1/1     Running   0          74s
openebs-ndm-operator-7c4f8fd547-hrsjz         1/1     Running   0          74s
openebs-ndm-prprv                             1/1     Running   0          74s
openebs-provisioner-65749f64fd-b28vk          1/1     Running   0          74s
openebs-snapshot-operator-5cfbb6fdb5-8qptr    2/2     Running   0          74s
```

Step 3: Confirm StorageClasses are created with `kubectl get sc`
```bash
NAME                        PROVISIONER                                                RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
openebs-device              openebs.io/local                                           Delete          WaitForFirstConsumer   false                  88s
openebs-hostpath            openebs.io/local                                           Delete          WaitForFirstConsumer   false                  88s
openebs-jiva-default        openebs.io/provisioner-iscsi                               Delete          Immediate              false                  89s
openebs-snapshot-promoter   volumesnapshot.external-storage.k8s.io/snapshot-promoter   Delete          Immediate              false                  88s
standard (default)          k8s.io/minikube-hostpath                                   Delete          Immediate              false                  7m44s
```

Step 4: Confirm the block (loop) devices created above are detected with `kubectl -n openebs get blockdevice`
```bash
NAME                                           NODENAME   SIZE          CLAIMSTATE   STATUS   AGE
blockdevice-50ef127b8853e09e1b519c6cc41f4d85   minikube   10485760000   Unclaimed    Active   2m18s
```

Step 5: Updated the `blockDeviceList` in `./cstor-pool-config.yaml` and run `kubectl -n openebs apply -f ./cstor-pool-config.yaml` to create a StoragePoolClaim
```yaml
apiVersion: openebs.io/v1alpha1
kind: StoragePoolClaim
metadata:
  name: cstor-disk-pool
  annotations:
    cas.openebs.io/config: |
      - name: PoolResourceRequests
        value: |-
            memory: 2Gi
      - name: PoolResourceLimits
        value: |-
            memory: 4Gi
spec:
  name: cstor-disk-pool
  type: disk
  poolSpec:
    poolType: striped
  blockDevices:
    blockDeviceList:   <----------- Update this list!!!!!!!!!!!!!!!
    - blockdevice-50ef127b8853e09e1b519c6cc41f4d85
---
```

Step 6: Confirm all pods are running with `watch kubectl -n openebs get pods`
```bash
NAME                                          READY   STATUS    RESTARTS   AGE
cstor-disk-pool-rzp4-6596c67c49-jnztf         3/3     Running   0          114s
...
```

Step 7: Confirm the cStor Storage Pool has been properly created with `kubectl get csp`
```bash
kubectl -n openebs get pods
NAME                   ALLOCATED   FREE    CAPACITY   STATUS    READONLY   TYPE      AGE
cstor-disk-pool-rzp4   86K         9.75G   9.75G      Healthy   false      striped   2m7s
```

Step 8: Create a StorageClass for the cStor Storage Pool named `openebs-sc-statefulset` with only one replica in minikube
```bash
kubectl apply -f ./openebs-sc-rep.yaml
```

Step 9: Confirm the StorageClass named `openebs-sc-statefulset` has been created with `kubectl get sc`
```bash
NAME                        PROVISIONER                                                RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
openebs-device              openebs.io/local                                           Delete          WaitForFirstConsumer   false                  10m
openebs-hostpath            openebs.io/local                                           Delete          WaitForFirstConsumer   false                  10m
openebs-jiva-default        openebs.io/provisioner-iscsi                               Delete          Immediate              false                  10m
openebs-sc-statefulset      openebs.io/provisioner-iscsi                               Delete          Immediate              false                  14s
openebs-snapshot-promoter   volumesnapshot.external-storage.k8s.io/snapshot-promoter   Delete          Immediate              false                  10m
standard (default)          k8s.io/minikube-hostpath                                   Delete          Immediate              false                  16m
```

## Installation of Prometheus and Grafana for Per-PVC Monitoring
Step 1: Run the following command
```bash
kubectl -n openebs apply -f ./monitor-openebs-pg.yaml -f ./monitor-kube-state-metrics.yaml
```

Step 2: Confirm all pods are running
```bash
watch kubectl -n openebs get pods
```

Step 3: Obtain the addresses of Prometheus and Grafana from minikube and you can use the OpenEBS dashboard `openebs-pg-dashboard.json` for monitoring in Grafana
```bash
minikube service list
```

## Demonstration
Step 1: Create a PersistentVolumeClaim with the `demo-cstor-pvc.yaml` file below using `kubectl -n openebs apply -f ./demo/demo-cstor-pvc.yaml`
```yaml
apiVersion: v1  
kind: PersistentVolumeClaim  
metadata:  
  name: cstor-pvc
spec:  
  storageClassName: openebs-sc-statefulset  
  accessModes:  
   - ReadWriteOnce  
  resources:  
   requests:  
    storage: 2Gi
```

Step 2: Create a Pod using the above PVC with the `demo-cstor-pod.yaml` file below using `kubectl -n openebs apply -f ./demo/demo-cstor-pod.yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello-cstor-pod
spec:
  volumes:
  - name: cstor-storage
    persistentVolumeClaim:
      claimName: cstor-pvc
  containers:
  - name: hello-container
    image: busybox
    command:
       - sh
       - -c
       - 'while true; do echo "`date` [`hostname`] Hello from OpenEBS cStor PV." >> /mnt/store/greet.txt; sleep $((60)); done'
    volumeMounts:
    - mountPath: /mnt/store
      name: cstor-storage
```

This pod writes a line to `/mnt/store/greet.txt` stored in the PVC every 60 seconds. We can verify the output with `kubectl -n openebs exec hello-cstor-pod -- cat /mnt/store/greet.txt`
```bash
Thu Jun 17 03:42:38 UTC 2021 [hello-cstor-pod] Hello from OpenEBS cStor PV.
Thu Jun 17 03:43:38 UTC 2021 [hello-cstor-pod] Hello from OpenEBS cStor PV.
```

If we delete the pod and start another one, the above command will show both lines written by the current pod and also the previous pods which were connected to the same PVC.

## Computing Environment

This repository has been tested on the host machine with the following configuration:
* OS: Ubuntu 18.04.3 LTS
* Memory: 31.2 GiB
* Processor: Intel® Core™ i7-9700K CPU @ 3.60GHz × 8
* Graphics: GeForce RTX 2080 SUPER/PCIe/SSE2
* OS Type: 64-bit
* Disk: 2.0 TB
* VM Driver: VirtualBox
