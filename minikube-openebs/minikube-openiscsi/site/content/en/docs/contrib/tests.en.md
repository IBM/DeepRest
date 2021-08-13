---
title: "Integration Tests"
description: >
  All minikube integration tests
---


## TestDownloadOnly
makes sure the --download-only parameter in minikube start caches the appropriate images and tarballs.

## TestDownloadOnlyKic
makes sure --download-only caches the docker driver images as well.

## TestOffline
makes sure minikube works without internet, once the user has cached the necessary images.
This test has to run after TestDownloadOnly.

## TestAddons
tests addons that require no special environment in parallel

#### validateIngressAddon
tests the ingress addon by deploying a default nginx pod

#### validateRegistryAddon
tests the registry addon

#### validateMetricsServerAddon
tests the metrics server addon by making sure "kubectl top pods" returns a sensible result

#### validateHelmTillerAddon
tests the helm tiller addon by running "helm version" inside the cluster

#### validateOlmAddon
tests the OLM addon

#### validateCSIDriverAndSnapshots
tests the csi hostpath driver by creating a persistent volume, snapshotting it and restoring it.

#### validateGCPAuthAddon
tests the GCP Auth addon with either phony or real credentials and makes sure the files are mounted into pods correctly

## TestCertOptions
makes sure minikube certs respect the --apiserver-ips and --apiserver-names parameters

## TestDockerFlags
makes sure the --docker-env and --docker-opt parameters are respected

## TestForceSystemdFlag
tests the --force-systemd flag, as one would expect.

#### validateDockerSystemd
makes sure the --force-systemd flag worked with the docker container runtime

#### validateContainerdSystemd
makes sure the --force-systemd flag worked with the containerd container runtime

## TestForceSystemdEnv
makes sure the MINIKUBE_FORCE_SYSTEMD environment variable works just as well as the --force-systemd flag

## TestKVMDriverInstallOrUpdate
makes sure our docker-machine-driver-kvm2 binary can be installed properly

## TestHyperKitDriverInstallOrUpdate
makes sure our docker-machine-driver-hyperkit binary can be installed properly

## TestHyperkitDriverSkipUpgrade
makes sure our docker-machine-driver-hyperkit binary can be installed properly

## TestErrorSpam
asserts that there are no unexpected errors displayed in minikube command outputs.

## TestFunctional
are functionality tests which can safely share a profile in parallel

#### validateNodeLabels
checks if minikube cluster is created with correct kubernetes's node label

#### validateLoadImage
makes sure that `minikube image load` works as expected

#### validateRemoveImage
makes sures that `minikube image rm` works as expected

#### validateBuildImage
makes sures that `minikube image build` works as expected

#### validateListImages
makes sures that `minikube image ls` works as expected

#### validateDockerEnv
check functionality of minikube after evaluating docker-env

#### validatePodmanEnv
check functionality of minikube after evaluating podman-env

#### validateStartWithProxy
makes sure minikube start respects the HTTP_PROXY environment variable

#### validateAuditAfterStart
makes sure the audit log contains the correct logging after minikube start

#### validateSoftStart
validates that after minikube already started, a "minikube start" should not change the configs.

#### validateKubeContext
asserts that kubectl is properly configured (race-condition prone!)

#### validateKubectlGetPods
asserts that `kubectl get pod -A` returns non-zero content

#### validateMinikubeKubectl
validates that the `minikube kubectl` command returns content

#### validateMinikubeKubectlDirectCall
validates that calling minikube's kubectl

#### validateExtraConfig
verifies minikube with --extra-config works as expected

#### validateComponentHealth
asserts that all Kubernetes components are healthy
NOTE: It expects all components to be Ready, so it makes sense to run it close after only those tests that include '--wait=all' start flag

#### validateStatusCmd
makes sure minikube status outputs correctly

#### validateDashboardCmd
asserts that the dashboard command works

#### validateDryRun
asserts that the dry-run mode quickly exits with the right code

#### validateCacheCmd
tests functionality of cache command (cache add, delete, list)

#### validateConfigCmd
asserts basic "config" command functionality

#### validateLogsCmd
asserts basic "logs" command functionality

#### validateLogsFileCmd
asserts "logs --file" command functionality

#### validateProfileCmd
asserts "profile" command functionality

#### validateServiceCmd
asserts basic "service" command functionality

#### validateAddonsCmd
asserts basic "addon" command functionality

#### validateSSHCmd
asserts basic "ssh" command functionality

#### validateCpCmd
asserts basic "cp" command functionality

#### validateMySQL
validates a minimalist MySQL deployment

#### validateFileSync
to check existence of the test file

#### validateCertSync
to check existence of the test certificate

#### validateUpdateContextCmd
asserts basic "update-context" command functionality

#### validateMountCmd
verifies the minikube mount command works properly

#### validatePersistentVolumeClaim
makes sure PVCs work properly

#### validateTunnelCmd
makes sure the minikube tunnel command works as expected

#### validateTunnelStart
starts `minikube tunnel`

#### validateServiceStable
starts nginx pod, nginx service and waits nginx having loadbalancer ingress IP

#### validateAccessDirect
validates if the test service can be accessed with LoadBalancer IP from host

#### validateDNSDig
validates if the DNS forwarding works by dig command DNS lookup
NOTE: DNS forwarding is experimental: https://minikube.sigs.k8s.io/docs/handbook/accessing/#dns-resolution-experimental

#### validateDNSDscacheutil
validates if the DNS forwarding works by dscacheutil command DNS lookup
NOTE: DNS forwarding is experimental: https://minikube.sigs.k8s.io/docs/handbook/accessing/#dns-resolution-experimental

#### validateAccessDNS
validates if the test service can be accessed with DNS forwarding from host
NOTE: DNS forwarding is experimental: https://minikube.sigs.k8s.io/docs/handbook/accessing/#dns-resolution-experimental

#### validateTunnelDelete
stops `minikube tunnel`

## TestGuestEnvironment
verifies files and packges installed inside minikube ISO/Base image

## TestGvisorAddon
tests the functionality of the gVisor addon

## TestJSONOutput
makes sure json output works properly for the start, pause, unpause, and stop commands

#### validateDistinctCurrentSteps
 validateDistinctCurrentSteps makes sure each step has a distinct step number

#### validateIncreasingCurrentSteps
verifies that for a successful minikube start, 'current step' should be increasing

## TestErrorJSONOutput
makes sure json output can print errors properly

## TestKicCustomNetwork
verifies the docker driver works with a custom network

## TestKicExistingNetwork
verifies the docker driver and run with an existing network

## TestingKicBaseImage
will return true if the integraiton test is running against a passed --base-image flag

## TestMultiNode
tests all multi node cluster functionality

#### validateMultiNodeStart
makes sure a 2 node cluster can start

#### validateAddNodeToMultiNode
uses the minikube node add command to add a node to an existing cluster

#### validateProfileListWithMultiNode
make sure minikube profile list outputs correct with multinode clusters

#### validateCopyFileWithMultiNode
validateProfileListWithMultiNode make sure minikube profile list outputs correct with multinode clusters

#### validateStopRunningNode
tests the minikube node stop command

#### validateStartNodeAfterStop
tests the minikube node start command on an existing stopped node

#### validateStopMultiNodeCluster
runs minikube stop on a multinode cluster

#### validateRestartMultiNodeCluster
verifies a soft restart on a multinode cluster works

#### validateDeleteNodeFromMultiNode
tests the minikube node delete command

#### validateNameConflict
tests that the node name verification works as expected

#### validateDeployAppToMultiNode
deploys an app to a multinode cluster and makes sure all nodes can serve traffic

## TestNetworkPlugins
tests all supported CNI options
Options tested: kubenet, bridge, flannel, kindnet, calico, cilium
Flags tested: enable-default-cni (legacy), false (CNI off), auto-detection

#### validateHairpinMode
makes sure the hairpinning (https://en.wikipedia.org/wiki/Hairpinning) is correctly configured for given CNI
try to access deployment/netcat pod using external, obtained from 'netcat' service dns resolution, IP address
should fail if hairpinMode is off

## TestChangeNoneUser
tests to make sure the CHANGE_MINIKUBE_NONE_USER environemt variable is respected
and changes the minikube file permissions from root to the correct user.

## TestPause
tests minikube pause functionality

#### validateFreshStart
just starts a new minikube cluster

#### validateStartNoReconfigure
validates that starting a running cluster does not invoke reconfiguration

#### validatePause
runs minikube pause

#### validateUnpause
runs minikube unpause

#### validateDelete
deletes the unpaused cluster

#### validateVerifyDeleted
makes sure no left over left after deleting a profile such as containers or volumes

#### validateStatus
makes sure paused clusters show up in minikube status correctly

## TestDebPackageInstall
TestPackageInstall tests installation of .deb packages with minikube itself and with kvm2 driver
on various debian/ubuntu docker images

## TestPreload
verifies the preload tarballs get pulled in properly by minikube

## TestScheduledStopWindows
tests the schedule stop functionality on Windows

## TestScheduledStopUnix
TestScheduledStopWindows tests the schedule stop functionality on Unix

## TestSkaffold
makes sure skaffold run can be run with minikube

## TestStartStop
tests starting, stopping and restarting a minikube clusters with various Kubernetes versions and configurations
The oldest supported, newest supported and default Kubernetes versions are always tested.

#### validateFirstStart
runs the initial minikube start

#### validateDeploying
deploys an app the minikube cluster

#### validateStop
tests minikube stop

#### validateEnableAddonAfterStop
makes sure addons can be enabled on a stopped cluster

#### validateSecondStart
verifies that starting a stopped cluster works

#### validateAppExistsAfterStop
verifies that a user's app will not vanish after a minikube stop

#### validateAddonAfterStop
validates that an addon which was enabled when minikube is stopped will be enabled and working..

#### validateKubernetesImages
verifies that a restarted cluster contains all the necessary images

#### validatePauseAfterStart
verifies that minikube pause works

## TestInsufficientStorage
makes sure minikube status displays the correct info if there is insufficient disk space on the machine

## TestRunningBinaryUpgrade
upgrades a running legacy cluster to minikube at HEAD

## TestStoppedBinaryUpgrade
starts a legacy minikube, stops it, and then upgrades to minikube at HEAD

## TestKubernetesUpgrade
upgrades Kubernetes from oldest to newest

## TestMissingContainerUpgrade
tests a Docker upgrade where the underlying container is missing

TEST COUNT: 115