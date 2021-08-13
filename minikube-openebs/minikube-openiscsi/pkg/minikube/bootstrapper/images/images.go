/*
Copyright 2019 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package images implements helpers for getting image names
package images

import (
	"fmt"
	"path"

	"github.com/blang/semver"

	"k8s.io/minikube/pkg/version"
)

// Pause returns the image name to pull for a given Kubernetes version
func Pause(v semver.Version, mirror string) string {
	// Should match `PauseVersion` in:
	// https://github.com/kubernetes/kubernetes/blob/master/cmd/kubeadm/app/constants/constants_unix.go
	pv := "3.2"
	if semver.MustParseRange("<1.18.0-alpha.0")(v) {
		pv = "3.1"
	}
	return path.Join(kubernetesRepo(mirror), "pause:"+pv)
}

// essentials returns images needed too bootstrap a Kubernetes
func essentials(mirror string, v semver.Version) []string {
	imgs := []string{
		componentImage("kube-proxy", v, mirror),
		componentImage("kube-scheduler", v, mirror),
		componentImage("kube-controller-manager", v, mirror),
		componentImage("kube-apiserver", v, mirror),
		coreDNS(v, mirror),
		etcd(v, mirror),
		Pause(v, mirror),
	}
	return imgs
}

// componentImage returns a Kubernetes component image to pull
func componentImage(name string, v semver.Version, mirror string) string {
	return fmt.Sprintf("%s:v%s", path.Join(kubernetesRepo(mirror), name), v)
}

// coreDNS returns the images used for CoreDNS
func coreDNS(v semver.Version, mirror string) string {
	// Should match `CoreDNSVersion` in
	// https://github.com/kubernetes/kubernetes/blob/master/cmd/kubeadm/app/constants/constants.go
	cv := "1.7.0"
	switch v.Minor {
	case 22:
		cv = "1.8.0"
	case 10, 20, 21:
		cv = "1.7.0"
	case 18:
		cv = "1.6.7"
	case 17:
		cv = "1.6.5"
	case 16:
		cv = "1.6.2"
	case 15, 14:
		cv = "1.3.1"
	case 13:
		cv = "1.2.6"
	case 12:
		cv = "1.2.2"
	case 11:
		cv = "1.1.3"
	}
	return path.Join(kubernetesRepo(mirror), "coredns:"+cv)
}

// etcd returns the image used for etcd
func etcd(v semver.Version, mirror string) string {
	// Should match `DefaultEtcdVersion` in:
	// https://github.com/kubernetes/kubernetes/blob/master/cmd/kubeadm/app/constants/constants.go
	ev := "3.4.13-0"

	switch v.Minor {
	case 17, 18:
		ev = "3.4.3-0"
	case 16:
		ev = "3.3.15-0"
	case 14, 15:
		ev = "3.3.10"
	case 12, 13:
		ev = "3.2.24"
	case 11:
		ev = "3.2.18"
	}

	// An awkward special case for v1.19.0 - do not imitate unless necessary
	if v.Equals(semver.MustParse("1.19.0")) {
		ev = "3.4.9-1"
	}

	return path.Join(kubernetesRepo(mirror), "etcd:"+ev)
}

// auxiliary returns images that are helpful for running minikube
func auxiliary(mirror string) []string {
	return []string{
		storageProvisioner(mirror),
		dashboardFrontend(mirror),
		dashboardMetrics(mirror),
		// NOTE: kindnet is also used when the Docker driver is used with a non-Docker runtime
	}
}

// storageProvisioner returns the minikube storage provisioner image
func storageProvisioner(mirror string) string {
	return path.Join(minikubeRepo(mirror), "storage-provisioner:"+version.GetStorageProvisionerVersion())
}

// dashboardFrontend returns the image used for the dashboard frontend
func dashboardFrontend(repo string) string {
	if repo == "" {
		repo = "docker.io/kubernetesui"
	}
	// See 'kubernetes-dashboard' in deploy/addons/dashboard/dashboard-dp.yaml
	return path.Join(repo, "dashboard:v2.1.0")
}

// dashboardMetrics returns the image used for the dashboard metrics scraper
func dashboardMetrics(repo string) string {
	if repo == "" {
		repo = "docker.io/kubernetesui"
	}
	// See 'dashboard-metrics-scraper' in deploy/addons/dashboard/dashboard-dp.yaml
	return path.Join(repo, "metrics-scraper:v1.0.4")
}

// KindNet returns the image used for kindnet
// ref: https://hub.docker.com/r/kindest/kindnetd/tags
// src: https://github.com/kubernetes-sigs/kind/tree/master/images/kindnetd
func KindNet(repo string) string {
	if repo == "" {
		repo = "kindest"
	}
	return path.Join(repo, "kindnetd:v20210326-1e038dc5")
}

// CalicoDaemonSet returns the image used for calicoDaemonSet
func CalicoDaemonSet(repo string) string {
	if repo == "" {
		repo = "calico"
	}
	return path.Join(repo, "node:v3.14.1")
}

// CalicoDeployment returns the image used for calicoDeployment
func CalicoDeployment(repo string) string {
	if repo == "" {
		repo = "calico"
	}
	return path.Join(repo, "kube-controllers:v3.14.1")
}
