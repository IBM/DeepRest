// +build integration

/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	retryablehttp "github.com/hashicorp/go-retryablehttp"
	"k8s.io/minikube/pkg/kapi"
	"k8s.io/minikube/pkg/minikube/detect"
	"k8s.io/minikube/pkg/util/retry"
)

// TestAddons tests addons that require no special environment in parallel
func TestAddons(t *testing.T) {
	profile := UniqueProfileName("addons")
	ctx, cancel := context.WithTimeout(context.Background(), Minutes(40))
	defer Cleanup(t, profile, cancel)

	// We don't need a dummy file is we're on GCE
	if !detect.IsOnGCE() || detect.IsCloudShell() {
		// Set an env var to point to our dummy credentials file
		err := os.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(*testdataDir, "gcp-creds.json"))
		defer os.Unsetenv("GOOGLE_APPLICATION_CREDENTIALS")
		if err != nil {
			t.Fatalf("Failed setting GOOGLE_APPLICATION_CREDENTIALS env var: %v", err)
		}

		err = os.Setenv("GOOGLE_CLOUD_PROJECT", "this_is_fake")
		defer os.Unsetenv("GOOGLE_CLOUD_PROJECT")
		if err != nil {
			t.Fatalf("Failed setting GOOGLE_CLOUD_PROJECT env var: %v", err)
		}
	}

	args := append([]string{"start", "-p", profile, "--wait=true", "--memory=4000", "--alsologtostderr", "--addons=registry", "--addons=metrics-server", "--addons=olm", "--addons=volumesnapshots", "--addons=csi-hostpath-driver"}, StartArgs()...)
	if !NoneDriver() && !(runtime.GOOS == "darwin" && KicDriver()) { // none driver and macos docker driver does not support ingress
		args = append(args, "--addons=ingress")
	}
	if !arm64Platform() {
		args = append(args, "--addons=helm-tiller")
	}
	if !detect.IsOnGCE() {
		args = append(args, "--addons=gcp-auth")
	}
	rr, err := Run(t, exec.CommandContext(ctx, Target(), args...))
	if err != nil {
		t.Fatalf("%s failed: %v", rr.Command(), err)
	}

	// If we're running the integration tests on GCE, which is frequently the case, first check to make sure we exit out properly,
	// then use force to actually test using creds.
	if detect.IsOnGCE() {
		args = []string{"-p", profile, "addons", "enable", "gcp-auth"}
		rr, err := Run(t, exec.CommandContext(ctx, Target(), args...))
		if err == nil {
			t.Errorf("Expected error but didn't get one. command %v, output %v", rr.Command(), rr.Output())
		} else {
			if !strings.Contains(rr.Output(), "It seems that you are running in GCE") {
				t.Errorf("Unexpected error message: %v", rr.Output())
			} else {
				// ok, use force here since we are in GCE
				// do not use --force unless absolutely necessary
				args = append(args, "--force")
				rr, err := Run(t, exec.CommandContext(ctx, Target(), args...))
				if err != nil {
					t.Errorf("%s failed: %v", rr.Command(), err)
				}
			}
		}
	}

	// Parallelized tests
	t.Run("parallel", func(t *testing.T) {
		tests := []struct {
			name      string
			validator validateFunc
		}{
			{"Registry", validateRegistryAddon},
			{"Ingress", validateIngressAddon},
			{"MetricsServer", validateMetricsServerAddon},
			{"HelmTiller", validateHelmTillerAddon},
			{"Olm", validateOlmAddon},
			{"CSI", validateCSIDriverAndSnapshots},
			{"GCPAuth", validateGCPAuthAddon},
		}
		for _, tc := range tests {
			tc := tc
			if ctx.Err() == context.DeadlineExceeded {
				t.Fatalf("Unable to run more tests (deadline exceeded)")
			}
			t.Run(tc.name, func(t *testing.T) {
				MaybeParallel(t)
				tc.validator(ctx, t, profile)
			})
		}
	})

	// Assert that disable/enable works offline
	rr, err = Run(t, exec.CommandContext(ctx, Target(), "stop", "-p", profile))
	if err != nil {
		t.Errorf("failed to stop minikube. args %q : %v", rr.Command(), err)
	}
	rr, err = Run(t, exec.CommandContext(ctx, Target(), "addons", "enable", "dashboard", "-p", profile))
	if err != nil {
		t.Errorf("failed to enable dashboard addon: args %q : %v", rr.Command(), err)
	}
	rr, err = Run(t, exec.CommandContext(ctx, Target(), "addons", "disable", "dashboard", "-p", profile))
	if err != nil {
		t.Errorf("failed to disable dashboard addon: args %q : %v", rr.Command(), err)
	}
}

// validateIngressAddon tests the ingress addon by deploying a default nginx pod
func validateIngressAddon(ctx context.Context, t *testing.T, profile string) {
	defer PostMortemLogs(t, profile)
	if NoneDriver() || (runtime.GOOS == "darwin" && KicDriver()) {
		t.Skipf("skipping: ingress not supported ")
	}

	client, err := kapi.Client(profile)
	if err != nil {
		t.Fatalf("failed to get Kubernetes client: %v", client)
	}

	if err := kapi.WaitForDeploymentToStabilize(client, "ingress-nginx", "ingress-nginx-controller", Minutes(6)); err != nil {
		t.Errorf("failed waiting for ingress-controller deployment to stabilize: %v", err)
	}
	if _, err := PodWait(ctx, t, profile, "ingress-nginx", "app.kubernetes.io/name=ingress-nginx", Minutes(12)); err != nil {
		t.Fatalf("failed waititing for ingress-nginx-controller : %v", err)
	}

	// create networking.k8s.io/v1beta1 ingress
	createv1betaIngress := func() error {
		// apply networking.k8s.io/v1beta1 ingress
		rr, err := Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "replace", "--force", "-f", filepath.Join(*testdataDir, "nginx-ingv1beta.yaml")))
		if err != nil {
			return err
		}
		if rr.Stderr.String() != "" {
			t.Logf("%v: unexpected stderr: %s (may be temporary)", rr.Command(), rr.Stderr)
		}
		return nil
	}

	// create networking.k8s.io/v1beta1 ingress
	if err := retry.Expo(createv1betaIngress, 1*time.Second, Seconds(90)); err != nil {
		t.Errorf("failed to create ingress: %v", err)
	}

	rr, err := Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "replace", "--force", "-f", filepath.Join(*testdataDir, "nginx-pod-svc.yaml")))
	if err != nil {
		t.Errorf("failed to kubectl replace nginx-pod-svc. args %q. %v", rr.Command(), err)
	}

	if _, err := PodWait(ctx, t, profile, "default", "run=nginx", Minutes(4)); err != nil {
		t.Fatalf("failed waiting for ngnix pod: %v", err)
	}
	if err := kapi.WaitForService(client, "default", "nginx", true, time.Millisecond*500, Minutes(10)); err != nil {
		t.Errorf("failed waiting for nginx service to be up: %v", err)
	}

	want := "Welcome to nginx!"
	addr := "http://127.0.0.1/"
	// check if the ingress can route nginx app with networking.k8s.io/v1beta1 ingress
	checkv1betaIngress := func() error {
		var rr *RunResult
		var err error
		if NoneDriver() { // just run curl directly on the none driver
			rr, err = Run(t, exec.CommandContext(ctx, "curl", "-s", addr, "-H", "'Host: nginx.example.com'"))
			if err != nil {
				return err
			}
		} else {
			rr, err = Run(t, exec.CommandContext(ctx, Target(), "-p", profile, "ssh", fmt.Sprintf("curl -s %s -H 'Host: nginx.example.com'", addr)))
			if err != nil {
				return err
			}
		}

		stderr := rr.Stderr.String()
		if rr.Stderr.String() != "" {
			t.Logf("debug: unexpected stderr for %v:\n%s", rr.Command(), stderr)
		}

		stdout := rr.Stdout.String()
		if !strings.Contains(stdout, want) {
			return fmt.Errorf("%v stdout = %q, want %q", rr.Command(), stdout, want)
		}
		return nil
	}

	// check if the ingress can route nginx app with networking.k8s.io/v1beta1 ingress
	if err := retry.Expo(checkv1betaIngress, 500*time.Millisecond, Seconds(90)); err != nil {
		t.Errorf("failed to get expected response from %s within minikube: %v", addr, err)
	}

	// create networking.k8s.io/v1 ingress
	createv1Ingress := func() error {
		// apply networking.k8s.io/v1beta1 ingress
		rr, err := Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "replace", "--force", "-f", filepath.Join(*testdataDir, "nginx-ingv1.yaml")))
		if err != nil {
			return err
		}
		if rr.Stderr.String() != "" {
			t.Logf("%v: unexpected stderr: %s (may be temporary)", rr.Command(), rr.Stderr)
		}
		return nil
	}

	// create networking.k8s.io/v1 ingress
	if err := retry.Expo(createv1Ingress, 1*time.Second, Seconds(90)); err != nil {
		t.Errorf("failed to create ingress: %v", err)
	}

	// check if the ingress can route nginx app with networking.k8s.io/v1 ingress
	checkv1Ingress := func() error {
		var rr *RunResult
		var err error
		if NoneDriver() { // just run curl directly on the none driver
			rr, err = Run(t, exec.CommandContext(ctx, "curl", "-s", addr, "-H", "'Host: nginx.example.com'"))
			if err != nil {
				return err
			}
		} else {
			rr, err = Run(t, exec.CommandContext(ctx, Target(), "-p", profile, "ssh", fmt.Sprintf("curl -s %s -H 'Host: nginx.example.com'", addr)))
			if err != nil {
				return err
			}
		}

		stderr := rr.Stderr.String()
		if rr.Stderr.String() != "" {
			t.Logf("debug: unexpected stderr for %v:\n%s", rr.Command(), stderr)
		}

		stdout := rr.Stdout.String()
		if !strings.Contains(stdout, want) {
			return fmt.Errorf("%v stdout = %q, want %q", rr.Command(), stdout, want)
		}
		return nil
	}

	// check if the ingress can route nginx app with networking.k8s.io/v1 ingress
	if err := retry.Expo(checkv1Ingress, 500*time.Millisecond, Seconds(90)); err != nil {
		t.Errorf("failed to get expected response from %s within minikube: %v", addr, err)
	}

	rr, err = Run(t, exec.CommandContext(ctx, Target(), "-p", profile, "addons", "disable", "ingress", "--alsologtostderr", "-v=1"))
	if err != nil {
		t.Errorf("failed to disable ingress addon. args %q : %v", rr.Command(), err)
	}
}

// validateRegistryAddon tests the registry addon
func validateRegistryAddon(ctx context.Context, t *testing.T, profile string) {
	defer PostMortemLogs(t, profile)

	client, err := kapi.Client(profile)
	if err != nil {
		t.Fatalf("failed to get Kubernetes client for %s : %v", profile, err)
	}

	start := time.Now()
	if err := kapi.WaitForRCToStabilize(client, "kube-system", "registry", Minutes(6)); err != nil {
		t.Errorf("failed waiting for registry replicacontroller to stabilize: %v", err)
	}
	t.Logf("registry stabilized in %s", time.Since(start))

	if _, err := PodWait(ctx, t, profile, "kube-system", "actual-registry=true", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for pod actual-registry: %v", err)
	}
	if _, err := PodWait(ctx, t, profile, "kube-system", "registry-proxy=true", Minutes(10)); err != nil {
		t.Fatalf("failed waiting for pod registry-proxy: %v", err)
	}

	// Test from inside the cluster (no curl available on busybox)
	rr, err := Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "delete", "po", "-l", "run=registry-test", "--now"))
	if err != nil {
		t.Logf("pre-cleanup %s failed: %v (not a problem)", rr.Command(), err)
	}

	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "run", "--rm", "registry-test", "--restart=Never", "--image=busybox", "-it", "--", "sh", "-c", "wget --spider -S http://registry.kube-system.svc.cluster.local"))
	if err != nil {
		t.Errorf("failed to hit registry.kube-system.svc.cluster.local. args %q failed: %v", rr.Command(), err)
	}
	want := "HTTP/1.1 200"
	if !strings.Contains(rr.Stdout.String(), want) {
		t.Errorf("expected curl response be %q, but got *%s*", want, rr.Stdout.String())
	}

	if NeedsPortForward() {
		t.Skipf("Unable to complete rest of the test due to connectivity assumptions")
	}

	// Test from outside the cluster
	rr, err = Run(t, exec.CommandContext(ctx, Target(), "-p", profile, "ip"))
	if err != nil {
		t.Fatalf("failed run minikube ip. args %q : %v", rr.Command(), err)
	}
	if rr.Stderr.String() != "" {
		t.Errorf("expected stderr to be -empty- but got: *%q* .  args %q", rr.Stderr, rr.Command())
	}

	endpoint := fmt.Sprintf("http://%s:%d", strings.TrimSpace(rr.Stdout.String()), 5000)
	u, err := url.Parse(endpoint)
	if err != nil {
		t.Fatalf("failed to parse %q: %v", endpoint, err)
	}

	checkExternalAccess := func() error {
		resp, err := retryablehttp.Get(u.String())
		if err != nil {
			return err
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("%s = status code %d, want %d", u, resp.StatusCode, http.StatusOK)
		}
		return nil
	}

	if err := retry.Expo(checkExternalAccess, 500*time.Millisecond, Seconds(150)); err != nil {
		t.Errorf("failed to check external access to %s: %v", u.String(), err.Error())
	}

	rr, err = Run(t, exec.CommandContext(ctx, Target(), "-p", profile, "addons", "disable", "registry", "--alsologtostderr", "-v=1"))
	if err != nil {
		t.Errorf("failed to disable registry addon. args %q: %v", rr.Command(), err)
	}
}

// validateMetricsServerAddon tests the metrics server addon by making sure "kubectl top pods" returns a sensible result
func validateMetricsServerAddon(ctx context.Context, t *testing.T, profile string) {
	defer PostMortemLogs(t, profile)

	client, err := kapi.Client(profile)
	if err != nil {
		t.Fatalf("failed to get Kubernetes client for %s: %v", profile, err)
	}

	start := time.Now()
	if err := kapi.WaitForDeploymentToStabilize(client, "kube-system", "metrics-server", Minutes(6)); err != nil {
		t.Errorf("failed waiting for metrics-server deployment to stabilize: %v", err)
	}
	t.Logf("metrics-server stabilized in %s", time.Since(start))

	if _, err := PodWait(ctx, t, profile, "kube-system", "k8s-app=metrics-server", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for k8s-app=metrics-server pod: %v", err)
	}

	want := "CPU(cores)"
	checkMetricsServer := func() error {
		rr, err := Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "top", "pods", "-n", "kube-system"))
		if err != nil {
			return err
		}
		if rr.Stderr.String() != "" {
			t.Logf("%v: unexpected stderr: %s", rr.Command(), rr.Stderr)
		}
		if !strings.Contains(rr.Stdout.String(), want) {
			return fmt.Errorf("%v stdout = %q, want %q", rr.Command(), rr.Stdout, want)
		}
		return nil
	}

	if err := retry.Expo(checkMetricsServer, time.Second*3, Minutes(6)); err != nil {
		t.Errorf("failed checking metric server: %v", err.Error())
	}

	rr, err := Run(t, exec.CommandContext(ctx, Target(), "-p", profile, "addons", "disable", "metrics-server", "--alsologtostderr", "-v=1"))
	if err != nil {
		t.Errorf("failed to disable metrics-server addon: args %q: %v", rr.Command(), err)
	}
}

// validateHelmTillerAddon tests the helm tiller addon by running "helm version" inside the cluster
func validateHelmTillerAddon(ctx context.Context, t *testing.T, profile string) {

	defer PostMortemLogs(t, profile)

	if arm64Platform() {
		t.Skip("skip Helm test on arm64")
	}

	client, err := kapi.Client(profile)
	if err != nil {
		t.Fatalf("failed to get Kubernetes client for %s: %v", profile, err)
	}

	start := time.Now()
	if err := kapi.WaitForDeploymentToStabilize(client, "kube-system", "tiller-deploy", Minutes(6)); err != nil {
		t.Errorf("failed waiting for tiller-deploy deployment to stabilize: %v", err)
	}
	t.Logf("tiller-deploy stabilized in %s", time.Since(start))

	if _, err := PodWait(ctx, t, profile, "kube-system", "app=helm", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for helm pod: %v", err)
	}

	if NoneDriver() {
		_, err := exec.LookPath("socat")
		if err != nil {
			t.Skipf("socat is required by kubectl to complete this test")
		}
	}

	want := "Server: &version.Version"
	// Test from inside the cluster (`helm version` use pod.list permission. we use tiller serviceaccount in kube-system to list pod)
	checkHelmTiller := func() error {

		rr, err := Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "run", "--rm", "helm-test", "--restart=Never", "--image=alpine/helm:2.16.3", "-it", "--namespace=kube-system", "--serviceaccount=tiller", "--", "version"))
		if err != nil {
			return err
		}
		if rr.Stderr.String() != "" {
			t.Logf("%v: unexpected stderr: %s", rr.Command(), rr.Stderr)
		}
		if !strings.Contains(rr.Stdout.String(), want) {
			return fmt.Errorf("%v stdout = %q, want %q", rr.Command(), rr.Stdout, want)
		}
		return nil
	}

	if err := retry.Expo(checkHelmTiller, 500*time.Millisecond, Minutes(2)); err != nil {
		t.Errorf("failed checking helm tiller: %v", err.Error())
	}

	rr, err := Run(t, exec.CommandContext(ctx, Target(), "-p", profile, "addons", "disable", "helm-tiller", "--alsologtostderr", "-v=1"))
	if err != nil {
		t.Errorf("failed disabling helm-tiller addon. arg %q.s %v", rr.Command(), err)
	}
}

// validateOlmAddon tests the OLM addon
func validateOlmAddon(ctx context.Context, t *testing.T, profile string) {
	defer PostMortemLogs(t, profile)

	client, err := kapi.Client(profile)
	if err != nil {
		t.Fatalf("failed to get Kubernetes client for %s: %v", profile, err)
	}

	start := time.Now()
	if err := kapi.WaitForDeploymentToStabilize(client, "olm", "catalog-operator", Minutes(6)); err != nil {
		t.Errorf("failed waiting for catalog-operator deployment to stabilize: %v", err)
	}
	t.Logf("catalog-operator stabilized in %s", time.Since(start))
	if err := kapi.WaitForDeploymentToStabilize(client, "olm", "olm-operator", Minutes(6)); err != nil {
		t.Errorf("failed waiting for olm-operator deployment to stabilize: %v", err)
	}
	t.Logf("olm-operator stabilized in %s", time.Since(start))
	if err := kapi.WaitForDeploymentToStabilize(client, "olm", "packageserver", Minutes(6)); err != nil {
		t.Errorf("failed waiting for packageserver deployment to stabilize: %v", err)
	}
	t.Logf("packageserver stabilized in %s", time.Since(start))

	if _, err := PodWait(ctx, t, profile, "olm", "app=catalog-operator", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for pod catalog-operator: %v", err)
	}
	if _, err := PodWait(ctx, t, profile, "olm", "app=olm-operator", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for pod olm-operator: %v", err)
	}
	if _, err := PodWait(ctx, t, profile, "olm", "app=packageserver", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for pod packageserver: %v", err)
	}
	if _, err := PodWait(ctx, t, profile, "olm", "olm.catalogSource=operatorhubio-catalog", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for pod operatorhubio-catalog: %v", err)
	}

	// Install one sample Operator such as etcd
	rr, err := Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "create", "-f", filepath.Join(*testdataDir, "etcd.yaml")))
	if err != nil {
		t.Logf("etcd operator installation with %s failed: %v", rr.Command(), err)
	}

	want := "Succeeded"
	checkOperatorInstalled := func() error {
		rr, err := Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "get", "csv", "-n", "my-etcd"))
		if err != nil {
			return err
		}
		if rr.Stderr.String() != "" {
			t.Logf("%v: unexpected stderr: %s", rr.Command(), rr.Stderr)
		}
		if !strings.Contains(rr.Stdout.String(), want) {
			return fmt.Errorf("%v stdout = %q, want %q", rr.Command(), rr.Stdout, want)
		}
		return nil
	}

	// Operator installation takes a while
	if err := retry.Expo(checkOperatorInstalled, time.Second*3, Minutes(10)); err != nil {
		t.Errorf("failed checking operator installed: %v", err.Error())
	}
}

// validateCSIDriverAndSnapshots tests the csi hostpath driver by creating a persistent volume, snapshotting it and restoring it.
func validateCSIDriverAndSnapshots(ctx context.Context, t *testing.T, profile string) {
	defer PostMortemLogs(t, profile)

	client, err := kapi.Client(profile)
	if err != nil {
		t.Fatalf("failed to get Kubernetes client for %s: %v", profile, err)
	}

	start := time.Now()
	if err := kapi.WaitForPods(client, "kube-system", "kubernetes.io/minikube-addons=csi-hostpath-driver", Minutes(6)); err != nil {
		t.Errorf("failed waiting for csi-hostpath-driver pods to stabilize: %v", err)
	}
	t.Logf("csi-hostpath-driver pods stabilized in %s", time.Since(start))

	// create sample PVC
	rr, err := Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "create", "-f", filepath.Join(*testdataDir, "csi-hostpath-driver", "pvc.yaml")))
	if err != nil {
		t.Logf("creating sample PVC with %s failed: %v", rr.Command(), err)
	}

	if err := PVCWait(ctx, t, profile, "default", "hpvc", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for PVC hpvc: %v", err)
	}

	// create sample pod with the PVC
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "create", "-f", filepath.Join(*testdataDir, "csi-hostpath-driver", "pv-pod.yaml")))
	if err != nil {
		t.Logf("creating pod with %s failed: %v", rr.Command(), err)
	}

	if _, err := PodWait(ctx, t, profile, "default", "app=task-pv-pod", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for pod task-pv-pod: %v", err)
	}

	// create volume snapshot
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "create", "-f", filepath.Join(*testdataDir, "csi-hostpath-driver", "snapshot.yaml")))
	if err != nil {
		t.Logf("creating pod with %s failed: %v", rr.Command(), err)
	}

	if err := VolumeSnapshotWait(ctx, t, profile, "default", "new-snapshot-demo", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for volume snapshot new-snapshot-demo: %v", err)
	}

	// delete pod
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "delete", "pod", "task-pv-pod"))
	if err != nil {
		t.Logf("deleting pod with %s failed: %v", rr.Command(), err)
	}

	// delete pvc
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "delete", "pvc", "hpvc"))
	if err != nil {
		t.Logf("deleting pod with %s failed: %v", rr.Command(), err)
	}

	// restore pv from snapshot
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "create", "-f", filepath.Join(*testdataDir, "csi-hostpath-driver", "pvc-restore.yaml")))
	if err != nil {
		t.Logf("creating pvc with %s failed: %v", rr.Command(), err)
	}

	if err = PVCWait(ctx, t, profile, "default", "hpvc-restore", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for PVC hpvc-restore: %v", err)
	}

	// create pod from restored snapshot
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "create", "-f", filepath.Join(*testdataDir, "csi-hostpath-driver", "pv-pod-restore.yaml")))
	if err != nil {
		t.Logf("creating pod with %s failed: %v", rr.Command(), err)
	}

	if _, err := PodWait(ctx, t, profile, "default", "app=task-pv-pod-restore", Minutes(6)); err != nil {
		t.Fatalf("failed waiting for pod task-pv-pod-restore: %v", err)
	}

	// CLEANUP
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "delete", "pod", "task-pv-pod-restore"))
	if err != nil {
		t.Logf("cleanup with %s failed: %v", rr.Command(), err)
	}
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "delete", "pvc", "hpvc-restore"))
	if err != nil {
		t.Logf("cleanup with %s failed: %v", rr.Command(), err)
	}
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "delete", "volumesnapshot", "new-snapshot-demo"))
	if err != nil {
		t.Logf("cleanup with %s failed: %v", rr.Command(), err)
	}
	rr, err = Run(t, exec.CommandContext(ctx, Target(), "-p", profile, "addons", "disable", "csi-hostpath-driver", "--alsologtostderr", "-v=1"))
	if err != nil {
		t.Errorf("failed to disable csi-hostpath-driver addon: args %q: %v", rr.Command(), err)
	}
	rr, err = Run(t, exec.CommandContext(ctx, Target(), "-p", profile, "addons", "disable", "volumesnapshots", "--alsologtostderr", "-v=1"))
	if err != nil {
		t.Errorf("failed to disable volumesnapshots addon: args %q: %v", rr.Command(), err)
	}
}

// validateGCPAuthAddon tests the GCP Auth addon with either phony or real credentials and makes sure the files are mounted into pods correctly
func validateGCPAuthAddon(ctx context.Context, t *testing.T, profile string) {
	defer PostMortemLogs(t, profile)

	// schedule a pod to check environment variables
	rr, err := Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "create", "-f", filepath.Join(*testdataDir, "busybox.yaml")))
	if err != nil {
		t.Fatalf("%s failed: %v", rr.Command(), err)
	}

	// 8 minutes, because 4 is not enough for images to pull in all cases.
	names, err := PodWait(ctx, t, profile, "default", "integration-test=busybox", Minutes(8))
	if err != nil {
		t.Fatalf("wait: %v", err)
	}

	// Use this pod to confirm that the env vars are set correctly
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "exec", names[0], "--", "/bin/sh", "-c", "printenv GOOGLE_APPLICATION_CREDENTIALS"))
	if err != nil {
		t.Fatalf("printenv creds: %v", err)
	}

	got := strings.TrimSpace(rr.Stdout.String())
	expected := "/google-app-creds.json"
	if got != expected {
		t.Errorf("'printenv GOOGLE_APPLICATION_CREDENTIALS' returned %s, expected %s", got, expected)
	}

	if !detect.IsOnGCE() || detect.IsCloudShell() {
		// Make sure the file contents are correct
		rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "exec", names[0], "--", "/bin/sh", "-c", "cat /google-app-creds.json"))
		if err != nil {
			t.Fatalf("cat creds: %v", err)
		}

		var gotJSON map[string]string
		err = json.Unmarshal(bytes.TrimSpace(rr.Stdout.Bytes()), &gotJSON)
		if err != nil {
			t.Fatalf("unmarshal json: %v", err)
		}
		expectedJSON := map[string]string{
			"client_id":        "haha",
			"client_secret":    "nice_try",
			"quota_project_id": "this_is_fake",
			"refresh_token":    "maybe_next_time",
			"type":             "authorized_user",
		}

		if !reflect.DeepEqual(gotJSON, expectedJSON) {
			t.Fatalf("unexpected creds file: got %v, expected %v", gotJSON, expectedJSON)
		}
	}

	// Check the GOOGLE_CLOUD_PROJECT env var as well
	rr, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "exec", names[0], "--", "/bin/sh", "-c", "printenv GOOGLE_CLOUD_PROJECT"))
	if err != nil {
		t.Fatalf("print env project: %v", err)
	}

	got = strings.TrimSpace(rr.Stdout.String())
	expected = "this_is_fake"
	if detect.IsOnGCE() && !detect.IsCloudShell() {
		expected = "k8s-minikube"
	}
	if got != expected {
		t.Errorf("'printenv GOOGLE_CLOUD_PROJECT' returned %s, expected %s", got, expected)
	}

	// If we're on GCE, we have proper credentials and can test the registry secrets with an artifact registry image
	if detect.IsOnGCE() && !detect.IsCloudShell() {
		_, err = Run(t, exec.CommandContext(ctx, "kubectl", "--context", profile, "apply", "-f", filepath.Join(*testdataDir, "private-image.yaml")))
		if err != nil {
			t.Fatalf("print env project: %v", err)
		}

		// Make sure the pod is up and running, which means we successfully pulled the private image down
		// 8 minutes, because 4 is not enough for images to pull in all cases.
		_, err := PodWait(ctx, t, profile, "default", "integration-test=private-image", Minutes(8))
		if err != nil {
			t.Fatalf("wait for private image: %v", err)
		}
	}

	rr, err = Run(t, exec.CommandContext(ctx, Target(), "-p", profile, "addons", "disable", "gcp-auth", "--alsologtostderr", "-v=1"))
	if err != nil {
		t.Errorf("failed disabling gcp-auth addon. arg %q.s %v", rr.Command(), err)
	}
}
