/*
Copyright 2020 The Kubernetes Authors All rights reserved.

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

// Package register contains all the logic to print out `minikube start` in JSON
package register

import (
	"fmt"

	"k8s.io/klog/v2"
	"k8s.io/minikube/pkg/trace"
)

// If you add a new step here, please also add it to register.Reg registry inside the init() function
const (
	InitialSetup                      RegStep = "Initial Minikube Setup"
	SelectingDriver                   RegStep = "Selecting Driver"
	DownloadingArtifacts              RegStep = "Downloading Artifacts"
	StartingNode                      RegStep = "Starting Node"
	UpdatingDriver                    RegStep = "Updating Driver"
	PullingBaseImage                  RegStep = "Pulling Base Image"
	RunningLocalhost                  RegStep = "Running on Localhost"
	RunningRemotely                   RegStep = "Running Remotely"
	LocalOSRelease                    RegStep = "Local OS Release"
	CreatingContainer                 RegStep = "Creating Container"
	CreatingVM                        RegStep = "Creating VM"
	ConfiguringLHEnv                  RegStep = "Configuring Localhost Environment"
	PreparingKubernetes               RegStep = "Preparing Kubernetes"
	PreparingKubernetesCerts          RegStep = "Generating certificates"
	PreparingKubernetesControlPlane   RegStep = "Booting control plane"
	PreparingKubernetesBootstrapToken RegStep = "Configuring RBAC rules"
	ConfiguringCNI                    RegStep = "Configuring CNI"
	VerifyingKubernetes               RegStep = "Verifying Kubernetes"
	EnablingAddons                    RegStep = "Enabling Addons"
	Done                              RegStep = "Done"

	Stopping  RegStep = "Stopping"
	PowerOff  RegStep = "PowerOff"
	Deleting  RegStep = "Deleting"
	Pausing   RegStep = "Pausing"
	Unpausing RegStep = "Unpausing"
)

// RegStep is a type representing a distinct step of `minikube start`
type RegStep string

// Register holds all of the steps we could see in `minikube start`
// and keeps track of the current step
type Register struct {
	steps   map[RegStep][]RegStep
	first   RegStep
	current RegStep
}

// Reg keeps track of all possible steps and the current step we are on
var Reg Register

func init() {
	Reg = Register{
		// Expected step orders, organized by the initial step seen
		steps: map[RegStep][]RegStep{
			InitialSetup: {
				InitialSetup,
				SelectingDriver,
				DownloadingArtifacts,
				StartingNode,
				UpdatingDriver,
				PullingBaseImage,
				RunningLocalhost,
				LocalOSRelease,
				CreatingContainer,
				CreatingVM,
				RunningRemotely,
				PreparingKubernetes,
				PreparingKubernetesCerts,
				PreparingKubernetesControlPlane,
				PreparingKubernetesBootstrapToken,
				ConfiguringCNI,
				ConfiguringLHEnv,
				VerifyingKubernetes,
				EnablingAddons,
				Done,
			},

			Stopping:  {Stopping, PowerOff, Done},
			Pausing:   {Pausing, Done},
			Unpausing: {Unpausing, Done},
			Deleting:  {Deleting, Stopping, Deleting, Done},
		},
	}
}

// totalSteps returns the total number of steps in the register
func (r *Register) totalSteps() string {
	return fmt.Sprintf("%d", len(r.steps[r.first])-1)
}

// currentStep returns the current step we are on
func (r *Register) currentStep() string {
	if r.first == RegStep("") {
		return ""
	}

	steps, ok := r.steps[r.first]
	if !ok {
		return "unknown"
	}

	for i, s := range r.steps[r.first] {
		if r.current == s {
			return fmt.Sprintf("%d", i)
		}
	}

	// Warn, as sometimes detours happen: "start" may cause "stopping" and "deleting"
	klog.Warningf("%q was not found within the registered steps for %q: %v", r.current, r.first, steps)
	return ""
}

// SetStep sets the current step
func (r *Register) SetStep(s RegStep) {
	defer trace.StartSpan(string(s))
	if r.first == RegStep("") {
		_, ok := r.steps[s]
		if ok {
			r.first = s
		} else {
			klog.Errorf("unexpected first step: %q", r.first)
		}
	} else {
		trace.EndSpan(string(r.current))
	}

	r.current = s
}
