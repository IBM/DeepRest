/*
Copyright 2021 The Kubernetes Authors All rights reserved.

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

package addons

import (
	"strconv"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"k8s.io/minikube/pkg/minikube/config"
	"k8s.io/minikube/pkg/minikube/machine"
	"k8s.io/minikube/pkg/minikube/storageclass"
)

const defaultStorageClassProvisioner = "standard"

// enableOrDisableStorageClasses enables or disables storage classes
func enableOrDisableStorageClasses(cc *config.ClusterConfig, name string, val string) error {
	klog.Infof("enableOrDisableStorageClasses %s=%v on %q", name, val, cc.Name)
	enable, err := strconv.ParseBool(val)
	if err != nil {
		return errors.Wrap(err, "Error parsing boolean")
	}

	class := defaultStorageClassProvisioner
	if name == "storage-provisioner-gluster" {
		class = "glusterfile"
	}

	api, err := machine.NewAPIClient()
	if err != nil {
		return errors.Wrap(err, "machine client")
	}
	defer api.Close()

	cp, err := config.PrimaryControlPlane(cc)
	if err != nil {
		return errors.Wrap(err, "getting control plane")
	}
	if !machine.IsRunning(api, config.MachineName(*cc, cp)) {
		klog.Warningf("%q is not running, writing %s=%v to disk and skipping enablement", config.MachineName(*cc, cp), name, val)
		return EnableOrDisableAddon(cc, name, val)
	}

	storagev1, err := storageclass.GetStoragev1(cc.Name)
	if err != nil {
		return errors.Wrapf(err, "Error getting storagev1 interface %v ", err)
	}

	if enable {
		// Only StorageClass for 'name' should be marked as default
		err = storageclass.SetDefaultStorageClass(storagev1, class)
		if err != nil {
			return errors.Wrapf(err, "Error making %s the default storage class", class)
		}
	} else {
		// Unset the StorageClass as default
		err := storageclass.DisableDefaultStorageClass(storagev1, class)
		if err != nil {
			return errors.Wrapf(err, "Error disabling %s as the default storage class", class)
		}
	}

	return EnableOrDisableAddon(cc, name, val)
}
