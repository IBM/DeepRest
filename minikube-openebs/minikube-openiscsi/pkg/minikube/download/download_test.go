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

package download

import (
	"fmt"
	"io/fs"
	"sync"
	"testing"
	"time"

	"k8s.io/minikube/pkg/minikube/constants"
)

// Force download tests to run in serial.
func TestDownload(t *testing.T) {
	t.Run("BinaryDownloadPreventsMultipleDownload", testBinaryDownloadPreventsMultipleDownload)
	t.Run("PreloadDownloadPreventsMultipleDownload", testPreloadDownloadPreventsMultipleDownload)
	t.Run("ImageToCache", testImageToCache)
	t.Run("ImageToDaemon", testImageToDaemon)
}

// Returns a mock function that sleeps before incrementing `downloadsCounter` and creates the requested file.
func mockSleepDownload(downloadsCounter *int) func(src, dst string) error {
	return func(src, dst string) error {
		// Sleep for 200ms to assure locking must have occurred.
		time.Sleep(time.Millisecond * 200)
		*downloadsCounter++
		return CreateDstDownloadMock(src, dst)
	}
}

func testBinaryDownloadPreventsMultipleDownload(t *testing.T) {
	downloadNum := 0
	DownloadMock = mockSleepDownload(&downloadNum)

	checkCache = func(file string) (fs.FileInfo, error) {
		if downloadNum == 0 {
			return nil, fmt.Errorf("some error")
		}
		return nil, nil
	}

	var group sync.WaitGroup
	group.Add(2)
	dlCall := func() {
		if _, err := Binary("kubectl", "v1.20.2", "linux", "amd64"); err != nil {
			t.Errorf("Failed to download binary: %+v", err)
		}
		group.Done()
	}

	go dlCall()
	go dlCall()

	group.Wait()

	if downloadNum != 1 {
		t.Errorf("Expected only 1 download attempt but got %v!", downloadNum)
	}
}

func testPreloadDownloadPreventsMultipleDownload(t *testing.T) {
	downloadNum := 0
	DownloadMock = mockSleepDownload(&downloadNum)

	checkCache = func(file string) (fs.FileInfo, error) {
		if downloadNum == 0 {
			return nil, fmt.Errorf("some error")
		}
		return nil, nil
	}
	checkPreloadExists = func(k8sVersion, containerRuntime string, forcePreload ...bool) bool { return true }
	compareChecksum = func(k8sVersion, containerRuntime, path string) error { return nil }

	var group sync.WaitGroup
	group.Add(2)
	dlCall := func() {
		if err := Preload(constants.DefaultKubernetesVersion, constants.DefaultContainerRuntime); err != nil {
			t.Errorf("Failed to download preload: %+v", err)
		}
		group.Done()
	}

	go dlCall()
	go dlCall()

	group.Wait()

	if downloadNum != 1 {
		t.Errorf("Expected only 1 download attempt but got %v!", downloadNum)
	}
}

func testImageToCache(t *testing.T) {
	downloadNum := 0
	DownloadMock = mockSleepDownload(&downloadNum)

	checkImageExistsInCache = func(img string) bool { return downloadNum > 0 }

	var group sync.WaitGroup
	group.Add(2)
	dlCall := func() {
		if err := ImageToCache("testimg"); err != nil {
			t.Errorf("Failed to download preload: %+v", err)
		}
		group.Done()
	}

	go dlCall()
	go dlCall()

	group.Wait()

	if downloadNum != 1 {
		t.Errorf("Expected only 1 download attempt but got %v!", downloadNum)
	}
}

func testImageToDaemon(t *testing.T) {
	downloadNum := 0
	DownloadMock = mockSleepDownload(&downloadNum)

	checkImageExistsInCache = func(img string) bool { return downloadNum > 0 }

	var group sync.WaitGroup
	group.Add(2)
	dlCall := func() {
		if err := ImageToCache("testimg"); err != nil {
			t.Errorf("Failed to download preload: %+v", err)
		}
		group.Done()
	}

	go dlCall()
	go dlCall()

	group.Wait()

	if downloadNum != 1 {
		t.Errorf("Expected only 1 download attempt but got %v!", downloadNum)
	}
}
