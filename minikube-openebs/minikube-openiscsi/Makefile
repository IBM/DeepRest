# Copyright 2016 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Bump these on release - and please check ISO_VERSION for correctness.
VERSION_MAJOR ?= 1
VERSION_MINOR ?= 20
VERSION_BUILD ?= 0
RAW_VERSION=$(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_BUILD)
VERSION ?= v$(RAW_VERSION)

KUBERNETES_VERSION ?= $(shell egrep "DefaultKubernetesVersion =" pkg/minikube/constants/constants.go | cut -d \" -f2)
KIC_VERSION ?= $(shell egrep "Version =" pkg/drivers/kic/types.go | cut -d \" -f2)

# Default to .0 for higher cache hit rates, as build increments typically don't require new ISO versions
ISO_VERSION ?= v1.20.0
# Dashes are valid in semver, but not Linux packaging. Use ~ to delimit alpha/beta
DEB_VERSION ?= $(subst -,~,$(RAW_VERSION))
DEB_REVISION ?= 0

RPM_VERSION ?= $(DEB_VERSION)
RPM_REVISION ?= 0

# used by hack/jenkins/release_build_and_upload.sh and KVM_BUILD_IMAGE, see also BUILD_IMAGE below
GO_VERSION ?= 1.16.4

# replace "x.y.0" => "x.y". kube-cross and golang.org/dl use different formats for x.y.0 go versions
KVM_GO_VERSION ?= $(GO_VERSION:.0=)


INSTALL_SIZE ?= $(shell du out/minikube-windows-amd64.exe | cut -f1)
BUILDROOT_BRANCH ?= 2020.02.12
REGISTRY?=gcr.io/k8s-minikube

# Get git commit id
COMMIT_NO := $(shell git rev-parse HEAD 2> /dev/null || true)
COMMIT ?= $(if $(shell git status --porcelain --untracked-files=no),"${COMMIT_NO}-dirty","${COMMIT_NO}")
COMMIT_SHORT = $(shell git rev-parse --short HEAD 2> /dev/null || true)
HYPERKIT_BUILD_IMAGE 	?= neilotoole/xcgo:go1.15

# NOTE: "latest" as of 2021-02-06. kube-cross images aren't updated as often as Kubernetes
# https://github.com/kubernetes/kubernetes/blob/master/build/build-image/cross/VERSION
#
BUILD_IMAGE 	?= us.gcr.io/k8s-artifacts-prod/build-image/kube-cross:v$(GO_VERSION)-1

ISO_BUILD_IMAGE ?= $(REGISTRY)/buildroot-image
KVM_BUILD_IMAGE ?= $(REGISTRY)/kvm-build-image:$(KVM_GO_VERSION)

ISO_BUCKET ?= minikube/iso

MINIKUBE_VERSION ?= $(ISO_VERSION)
MINIKUBE_BUCKET ?= minikube/releases
MINIKUBE_UPLOAD_LOCATION := gs://${MINIKUBE_BUCKET}
MINIKUBE_RELEASES_URL=https://github.com/kubernetes/minikube/releases/download

KERNEL_VERSION ?= 4.19.182
# latest from https://github.com/golangci/golangci-lint/releases
GOLINT_VERSION ?= v1.39.0
# Limit number of default jobs, to avoid the CI builds running out of memory
GOLINT_JOBS ?= 4
# see https://github.com/golangci/golangci-lint#memory-usage-of-golangci-lint
GOLINT_GOGC ?= 100
# options for lint (golangci-lint)
GOLINT_OPTIONS = --timeout 7m \
	  --build-tags "${MINIKUBE_INTEGRATION_BUILD_TAGS}" \
	  --enable gofmt,goimports,gocritic,golint,gocyclo,misspell,nakedret,stylecheck,unconvert,unparam,dogsled \
	  --exclude 'variable on range scope.*in function literal|ifElseChain' \
	  --skip-files "pkg/minikube/translate/translations.go|pkg/minikube/assets/assets.go"

export GO111MODULE := on

GOOS ?= $(shell go env GOOS)
GOARCH ?= $(shell go env GOARCH)
GOARM ?= 7 # the default is 5
GOPATH ?= $(shell go env GOPATH)
BUILD_DIR ?= ./out
$(shell mkdir -p $(BUILD_DIR))
CURRENT_GIT_BRANCH ?= $(shell git branch | grep \* | cut -d ' ' -f2)

# Use system python if it exists, otherwise use Docker.
PYTHON := $(shell command -v python || echo "docker run --rm -it -v $(shell pwd):/minikube -w /minikube python python")
BUILD_OS := $(shell uname -s)

SHA512SUM=$(shell command -v sha512sum || echo "shasum -a 512")

# gvisor tag to automatically push changes to
# to update minikubes default, update deploy/addons/gvisor
GVISOR_TAG ?= latest

# auto-pause-hook tag to push changes to
AUTOPAUSE_HOOK_TAG ?= v0.0.2

# prow-test tag to push changes to
PROW_TEST_TAG ?= v0.0.1

# storage provisioner tag to push changes to
# NOTE: you will need to bump the PreloadVersion if you change this
STORAGE_PROVISIONER_TAG ?= v5

STORAGE_PROVISIONER_MANIFEST ?= $(REGISTRY)/storage-provisioner:$(STORAGE_PROVISIONER_TAG)
STORAGE_PROVISIONER_IMAGE ?= $(REGISTRY)/storage-provisioner-$(GOARCH):$(STORAGE_PROVISIONER_TAG)

# Set the version information for the Kubernetes servers
MINIKUBE_LDFLAGS := -X k8s.io/minikube/pkg/version.version=$(VERSION) -X k8s.io/minikube/pkg/version.isoVersion=$(ISO_VERSION) -X k8s.io/minikube/pkg/version.gitCommitID=$(COMMIT) -X k8s.io/minikube/pkg/version.storageProvisionerVersion=$(STORAGE_PROVISIONER_TAG)
PROVISIONER_LDFLAGS := "-X k8s.io/minikube/pkg/storage.version=$(STORAGE_PROVISIONER_TAG) -s -w -extldflags '-static'"

MINIKUBEFILES := ./cmd/minikube/
HYPERKIT_FILES := ./cmd/drivers/hyperkit
STORAGE_PROVISIONER_FILES := ./cmd/storage-provisioner
KVM_DRIVER_FILES := ./cmd/drivers/kvm/

MINIKUBE_TEST_FILES := ./cmd/... ./pkg/...

# npm install -g markdownlint-cli
MARKDOWNLINT ?= markdownlint


MINIKUBE_MARKDOWN_FILES := README.md CONTRIBUTING.md CHANGELOG.md

MINIKUBE_BUILD_TAGS :=
MINIKUBE_INTEGRATION_BUILD_TAGS := integration $(MINIKUBE_BUILD_TAGS)

CMD_SOURCE_DIRS = cmd pkg
SOURCE_DIRS = $(CMD_SOURCE_DIRS) test
SOURCE_PACKAGES = ./cmd/... ./pkg/... ./test/...

SOURCE_GENERATED = pkg/minikube/assets/assets.go pkg/minikube/translate/translations.go
SOURCE_FILES = $(shell find $(CMD_SOURCE_DIRS) -type f -name "*.go" | grep -v _test.go)
GOTEST_FILES = $(shell find $(CMD_SOURCE_DIRS) -type f -name "*.go" | grep _test.go)

# kvm2 ldflags
KVM2_LDFLAGS := -X k8s.io/minikube/pkg/drivers/kvm.version=$(VERSION) -X k8s.io/minikube/pkg/drivers/kvm.gitCommitID=$(COMMIT)

# hyperkit ldflags
HYPERKIT_LDFLAGS := -X k8s.io/minikube/pkg/drivers/hyperkit.version=$(VERSION) -X k8s.io/minikube/pkg/drivers/hyperkit.gitCommitID=$(COMMIT)

# autopush artefacts
AUTOPUSH ?=

# don't ask for user confirmation
IN_CI := false

# $(call user_confirm, message)
define user_confirm
	@if [ "${IN_CI}" = "false" ]; then\
		echo "⚠️ $(1)";\
		read -p "Do you want to proceed? (Y/N): " confirm && echo $$confirm | grep -iq "^[yY]" || exit 1;\
	fi
endef

# $(call DOCKER, image, command)
define DOCKER
	docker run --rm -e GOCACHE=/app/.cache -e IN_DOCKER=1 --user $(shell id -u):$(shell id -g) -w /app -v $(PWD):/app -v $(GOPATH):/go --init $(1) /bin/bash -c '$(2)'
endef

ifeq ($(BUILD_IN_DOCKER),y)
	MINIKUBE_BUILD_IN_DOCKER=y
endif

# If we are already running in docker,
# prevent recursion by unsetting the BUILD_IN_DOCKER directives.
# The _BUILD_IN_DOCKER variables should not be modified after this conditional.
ifeq ($(IN_DOCKER),1)
	MINIKUBE_BUILD_IN_DOCKER=n
endif

ifeq ($(GOOS),windows)
	IS_EXE = .exe
	DIRSEP_ = \\
	DIRSEP = $(strip $(DIRSEP_))
	PATHSEP = ;
else
	DIRSEP = /
	PATHSEP = :
endif

v_at_0 = yes
v_at_ = $(v_at_1)
quiet := $(v_at_$(V))
Q=$(if $(quiet),@)

INTEGRATION_TESTS_TO_RUN := ./test/integration
ifneq ($(TEST_FILES),)
	TEST_HELPERS = main_test.go util_test.go helpers_test.go
	INTEGRATION_TESTS_TO_RUN := $(addprefix ./test/integration/, $(TEST_HELPERS) $(TEST_FILES))
endif

out/minikube$(IS_EXE): $(SOURCE_GENERATED) $(SOURCE_FILES) go.mod
ifeq ($(MINIKUBE_BUILD_IN_DOCKER),y)
	$(call DOCKER,$(BUILD_IMAGE),GOOS=$(GOOS) GOARCH=$(GOARCH) GOARM=$(GOARM) /usr/bin/make $@)
else
	$(if $(quiet),@echo "  GO       $@")
	$(Q)go build $(MINIKUBE_GOFLAGS) -tags "$(MINIKUBE_BUILD_TAGS)" -ldflags="$(MINIKUBE_LDFLAGS)" -o $@ k8s.io/minikube/cmd/minikube
endif

out/minikube-windows-amd64.exe: out/minikube-windows-amd64
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

out/minikube-linux-i686: out/minikube-linux-386
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

out/minikube-linux-x86_64: out/minikube-linux-amd64
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

out/minikube-linux-armhf: out/minikube-linux-arm
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

out/minikube-linux-armv7hl: out/minikube-linux-arm
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

out/minikube-linux-aarch64: out/minikube-linux-arm64
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

out/minikube-linux-ppc64el: out/minikube-linux-ppc64le
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

.PHONY: minikube-linux-amd64 minikube-linux-arm64
minikube-linux-amd64: out/minikube-linux-amd64 ## Build Minikube for Linux x86 64bit
minikube-linux-arm64: out/minikube-linux-arm64 ## Build Minikube for Linux ARM 64bit

.PHONY: minikube-darwin-amd64 minikube-darwin-arm64
minikube-darwin-amd64: out/minikube-darwin-amd64 ## Build Minikube for Darwin x86 64bit
minikube-darwin-arm64: out/minikube-darwin-arm64 ## Build Minikube for Darwin ARM 64bit

.PHONY: minikube-windows-amd64.exe
minikube-windows-amd64.exe: out/minikube-windows-amd64.exe ## Build Minikube for Windows 64bit

eq = $(and $(findstring x$(1),x$(2)),$(findstring x$(2),x$(1)))

out/minikube-%: $(SOURCE_GENERATED) $(SOURCE_FILES)
ifeq ($(MINIKUBE_BUILD_IN_DOCKER),y)
	$(call DOCKER,$(BUILD_IMAGE),/usr/bin/make $@)
else
	$(if $(quiet),@echo "  GO       $@")
	$(Q)GOOS="$(firstword $(subst -, ,$*))" GOARCH="$(lastword $(subst -, ,$(subst $(IS_EXE), ,$*)))" $(if $(call eq,$(lastword $(subst -, ,$(subst $(IS_EXE), ,$*))),arm),GOARM=$(GOARM)) \
	go build -tags "$(MINIKUBE_BUILD_TAGS)" -ldflags="$(MINIKUBE_LDFLAGS)" -a -o $@ k8s.io/minikube/cmd/minikube
endif

out/minikube-linux-armv6: $(SOURCE_GENERATED) $(SOURCE_FILES)
	$(Q)GOOS=linux GOARCH=arm GOARM=6 \
	go build -tags "$(MINIKUBE_BUILD_TAGS)" -ldflags="$(MINIKUBE_LDFLAGS)" -a -o $@ k8s.io/minikube/cmd/minikube

.PHONY: e2e-linux-amd64 e2e-linux-arm64 e2e-darwin-amd64 e2e-windows-amd64.exe
e2e-linux-amd64: out/e2e-linux-amd64 ## build end2end binary for Linux x86 64bit
e2e-linux-arm64: out/e2e-linux-arm64 ## build end2end binary for Linux ARM 64bit
e2e-darwin-amd64: out/e2e-darwin-amd64 ## build end2end binary for Darwin x86 64bit
e2e-darwin-arm64: out/e2e-darwin-arm64 ## build end2end binary for Darwin ARM 64bit
e2e-windows-amd64.exe: out/e2e-windows-amd64.exe ## build end2end binary for Windows 64bit

out/e2e-%: out/minikube-%
	GOOS="$(firstword $(subst -, ,$*))" GOARCH="$(lastword $(subst -, ,$(subst $(IS_EXE), ,$*)))" go test -ldflags="${MINIKUBE_LDFLAGS}" -c k8s.io/minikube/test/integration --tags="$(MINIKUBE_INTEGRATION_BUILD_TAGS)" -o $@

out/e2e-windows-amd64.exe: out/e2e-windows-amd64
	cp $< $@

minikube_iso: deploy/iso/minikube-iso/board/coreos/minikube/rootfs-overlay/usr/bin/auto-pause # build minikube iso
	echo $(ISO_VERSION) > deploy/iso/minikube-iso/board/coreos/minikube/rootfs-overlay/etc/VERSION
	if [ ! -d $(BUILD_DIR)/buildroot ]; then \
		mkdir -p $(BUILD_DIR); \
		git clone --depth=1 --branch=$(BUILDROOT_BRANCH) https://github.com/buildroot/buildroot $(BUILD_DIR)/buildroot; \
	fi;
	$(MAKE) BR2_EXTERNAL=../../deploy/iso/minikube-iso minikube_defconfig -C $(BUILD_DIR)/buildroot
	mkdir -p $(BUILD_DIR)/buildroot/output/build
	echo "module buildroot.org/go" > $(BUILD_DIR)/buildroot/output/build/go.mod
	$(MAKE) -C $(BUILD_DIR)/buildroot host-python
	$(MAKE) -C $(BUILD_DIR)/buildroot
	mv $(BUILD_DIR)/buildroot/output/images/rootfs.iso9660 $(BUILD_DIR)/minikube.iso

# Change buildroot configuration for the minikube ISO
.PHONY: iso-menuconfig
iso-menuconfig: ## Configure buildroot configuration
	$(MAKE) -C $(BUILD_DIR)/buildroot menuconfig
	$(MAKE) -C $(BUILD_DIR)/buildroot savedefconfig

# Change the kernel configuration for the minikube ISO
.PHONY: linux-menuconfig
linux-menuconfig:  ## Configure Linux kernel configuration
	$(MAKE) -C $(BUILD_DIR)/buildroot/output/build/linux-$(KERNEL_VERSION)/ menuconfig
	$(MAKE) -C $(BUILD_DIR)/buildroot/output/build/linux-$(KERNEL_VERSION)/ savedefconfig
	cp $(BUILD_DIR)/buildroot/output/build/linux-$(KERNEL_VERSION)/defconfig deploy/iso/minikube-iso/board/coreos/minikube/linux_defconfig

out/minikube.iso: $(shell find "deploy/iso/minikube-iso" -type f)
ifeq ($(IN_DOCKER),1)
	$(MAKE) minikube_iso
else
	docker run --rm --workdir /mnt --volume $(CURDIR):/mnt $(ISO_DOCKER_EXTRA_ARGS) \
		--user $(shell id -u):$(shell id -g) --env HOME=/tmp --env IN_DOCKER=1 \
		$(ISO_BUILD_IMAGE) /usr/bin/make out/minikube.iso
endif

iso_in_docker:
	docker run -it --rm --workdir /mnt --volume $(CURDIR):/mnt $(ISO_DOCKER_EXTRA_ARGS) \
		--user $(shell id -u):$(shell id -g) --env HOME=/tmp --env IN_DOCKER=1 \
		$(ISO_BUILD_IMAGE) /bin/bash

test-iso: $(SOURCE_GENERATED)
	go test -v $(INTEGRATION_TESTS_TO_RUN) --tags=iso --minikube-start-args="--iso-url=file://$(shell pwd)/out/buildroot/output/images/rootfs.iso9660"

.PHONY: test-pkg
test-pkg/%: $(SOURCE_GENERATED) ## Trigger packaging test
	go test -v -test.timeout=60m ./$* --tags="$(MINIKUBE_BUILD_TAGS)"

.PHONY: all
all: cross drivers e2e-cross cross-tars exotic retro out/gvisor-addon ## Build all different minikube components

.PHONY: drivers
drivers: docker-machine-driver-hyperkit docker-machine-driver-kvm2 ## Build Hyperkit and KVM2 drivers

.PHONY: docker-machine-driver-hyperkit
docker-machine-driver-hyperkit: out/docker-machine-driver-hyperkit ## Build Hyperkit driver

.PHONY: docker-machine-driver-kvm2
docker-machine-driver-kvm2: out/docker-machine-driver-kvm2 ## Build KVM2 driver

.PHONY: integration
integration: out/minikube$(IS_EXE) ## Trigger minikube integration test, logs to ./out/testout_COMMIT.txt
	go test -ldflags="${MINIKUBE_LDFLAGS}" -v -test.timeout=90m $(INTEGRATION_TESTS_TO_RUN) --tags="$(MINIKUBE_INTEGRATION_BUILD_TAGS)" $(TEST_ARGS) 2>&1 | tee "./out/testout_$(COMMIT_SHORT).txt"

.PHONY: integration-none-driver
integration-none-driver: e2e-linux-$(GOARCH) out/minikube-linux-$(GOARCH)  ## Trigger minikube none driver test, logs to ./out/testout_COMMIT.txt
	sudo -E out/e2e-linux-$(GOARCH) -testdata-dir "test/integration/testdata" -minikube-start-args="--driver=none" -test.v -test.timeout=60m -binary=out/minikube-linux-amd64 $(TEST_ARGS) 2>&1 | tee "./out/testout_$(COMMIT_SHORT).txt"

.PHONY: integration-versioned
integration-versioned: out/minikube ## Trigger minikube integration testing, logs to ./out/testout_COMMIT.txt
	go test -ldflags="${MINIKUBE_LDFLAGS}" -v -test.timeout=90m $(INTEGRATION_TESTS_TO_RUN) --tags="$(MINIKUBE_INTEGRATION_BUILD_TAGS) versioned" $(TEST_ARGS) 2>&1 | tee "./out/testout_$(COMMIT_SHORT).txt"

.PHONY: functional
functional: integration-functional-only

.PHONY: integration-functional-only
integration-functional-only: out/minikube$(IS_EXE) ## Trigger only functioanl tests in integration test, logs to ./out/testout_COMMIT.txt
	go test -ldflags="${MINIKUBE_LDFLAGS}" -v -test.timeout=20m $(INTEGRATION_TESTS_TO_RUN) --tags="$(MINIKUBE_INTEGRATION_BUILD_TAGS)" $(TEST_ARGS) -test.run TestFunctional 2>&1 | tee "./out/testout_$(COMMIT_SHORT).txt"

.PHONY: html_report
html_report: ## Generate HTML  report out of the last ran integration test logs.
	@go tool test2json -t < "./out/testout_$(COMMIT_SHORT).txt" > "./out/testout_$(COMMIT_SHORT).json"
	@gopogh -in "./out/testout_$(COMMIT_SHORT).json" -out ./out/testout_$(COMMIT_SHORT).html -name "$(shell git rev-parse --abbrev-ref HEAD)" -pr "" -repo github.com/kubernetes/minikube/  -details "${COMMIT_SHORT}"
	@echo "-------------------------- Open HTML Report in Browser: ---------------------------"
ifeq ($(GOOS),windows)
	@echo start $(CURDIR)/out/testout_$(COMMIT_SHORT).html
	@echo "-----------------------------------------------------------------------------------"
	@start $(CURDIR)/out/testout_$(COMMIT_SHORT).html || true
else
	@echo open $(CURDIR)/out/testout_$(COMMIT_SHORT).html
	@echo "-----------------------------------------------------------------------------------"
	@open $(CURDIR)/out/testout_$(COMMIT_SHORT).html || true
endif

.PHONY: test
test: $(SOURCE_GENERATED) ## Trigger minikube test
	MINIKUBE_LDFLAGS="${MINIKUBE_LDFLAGS}" ./test.sh

.PHONY: generate-docs
generate-docs: extract out/minikube ## Automatically generate commands documentation.
	out/minikube generate-docs --path ./site/content/en/docs/commands/ --test-path ./site/content/en/docs/contrib/tests.en.md

.PHONY: gotest
gotest: $(SOURCE_GENERATED) ## Trigger minikube test
	$(if $(quiet),@echo "  TEST     $@")
	$(Q)go test -tags "$(MINIKUBE_BUILD_TAGS)" -ldflags="$(MINIKUBE_LDFLAGS)" $(MINIKUBE_TEST_FILES)

# Run the gotest, while recording JSON report and coverage
out/unittest.json: $(SOURCE_FILES) $(GOTEST_FILES)
	$(if $(quiet),@echo "  TEST     $@")
	$(Q)go test -tags "$(MINIKUBE_BUILD_TAGS)" -ldflags="$(MINIKUBE_LDFLAGS)" $(MINIKUBE_TEST_FILES) \
	-coverprofile=out/coverage.out -json > out/unittest.json
out/coverage.out: out/unittest.json

# Generate go test report (from gotest) as a a HTML page
out/unittest.html: out/unittest.json
	$(if $(quiet),@echo "  REPORT   $@")
	$(Q)go-test-report < $< -o $@

# Generate go coverage report (from gotest) as a HTML page
out/coverage.html: out/coverage.out
	$(if $(quiet),@echo "  COVER    $@")
	$(Q)go tool cover -html=$< -o $@

.PHONY: extract 
extract: ## extract internationalization words for translations
	go run cmd/extract/extract.go

# Regenerates assets.go when template files have been updated
pkg/minikube/assets/assets.go: $(shell find "deploy/addons" -type f)
ifeq ($(MINIKUBE_BUILD_IN_DOCKER),y)
	$(call DOCKER,$(BUILD_IMAGE),/usr/bin/make $@)
endif
	@which go-bindata >/dev/null 2>&1 || GO111MODULE=off GOBIN="$(GOPATH)$(DIRSEP)bin" go get github.com/go-bindata/go-bindata/...
	$(if $(quiet),@echo "  GEN      $@")
	$(Q)PATH="$(PATH)$(PATHSEP)$(GOPATH)$(DIRSEP)bin" go-bindata -nomemcopy -o $@ -pkg assets deploy/addons/...
	$(Q)-gofmt -s -w $@
	@#golint: Dns should be DNS (compat sed)
	@sed -i -e 's/Dns/DNS/g' $@ && rm -f ./-e
	@#golint: Html should be HTML (compat sed)
	@sed -i -e 's/Html/HTML/g' $@ && rm -f ./-e
	@#golint: don't use underscores in Go names
	@sed -i -e 's/SnapshotStorageK8sIo_volumesnapshot/SnapshotStorageK8sIoVolumesnapshot/g' $@ && rm -f ./-e

pkg/minikube/translate/translations.go: $(shell find "translations/" -type f)
ifeq ($(MINIKUBE_BUILD_IN_DOCKER),y)
	$(call DOCKER,$(BUILD_IMAGE),/usr/bin/make $@)
endif
	@which go-bindata >/dev/null 2>&1 || GO111MODULE=off GOBIN="$(GOPATH)$(DIRSEP)bin" go get github.com/go-bindata/go-bindata/...
	$(if $(quiet),@echo "  GEN      $@")
	$(Q)PATH="$(PATH)$(PATHSEP)$(GOPATH)$(DIRSEP)bin" go-bindata -nomemcopy -o $@ -pkg translate translations/...
	$(Q)-gofmt -s -w $@
	@#golint: Json should be JSON (compat sed)
	@sed -i -e 's/Json/JSON/' $@ && rm -f ./-e

.PHONY: cross
cross: minikube-linux-amd64 minikube-darwin-amd64 minikube-windows-amd64.exe ## Build minikube for all platform

.PHONY: exotic
exotic: out/minikube-linux-arm out/minikube-linux-arm64 out/minikube-linux-ppc64le out/minikube-linux-s390x ## Build minikube for non-amd64 linux

.PHONY: retro
retro: out/minikube-linux-386 out/minikube-linux-armv6 ## Build minikube for legacy 32-bit linux

.PHONY: windows
windows: minikube-windows-amd64.exe ## Build minikube for Windows 64bit

.PHONY: darwin
darwin: minikube-darwin-amd64 ## Build minikube for Darwin 64bit

.PHONY: linux
linux: minikube-linux-amd64 ## Build minikube for Linux 64bit

.PHONY: e2e-cross
e2e-cross: e2e-linux-amd64 e2e-linux-arm64 e2e-darwin-amd64 e2e-windows-amd64.exe ## End-to-end cross test

.PHONY: checksum
checksum: ## Generate checksums
	for f in out/minikube.iso out/minikube-linux-amd64 out/minikube-linux-arm \
		 out/minikube-linux-arm64 out/minikube-linux-ppc64le out/minikube-linux-s390x \
		 out/minikube-darwin-amd64 out/minikube-windows-amd64.exe \
		 out/docker-machine-driver-kvm2 out/docker-machine-driver-hyperkit; do \
		if [ -f "$${f}" ]; then \
			openssl sha256 "$${f}" | awk '{print $$2}' > "$${f}.sha256" ; \
		fi ; \
	done

.PHONY: clean
clean: ## Clean build
	rm -rf $(BUILD_DIR)
	rm -f pkg/minikube/assets/assets.go
	rm -f pkg/minikube/translate/translations.go
	rm -rf ./vendor
	rm -rf /tmp/tmp.*.minikube_*

.PHONY: gendocs
gendocs: out/docs/minikube.md  ## Generate documentation

.PHONY: fmt
fmt: ## Run go fmt and modify files in place
	@gofmt -s -w $(SOURCE_DIRS)

.PHONY: gofmt
gofmt: ## Run go fmt and list the files differs from gofmt's
	@gofmt -s -l $(SOURCE_DIRS)
	@test -z "`gofmt -s -l $(SOURCE_DIRS)`"

.PHONY: vet
vet: ## Run go vet
	@go vet $(SOURCE_PACKAGES)

.PHONY: imports
imports: ## Run goimports and modify files in place
	@goimports -w $(SOURCE_DIRS)

.PHONY: goimports
goimports: ## Run goimports and list the files differs from goimport's
	@goimports -l $(SOURCE_DIRS)
	@test -z "`goimports -l $(SOURCE_DIRS)`"

.PHONY: golint
golint: $(SOURCE_GENERATED) ## Run golint
	@golint -set_exit_status $(SOURCE_PACKAGES)

.PHONY: gocyclo
gocyclo: ## Run gocyclo (calculates cyclomatic complexities)
	@gocyclo -over 15 `find $(SOURCE_DIRS) -type f -name "*.go"`

out/linters/golangci-lint-$(GOLINT_VERSION):
	mkdir -p out/linters
	curl -sfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b out/linters $(GOLINT_VERSION)
	mv out/linters/golangci-lint out/linters/golangci-lint-$(GOLINT_VERSION)

# this one is meant for local use
.PHONY: lint
ifeq ($(MINIKUBE_BUILD_IN_DOCKER),y)
lint: $(SOURCE_GENERATED)
	docker run --rm -v $(pwd):/app -w /app golangci/golangci-lint:$(GOLINT_VERSION) \
	golangci-lint run ${GOLINT_OPTIONS} --skip-dirs "cmd/drivers/kvm|cmd/drivers/hyperkit|pkg/drivers/kvm|pkg/drivers/hyperkit" ./...
else
lint: $(SOURCE_GENERATED) out/linters/golangci-lint-$(GOLINT_VERSION) ## Run lint
	./out/linters/golangci-lint-$(GOLINT_VERSION) run ${GOLINT_OPTIONS} ./...
endif

# lint-ci is slower version of lint and is meant to be used in ci (travis) to avoid out of memory leaks.
.PHONY: lint-ci
lint-ci: $(SOURCE_GENERATED) out/linters/golangci-lint-$(GOLINT_VERSION) ## Run lint-ci
	GOGC=${GOLINT_GOGC} ./out/linters/golangci-lint-$(GOLINT_VERSION) run \
	--concurrency ${GOLINT_JOBS} ${GOLINT_OPTIONS} ./...

.PHONY: reportcard
reportcard: ## Run goreportcard for minikube
	goreportcard-cli -v
	# "disabling misspell on large repo..."
	-misspell -error $(SOURCE_DIRS)

.PHONY: mdlint
mdlint:
	@$(MARKDOWNLINT) $(MINIKUBE_MARKDOWN_FILES)

.PHONY: verify-iso
verify-iso: # Make sure the current ISO exists in the expected bucket
	gsutil stat gs://$(ISO_BUCKET)/minikube-$(ISO_VERSION).iso

out/docs/minikube.md: $(shell find "cmd") $(shell find "pkg/minikube/constants") $(SOURCE_GENERATED)
	go run -ldflags="$(MINIKUBE_LDFLAGS)" -tags gendocs hack/help_text/gen_help_text.go


.PHONY: debs ## Build all deb packages
debs: out/minikube_$(DEB_VERSION)-$(DEB_REVISION)_amd64.deb \
	  out/minikube_$(DEB_VERSION)-$(DEB_REVISION)_arm64.deb \
	  out/docker-machine-driver-kvm2_$(DEB_VERSION).deb


.PHONY: deb_version
deb_version:
	@echo $(DEB_VERSION)-$(DEB_REVISION)

.PHONY: deb_version_base
deb_version_base:
	@echo $(DEB_VERSION)

out/minikube_$(DEB_VERSION).deb: out/minikube_$(DEB_VERSION)-$(DEB_REVISION)_amd64.deb
	cp $< $@

out/minikube_$(DEB_VERSION)-$(DEB_REVISION)_%.deb: out/minikube-linux-%
	$(eval DEB_PACKAGING_DIRECTORY_$*=$(shell mktemp -d --suffix ".minikube_$(DEB_VERSION)-$*-deb"))
	cp -r installers/linux/deb/minikube_deb_template/* $(DEB_PACKAGING_DIRECTORY_$*)/
	chmod 0755 $(DEB_PACKAGING_DIRECTORY_$*)/DEBIAN
	sed -E -i 's/--VERSION--/'$(DEB_VERSION)'/g' $(DEB_PACKAGING_DIRECTORY_$*)/DEBIAN/control
	sed -E -i 's/--REVISION--/'$(DEB_REVISION)'/g' $(DEB_PACKAGING_DIRECTORY_$*)/DEBIAN/control
	sed -E -i 's/--ARCH--/'$*'/g' $(DEB_PACKAGING_DIRECTORY_$*)/DEBIAN/control
  
	if [ "$*" = "amd64" ]; then \
	    sed -E -i 's/--RECOMMENDS--/virtualbox/' $(DEB_PACKAGING_DIRECTORY_$*)/DEBIAN/control; \
	else \
	    sed -E -i '/Recommends: --RECOMMENDS--/d' $(DEB_PACKAGING_DIRECTORY_$*)/DEBIAN/control; \
	fi
  
	mkdir -p $(DEB_PACKAGING_DIRECTORY_$*)/usr/bin
	cp $< $(DEB_PACKAGING_DIRECTORY_$*)/usr/bin/minikube
	fakeroot dpkg-deb --build $(DEB_PACKAGING_DIRECTORY_$*) $@
	rm -rf $(DEB_PACKAGING_DIRECTORY_$*)

rpm_version:
	@echo $(RPM_VERSION)-$(RPM_REVISION)

out/minikube-$(RPM_VERSION).rpm: out/minikube-$(RPM_VERSION)-$(RPM_REVISION).x86_64.rpm
	cp $< $@

out/minikube-$(RPM_VERSION)-0.%.rpm: out/minikube-linux-%
	$(eval RPM_PACKAGING_DIRECTORY_$*=$(shell mktemp -d --suffix ".minikube_$(RPM_VERSION)-$*-rpm"))
	cp -r installers/linux/rpm/minikube_rpm_template/* $(RPM_PACKAGING_DIRECTORY_$*)/
	sed -E -i 's/--VERSION--/'$(RPM_VERSION)'/g' $(RPM_PACKAGING_DIRECTORY_$*)/minikube.spec
	sed -E -i 's/--REVISION--/'$(RPM_REVISION)'/g' $(RPM_PACKAGING_DIRECTORY_$*)/minikube.spec
	sed -E -i 's|--OUT--|'$(PWD)/out'|g' $(RPM_PACKAGING_DIRECTORY_$*)/minikube.spec
	rpmbuild -bb -D "_rpmdir $(PWD)/out" --target $* \
		 $(RPM_PACKAGING_DIRECTORY_$*)/minikube.spec
	@mv out/$*/minikube-$(RPM_VERSION)-$(RPM_REVISION).$*.rpm out/ && rmdir out/$*
	rm -rf $(RPM_PACKAGING_DIRECTORY_$*)

.PHONY: apt
apt: out/Release ## Generate apt package file

out/Release: out/minikube_$(DEB_VERSION).deb
	( cd out && apt-ftparchive packages . ) | gzip -c > out/Packages.gz
	( cd out && apt-ftparchive release . ) > out/Release

.PHONY: yum
yum: out/repodata/repomd.xml

out/repodata/repomd.xml: out/minikube-$(RPM_VERSION).rpm
	createrepo --simple-md-filenames --no-database \
	-u "$(MINIKUBE_RELEASES_URL)/$(VERSION)/" out

.SECONDEXPANSION:
TAR_TARGETS_linux-amd64   := out/minikube-linux-amd64 out/docker-machine-driver-kvm2
TAR_TARGETS_linux-arm64   := out/minikube-linux-arm64 #out/docker-machine-driver-kvm2
TAR_TARGETS_darwin-amd64  := out/minikube-darwin-amd64 out/docker-machine-driver-hyperkit
TAR_TARGETS_darwin-arm64  := out/minikube-darwin-arm64 #out/docker-machine-driver-hyperkit
TAR_TARGETS_windows-amd64 := out/minikube-windows-amd64.exe
out/minikube-%.tar.gz: $$(TAR_TARGETS_$$*)
	$(if $(quiet),@echo "  TAR      $@")
	$(Q)tar -cvzf $@ $^

.PHONY: cross-tars
cross-tars: out/minikube-linux-amd64.tar.gz out/minikube-windows-amd64.tar.gz out/minikube-darwin-amd64.tar.gz ## Cross-compile minikube
	-cd out && $(SHA512SUM) *.tar.gz > SHA512SUM

out/minikube-installer.exe: out/minikube-windows-amd64.exe
	rm -rf out/windows_tmp
	cp -r installers/windows/ out/windows_tmp
	cp -r LICENSE out/windows_tmp/LICENSE
	awk 'sub("$$", "\r")' out/windows_tmp/LICENSE > out/windows_tmp/LICENSE.txt
	sed -E -i 's/--VERSION_MAJOR--/'$(VERSION_MAJOR)'/g' out/windows_tmp/minikube.nsi
	sed -E -i 's/--VERSION_MINOR--/'$(VERSION_MINOR)'/g' out/windows_tmp/minikube.nsi
	sed -E -i 's/--VERSION_BUILD--/'$(VERSION_BUILD)'/g' out/windows_tmp/minikube.nsi
	sed -E -i 's/--INSTALL_SIZE--/'$(INSTALL_SIZE)'/g' out/windows_tmp/minikube.nsi
	cp out/minikube-windows-amd64.exe out/windows_tmp/minikube.exe
	makensis out/windows_tmp/minikube.nsi
	mv out/windows_tmp/minikube-installer.exe out/minikube-installer.exe
	rm -rf out/windows_tmp

out/docker-machine-driver-hyperkit:
ifeq ($(MINIKUBE_BUILD_IN_DOCKER),y)
	docker run --rm -e GOCACHE=/app/.cache -e IN_DOCKER=1 \
		--user $(shell id -u):$(shell id -g) -w /app \
		-v $(PWD):/app -v $(GOPATH):/go --init --entrypoint "" \
		$(HYPERKIT_BUILD_IMAGE) /bin/bash -c 'CC=o64-clang CXX=o64-clang++ /usr/bin/make $@'
else
	$(if $(quiet),@echo "  GO       $@")
	$(Q)GOOS=darwin CGO_ENABLED=1 go build \
		-ldflags="$(HYPERKIT_LDFLAGS)"   \
		-o $@ k8s.io/minikube/cmd/drivers/hyperkit
endif

hyperkit_in_docker:
	rm -f out/docker-machine-driver-hyperkit
	$(MAKE) MINIKUBE_BUILD_IN_DOCKER=y out/docker-machine-driver-hyperkit

.PHONY: install-hyperkit-driver
install-hyperkit-driver: out/docker-machine-driver-hyperkit ## Install hyperkit to local machine
	mkdir -p $(HOME)/bin
	sudo cp out/docker-machine-driver-hyperkit $(HOME)/bin/docker-machine-driver-hyperkit
	sudo chown root:wheel $(HOME)/bin/docker-machine-driver-hyperkit
	sudo chmod u+s $(HOME)/bin/docker-machine-driver-hyperkit

.PHONY: release-hyperkit-driver
release-hyperkit-driver: install-hyperkit-driver checksum ## Copy hyperkit using gsutil
	gsutil cp $(GOBIN)/docker-machine-driver-hyperkit gs://minikube/drivers/hyperkit/$(VERSION)/
	gsutil cp $(GOBIN)/docker-machine-driver-hyperkit.sha256 gs://minikube/drivers/hyperkit/$(VERSION)/

.PHONY: check-release
check-release: $(SOURCE_GENERATED) ## Execute go test
	go test -v ./deploy/minikube/release_sanity_test.go -tags=release

buildroot-image: $(ISO_BUILD_IMAGE) # convenient alias to build the docker container
$(ISO_BUILD_IMAGE): deploy/iso/minikube-iso/Dockerfile
	docker build $(ISO_DOCKER_EXTRA_ARGS) -t $@ -f $< $(dir $<)
	@echo ""
	@echo "$(@) successfully built"

out/storage-provisioner: out/storage-provisioner-$(GOARCH)
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

out/storage-provisioner-%: cmd/storage-provisioner/main.go pkg/storage/storage_provisioner.go
ifeq ($(MINIKUBE_BUILD_IN_DOCKER),y)
	$(call DOCKER,$(BUILD_IMAGE),/usr/bin/make $@)
else
	$(if $(quiet),@echo "  GO       $@")
	$(Q)CGO_ENABLED=0 GOOS=linux GOARCH=$* go build -o $@ -ldflags=$(PROVISIONER_LDFLAGS) cmd/storage-provisioner/main.go
endif

.PHONY: storage-provisioner-image
storage-provisioner-image: storage-provisioner-image-$(GOARCH) ## Build storage-provisioner docker image
	docker tag $(REGISTRY)/storage-provisioner-$(GOARCH):$(STORAGE_PROVISIONER_TAG) $(REGISTRY)/storage-provisioner:$(STORAGE_PROVISIONER_TAG)

storage-provisioner-image-%: out/storage-provisioner-%
	docker build -t $(REGISTRY)/storage-provisioner-$*:$(STORAGE_PROVISIONER_TAG) -f deploy/storage-provisioner/Dockerfile  --build-arg arch=$* .


X_DOCKER_BUILDER ?= minikube-builder
X_BUILD_ENV ?= DOCKER_CLI_EXPERIMENTAL=enabled

.PHONY: docker-multi-arch-builder
docker-multi-arch-builder:
	env $(X_BUILD_ENV) docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
	env $(X_BUILD_ENV) docker buildx rm --builder $(X_DOCKER_BUILDER) || true
	env $(X_BUILD_ENV) docker buildx create --name $(X_DOCKER_BUILDER) --buildkitd-flags '--debug' || true

KICBASE_ARCH = linux/arm64,linux/amd64
KICBASE_IMAGE_GCR ?= $(REGISTRY)/kicbase:$(KIC_VERSION)
KICBASE_IMAGE_HUB ?= kicbase/stable:$(KIC_VERSION)
KICBASE_IMAGE_REGISTRIES ?= $(KICBASE_IMAGE_GCR) $(KICBASE_IMAGE_HUB)

.PHONY: push-kic-base-image 
push-kic-base-image: deploy/kicbase/auto-pause docker-multi-arch-builder ## Push multi-arch local/kicbase:latest to all remote registries
ifdef AUTOPUSH
	docker login gcr.io/k8s-minikube
	docker login docker.pkg.github.com
	docker login
endif
	$(foreach REG,$(KICBASE_IMAGE_REGISTRIES), \
		@docker pull $(REG) && echo "Image already exist in registry" && exit 1 || echo "Image doesn't exist in registry";)
ifndef CIBUILD
	$(call user_confirm, 'Are you sure you want to push $(KICBASE_IMAGE_REGISTRIES) ?')
endif
	env $(X_BUILD_ENV) docker buildx build --builder $(X_DOCKER_BUILDER) --platform $(KICBASE_ARCH) $(addprefix -t ,$(KICBASE_IMAGE_REGISTRIES)) --push  --build-arg COMMIT_SHA=${VERSION}-$(COMMIT) ./deploy/kicbase

out/preload-tool:
	go build -ldflags="$(MINIKUBE_LDFLAGS)" -o $@ ./hack/preload-images/*.go

.PHONY: upload-preloaded-images-tar
upload-preloaded-images-tar: out/minikube out/preload-tool ## Upload the preloaded images for oldest supported, newest supported, and default kubernetes versions to GCS.
	out/preload-tool

.PHONY: generate-preloaded-images-tar
generate-preloaded-images-tar: out/minikube out/preload-tool ## Generates the preloaded images for oldest supported, newest supported, and default kubernetes versions
	out/preload-tool --no-upload

ALL_ARCH = amd64 arm arm64 ppc64le s390x
IMAGE = $(REGISTRY)/storage-provisioner
TAG = $(STORAGE_PROVISIONER_TAG)

.PHONY: push-storage-provisioner-manifest
push-storage-provisioner-manifest: $(shell echo $(ALL_ARCH) | sed -e "s~[^ ]*~storage\-provisioner\-image\-&~g") ## Push multi-arch storage-provisioner image
ifndef CIBUILD
	docker login gcr.io/k8s-minikube
endif
	set -x; for arch in $(ALL_ARCH); do docker push ${IMAGE}-$${arch}:${TAG}; done
	$(X_BUILD_ENV) docker manifest create --amend $(IMAGE):$(TAG) $(shell echo $(ALL_ARCH) | sed -e "s~[^ ]*~$(IMAGE)\-&:$(TAG)~g")
	set -x; for arch in $(ALL_ARCH); do $(X_BUILD_ENV) docker manifest annotate --arch $${arch} ${IMAGE}:${TAG} ${IMAGE}-$${arch}:${TAG}; done
	$(X_BUILD_ENV) docker manifest push $(STORAGE_PROVISIONER_MANIFEST)

.PHONY: push-docker
push-docker: # Push docker image base on to IMAGE variable (used internally by other targets)
	@docker pull $(IMAGE) && echo "Image already exist in registry" && exit 1 || echo "Image doesn't exist in registry"
ifndef AUTOPUSH
	$(call user_confirm, 'Are you sure you want to push $(IMAGE) ?')
endif
	docker push $(IMAGE)

.PHONY: out/gvisor-addon
out/gvisor-addon: $(SOURCE_GENERATED) ## Build gvisor addon
	$(if $(quiet),@echo "  GO       $@")
	$(Q)GOOS=linux CGO_ENABLED=0 go build -o $@ cmd/gvisor/gvisor.go

.PHONY: gvisor-addon-image
gvisor-addon-image: out/gvisor-addon  ## Build docker image for gvisor
	docker build -t $(REGISTRY)/gvisor-addon:$(GVISOR_TAG) -f deploy/gvisor/Dockerfile .

.PHONY: push-gvisor-addon-image
push-gvisor-addon-image: gvisor-addon-image
	docker login gcr.io/k8s-minikube
	$(MAKE) push-docker IMAGE=$(REGISTRY)/gvisor-addon:$(GVISOR_TAG)

.PHONY: release-iso
release-iso: minikube_iso checksum  ## Build and release .iso file
	gsutil cp out/minikube.iso gs://$(ISO_BUCKET)/minikube-$(ISO_VERSION).iso
	gsutil cp out/minikube.iso.sha256 gs://$(ISO_BUCKET)/minikube-$(ISO_VERSION).iso.sha256

.PHONY: release-minikube
release-minikube: out/minikube checksum ## Minikube release
	gsutil cp out/minikube-$(GOOS)-$(GOARCH) $(MINIKUBE_UPLOAD_LOCATION)/$(MINIKUBE_VERSION)/minikube-$(GOOS)-$(GOARCH)
	gsutil cp out/minikube-$(GOOS)-$(GOARCH).sha256 $(MINIKUBE_UPLOAD_LOCATION)/$(MINIKUBE_VERSION)/minikube-$(GOOS)-$(GOARCH).sha256

out/docker-machine-driver-kvm2: out/docker-machine-driver-kvm2-amd64
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

out/docker-machine-driver-kvm2-x86_64: out/docker-machine-driver-kvm2-amd64
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

out/docker-machine-driver-kvm2-aarch64: out/docker-machine-driver-kvm2-arm64
	$(if $(quiet),@echo "  CP       $@")
	$(Q)cp $< $@

out/docker-machine-driver-kvm2-%:
ifeq ($(MINIKUBE_BUILD_IN_DOCKER),y)
	docker image inspect -f '{{.Id}} {{.RepoTags}}' $(KVM_BUILD_IMAGE) || $(MAKE) kvm-image
	$(call DOCKER,$(KVM_BUILD_IMAGE),/usr/bin/make $@ COMMIT=$(COMMIT))
	# make extra sure that we are linking with the older version of libvirt (1.3.1)
	test "`strings $@ | grep '^LIBVIRT_[0-9]' | sort | tail -n 1`" = "LIBVIRT_1.2.9"
else
	$(if $(quiet),@echo "  GO       $@")
	$(Q)GOARCH=$* \
	go build \
		-installsuffix "static" \
		-ldflags="$(KVM2_LDFLAGS)" \
		-tags "libvirt.1.3.1 without_lxc" \
		-o $@ \
		k8s.io/minikube/cmd/drivers/kvm
endif
	chmod +X $@

out/docker-machine-driver-kvm2_$(DEB_VERSION).deb: out/docker-machine-driver-kvm2_$(DEB_VERSION)-0_amd64.deb
	cp $< $@

out/docker-machine-driver-kvm2_$(DEB_VERSION)-0_%.deb: out/docker-machine-driver-kvm2-%
	cp -r installers/linux/deb/kvm2_deb_template out/docker-machine-driver-kvm2_$(DEB_VERSION)
	chmod 0755 out/docker-machine-driver-kvm2_$(DEB_VERSION)/DEBIAN
	sed -E -i 's/--VERSION--/'$(DEB_VERSION)'/g' out/docker-machine-driver-kvm2_$(DEB_VERSION)/DEBIAN/control
	sed -E -i 's/--ARCH--/'$*'/g' out/docker-machine-driver-kvm2_$(DEB_VERSION)/DEBIAN/control
	mkdir -p out/docker-machine-driver-kvm2_$(DEB_VERSION)/usr/bin
	cp $< out/docker-machine-driver-kvm2_$(DEB_VERSION)/usr/bin/docker-machine-driver-kvm2
	fakeroot dpkg-deb --build out/docker-machine-driver-kvm2_$(DEB_VERSION) $@
	rm -rf out/docker-machine-driver-kvm2_$(DEB_VERSION)

out/docker-machine-driver-kvm2-$(RPM_VERSION).rpm: out/docker-machine-driver-kvm2-$(RPM_VERSION)-0.x86_64.deb
	cp $< $@

out/docker-machine-driver-kvm2-$(RPM_VERSION)-0.%.rpm: out/docker-machine-driver-kvm2-%
	cp -r installers/linux/rpm/kvm2_rpm_template out/docker-machine-driver-kvm2-$(RPM_VERSION)
	sed -E -i 's/--VERSION--/'$(RPM_VERSION)'/g' out/docker-machine-driver-kvm2-$(RPM_VERSION)/docker-machine-driver-kvm2.spec
	sed -E -i 's|--OUT--|'$(PWD)/out'|g' out/docker-machine-driver-kvm2-$(RPM_VERSION)/docker-machine-driver-kvm2.spec
	rpmbuild -bb -D "_rpmdir $(PWD)/out" --target $* \
		out/docker-machine-driver-kvm2-$(RPM_VERSION)/docker-machine-driver-kvm2.spec
	@mv out/$*/docker-machine-driver-kvm2-$(RPM_VERSION)-0.$*.rpm out/ && rmdir out/$*
	rm -rf out/docker-machine-driver-kvm2-$(RPM_VERSION)

.PHONY: kvm-image
kvm-image: installers/linux/kvm/Dockerfile  ## Convenient alias to build the docker container
	docker build --build-arg "GO_VERSION=$(KVM_GO_VERSION)" -t $(KVM_BUILD_IMAGE) -f $< $(dir $<)
	@echo ""
	@echo "$(@) successfully built"

kvm_in_docker:
	docker image inspect -f '{{.Id}} {{.RepoTags}}' $(KVM_BUILD_IMAGE) || $(MAKE) kvm-image
	rm -f out/docker-machine-driver-kvm2
	$(call DOCKER,$(KVM_BUILD_IMAGE),/usr/bin/make out/docker-machine-driver-kvm2 COMMIT=$(COMMIT))

.PHONY: install-kvm-driver
install-kvm-driver: out/docker-machine-driver-kvm2  ## Install KVM Driver
	mkdir -p $(GOBIN)
	cp out/docker-machine-driver-kvm2 $(GOBIN)/docker-machine-driver-kvm2

.PHONY: release-kvm-driver
release-kvm-driver: install-kvm-driver checksum  ## Release KVM Driver
	gsutil cp $(GOBIN)/docker-machine-driver-kvm2 gs://minikube/drivers/kvm/$(VERSION)/
	gsutil cp $(GOBIN)/docker-machine-driver-kvm2.sha256 gs://minikube/drivers/kvm/$(VERSION)/

site/themes/docsy/assets/vendor/bootstrap/package.js: ## update the website docsy theme git submodule 
	git submodule update -f --init --recursive

out/hugo/hugo:
	mkdir -p out
	test -d out/hugo || git clone https://github.com/gohugoio/hugo.git out/hugo
	go get golang.org/dl/go1.16 && go1.16 download
	(cd out/hugo && go1.16 build --tags extended)

.PHONY: site
site: site/themes/docsy/assets/vendor/bootstrap/package.js out/hugo/hugo ## Serve the documentation site to localhost
	(cd site && ../out/hugo/hugo serve \
	  --disableFastRender \
	  --navigateToChanged \
	  --ignoreCache \
	  --buildFuture)

.PHONY: out/mkcmp
out/mkcmp:
	GOOS=$(GOOS) GOARCH=$(GOARCH) go build -o $@ cmd/performance/mkcmp/main.go

.PHONY: deploy/kicbase/auto-pause # auto pause binary to be used for kic image work around for not passing the whole repo as docker context
deploy/kicbase/auto-pause: $(SOURCE_GENERATED) $(SOURCE_FILES)
	GOOS=linux GOARCH=$(GOARCH) go build -o $@ cmd/auto-pause/auto-pause.go

# auto pause binary to be used for ISO
deploy/iso/minikube-iso/board/coreos/minikube/rootfs-overlay/usr/bin/auto-pause: $(SOURCE_GENERATED) $(SOURCE_FILES)
	GOOS=linux GOARCH=$(GOARCH) go build -o $@ cmd/auto-pause/auto-pause.go


.PHONY: deploy/addons/auto-pause/auto-pause-hook
deploy/addons/auto-pause/auto-pause-hook: $(SOURCE_GENERATED) ## Build auto-pause hook addon
	$(if $(quiet),@echo "  GO       $@")
	$(Q)GOOS=linux CGO_ENABLED=0 go build -a --ldflags '-extldflags "-static"' -tags netgo -installsuffix netgo -o $@ cmd/auto-pause/auto-pause-hook/main.go cmd/auto-pause/auto-pause-hook/config.go cmd/auto-pause/auto-pause-hook/certs.go

.PHONY: auto-pause-hook-image
auto-pause-hook-image: deploy/addons/auto-pause/auto-pause-hook ## Build docker image for auto-pause hook
	docker build -t $(REGISTRY)/auto-pause-hook:$(AUTOPAUSE_HOOK_TAG) ./deploy/addons/auto-pause

.PHONY: push-auto-pause-hook-image
push-auto-pause-hook-image: auto-pause-hook-image
	docker login gcr.io/k8s-minikube
	$(MAKE) push-docker IMAGE=$(REGISTRY)/auto-pause-hook:$(AUTOPAUSE_HOOK_TAG)

.PHONY: prow-test-image
prow-test-image:
	docker build --build-arg "GO_VERSION=$(GO_VERSION)"  -t $(REGISTRY)/prow-test:$(PROW_TEST_TAG) ./deploy/prow

.PHONY: push-prow-test-image
push-prow-test-image: prow-test-image
	docker login gcr.io/k8s-minikube
	$(MAKE) push-docker IMAGE=$(REGISTRY)/prow-test:$(PROW_TEST_TAG)

.PHONY: out/performance-bot
out/performance-bot:
	GOOS=$(GOOS) GOARCH=$(GOARCH) go build -o $@ cmd/performance/pr-bot/bot.go

.PHONY: out/metrics-collector
out/metrics-collector:
	GOOS=$(GOOS) GOARCH=$(GOARCH) go build -o $@ hack/metrics/*.go


.PHONY: compare
compare: out/mkcmp out/minikube
	mv out/minikube out/$(CURRENT_GIT_BRANCH).minikube
	git checkout master
	make out/minikube
	mv out/minikube out/master.minikube
	git checkout $(CURRENT_GIT_BRANCH)
	out/mkcmp out/master.minikube out/$(CURRENT_GIT_BRANCH).minikube
	

.PHONY: help
help:
	@printf "\033[1mAvailable targets for minikube ${VERSION}\033[21m\n"
	@printf "\033[1m--------------------------------------\033[21m\n"
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


.PHONY: update-kubernetes-version
update-kubernetes-version:
	(cd hack/update/kubernetes_version && \
	 go run update_kubernetes_version.go)

.PHONY: update-kubernetes-version-pr
update-kubernetes-version-pr:
ifndef GITHUB_TOKEN
	@echo "⚠️ please set GITHUB_TOKEN environment variable with your GitHub token"
	@echo "you can use https://github.com/settings/tokens/new?scopes=repo,write:packages to create new one"
else
	(cd hack/update/kubernetes_version && \
	 export UPDATE_TARGET="all" && \
	 go run update_kubernetes_version.go)
endif

.PHONY: stress
stress: ## run the stress tests
	go test -test.v -test.timeout=2h ./test/stress -loops=10 | tee "./out/testout_$(COMMIT_SHORT).txt"

.PHONY: cpu-benchmark-idle
cpu-benchmark-idle: ## run the cpu usage 5 minutes idle benchmark
	./hack/benchmark/cpu_usage/idle_only/benchmark_local_k8s.sh

.PHONY: cpu-benchmark-autopause
cpu-benchmark-autopause: ## run the cpu usage auto-pause benchmark
	./hack/benchmark/cpu_usage/auto_pause/benchmark_local_k8s.sh

.PHONY: update-gopogh-version
update-gopogh-version: ## update gopogh version
	(cd hack/update/gopogh_version && \
	 go run update_gopogh_version.go)
