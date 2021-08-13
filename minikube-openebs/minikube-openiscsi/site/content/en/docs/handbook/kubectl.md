---
title: "Kubectl"
weight: 2
description: >
  Use kubectl inside minikube
aliases:
  - /docs/kubectl/
---

By default, [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) gets configured to access the kubernetes cluster control plane
inside minikube when the `minikube start` command is executed.

However if `kubectl` is not installed locally, minikube already includes kubectl which can be used like this:

```shell
minikube kubectl -- <kubectl commands>
```

You can also `alias kubectl="minikube kubectl --"` for easier usage.

Alternatively, you can create a symbolic link to minikube's binary named 'kubectl'.

```shell
ln -s $(which minikube) /usr/local/bin/kubectl
```

Get pods

```shell
minikube kubectl -- get pods
```

Creating a deployment inside kubernetes cluster

```shell
minikube kubectl -- create deployment hello-minikube --image=k8s.gcr.io/echoserver:1.4
```

Exposing the deployment with a NodePort service

```shell
minikube kubectl -- expose deployment hello-minikube --type=NodePort --port=8080
```

For more help

```shell
minikube kubectl -- --help
```

### Shell autocompletion

After applying the alias or the symbolic link you can follow https://kubernetes.io/docs/tasks/tools/included/optional-kubectl-configs-bash-linux/ to enable shell-autocompletion.
