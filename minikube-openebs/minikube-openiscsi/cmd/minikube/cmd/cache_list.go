/*
Copyright 2017 The Kubernetes Authors All rights reserved.

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

package cmd

import (
	"os"
	"text/template"

	"github.com/spf13/cobra"
	cmdConfig "k8s.io/minikube/cmd/minikube/cmd/config"
	"k8s.io/minikube/pkg/minikube/exit"
	"k8s.io/minikube/pkg/minikube/reason"
)

const defaultCacheListFormat = "{{.CacheImage}}\n"

var cacheListFormat string

// CacheListTemplate represents the cache list template
type CacheListTemplate struct {
	CacheImage string
}

// listCacheCmd represents the cache list command
var listCacheCmd = &cobra.Command{
	Use:   "list",
	Short: "List all available images from the local cache.",
	Long:  "List all available images from the local cache.",
	Run: func(cmd *cobra.Command, args []string) {
		images, err := cmdConfig.ListConfigMap(cacheImageConfigKey)
		if err != nil {
			exit.Error(reason.InternalListConfig, "Failed to get image map", err)
		}
		if err := cacheList(images); err != nil {
			exit.Error(reason.InternalCacheList, "Failed to list cached images", err)
		}
	},
}

func init() {
	listCacheCmd.Flags().StringVar(&cacheListFormat, "format", defaultCacheListFormat,
		`Go template format string for the cache list output.  The format for Go templates can be found here: https://golang.org/pkg/text/template/
For the list of accessible variables for the template, see the struct values here: https://godoc.org/k8s.io/minikube/cmd/minikube/cmd#CacheListTemplate`)
	cacheCmd.AddCommand(listCacheCmd)
}

// cacheList returns a formatted list of images found within the local cache
func cacheList(images []string) error {
	for _, image := range images {
		tmpl, err := template.New("list").Parse(cacheListFormat)
		if err != nil {
			return err
		}
		listTmplt := CacheListTemplate{image}
		if err := tmpl.Execute(os.Stdout, listTmplt); err != nil {
			return err
		}
	}
	return nil
}
