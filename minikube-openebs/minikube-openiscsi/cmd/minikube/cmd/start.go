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

package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"net/url"
	"os"
	"os/exec"
	"os/user"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/blang/semver"
	"github.com/docker/machine/libmachine/ssh"
	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote"
	"github.com/pkg/errors"
	"github.com/shirou/gopsutil/v3/cpu"
	gopshost "github.com/shirou/gopsutil/v3/host"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"k8s.io/klog/v2"
	cmdcfg "k8s.io/minikube/cmd/minikube/cmd/config"
	"k8s.io/minikube/pkg/drivers/kic/oci"
	"k8s.io/minikube/pkg/minikube/bootstrapper/bsutil"
	"k8s.io/minikube/pkg/minikube/bootstrapper/images"
	"k8s.io/minikube/pkg/minikube/config"
	"k8s.io/minikube/pkg/minikube/constants"
	"k8s.io/minikube/pkg/minikube/cruntime"
	"k8s.io/minikube/pkg/minikube/detect"
	"k8s.io/minikube/pkg/minikube/download"
	"k8s.io/minikube/pkg/minikube/driver"
	"k8s.io/minikube/pkg/minikube/exit"
	"k8s.io/minikube/pkg/minikube/kubeconfig"
	"k8s.io/minikube/pkg/minikube/localpath"
	"k8s.io/minikube/pkg/minikube/machine"
	"k8s.io/minikube/pkg/minikube/mustload"
	"k8s.io/minikube/pkg/minikube/node"
	"k8s.io/minikube/pkg/minikube/notify"
	"k8s.io/minikube/pkg/minikube/out"
	"k8s.io/minikube/pkg/minikube/out/register"
	"k8s.io/minikube/pkg/minikube/reason"
	"k8s.io/minikube/pkg/minikube/style"
	pkgtrace "k8s.io/minikube/pkg/trace"

	"k8s.io/minikube/pkg/minikube/registry"
	"k8s.io/minikube/pkg/minikube/translate"
	"k8s.io/minikube/pkg/util"
	"k8s.io/minikube/pkg/version"
)

var (
	registryMirror   []string
	insecureRegistry []string
	apiServerNames   []string
	apiServerIPs     []net.IP
	hostRe           = regexp.MustCompile(`^[^-][\w\.-]+$`)
)

func init() {
	initMinikubeFlags()
	initKubernetesFlags()
	initDriverFlags()
	initNetworkingFlags()
	if err := viper.BindPFlags(startCmd.Flags()); err != nil {
		exit.Error(reason.InternalBindFlags, "unable to bind flags", err)
	}
}

// startCmd represents the start command
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Starts a local Kubernetes cluster",
	Long:  "Starts a local Kubernetes cluster",
	Run:   runStart,
}

// platform generates a user-readable platform message
func platform() string {
	var s strings.Builder

	// Show the distro version if possible
	hi, err := gopshost.Info()
	if err == nil {
		s.WriteString(fmt.Sprintf("%s %s", strings.Title(hi.Platform), hi.PlatformVersion))
		klog.Infof("hostinfo: %+v", hi)
	} else {
		klog.Warningf("gopshost.Info returned error: %v", err)
		s.WriteString(runtime.GOOS)
	}

	vsys, vrole, err := gopshost.Virtualization()
	if err != nil {
		klog.Warningf("gopshost.Virtualization returned error: %v", err)
	} else {
		klog.Infof("virtualization: %s %s", vsys, vrole)
	}

	// This environment is exotic, let's output a bit more.
	if vrole == "guest" || runtime.GOARCH != "amd64" {
		if vrole == "guest" && vsys != "" {
			s.WriteString(fmt.Sprintf(" (%s/%s)", vsys, runtime.GOARCH))
		} else {
			s.WriteString(fmt.Sprintf(" (%s)", runtime.GOARCH))
		}
	}
	return s.String()
}

// runStart handles the executes the flow of "minikube start"
func runStart(cmd *cobra.Command, args []string) {
	register.SetEventLogPath(localpath.EventLog(ClusterFlagValue()))
	ctx := context.Background()
	out.SetJSON(outputFormat == "json")
	if err := pkgtrace.Initialize(viper.GetString(trace)); err != nil {
		exit.Message(reason.Usage, "error initializing tracing: {{.Error}}", out.V{"Error": err.Error()})
	}
	defer pkgtrace.Cleanup()
	displayVersion(version.GetVersion())

	// No need to do the update check if no one is going to see it
	if !viper.GetBool(interactive) || !viper.GetBool(dryRun) {
		// Avoid blocking execution on optional HTTP fetches
		go notify.MaybePrintUpdateTextFromGithub()
	}

	displayEnviron(os.Environ())
	if viper.GetBool(force) {
		out.WarningT("minikube skips various validations when --force is supplied; this may lead to unexpected behavior")
	}

	// if --registry-mirror specified when run minikube start,
	// take arg precedence over MINIKUBE_REGISTRY_MIRROR
	// actually this is a hack, because viper 1.0.0 can assign env to variable if StringSliceVar
	// and i can't update it to 1.4.0, it affects too much code
	// other types (like String, Bool) of flag works, so imageRepository, imageMirrorCountry
	// can be configured as MINIKUBE_IMAGE_REPOSITORY and IMAGE_MIRROR_COUNTRY
	// this should be updated to documentation
	if len(registryMirror) == 0 {
		registryMirror = viper.GetStringSlice("registry_mirror")
	}

	if !config.ProfileNameValid(ClusterFlagValue()) {
		out.WarningT("Profile name '{{.name}}' is not valid", out.V{"name": ClusterFlagValue()})
		exit.Message(reason.Usage, "Only alphanumeric and dashes '-' are permitted. Minimum 2 characters, starting with alphanumeric.")
	}

	existing, err := config.Load(ClusterFlagValue())
	if err != nil && !config.IsNotExist(err) {
		kind := reason.HostConfigLoad
		if config.IsPermissionDenied(err) {
			kind = reason.HostHomePermission
		}
		exit.Message(kind, "Unable to load config: {{.error}}", out.V{"error": err})
	}

	if existing != nil {
		upgradeExistingConfig(cmd, existing)
	} else {
		validateProfileName()
	}

	validateSpecifiedDriver(existing)
	validateKubernetesVersion(existing)

	ds, alts, specified := selectDriver(existing)
	if cmd.Flag(kicBaseImage).Changed {
		if !isBaseImageApplicable(ds.Name) {
			exit.Message(reason.Usage,
				"flag --{{.imgFlag}} is not available for driver '{{.driver}}'. Did you mean to use '{{.docker}}' or '{{.podman}}' driver instead?\n"+
					"Please use --{{.isoFlag}} flag to configure VM based drivers",
				out.V{
					"imgFlag": kicBaseImage,
					"driver":  ds.Name,
					"docker":  registry.Docker,
					"podman":  registry.Podman,
					"isoFlag": isoURL,
				},
			)
		}
	}

	starter, err := provisionWithDriver(cmd, ds, existing)
	if err != nil {
		node.ExitIfFatal(err)
		machine.MaybeDisplayAdvice(err, ds.Name)
		if specified {
			// If the user specified a driver, don't fallback to anything else
			exitGuestProvision(err)
		} else {
			success := false
			// Walk down the rest of the options
			for _, alt := range alts {
				// Skip non-default drivers
				if !ds.Default {
					continue
				}
				out.WarningT("Startup with {{.old_driver}} driver failed, trying with alternate driver {{.new_driver}}: {{.error}}", out.V{"old_driver": ds.Name, "new_driver": alt.Name, "error": err})
				ds = alt
				// Delete the existing cluster and try again with the next driver on the list
				profile, err := config.LoadProfile(ClusterFlagValue())
				if err != nil {
					klog.Warningf("%s profile does not exist, trying anyways.", ClusterFlagValue())
				}

				err = deleteProfile(ctx, profile)
				if err != nil {
					out.WarningT("Failed to delete cluster {{.name}}, proceeding with retry anyway.", out.V{"name": ClusterFlagValue()})
				}
				starter, err = provisionWithDriver(cmd, ds, existing)
				if err != nil {
					continue
				} else {
					// Success!
					success = true
					break
				}
			}
			if !success {
				exitGuestProvision(err)
			}
		}
	}

	if existing != nil && driver.IsKIC(existing.Driver) {
		if viper.GetBool(createMount) {
			old := ""
			if len(existing.ContainerVolumeMounts) > 0 {
				old = existing.ContainerVolumeMounts[0]
			}
			if mount := viper.GetString(mountString); old != mount {
				exit.Message(reason.GuestMountConflict, "Sorry, {{.driver}} does not allow mounts to be changed after container creation (previous mount: '{{.old}}', new mount: '{{.new}})'", out.V{
					"driver": existing.Driver,
					"new":    mount,
					"old":    old,
				})
			}
		}
	}

	kubeconfig, err := startWithDriver(cmd, starter, existing)
	if err != nil {
		node.ExitIfFatal(err)
		exit.Error(reason.GuestStart, "failed to start node", err)
	}

	if err := showKubectlInfo(kubeconfig, starter.Node.KubernetesVersion, starter.Cfg.Name); err != nil {
		klog.Errorf("kubectl info: %v", err)
	}
}

func provisionWithDriver(cmd *cobra.Command, ds registry.DriverState, existing *config.ClusterConfig) (node.Starter, error) {
	driverName := ds.Name
	klog.Infof("selected driver: %s", driverName)
	validateDriver(ds, existing)
	err := autoSetDriverOptions(cmd, driverName)
	if err != nil {
		klog.Errorf("Error autoSetOptions : %v", err)
	}

	validateFlags(cmd, driverName)
	validateUser(driverName)
	if driverName == oci.Docker {
		validateDockerStorageDriver(driverName)
	}

	// Download & update the driver, even in --download-only mode
	if !viper.GetBool(dryRun) {
		updateDriver(driverName)
	}

	k8sVersion := getKubernetesVersion(existing)
	cc, n, err := generateClusterConfig(cmd, existing, k8sVersion, driverName)
	if err != nil {
		return node.Starter{}, errors.Wrap(err, "Failed to generate config")
	}

	// This is about as far as we can go without overwriting config files
	if viper.GetBool(dryRun) {
		out.Step(style.DryRun, `dry-run validation complete!`)
		os.Exit(0)
	}

	if driver.IsVM(driverName) && !driver.IsSSH(driverName) {
		url, err := download.ISO(viper.GetStringSlice(isoURL), cmd.Flags().Changed(isoURL))
		if err != nil {
			return node.Starter{}, errors.Wrap(err, "Failed to cache ISO")
		}
		cc.MinikubeISO = url
	}

	var existingAddons map[string]bool
	if viper.GetBool(installAddons) {
		existingAddons = map[string]bool{}
		if existing != nil && existing.Addons != nil {
			existingAddons = existing.Addons
		}
	}

	if viper.GetBool(nativeSSH) {
		ssh.SetDefaultClient(ssh.Native)
	} else {
		ssh.SetDefaultClient(ssh.External)
	}

	mRunner, preExists, mAPI, host, err := node.Provision(&cc, &n, true, viper.GetBool(deleteOnFailure))
	if err != nil {
		return node.Starter{}, err
	}

	return node.Starter{
		Runner:         mRunner,
		PreExists:      preExists,
		MachineAPI:     mAPI,
		Host:           host,
		ExistingAddons: existingAddons,
		Cfg:            &cc,
		Node:           &n,
	}, nil
}

func startWithDriver(cmd *cobra.Command, starter node.Starter, existing *config.ClusterConfig) (*kubeconfig.Settings, error) {
	kubeconfig, err := node.Start(starter, true)
	if err != nil {
		kubeconfig, err = maybeDeleteAndRetry(cmd, *starter.Cfg, *starter.Node, starter.ExistingAddons, err)
		if err != nil {
			return nil, err
		}
	}

	numNodes := viper.GetInt(nodes)
	if existing != nil {
		if numNodes > 1 {
			// We ignore the --nodes parameter if we're restarting an existing cluster
			out.WarningT(`The cluster {{.cluster}} already exists which means the --nodes parameter will be ignored. Use "minikube node add" to add nodes to an existing cluster.`, out.V{"cluster": existing.Name})
		}
		numNodes = len(existing.Nodes)
	}
	if numNodes > 1 {
		if driver.BareMetal(starter.Cfg.Driver) {
			exit.Message(reason.DrvUnsupportedMulti, "The none driver is not compatible with multi-node clusters.")
		} else {
			if existing == nil {
				for i := 1; i < numNodes; i++ {
					nodeName := node.Name(i + 1)
					n := config.Node{
						Name:              nodeName,
						Worker:            true,
						ControlPlane:      false,
						KubernetesVersion: starter.Cfg.KubernetesConfig.KubernetesVersion,
					}
					out.Ln("") // extra newline for clarity on the command line
					err := node.Add(starter.Cfg, n, viper.GetBool(deleteOnFailure))
					if err != nil {
						return nil, errors.Wrap(err, "adding node")
					}
				}
			} else {
				for _, n := range existing.Nodes {
					if !n.ControlPlane {
						err := node.Add(starter.Cfg, n, viper.GetBool(deleteOnFailure))
						if err != nil {
							return nil, errors.Wrap(err, "adding node")
						}
					}
				}
			}
		}
	}

	return kubeconfig, nil
}

func warnAboutMultiNodeCNI() {
	out.WarningT("Cluster was created without any CNI, adding a node to it might cause broken networking.")
}

func updateDriver(driverName string) {
	v, err := version.GetSemverVersion()
	if err != nil {
		out.WarningT("Error parsing minikube version: {{.error}}", out.V{"error": err})
	} else if err := driver.InstallOrUpdate(driverName, localpath.MakeMiniPath("bin"), v, viper.GetBool(interactive), viper.GetBool(autoUpdate)); err != nil {
		out.WarningT("Unable to update {{.driver}} driver: {{.error}}", out.V{"driver": driverName, "error": err})
	}
}

func displayVersion(version string) {
	prefix := ""
	if ClusterFlagValue() != constants.DefaultClusterName {
		prefix = fmt.Sprintf("[%s] ", ClusterFlagValue())
	}

	register.Reg.SetStep(register.InitialSetup)
	out.Step(style.Happy, "{{.prefix}}minikube {{.version}} on {{.platform}}", out.V{"prefix": prefix, "version": version, "platform": platform()})
}

// displayEnviron makes the user aware of environment variables that will affect how minikube operates
func displayEnviron(env []string) {
	for _, kv := range env {
		bits := strings.SplitN(kv, "=", 2)
		k := bits[0]
		v := bits[1]
		if strings.HasPrefix(k, "MINIKUBE_") || k == constants.KubeconfigEnvVar {
			out.Infof("{{.key}}={{.value}}", out.V{"key": k, "value": v})
		}
	}
}

func showKubectlInfo(kcs *kubeconfig.Settings, k8sVersion string, machineName string) error {
	// To be shown at the end, regardless of exit path
	defer func() {
		register.Reg.SetStep(register.Done)
		if kcs.KeepContext {
			out.Step(style.Kubectl, "To connect to this cluster, use:  --context={{.name}}", out.V{"name": kcs.ClusterName})
		} else {
			out.Step(style.Ready, `Done! kubectl is now configured to use "{{.name}}" cluster and "{{.ns}}" namespace by default`, out.V{"name": machineName, "ns": kcs.Namespace})
		}
	}()

	path, err := exec.LookPath("kubectl")
	if err != nil {
		out.Styled(style.Tip, "kubectl not found. If you need it, try: 'minikube kubectl -- get pods -A'")
		return nil
	}

	gitVersion, err := kubectlVersion(path)
	if err != nil {
		return err
	}

	client, err := semver.Make(strings.TrimPrefix(gitVersion, version.VersionPrefix))
	if err != nil {
		return errors.Wrap(err, "client semver")
	}

	cluster := semver.MustParse(strings.TrimPrefix(k8sVersion, version.VersionPrefix))
	minorSkew := int(math.Abs(float64(int(client.Minor) - int(cluster.Minor))))
	klog.Infof("kubectl: %s, cluster: %s (minor skew: %d)", client, cluster, minorSkew)

	if client.Major != cluster.Major || minorSkew > 1 {
		out.Ln("")
		out.WarningT("{{.path}} is version {{.client_version}}, which may have incompatibilites with Kubernetes {{.cluster_version}}.",
			out.V{"path": path, "client_version": client, "cluster_version": cluster})
		out.Infof("Want kubectl {{.version}}? Try 'minikube kubectl -- get pods -A'", out.V{"version": k8sVersion})
	}
	return nil
}

func maybeDeleteAndRetry(cmd *cobra.Command, existing config.ClusterConfig, n config.Node, existingAddons map[string]bool, originalErr error) (*kubeconfig.Settings, error) {
	if viper.GetBool(deleteOnFailure) {
		out.WarningT("Node {{.name}} failed to start, deleting and trying again.", out.V{"name": n.Name})
		// Start failed, delete the cluster and try again
		profile, err := config.LoadProfile(existing.Name)
		if err != nil {
			out.ErrT(style.Meh, `"{{.name}}" profile does not exist, trying anyways.`, out.V{"name": existing.Name})
		}

		err = deleteProfile(context.Background(), profile)
		if err != nil {
			out.WarningT("Failed to delete cluster {{.name}}, proceeding with retry anyway.", out.V{"name": existing.Name})
		}

		// Re-generate the cluster config, just in case the failure was related to an old config format
		cc := updateExistingConfigFromFlags(cmd, &existing)
		var kubeconfig *kubeconfig.Settings
		for _, n := range cc.Nodes {
			r, p, m, h, err := node.Provision(&cc, &n, n.ControlPlane, false)
			s := node.Starter{
				Runner:         r,
				PreExists:      p,
				MachineAPI:     m,
				Host:           h,
				Cfg:            &cc,
				Node:           &n,
				ExistingAddons: existingAddons,
			}
			if err != nil {
				// Ok we failed again, let's bail
				return nil, err
			}

			k, err := node.Start(s, n.ControlPlane)
			if n.ControlPlane {
				kubeconfig = k
			}
			if err != nil {
				// Ok we failed again, let's bail
				return nil, err
			}
		}
		return kubeconfig, nil
	}
	// Don't delete the cluster unless they ask
	return nil, originalErr
}

func kubectlVersion(path string) (string, error) {
	j, err := exec.Command(path, "version", "--client", "--output=json").Output()
	if err != nil {
		// really old Kubernetes clients did not have the --output parameter
		b, err := exec.Command(path, "version", "--client", "--short").Output()
		if err != nil {
			return "", errors.Wrap(err, "exec")
		}
		s := strings.TrimSpace(string(b))
		return strings.Replace(s, "Client Version: ", "", 1), nil
	}

	cv := struct {
		ClientVersion struct {
			GitVersion string `json:"gitVersion"`
		} `json:"clientVersion"`
	}{}
	err = json.Unmarshal(j, &cv)
	if err != nil {
		return "", errors.Wrap(err, "unmarshal")
	}

	return cv.ClientVersion.GitVersion, nil
}

// returns (current_driver, suggested_drivers, "true, if the driver is set by command line arg or in the config file")
func selectDriver(existing *config.ClusterConfig) (registry.DriverState, []registry.DriverState, bool) {
	// Technically unrelated, but important to perform before detection
	driver.SetLibvirtURI(viper.GetString(kvmQemuURI))
	register.Reg.SetStep(register.SelectingDriver)
	// By default, the driver is whatever we used last time
	if existing != nil {
		old := hostDriver(existing)
		ds := driver.Status(old)
		out.Step(style.Sparkle, `Using the {{.driver}} driver based on existing profile`, out.V{"driver": ds.String()})
		return ds, nil, true
	}

	// Default to looking at the new driver parameter
	if d := viper.GetString("driver"); d != "" {
		if vmd := viper.GetString("vm-driver"); vmd != "" {
			// Output a warning
			warning := `Both driver={{.driver}} and vm-driver={{.vmd}} have been set.

    Since vm-driver is deprecated, minikube will default to driver={{.driver}}.

    If vm-driver is set in the global config, please run "minikube config unset vm-driver" to resolve this warning.
			`
			out.WarningT(warning, out.V{"driver": d, "vmd": vmd})
		}
		ds := driver.Status(d)
		if ds.Name == "" {
			exit.Message(reason.DrvUnsupportedOS, "The driver '{{.driver}}' is not supported on {{.os}}/{{.arch}}", out.V{"driver": d, "os": runtime.GOOS, "arch": runtime.GOARCH})
		}
		out.Step(style.Sparkle, `Using the {{.driver}} driver based on user configuration`, out.V{"driver": ds.String()})
		return ds, nil, true
	}

	// Fallback to old driver parameter
	if d := viper.GetString("vm-driver"); d != "" {
		ds := driver.Status(viper.GetString("vm-driver"))
		if ds.Name == "" {
			exit.Message(reason.DrvUnsupportedOS, "The driver '{{.driver}}' is not supported on {{.os}}/{{.arch}}", out.V{"driver": d, "os": runtime.GOOS, "arch": runtime.GOARCH})
		}
		out.Step(style.Sparkle, `Using the {{.driver}} driver based on user configuration`, out.V{"driver": ds.String()})
		return ds, nil, true
	}

	choices := driver.Choices(viper.GetBool("vm"))
	pick, alts, rejects := driver.Suggest(choices)
	if pick.Name == "" {
		out.Step(style.ThumbsDown, "Unable to pick a default driver. Here is what was considered, in preference order:")
		for _, r := range rejects {
			out.Infof("{{ .name }}: {{ .rejection }}", out.V{"name": r.Name, "rejection": r.Rejection})
			if r.Suggestion != "" {
				out.Infof("{{ .name }}: Suggestion: {{ .suggestion}}", out.V{"name": r.Name, "suggestion": r.Suggestion})
			}
		}
		exit.Message(reason.DrvNotDetected, "No possible driver was detected. Try specifying --driver, or see https://minikube.sigs.k8s.io/docs/start/")
	}

	if len(alts) > 1 {
		altNames := []string{}
		for _, a := range alts {
			altNames = append(altNames, a.String())
		}
		out.Step(style.Sparkle, `Automatically selected the {{.driver}} driver. Other choices: {{.alternates}}`, out.V{"driver": pick.Name, "alternates": strings.Join(altNames, ", ")})
	} else {
		out.Step(style.Sparkle, `Automatically selected the {{.driver}} driver`, out.V{"driver": pick.String()})
	}
	return pick, alts, false
}

// hostDriver returns the actual driver used by a libmachine host, which can differ from our config
func hostDriver(existing *config.ClusterConfig) string {
	if existing == nil {
		return ""
	}
	api, err := machine.NewAPIClient()
	if err != nil {
		klog.Warningf("selectDriver NewAPIClient: %v", err)
		return existing.Driver
	}

	cp, err := config.PrimaryControlPlane(existing)
	if err != nil {
		klog.Warningf("Unable to get control plane from existing config: %v", err)
		return existing.Driver
	}
	machineName := config.MachineName(*existing, cp)
	h, err := api.Load(machineName)
	if err != nil {
		klog.Warningf("api.Load failed for %s: %v", machineName, err)
		if existing.VMDriver != "" {
			return existing.VMDriver
		}
		return existing.Driver
	}

	return h.Driver.DriverName()
}

// validateProfileName makes sure that new profile name not duplicated with any of machine names in existing multi-node clusters.
func validateProfileName() {
	profiles, err := config.ListValidProfiles()
	if err != nil {
		exit.Message(reason.InternalListConfig, "Unable to list profiles: {{.error}}", out.V{"error": err})
	}
	for _, p := range profiles {
		for _, n := range p.Config.Nodes {
			machineName := config.MachineName(*p.Config, n)
			if ClusterFlagValue() == machineName {
				out.WarningT("Profile name '{{.name}}' is duplicated with machine name '{{.machine}}' in profile '{{.profile}}'", out.V{"name": ClusterFlagValue(),
					"machine": machineName,
					"profile": p.Name})
				exit.Message(reason.Usage, "Profile name should be unique")
			}
		}
	}
}

// validateSpecifiedDriver makes sure that if a user has passed in a driver
// it matches the existing cluster if there is one
func validateSpecifiedDriver(existing *config.ClusterConfig) {
	if existing == nil {
		return
	}

	var requested string
	if d := viper.GetString("driver"); d != "" {
		requested = d
	} else if d := viper.GetString("vm-driver"); d != "" {
		requested = d
	}

	// Neither --vm-driver or --driver was specified
	if requested == "" {
		return
	}

	old := hostDriver(existing)
	if requested == old {
		return
	}

	// hostDriver always returns original driver name even if an alias is used to start minikube.
	// For all next start with alias needs to be check against the host driver aliases.
	if driver.IsAlias(old, requested) {
		return
	}

	out.WarningT("Deleting existing cluster {{.name}} with different driver {{.driver_name}} due to --delete-on-failure flag set by the user. ", out.V{"name": existing.Name, "driver_name": old})
	if viper.GetBool(deleteOnFailure) {
		// Start failed, delete the cluster
		profile, err := config.LoadProfile(existing.Name)
		if err != nil {
			out.ErrT(style.Meh, `"{{.name}}" profile does not exist, trying anyways.`, out.V{"name": existing.Name})
		}

		err = deleteProfile(context.Background(), profile)
		if err != nil {
			out.WarningT("Failed to delete cluster {{.name}}.", out.V{"name": existing.Name})
		}
	}

	exit.Advice(
		reason.GuestDrvMismatch,
		`The existing "{{.name}}" cluster was created using the "{{.old}}" driver, which is incompatible with requested "{{.new}}" driver.`,
		"Delete the existing '{{.name}}' cluster using: '{{.delcommand}}', or start the existing '{{.name}}' cluster using: '{{.command}} --driver={{.old}}'",
		out.V{
			"name":       existing.Name,
			"new":        requested,
			"old":        old,
			"command":    mustload.ExampleCmd(existing.Name, "start"),
			"delcommand": mustload.ExampleCmd(existing.Name, "delete"),
		},
	)
}

// validateDriver validates that the selected driver appears sane, exits if not
func validateDriver(ds registry.DriverState, existing *config.ClusterConfig) {
	name := ds.Name
	os := detect.RuntimeOS()
	arch := detect.RuntimeArch()
	klog.Infof("validating driver %q against %+v", name, existing)
	if !driver.Supported(name) {
		exit.Message(reason.DrvUnsupportedOS, "The driver '{{.driver}}' is not supported on {{.os}}/{{.arch}}", out.V{"driver": name, "os": os, "arch": arch})
	}

	// if we are only downloading artifacts for a driver, we can stop validation here
	if viper.GetBool("download-only") {
		return
	}

	st := ds.State
	klog.Infof("status for %s: %+v", name, st)

	if st.NeedsImprovement {
		out.Styled(style.Improvement, `For improved {{.driver}} performance, {{.fix}}`, out.V{"driver": driver.FullName(ds.Name), "fix": translate.T(st.Fix)})
	}

	if ds.Priority == registry.Obsolete {
		exit.Message(reason.Kind{
			ID:       fmt.Sprintf("PROVIDER_%s_OBSOLETE", strings.ToUpper(name)),
			Advice:   translate.T(st.Fix),
			ExitCode: reason.ExProviderUnsupported,
			URL:      st.Doc,
			Style:    style.Shrug,
		}, st.Error.Error())
	}

	if st.Error == nil {
		return
	}

	r := reason.MatchKnownIssue(reason.Kind{}, st.Error, runtime.GOOS)
	if r != nil && r.ID != "" {
		exitIfNotForced(*r, st.Error.Error())
	}

	if !st.Installed {
		exit.Message(reason.Kind{
			ID:       fmt.Sprintf("PROVIDER_%s_NOT_FOUND", strings.ToUpper(name)),
			Advice:   translate.T(st.Fix),
			ExitCode: reason.ExProviderNotFound,
			URL:      st.Doc,
			Style:    style.Shrug,
		}, `The '{{.driver}}' provider was not found: {{.error}}`, out.V{"driver": name, "error": st.Error})
	}

	id := st.Reason
	if id == "" {
		id = fmt.Sprintf("PROVIDER_%s_ERROR", strings.ToUpper(name))
	}

	code := reason.ExProviderUnavailable

	if !st.Running {
		id = fmt.Sprintf("PROVIDER_%s_NOT_RUNNING", strings.ToUpper(name))
		code = reason.ExProviderNotRunning
	}

	exitIfNotForced(reason.Kind{
		ID:       id,
		Advice:   translate.T(st.Fix),
		ExitCode: code,
		URL:      st.Doc,
		Style:    style.Fatal,
	}, st.Error.Error())
}

func selectImageRepository(mirrorCountry string, v semver.Version) (bool, string, error) {
	var tryCountries []string
	var fallback string
	klog.Infof("selecting image repository for country %s ...", mirrorCountry)

	if mirrorCountry != "" {
		localRepos, ok := constants.ImageRepositories[mirrorCountry]
		if !ok || len(localRepos) == 0 {
			return false, "", fmt.Errorf("invalid image mirror country code: %s", mirrorCountry)
		}

		tryCountries = append(tryCountries, mirrorCountry)

		// we'll use the first repository as fallback
		// when none of the mirrors in the given location is available
		fallback = localRepos[0]

	} else {
		// always make sure global is preferred
		tryCountries = append(tryCountries, "global")
		for k := range constants.ImageRepositories {
			if strings.ToLower(k) != "global" {
				tryCountries = append(tryCountries, k)
			}
		}
	}

	for _, code := range tryCountries {
		localRepos := constants.ImageRepositories[code]
		for _, repo := range localRepos {
			err := checkRepository(v, repo)
			if err == nil {
				return true, repo, nil
			}
		}
	}

	return false, fallback, nil
}

var checkRepository = func(v semver.Version, repo string) error {
	pauseImage := images.Pause(v, repo)
	ref, err := name.ParseReference(pauseImage, name.WeakValidation)
	if err != nil {
		return err
	}

	_, err = remote.Image(ref, remote.WithAuthFromKeychain(authn.DefaultKeychain))
	return err
}

// validateUser validates minikube is run by the recommended user (privileged or regular)
func validateUser(drvName string) {
	u, err := user.Current()
	if err != nil {
		klog.Errorf("Error getting the current user: %v", err)
		return
	}

	useForce := viper.GetBool(force)

	// None driver works with root and without root on Linux
	if runtime.GOOS == "linux" && drvName == driver.None {
		if !viper.GetBool(interactive) {
			test := exec.Command("sudo", "-n", "echo", "-n")
			if err := test.Run(); err != nil {
				exit.Message(reason.DrvNeedsRoot, `sudo requires a password, and --interactive=false`)
			}
		}
		return
	}

	// If we are not root, exit early
	if u.Uid != "0" {
		return
	}

	out.ErrT(style.Stopped, `The "{{.driver_name}}" driver should not be used with root privileges.`, out.V{"driver_name": drvName})
	out.ErrT(style.Tip, "If you are running minikube within a VM, consider using --driver=none:")
	out.ErrT(style.Documentation, "  {{.url}}", out.V{"url": "https://minikube.sigs.k8s.io/docs/reference/drivers/none/"})

	cname := ClusterFlagValue()
	_, err = config.Load(cname)
	if err == nil || !config.IsNotExist(err) {
		out.ErrT(style.Tip, "Tip: To remove this root owned cluster, run: sudo {{.cmd}}", out.V{"cmd": mustload.ExampleCmd(cname, "delete")})
	}

	if !useForce {
		exit.Message(reason.DrvAsRoot, `The "{{.driver_name}}" driver should not be used with root privileges.`, out.V{"driver_name": drvName})
	}
}

// memoryLimits returns the amount of memory allocated to the system and hypervisor, the return value is in MiB
func memoryLimits(drvName string) (int, int, error) {
	info, cpuErr, memErr, diskErr := machine.LocalHostInfo()
	if cpuErr != nil {
		klog.Warningf("could not get system cpu info while verifying memory limits, which might be okay: %v", cpuErr)
	}
	if diskErr != nil {
		klog.Warningf("could not get system disk info while verifying memory limits, which might be okay: %v", diskErr)
	}

	if memErr != nil {
		return -1, -1, memErr
	}

	sysLimit := int(info.Memory)
	containerLimit := 0

	if driver.IsKIC(drvName) {
		s, err := oci.CachedDaemonInfo(drvName)
		if err != nil {
			return -1, -1, err
		}
		containerLimit = util.ConvertBytesToMB(s.TotalMemory)
	}

	return sysLimit, containerLimit, nil
}

// suggestMemoryAllocation calculates the default memory footprint in MiB
func suggestMemoryAllocation(sysLimit int, containerLimit int, nodes int) int {
	if mem := viper.GetInt(memory); mem != 0 {
		return mem
	}
	fallback := 2200
	maximum := 6000

	if sysLimit > 0 && fallback > sysLimit {
		return sysLimit
	}

	// If there are container limits, add tiny bit of slack for non-minikube components
	if containerLimit > 0 {
		if fallback > containerLimit {
			return containerLimit
		}
		maximum = containerLimit - 48
	}

	// Suggest 25% of RAM, rounded to nearest 100MB. Hyper-V requires an even number!
	suggested := int(float32(sysLimit)/400.0) * 100

	if nodes > 1 {
		suggested /= nodes
	}

	if suggested > maximum {
		return maximum
	}

	if suggested < fallback {
		return fallback
	}

	return suggested
}

// validateRequestedMemorySize validates the memory size matches the minimum recommended
func validateRequestedMemorySize(req int, drvName string) {
	// TODO: Fix MB vs MiB confusion
	sysLimit, containerLimit, err := memoryLimits(drvName)
	if err != nil {
		klog.Warningf("Unable to query memory limits: %v", err)
	}

	// Detect if their system doesn't have enough memory to work with.
	if driver.IsKIC(drvName) && containerLimit < minUsableMem {
		if driver.IsDockerDesktop(drvName) {
			if runtime.GOOS == "darwin" {
				exitIfNotForced(reason.RsrcInsufficientDarwinDockerMemory, "Docker Desktop only has {{.size}}MiB available, less than the required {{.req}}MiB for Kubernetes", out.V{"size": containerLimit, "req": minUsableMem, "recommend": "2.25 GB"})
			} else {
				exitIfNotForced(reason.RsrcInsufficientWindowsDockerMemory, "Docker Desktop only has {{.size}}MiB available, less than the required {{.req}}MiB for Kubernetes", out.V{"size": containerLimit, "req": minUsableMem, "recommend": "2.25 GB"})
			}
		}
		exitIfNotForced(reason.RsrcInsufficientContainerMemory, "{{.driver}} only has {{.size}}MiB available, less than the required {{.req}}MiB for Kubernetes", out.V{"size": containerLimit, "driver": drvName, "req": minUsableMem})
	}

	if sysLimit < minUsableMem {
		exitIfNotForced(reason.RsrcInsufficientSysMemory, "System only has {{.size}}MiB available, less than the required {{.req}}MiB for Kubernetes", out.V{"size": sysLimit, "req": minUsableMem})
	}

	if req < minUsableMem {
		exitIfNotForced(reason.RsrcInsufficientReqMemory, "Requested memory allocation {{.requested}}MiB is less than the usable minimum of {{.minimum_memory}}MB", out.V{"requested": req, "minimum_memory": minUsableMem})
	}
	if req < minRecommendedMem {
		if driver.IsDockerDesktop(drvName) {
			if runtime.GOOS == "darwin" {
				out.WarnReason(reason.RsrcInsufficientDarwinDockerMemory, "Docker Desktop only has {{.size}}MiB available, you may encounter application deployment failures.", out.V{"size": containerLimit, "req": minUsableMem, "recommend": "2.25 GB"})
			} else {
				out.WarnReason(reason.RsrcInsufficientWindowsDockerMemory, "Docker Desktop only has {{.size}}MiB available, you may encounter application deployment failures.", out.V{"size": containerLimit, "req": minUsableMem, "recommend": "2.25 GB"})
			}
		} else {
			out.WarnReason(reason.RsrcInsufficientReqMemory, "Requested memory allocation ({{.requested}}MB) is less than the recommended minimum {{.recommend}}MB. Deployments may fail.", out.V{"requested": req, "recommend": minRecommendedMem})
		}
	}

	advised := suggestMemoryAllocation(sysLimit, containerLimit, viper.GetInt(nodes))
	if req > sysLimit {
		exitIfNotForced(reason.Kind{ID: "RSRC_OVER_ALLOC_MEM", Advice: "Start minikube with less memory allocated: 'minikube start --memory={{.advised}}mb'"},
			`Requested memory allocation {{.requested}}MB is more than your system limit {{.system_limit}}MB.`,
			out.V{"requested": req, "system_limit": sysLimit, "advised": advised})
	}

	// Recommend 1GB to handle OS/VM overhead
	maxAdvised := sysLimit - 1024
	if req > maxAdvised {
		out.WarnReason(reason.Kind{ID: "RSRC_OVER_ALLOC_MEM", Advice: "Start minikube with less memory allocated: 'minikube start --memory={{.advised}}mb'"},
			`The requested memory allocation of {{.requested}}MiB does not leave room for system overhead (total system memory: {{.system_limit}}MiB). You may face stability issues.`,
			out.V{"requested": req, "system_limit": sysLimit, "advised": advised})
	}
}

// validateCPUCount validates the cpu count matches the minimum recommended & not exceeding the available cpu count
func validateCPUCount(drvName string) {
	var cpuCount int
	if driver.BareMetal(drvName) {

		// Uses the gopsutil cpu package to count the number of logical cpu cores
		ci, err := cpu.Counts(true)
		if err != nil {
			klog.Warningf("Unable to get CPU info: %v", err)
		} else {
			cpuCount = ci
		}
	} else {
		cpuCount = viper.GetInt(cpus)
	}

	if cpuCount < minimumCPUS {
		exitIfNotForced(reason.RsrcInsufficientCores, "Requested cpu count {{.requested_cpus}} is less than the minimum allowed of {{.minimum_cpus}}", out.V{"requested_cpus": cpuCount, "minimum_cpus": minimumCPUS})
	}

	if !driver.IsKIC((drvName)) {
		return
	}

	si, err := oci.CachedDaemonInfo(drvName)
	if err != nil {
		out.Styled(style.Confused, "Failed to verify '{{.driver_name}} info' will try again ...", out.V{"driver_name": drvName})
		si, err = oci.DaemonInfo(drvName)
		if err != nil {
			exit.Message(reason.Usage, "Ensure your {{.driver_name}} is running and is healthy.", out.V{"driver_name": driver.FullName(drvName)})
		}

	}

	if si.CPUs < cpuCount {

		if driver.IsDockerDesktop(drvName) {
			out.Styled(style.Empty, `- Ensure your {{.driver_name}} daemon has access to enough CPU/memory resources.`, out.V{"driver_name": drvName})
			if runtime.GOOS == "darwin" {
				out.Styled(style.Empty, `- Docs https://docs.docker.com/docker-for-mac/#resources`, out.V{"driver_name": drvName})
			}
			if runtime.GOOS == "windows" {
				out.String("\n\t")
				out.Styled(style.Empty, `- Docs https://docs.docker.com/docker-for-windows/#resources`, out.V{"driver_name": drvName})
			}
		}

		exitIfNotForced(reason.RsrcInsufficientCores, "Requested cpu count {{.requested_cpus}} is greater than the available cpus of {{.avail_cpus}}", out.V{"requested_cpus": cpuCount, "avail_cpus": si.CPUs})
	}

	// looks good
	if si.CPUs >= 2 {
		return
	}

	if drvName == oci.Docker && runtime.GOOS == "darwin" {
		exitIfNotForced(reason.RsrcInsufficientDarwinDockerCores, "Docker Desktop has less than 2 CPUs configured, but Kubernetes requires at least 2 to be available")
	} else if drvName == oci.Docker && runtime.GOOS == "windows" {
		exitIfNotForced(reason.RsrcInsufficientWindowsDockerCores, "Docker Desktop has less than 2 CPUs configured, but Kubernetes requires at least 2 to be available")
	} else {
		exitIfNotForced(reason.RsrcInsufficientCores, "{{.driver_name}} has less than 2 CPUs available, but Kubernetes requires at least 2 to be available", out.V{"driver_name": driver.FullName(viper.GetString("driver"))})
	}
}

// validateFlags validates the supplied flags against known bad combinations
func validateFlags(cmd *cobra.Command, drvName string) {
	if cmd.Flags().Changed(humanReadableDiskSize) {
		diskSizeMB, err := util.CalculateSizeInMB(viper.GetString(humanReadableDiskSize))
		if err != nil {
			exitIfNotForced(reason.Usage, "Validation unable to parse disk size '{{.diskSize}}': {{.error}}", out.V{"diskSize": viper.GetString(humanReadableDiskSize), "error": err})
		}

		if diskSizeMB < minimumDiskSize {
			exitIfNotForced(reason.RsrcInsufficientStorage, "Requested disk size {{.requested_size}} is less than minimum of {{.minimum_size}}", out.V{"requested_size": diskSizeMB, "minimum_size": minimumDiskSize})
		}
	}

	if cmd.Flags().Changed(cpus) {
		if !driver.HasResourceLimits(drvName) {
			out.WarningT("The '{{.name}}' driver does not respect the --cpus flag", out.V{"name": drvName})
		}
	}

	validateCPUCount(drvName)

	if cmd.Flags().Changed(memory) {
		validateChangedMemoryFlags(drvName)
	}

	if cmd.Flags().Changed(listenAddress) {
		validateListenAddress(viper.GetString(listenAddress))
	}

	if cmd.Flags().Changed(imageRepository) {
		viper.Set(imageRepository, validateImageRepository(viper.GetString(imageRepository)))
	}

	if cmd.Flags().Changed(containerRuntime) {
		runtime := strings.ToLower(viper.GetString(containerRuntime))

		validOptions := cruntime.ValidRuntimes()
		// `crio` is accepted as an alternative spelling to `cri-o`
		validOptions = append(validOptions, constants.CRIO)

		var validRuntime bool
		for _, option := range validOptions {
			if runtime == option {
				validRuntime = true
			}

			// Convert `cri-o` to `crio` as the K8s config uses the `crio` spelling
			if runtime == "cri-o" {
				viper.Set(containerRuntime, constants.CRIO)
			}
		}

		if !validRuntime {
			exit.Message(reason.Usage, `Invalid Container Runtime: "{{.runtime}}". Valid runtimes are: {{.validOptions}}`, out.V{"runtime": runtime, "validOptions": strings.Join(cruntime.ValidRuntimes(), ", ")})
		}
	}

	if driver.BareMetal(drvName) {
		if ClusterFlagValue() != constants.DefaultClusterName {
			exit.Message(reason.DrvUnsupportedProfile, "The '{{.name}} driver does not support multiple profiles: https://minikube.sigs.k8s.io/docs/reference/drivers/none/", out.V{"name": drvName})
		}

		runtime := viper.GetString(containerRuntime)
		if runtime != "docker" {
			out.WarningT("Using the '{{.runtime}}' runtime with the 'none' driver is an untested configuration!", out.V{"runtime": runtime})
		}

		// conntrack is required starting with Kubernetes 1.18, include the release candidates for completion
		version, _ := util.ParseKubernetesVersion(getKubernetesVersion(nil))
		if version.GTE(semver.MustParse("1.18.0-beta.1")) {
			if _, err := exec.LookPath("conntrack"); err != nil {
				exit.Message(reason.GuestMissingConntrack, "Sorry, Kubernetes {{.k8sVersion}} requires conntrack to be installed in root's path", out.V{"k8sVersion": version.String()})
			}
		}
	}

	if driver.IsSSH(drvName) {
		sshIPAddress := viper.GetString(sshIPAddress)
		if sshIPAddress == "" {
			exit.Message(reason.Usage, "No IP address provided. Try specifying --ssh-ip-address, or see https://minikube.sigs.k8s.io/docs/drivers/ssh/")
		}

		if net.ParseIP(sshIPAddress) == nil {
			_, err := net.LookupIP(sshIPAddress)
			if err != nil {
				exit.Error(reason.Usage, "Could not resolve IP address", err)
			}
		}
	}

	// validate kubeadm extra args
	if invalidOpts := bsutil.FindInvalidExtraConfigFlags(config.ExtraOptions); len(invalidOpts) > 0 {
		out.WarningT(
			"These --extra-config parameters are invalid: {{.invalid_extra_opts}}",
			out.V{"invalid_extra_opts": invalidOpts},
		)
		exit.Message(
			reason.Usage,
			"Valid components are: {{.valid_extra_opts}}",
			out.V{"valid_extra_opts": bsutil.KubeadmExtraConfigOpts},
		)
	}

	// check that kubeadm extra args contain only allowed parameters
	for param := range config.ExtraOptions.AsMap().Get(bsutil.Kubeadm) {
		if !config.ContainsParam(bsutil.KubeadmExtraArgsAllowed[bsutil.KubeadmCmdParam], param) &&
			!config.ContainsParam(bsutil.KubeadmExtraArgsAllowed[bsutil.KubeadmConfigParam], param) {
			exit.Message(reason.Usage, "Sorry, the kubeadm.{{.parameter_name}} parameter is currently not supported by --extra-config", out.V{"parameter_name": param})
		}
	}

	if outputFormat != "text" && outputFormat != "json" {
		exit.Message(reason.Usage, "Sorry, please set the --output flag to one of the following valid options: [text,json]")
	}

	validateRegistryMirror()
	validateInsecureRegistry()

}

// validateChangedMemoryFlags validates memory related flags.
func validateChangedMemoryFlags(drvName string) {
	if driver.IsKIC(drvName) && !oci.HasMemoryCgroup() {
		out.WarningT("Your cgroup does not allow setting memory.")
		out.Infof("More information: https://docs.docker.com/engine/install/linux-postinstall/#your-kernel-does-not-support-cgroup-swap-limit-capabilities")
	}
	if !driver.HasResourceLimits(drvName) {
		out.WarningT("The '{{.name}}' driver does not respect the --memory flag", out.V{"name": drvName})
	}
	req, err := util.CalculateSizeInMB(viper.GetString(memory))
	if err != nil {
		exitIfNotForced(reason.Usage, "Unable to parse memory '{{.memory}}': {{.error}}", out.V{"memory": viper.GetString(memory), "error": err})
	}
	validateRequestedMemorySize(req, drvName)
}

// This function validates if the --registry-mirror
// args match the format of http://localhost
func validateRegistryMirror() {
	if len(registryMirror) > 0 {
		for _, loc := range registryMirror {
			URL, err := url.Parse(loc)
			if err != nil {
				klog.Errorln("Error Parsing URL: ", err)
			}
			if (URL.Scheme != "http" && URL.Scheme != "https") || URL.Path != "" {
				exit.Message(reason.Usage, "Sorry, the url provided with the --registry-mirror flag is invalid: {{.url}}", out.V{"url": loc})
			}

		}
	}
}

// This function validates if the --image-repository
// args match the format of registry.cn-hangzhou.aliyuncs.com/google_containers
func validateImageRepository(imagRepo string) (vaildImageRepo string) {

	if strings.ToLower(imagRepo) == "auto" {
		vaildImageRepo = "auto"
	}
	URL, err := url.Parse(imagRepo)
	if err != nil {
		klog.Errorln("Error Parsing URL: ", err)
	}
	// tips when imagRepo ended with a trailing /.
	if strings.HasSuffix(imagRepo, "/") {
		out.Infof("The --image-repository flag your provided ended with a trailing / that could cause conflict in kuberentes, removed automatically")
	}
	// tips when imageRepo started with scheme.
	if URL.Scheme != "" {
		out.Infof("The --image-repository flag your provided contains Scheme: {{.scheme}}, it will be as a domian, removed automatically", out.V{"scheme": URL.Scheme})
	}

	vaildImageRepo = URL.Hostname() + strings.TrimSuffix(URL.Path, "/")
	return
}

// This function validates if the --listen-address
// match the format 0.0.0.0
func validateListenAddress(listenAddr string) {
	if len(listenAddr) > 0 && net.ParseIP(listenAddr) == nil {
		exit.Message(reason.Usage, "Sorry, the IP provided with the --listen-address flag is invalid: {{.listenAddr}}.", out.V{"listenAddr": listenAddr})
	}
}

// This function validates that the --insecure-registry follows one of the following formats:
// "<ip>[:<port>]" "<hostname>[:<port>]" "<network>/<netmask>"
func validateInsecureRegistry() {
	if len(insecureRegistry) > 0 {
		for _, addr := range insecureRegistry {
			// Remove http or https from registryMirror
			if strings.HasPrefix(strings.ToLower(addr), "http://") || strings.HasPrefix(strings.ToLower(addr), "https://") {
				i := strings.Index(addr, "//")
				addr = addr[i+2:]
			} else if strings.Contains(addr, "://") || strings.HasSuffix(addr, ":") {
				exit.Message(reason.Usage, "Sorry, the address provided with the --insecure-registry flag is invalid: {{.addr}}. Expected formats are: <ip>[:<port>], <hostname>[:<port>] or <network>/<netmask>", out.V{"addr": addr})
			}
			hostnameOrIP, port, err := net.SplitHostPort(addr)
			if err != nil {
				_, _, err := net.ParseCIDR(addr)
				if err == nil {
					continue
				}
				hostnameOrIP = addr
			}
			if !hostRe.MatchString(hostnameOrIP) && net.ParseIP(hostnameOrIP) == nil {
				//		fmt.Printf("This is not hostname or ip %s", hostnameOrIP)
				exit.Message(reason.Usage, "Sorry, the address provided with the --insecure-registry flag is invalid: {{.addr}}. Expected formats are: <ip>[:<port>], <hostname>[:<port>] or <network>/<netmask>", out.V{"addr": addr})
			}
			if port != "" {
				v, err := strconv.Atoi(port)
				if err != nil {
					exit.Message(reason.Usage, "Sorry, the address provided with the --insecure-registry flag is invalid: {{.addr}}. Expected formats are: <ip>[:<port>], <hostname>[:<port>] or <network>/<netmask>", out.V{"addr": addr})
				}
				if v < 0 || v > 65535 {
					exit.Message(reason.Usage, "Sorry, the address provided with the --insecure-registry flag is invalid: {{.addr}}. Expected formats are: <ip>[:<port>], <hostname>[:<port>] or <network>/<netmask>", out.V{"addr": addr})
				}
			}
		}
	}
}

func createNode(cc config.ClusterConfig, kubeNodeName string, existing *config.ClusterConfig) (config.ClusterConfig, config.Node, error) {
	// Create the initial node, which will necessarily be a control plane
	if existing != nil {
		cp, err := config.PrimaryControlPlane(existing)
		cp.KubernetesVersion = getKubernetesVersion(&cc)
		if err != nil {
			return cc, config.Node{}, err
		}

		// Make sure that existing nodes honor if KubernetesVersion gets specified on restart
		// KubernetesVersion is the only attribute that the user can override in the Node object
		nodes := []config.Node{}
		for _, n := range existing.Nodes {
			n.KubernetesVersion = getKubernetesVersion(&cc)
			nodes = append(nodes, n)
		}
		cc.Nodes = nodes

		return cc, cp, nil
	}

	cp := config.Node{
		Port:              cc.KubernetesConfig.NodePort,
		KubernetesVersion: getKubernetesVersion(&cc),
		Name:              kubeNodeName,
		ControlPlane:      true,
		Worker:            true,
	}
	cc.Nodes = []config.Node{cp}
	return cc, cp, nil
}

// autoSetDriverOptions sets the options needed for specific driver automatically.
func autoSetDriverOptions(cmd *cobra.Command, drvName string) (err error) {
	err = nil
	hints := driver.FlagDefaults(drvName)
	if len(hints.ExtraOptions) > 0 {
		for _, eo := range hints.ExtraOptions {
			if config.ExtraOptions.Exists(eo) {
				klog.Infof("skipping extra-config %q.", eo)
				continue
			}
			klog.Infof("auto setting extra-config to %q.", eo)
			err = config.ExtraOptions.Set(eo)
			if err != nil {
				err = errors.Wrapf(err, "setting extra option %s", eo)
			}
		}
	}

	if !cmd.Flags().Changed(cacheImages) {
		viper.Set(cacheImages, hints.CacheImages)
	}

	if !cmd.Flags().Changed(containerRuntime) && hints.ContainerRuntime != "" {
		viper.Set(containerRuntime, hints.ContainerRuntime)
		klog.Infof("auto set %s to %q.", containerRuntime, hints.ContainerRuntime)
	}

	if !cmd.Flags().Changed(cmdcfg.Bootstrapper) && hints.Bootstrapper != "" {
		viper.Set(cmdcfg.Bootstrapper, hints.Bootstrapper)
		klog.Infof("auto set %s to %q.", cmdcfg.Bootstrapper, hints.Bootstrapper)

	}

	return err
}

// validateKubernetesVersion ensures that the requested version is reasonable
func validateKubernetesVersion(old *config.ClusterConfig) {
	nvs, _ := semver.Make(strings.TrimPrefix(getKubernetesVersion(old), version.VersionPrefix))

	oldestVersion, err := semver.Make(strings.TrimPrefix(constants.OldestKubernetesVersion, version.VersionPrefix))
	if err != nil {
		exit.Message(reason.InternalSemverParse, "Unable to parse oldest Kubernetes version from constants: {{.error}}", out.V{"error": err})
	}
	defaultVersion, err := semver.Make(strings.TrimPrefix(constants.DefaultKubernetesVersion, version.VersionPrefix))
	if err != nil {
		exit.Message(reason.InternalSemverParse, "Unable to parse default Kubernetes version from constants: {{.error}}", out.V{"error": err})
	}

	if nvs.LT(oldestVersion) {
		out.WarningT("Specified Kubernetes version {{.specified}} is less than the oldest supported version: {{.oldest}}", out.V{"specified": nvs, "oldest": constants.OldestKubernetesVersion})
		if !viper.GetBool(force) {
			out.WarningT("You can force an unsupported Kubernetes version via the --force flag")
		}
		exitIfNotForced(reason.KubernetesTooOld, "Kubernetes {{.version}} is not supported by this release of minikube", out.V{"version": nvs})
	}

	// If the version of Kubernetes has a known issue, print a warning out to the screen
	if issue := reason.ProblematicK8sVersion(nvs); issue != nil {
		out.WarningT(issue.Description, out.V{"version": nvs.String()})
		if issue.URL != "" {
			out.WarningT("For more information, see: {{.url}}", out.V{"url": issue.URL})
		}
	}

	if old == nil || old.KubernetesConfig.KubernetesVersion == "" {
		return
	}

	ovs, err := semver.Make(strings.TrimPrefix(old.KubernetesConfig.KubernetesVersion, version.VersionPrefix))
	if err != nil {
		klog.Errorf("Error parsing old version %q: %v", old.KubernetesConfig.KubernetesVersion, err)
	}

	if nvs.LT(ovs) {
		profileArg := ""
		if old.Name != constants.DefaultClusterName {
			profileArg = fmt.Sprintf(" -p %s", old.Name)
		}

		suggestedName := old.Name + "2"
		exit.Message(reason.KubernetesDowngrade, "Unable to safely downgrade existing Kubernetes v{{.old}} cluster to v{{.new}}",
			out.V{"prefix": version.VersionPrefix, "new": nvs, "old": ovs, "profile": profileArg, "suggestedName": suggestedName})

	}
	if defaultVersion.GT(nvs) {
		out.Styled(style.New, "Kubernetes {{.new}} is now available. If you would like to upgrade, specify: --kubernetes-version={{.prefix}}{{.new}}", out.V{"prefix": version.VersionPrefix, "new": defaultVersion})
	}
}

func isBaseImageApplicable(drv string) bool {
	return registry.IsKIC(drv)
}

func getKubernetesVersion(old *config.ClusterConfig) string {
	paramVersion := viper.GetString(kubernetesVersion)

	// try to load the old version first if the user didn't specify anything
	if paramVersion == "" && old != nil {
		paramVersion = old.KubernetesConfig.KubernetesVersion
	}

	if paramVersion == "" || strings.EqualFold(paramVersion, "stable") {
		paramVersion = constants.DefaultKubernetesVersion
	} else if strings.EqualFold(paramVersion, "latest") {
		paramVersion = constants.NewestKubernetesVersion
	}

	nvs, err := semver.Make(strings.TrimPrefix(paramVersion, version.VersionPrefix))
	if err != nil {
		exit.Message(reason.Usage, `Unable to parse "{{.kubernetes_version}}": {{.error}}`, out.V{"kubernetes_version": paramVersion, "error": err})
	}

	return version.VersionPrefix + nvs.String()
}

// validateDockerStorageDriver checks that docker is using overlay2
// if not, set preload=false (see #7626)
func validateDockerStorageDriver(drvName string) {
	if !driver.IsKIC(drvName) {
		return
	}
	if _, err := exec.LookPath(drvName); err != nil {
		exit.Error(reason.DrvNotFound, fmt.Sprintf("%s not found on PATH", drvName), err)
	}
	si, err := oci.DaemonInfo(drvName)
	if err != nil {
		klog.Warningf("Unable to confirm that %s is using overlay2 storage driver; setting preload=false", drvName)
		viper.Set(preload, false)
		return
	}
	if si.StorageDriver == "overlay2" {
		return
	}
	out.WarningT("{{.Driver}} is currently using the {{.StorageDriver}} storage driver, consider switching to overlay2 for better performance", out.V{"StorageDriver": si.StorageDriver, "Driver": drvName})
	viper.Set(preload, false)
}

func exitIfNotForced(r reason.Kind, message string, v ...out.V) {
	if !viper.GetBool(force) {
		exit.Message(r, message, v...)
	}
	out.Error(r, message, v...)
}

func exitGuestProvision(err error) {
	if errors.Cause(err) == oci.ErrInsufficientDockerStorage {
		exit.Message(reason.RsrcInsufficientDockerStorage, "preload extraction failed: \"No space left on device\"")
	}
	exit.Error(reason.GuestProvision, "error provisioning host", err)
}
