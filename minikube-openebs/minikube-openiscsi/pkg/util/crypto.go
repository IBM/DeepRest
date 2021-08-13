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

package util

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io/ioutil"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"k8s.io/minikube/pkg/util/lock"
)

// GenerateCACert generates a CA certificate and RSA key for a common name
func GenerateCACert(certPath, keyPath string, name string) error {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return errors.Wrap(err, "Error generating rsa key")
	}

	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: name,
		},
		NotBefore: time.Now().Add(time.Hour * -24),
		NotAfter:  time.Now().Add(time.Hour * 24 * 365 * 10),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	return writeCertsAndKeys(&template, certPath, priv, keyPath, &template, priv)
}

// You may also specify additional subject alt names (either ip or dns names) for the certificate
// The certificate will be created with file mode 0644. The key will be created with file mode 0600.
// If the certificate or key files already exist, they will be overwritten.
// Any parent directories of the certPath or keyPath will be created as needed with file mode 0755.

// GenerateSignedCert generates a signed certificate and key
func GenerateSignedCert(certPath, keyPath, cn string, ips []net.IP, alternateDNS []string, signerCertPath, signerKeyPath string) error {
	klog.Infof("Generating cert %s with IP's: %s", certPath, ips)
	signerCertBytes, err := ioutil.ReadFile(signerCertPath)
	if err != nil {
		return errors.Wrap(err, "Error reading file: signerCertPath")
	}
	decodedSignerCert, _ := pem.Decode(signerCertBytes)
	if decodedSignerCert == nil {
		return errors.New("Unable to decode certificate")
	}
	signerCert, err := x509.ParseCertificate(decodedSignerCert.Bytes)
	if err != nil {
		return errors.Wrap(err, "Error parsing certificate: decodedSignerCert.Bytes")
	}
	signerKeyBytes, err := ioutil.ReadFile(signerKeyPath)
	if err != nil {
		return errors.Wrap(err, "Error reading file: signerKeyPath")
	}
	decodedSignerKey, _ := pem.Decode(signerKeyBytes)
	if decodedSignerKey == nil {
		return errors.New("Unable to decode key")
	}
	signerKey, err := x509.ParsePKCS1PrivateKey(decodedSignerKey.Bytes)
	if err != nil {
		return errors.Wrap(err, "Error parsing private key: decodedSignerKey.Bytes")
	}

	template := x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject: pkix.Name{
			CommonName:   cn,
			Organization: []string{"system:masters"},
		},
		NotBefore: time.Now().Add(time.Hour * -24),
		NotAfter:  time.Now().Add(time.Hour * 24 * 365),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
	}

	template.IPAddresses = append(template.IPAddresses, ips...)
	template.DNSNames = append(template.DNSNames, alternateDNS...)

	priv, err := loadOrGeneratePrivateKey(keyPath)
	if err != nil {
		return errors.Wrap(err, "Error loading or generating private key: keyPath")
	}

	return writeCertsAndKeys(&template, certPath, priv, keyPath, signerCert, signerKey)
}

func loadOrGeneratePrivateKey(keyPath string) (*rsa.PrivateKey, error) {
	keyBytes, err := ioutil.ReadFile(keyPath)
	if err == nil {
		decodedKey, _ := pem.Decode(keyBytes)
		if decodedKey != nil {
			priv, err := x509.ParsePKCS1PrivateKey(decodedKey.Bytes)
			if err == nil {
				return priv, nil
			}
		}
	}
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, errors.Wrap(err, "Error generating RSA key")
	}
	return priv, nil
}

func writeCertsAndKeys(template *x509.Certificate, certPath string, signeeKey *rsa.PrivateKey, keyPath string, parent *x509.Certificate, signingKey *rsa.PrivateKey) error {
	derBytes, err := x509.CreateCertificate(rand.Reader, template, parent, &signeeKey.PublicKey, signingKey)
	if err != nil {
		return errors.Wrap(err, "Error creating certificate")
	}

	certBuffer := bytes.Buffer{}
	if err := pem.Encode(&certBuffer, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes}); err != nil {
		return errors.Wrap(err, "Error encoding certificate")
	}

	keyBuffer := bytes.Buffer{}
	if err := pem.Encode(&keyBuffer, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(signeeKey)}); err != nil {
		return errors.Wrap(err, "Error encoding key")
	}

	if err := os.MkdirAll(filepath.Dir(certPath), os.FileMode(0755)); err != nil {
		return errors.Wrap(err, "Error creating certificate directory")
	}
	klog.Infof("Writing cert to %s ...", certPath)
	if err := lock.WriteFile(certPath, certBuffer.Bytes(), os.FileMode(0644)); err != nil {
		return errors.Wrap(err, "Error writing certificate to cert path")
	}

	if err := os.MkdirAll(filepath.Dir(keyPath), os.FileMode(0755)); err != nil {
		return errors.Wrap(err, "Error creating key directory")
	}
	klog.Infof("Writing key to %s ...", keyPath)
	if err := lock.WriteFile(keyPath, keyBuffer.Bytes(), os.FileMode(0600)); err != nil {
		return errors.Wrap(err, "Error writing key file")
	}

	return nil
}
