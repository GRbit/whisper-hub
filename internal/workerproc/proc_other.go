//go:build !linux

package workerproc

import "os/exec"

// setProcAttrs is a no-op outside Linux; the images are Linux only.
func setProcAttrs(*exec.Cmd) {}
