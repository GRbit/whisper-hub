package workerproc

import (
	"os/exec"
	"syscall"
)

// setProcAttrs puts the worker in its own process group, so a terminal's
// Ctrl-C reaches only the gateway, which then stops the worker last.
//
// No Pdeathsig: it fires when the starting OS thread exits, which the Go
// runtime may do at any time (golang/go#27505). In a container the kernel
// kills every process when PID 1 dies anyway.
func setProcAttrs(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}
