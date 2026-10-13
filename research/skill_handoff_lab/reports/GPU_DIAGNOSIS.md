# GPU diagnosis — read-only 2026-10-13

No repair, package operation, module unload, or reboot was performed.

| Item | Observed |
|---|---|
| Kernel | `6.8.0-138-generic` |
| Loaded NVIDIA module | `580.173.02` |
| Installed DKMS/userspace | `580.178.04` |
| `nvidia-smi` | fails to communicate with driver |
| DKMS for running kernel | `580.178.04`, installed for `6.8.0-138-generic` |
| RAM | 31 GiB total; about 8.6 GiB available at latest audit |
| Swap | 2 GiB total; about 2 GiB used (23 MiB free) |
| Root/data filesystem | 916 GiB; 772 GiB used; 98 GiB available (89%) |

## Diagnosis and minimum maintenance action

The loaded kernel module and installed DKMS/userspace versions differ. DKMS
does contain a 580.178.04 build for the running kernel. The least invasive
likely recovery is therefore a controlled maintenance reboot. If the
post-reboot module remains 580.173.02, align all NVIDIA driver packages and the
kernel module to one exact version during a maintenance window. Do not mix
580.173.02 and 580.178.04 libraries, and do not start by reinstalling CUDA.

Before any GPU oracle: verify `nvidia-smi`, driver/module equality, free RAM and
swap pressure, then run a tiny cuRobo import/one-segment qualification. Disk is
not yet the direct blocker but needs monitoring before Isaac cache growth.
