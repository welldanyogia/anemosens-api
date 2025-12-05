# VPS Setup Guide (Ubuntu 20.04/22.04 LTS)

This guide covers the initial configuration of a fresh VPS before deploying the AneMoSense API.

## 1. Initial Access & User Setup
Login to your fresh VPS as root:
```bash
ssh root@103.23.199.172
```

Create a new sudouser (replace `anemo` with your preferred username):
```bash
adduser anemo
# Enter password and skip details with ENTER
usermod -aG sudo anemo
```

## 2. Security Configuration

### SSH Hardening
Copy your local SSH key to the new user (run this **from your local machine**, not the VPS):
```bash
ssh-copy-id anemo@103.23.199.172
```

Back on the VPS, disable root login and password authentication:
```bash
nano /etc/ssh/sshd_config
```
Find and change these lines:
```
PermitRootLogin no
PasswordAuthentication no
```
Restart SSH:
```bash
sudo systemctl restart ssh
```

### Firewall (UFW)
Allow SSH, HTTP, and HTTPS:
```bash
    sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
# Type 'y' to confirm
```

### Fail2Ban (Optional but Recommended)
Protect against brute-force attacks:
```bash
sudo apt update
sudo apt install fail2ban -y
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

## 3. Install Docker & Docker Compose

Uninstall old versions:
```bash
sudo apt remove docker docker-engine docker.io containerd runc
```

Install prerequisites:
```bash
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release
```

Add Docker's official GPG key:
```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

Set up the repository:
```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

Install Docker Engine:
```bash
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

Add your user to the docker group (avoid using sudo for docker commands):
```bash
sudo usermod -aG docker ${USER}
```
**Log out and log back in** for this to take effect.

## 4. Project Directory Setup

Create the directory structure expected by the deployment script:
```bash
mkdir -p ~/anemosens-api
cd ~/anemosens-api
```

## 5. Next Steps
Now your VPS is ready. Retrieve the `DEPLOY.md` file and proceed with the **Initial Setup** section to:
1. Clone the repo / copy files.
2. Generate SSL certificates.
3. Configure GitHub Secrets for CI/CD.
